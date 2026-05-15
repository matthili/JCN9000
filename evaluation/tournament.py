"""Tournament-Runner: laesst mehrere Player-Typen gegeneinander spielen
und aggregiert Elo-Ratings und Team-Statistiken.

Modi:
- "two_teams": Spieler-A-Typ als Team 0+2, Spieler-B-Typ als Team 1+3 (Standard)
- "round_robin": Alle Player-Typen einer Liste paarweise gegeneinander (kommt spaeter)

Parallel-Variante:
- `two_team_match_parallel`: spaltet die Spiele auf N Worker-Prozesse auf.
  Jeder Worker laedt sein NN-Modell selbst (auf CPU, um GPU-Memory-Konflikte
  zwischen Workern zu vermeiden) und liefert TeamStats zurueck, die im
  Hauptprozess gemerged werden. Elo wird im Parallel-Modus nicht gepflegt
  (Elo-Updates sind iterative und nicht ohne Verlust mergeable).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jass_engine.player import Player
from jass_engine.variants.kreuz_jass import play_kreuz_jass
from evaluation.elo import EloRating
from evaluation.stats import TeamStats, update_stats_from_game


PlayerFactory = Callable[[int, random.Random], Player]


@dataclass
class TournamentResult:
    label_a: str
    label_b: str
    stats_a: TeamStats
    stats_b: TeamStats
    elo: EloRating
    games_played: int


def two_team_match(
    label_a: str,
    factory_a: PlayerFactory,
    label_b: str,
    factory_b: PlayerFactory,
    num_games: int,
    target_score: int = 1000,
    seed: int = 0,
    swap_seats_each_half: bool = True,
    elo: EloRating | None = None,
) -> TournamentResult:
    """Spielt N Partien Team-A vs Team-B.

    Args:
        label_a / label_b: Anzeigenamen
        factory_a / factory_b: Funktion, die einen Player erzeugt
        num_games: Gesamtanzahl Partien
        target_score: Punkteziel pro Partie (Default 1000)
        seed: RNG-Seed fuer Reproduzierbarkeit
        swap_seats_each_half: Tauscht in der zweiten Haelfte die Sitzplaetze,
            damit moegliche Sitz-Boni egalisiert sind
        elo: optionales bestehendes Elo, wird mit den Resultaten aktualisiert
    """
    rng = random.Random(seed)
    stats_a = TeamStats()
    stats_b = TeamStats()
    if elo is None:
        elo = EloRating()

    half = num_games // 2 if swap_seats_each_half else num_games

    for game_idx in range(num_games):
        if swap_seats_each_half and game_idx >= half:
            # In der zweiten Haelfte: Team A sitzt auf 1+3 statt 0+2
            seat_to_factory = {0: factory_b, 1: factory_a, 2: factory_b, 3: factory_a}
            team_a_team_id = 1
            team_b_team_id = 0
        else:
            seat_to_factory = {0: factory_a, 1: factory_b, 2: factory_a, 3: factory_b}
            team_a_team_id = 0
            team_b_team_id = 1

        players = [
            seat_to_factory[i](i, random.Random(rng.randint(0, 10**9)))
            for i in range(4)
        ]
        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )

        update_stats_from_game(
            stats_a, stats_b, game,
            team_a_id=team_a_team_id, team_b_id=team_b_team_id,
        )

        # Elo-Update: zwei Instanzen pro Team
        score_a = game.final_scores.get(team_a_team_id, 0)
        score_b = game.final_scores.get(team_b_team_id, 0)
        if score_a > score_b:
            elo_score_a = 1.0
        elif score_b > score_a:
            elo_score_a = 0.0
        else:
            elo_score_a = 0.5
        elo.update_team_match(
            team_a_players=[label_a, label_a],
            team_b_players=[label_b, label_b],
            score_a=elo_score_a,
        )

    return TournamentResult(
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a,
        stats_b=stats_b,
        elo=elo,
        games_played=num_games,
    )


# ---------- Parallel-Variante ----------
#
# Die Worker-Funktion muss auf Modul-Ebene definiert sein, damit
# multiprocessing sie pickeln kann. Sie erhaelt die "Spec" der Factories
# (Kind + Modell-Pfad) und baut sie selbst auf -- Closures aus _make_factory
# in run_eval.py sind nicht ueber Process-Grenzen serialisierbar.


def _build_factory_in_worker(kind: str, model_path: str | None) -> PlayerFactory:
    """Baut eine PlayerFactory im Worker. CPU-only TF, kein GPU-Konflikt
    zwischen parallelen Workern."""
    if kind == "random":
        from players.random_player import RandomPlayer
        def _f(seat: int, rng: random.Random) -> Player:
            return RandomPlayer(name=f"R{seat}", rng=rng)
        return _f
    if kind == "heuristic":
        from players.heuristic_player import HeuristicPlayer
        def _f(seat: int, rng: random.Random) -> Player:
            return HeuristicPlayer(name=f"H{seat}", rng=rng)
        return _f
    if kind == "nn":
        if model_path is None:
            raise ValueError("NN-Player im Worker braucht model_path")
        # GPU im Worker komplett deaktivieren -- bei N parallelen Workers
        # wuerden sonst N CUDA-Contexte gleichzeitig VRAM allokieren und
        # entweder OOM machen oder einander blockieren. Inference auf CPU
        # ist bei dem kleinen Modell schnell genug (~5ms statt 1ms), und
        # 16 CPU-Inferences parallel schlagen 1 GPU-Inference seriell.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        from players.nn_player import NNPlayer
        from players.heuristic_player import HeuristicPlayer
        from jass_engine.player import Player as _P
        shared_model = keras.models.load_model(model_path)

        def _f(seat: int, rng: random.Random) -> Player:
            class _FastNN(NNPlayer):
                def __init__(self, name: str):
                    _P.__init__(self, name)
                    self.model = shared_model
                    self.fallback = HeuristicPlayer(name + "_fb")
                    self.greedy = True
            return _FastNN(name=f"NN{seat}")
        return _f
    raise ValueError(f"Unbekannter Player-Typ im Worker: {kind}")


def _eval_worker(args: tuple) -> tuple[TeamStats, TeamStats]:
    """Top-level-Funktion fuer multiprocessing.Pool. Spielt einen Batch
    Partien und gibt (stats_a, stats_b) zurueck."""
    (
        kind_a, model_a, kind_b, model_b,
        num_games, target_score, seed, swap_seats_each_half,
        label_a, label_b,
    ) = args
    factory_a = _build_factory_in_worker(kind_a, model_a)
    factory_b = _build_factory_in_worker(kind_b, model_b)
    result = two_team_match(
        label_a=label_a,
        factory_a=factory_a,
        label_b=label_b,
        factory_b=factory_b,
        num_games=num_games,
        target_score=target_score,
        seed=seed,
        swap_seats_each_half=swap_seats_each_half,
        elo=EloRating(),  # Worker-Elo wird verworfen
    )
    return result.stats_a, result.stats_b


def two_team_match_parallel(
    label_a: str,
    kind_a: str,
    model_a: Path | None,
    label_b: str,
    kind_b: str,
    model_b: Path | None,
    num_games: int,
    workers: int,
    target_score: int = 1000,
    seed: int = 0,
    swap_seats_each_half: bool = True,
) -> TournamentResult:
    """Parallel-Variante von two_team_match.

    Spiele werden in `workers` Batches geteilt; jeder Batch laeuft in einem
    separaten Subprocess (spawn-Context, CPU-only TF). Stats werden im
    Hauptprozess zu einem TournamentResult zusammengefuehrt.

    Wichtig: Elo-Update entfaellt im Parallel-Modus, weil Elo iterative
    Spiel-fuer-Spiel-Updates braucht und das nicht ohne Datenverlust ueber
    Worker-Grenzen aggregierbar ist. Wenn du Elo brauchst, lass workers=1.
    """
    # Spiele aufteilen
    base = num_games // workers
    remainder = num_games % workers
    batch_sizes = [base + (1 if i < remainder else 0) for i in range(workers)]
    # Pro Worker eindeutiger Seed, abgeleitet vom Master-Seed
    seeds = [seed + i * 10_000_003 for i in range(workers)]

    tasks = []
    for i, (batch, sub_seed) in enumerate(zip(batch_sizes, seeds)):
        if batch == 0:
            continue
        tasks.append((
            kind_a, str(model_a) if model_a else None,
            kind_b, str(model_b) if model_b else None,
            batch, target_score, sub_seed, swap_seats_each_half,
            label_a, label_b,
        ))

    # spawn-Context: jeder Worker startet einen frischen Python-Interpreter.
    # Notwendig, weil TF nicht fork-safe ist, und damit CUDA_VISIBLE_DEVICES=-1
    # im Worker tatsaechlich greift.
    ctx = mp.get_context("spawn")
    stats_a_total = TeamStats()
    stats_b_total = TeamStats()

    with ctx.Pool(processes=len(tasks)) as pool:
        for sa, sb in pool.imap_unordered(_eval_worker, tasks):
            stats_a_total.merge(sa)
            stats_b_total.merge(sb)

    return TournamentResult(
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a_total,
        stats_b=stats_b_total,
        elo=EloRating(),  # leeres Elo im Parallel-Modus
        games_played=num_games,
    )


def format_tournament_summary(res: TournamentResult) -> str:
    """Kompakte Zusammenfassung als Konsolen-Text."""
    from evaluation.stats import format_stats_table

    lines = []
    lines.append(f"Tournament: {res.label_a}  vs.  {res.label_b}")
    lines.append(f"Partien: {res.games_played}")
    lines.append(format_stats_table(res.label_a, res.stats_a, res.label_b, res.stats_b))
    lines.append("")
    lines.append("Elo-Leaderboard:")
    for name, rating, games in res.elo.leaderboard():
        lines.append(f"  {name:<24} {rating:>7.1f}   (n={games})")
    return "\n".join(lines)
