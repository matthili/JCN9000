"""Batched-GPU-Eval: viele Spiele parallel in einem Process, GPU macht
die NN-Inferenzen als Batch.

Architektur analog zu training/rl/batched_selfplay.py:
- Modelle werden EINMAL geladen (auf GPU)
- Pro NN-Team ein InferenceServer (Thread) im Hauptprozess
- N parallele Game-Threads spielen die Eval-Spiele
- BatchedEvalNNPlayer routet seine choose_card-Anfragen via Server

Im Vergleich zum Multi-Process-Eval (run_eval mit --workers > 1):
- Eval mit `--inference-mode batched-gpu` ist 5-10x schneller bei 2000 Spielen,
  weil pro Tick 32-64 Inferenzen als Batch durchgehen statt einzeln auf CPU.
- Nutzt die schon geladenen Modelle auf der GPU (kein Pro-Worker-Reload).

Limitation:
- Threading mit GIL: 64 Game-Threads konkurrieren um den GIL fuer die
  Python-Spiellogik. Aber: choose_card-Anfragen blockieren auf dem Server
  und geben dabei den GIL frei, sodass andere Threads weiterarbeiten koennen.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluation.elo import EloRating
from evaluation.stats import TeamStats, update_stats_from_game
from evaluation.tournament import TournamentResult
from jass_engine.game import GameResult
from jass_engine.player import GameState, Player
from jass_engine.variants.kreuz_jass import play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


class BatchedEvalNNPlayer(Player):
    """Greedy NN-Player fuer Eval, der einen InferenceServer benutzt.

    Anders als RLPlayer im Self-Play wird hier keine Trajectory aufgezeichnet
    und es wird greedy (argmax) statt stochastisch gespielt.
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        fallback: Player | None = None,
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.fallback = fallback or HeuristicPlayer(name + "_fb")

    def choose_announcement(self, hand, round_idx, can_push):
        return self.fallback.choose_announcement(hand, round_idx, can_push)

    def announce_weise(self, hand, variant, possible_weise):
        return self.fallback.announce_weise(hand, variant, possible_weise)

    def choose_card(self, hand, state: GameState):
        x = encode_state(hand, state).astype(np.float32)
        mask = legal_action_mask(hand, state).astype(np.float32)

        policy, _value = self.inference_server.request(x, mask)

        # Greedy: höchste Wahrscheinlichkeit aus den legalen Aktionen
        legal_policy = policy * mask
        s = legal_policy.sum()
        if s <= 0:
            # Sehr seltener Fall (Modell hat alle legalen Karten auf ~0 gedrückt)
            return self.fallback.choose_card(hand, state)
        action_idx = int(np.argmax(legal_policy))
        return index_to_card(action_idx)


# ----- Player-Factories pro Spieler-Typ -----


def _make_player_for_kind(
    kind: str,
    seat: int,
    server: InferenceServer | None,
    rng: random.Random,
) -> Player:
    if kind == "random":
        return RandomPlayer(name=f"R{seat}", rng=rng)
    if kind == "heuristic":
        return HeuristicPlayer(name=f"H{seat}", rng=rng)
    if kind == "nn":
        assert server is not None, "NN-Player braucht einen InferenceServer"
        return BatchedEvalNNPlayer(name=f"NN{seat}", inference_server=server)
    raise ValueError(f"Unbekannter Player-Kind: {kind!r}")


# ----- Game-Setup pro Partie + Result-Sammlung -----


def _play_one_game(
    game_idx: int,
    kind_a: str, server_a: InferenceServer | None,
    kind_b: str, server_b: InferenceServer | None,
    target_score: int,
    sub_seed: int,
    swap_seats: bool,
) -> tuple[GameResult, int, int]:
    """Spielt EINE Partie, returnt (GameResult, team_a_id, team_b_id)."""
    rng = random.Random(sub_seed)

    if swap_seats:
        # Team A sitzt auf Sitz 1+3 (gegen den ueblichen 0+2)
        kinds_per_seat = [kind_b, kind_a, kind_b, kind_a]
        servers_per_seat = [server_b, server_a, server_b, server_a]
        team_a_id, team_b_id = 1, 0
    else:
        kinds_per_seat = [kind_a, kind_b, kind_a, kind_b]
        servers_per_seat = [server_a, server_b, server_a, server_b]
        team_a_id, team_b_id = 0, 1

    players = [
        _make_player_for_kind(
            kind=kinds_per_seat[seat],
            seat=seat,
            server=servers_per_seat[seat],
            rng=random.Random(rng.randint(0, 10**9)),
        )
        for seat in range(4)
    ]

    game = play_kreuz_jass(
        players,
        target_score=target_score,
        rng=random.Random(rng.randint(0, 10**9)),
    )
    return game, team_a_id, team_b_id


# ----- Top-level: Two-Team-Match mit GPU-Batching -----


def two_team_match_batched_gpu(
    label_a: str,
    kind_a: str,
    model_a: Path | None,
    label_b: str,
    kind_b: str,
    model_b: Path | None,
    num_games: int,
    target_score: int = 1000,
    seed: int = 0,
    swap_seats_each_half: bool = True,
    inference_batch_size: int = 64,
    parallel_threads: int = 128,
    paired_eval: bool = False,
) -> TournamentResult:
    """Eval mit Threading + GPU-Inferenz-Server.

    Args:
        label_a / label_b: Anzeigenamen.
        kind_a / kind_b: "random" | "heuristic" | "nn".
        model_a / model_b: Modell-Pfade (nur wenn entsprechender kind == "nn").
        num_games: Anzahl Eval-Partien.
        target_score: Punkteziel pro Partie (Default 1000).
        seed: Master-Seed.
        swap_seats_each_half: in der zweiten Haelfte Team A und B tauschen.
        inference_batch_size: max. Batch im InferenceServer.
        parallel_threads: max. gleichzeitig spielende Game-Threads. Sollte
            ungefaehr inference_batch_size sein -- dann ist der Server gut
            ausgelastet, aber nicht ueberlaufen.

    Returns:
        TournamentResult mit aggregierten Stats (Elo wird nicht gepflegt --
        wie auch im Multi-Process-Eval, weil Elo iterative Updates braucht).
    """
    # Modelle laden + Server starten (nur fuer NN-Teams)
    server_a: InferenceServer | None = None
    server_b: InferenceServer | None = None

    if kind_a == "nn":
        if model_a is None:
            raise ValueError("kind_a='nn' braucht model_a")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401 -- Custom-Layer registrieren
        print(f"  Lade Modell A: {model_a}")
        model_a_obj = keras.models.load_model(str(model_a))
        server_a = InferenceServer(
            model_a_obj, max_batch_size=inference_batch_size
        )

    if kind_b == "nn":
        if model_b is None:
            raise ValueError("kind_b='nn' braucht model_b")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        print(f"  Lade Modell B: {model_b}")
        model_b_obj = keras.models.load_model(str(model_b))
        server_b = InferenceServer(
            model_b_obj, max_batch_size=inference_batch_size
        )

    try:
        rng = random.Random(seed)

        # Game-Jobs vorbereiten (mit deterministischem Seed pro Spiel).
        # Bei paired_eval=True: pro Paar zwei Jobs mit demselben Seed --
        # einmal swap=False, einmal swap=True. So sehen beide Modelle exakt
        # dieselbe Kartenverteilung, nur in vertauschten Sitzplaetzen.
        jobs = []
        if paired_eval:
            if num_games % 2 != 0:
                raise ValueError(
                    "paired_eval=True braucht eine gerade Zahl an Partien "
                    f"(num_games={num_games})."
                )
            for pair_idx in range(num_games // 2):
                pair_seed = rng.randint(0, 10**9)
                jobs.append({
                    "game_idx": pair_idx * 2,
                    "swap_seats": False,
                    "sub_seed": pair_seed,
                })
                jobs.append({
                    "game_idx": pair_idx * 2 + 1,
                    "swap_seats": True,
                    "sub_seed": pair_seed,
                })
        else:
            half = num_games // 2 if swap_seats_each_half else num_games
            for game_idx in range(num_games):
                jobs.append({
                    "game_idx": game_idx,
                    "swap_seats": swap_seats_each_half and game_idx >= half,
                    "sub_seed": rng.randint(0, 10**9),
                })

        # Threads: Pool mit `parallel_threads` Workern, sammelt Results
        results: list[tuple[GameResult, int, int]] = [None] * num_games  # type: ignore
        start_time = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=parallel_threads, thread_name_prefix="EvalGame"
        ) as pool:
            future_to_idx = {
                pool.submit(
                    _play_one_game,
                    job["game_idx"],
                    kind_a, server_a,
                    kind_b, server_b,
                    target_score,
                    job["sub_seed"],
                    job["swap_seats"],
                ): job["game_idx"]
                for job in jobs
            }
            done_count = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                done_count += 1
                if done_count % max(1, num_games // 10) == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(
                        f"  Fortschritt: {done_count}/{num_games} Spiele "
                        f"({rate:.1f}/s)"
                    )

        # Stats aus den Game-Results aggregieren
        stats_a = TeamStats()
        stats_b = TeamStats()
        for game_result, team_a_id, team_b_id in results:
            update_stats_from_game(
                stats_a, stats_b, game_result,
                team_a_id=team_a_id, team_b_id=team_b_id,
            )

        return TournamentResult(
            label_a=label_a,
            label_b=label_b,
            stats_a=stats_a,
            stats_b=stats_b,
            elo=EloRating(),  # leeres Elo wie im Multi-Process-Eval
            games_played=num_games,
        )
    finally:
        if server_a is not None:
            server_a.shutdown()
        if server_b is not None:
            server_b.shutdown()
