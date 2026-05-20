"""Bodensee-Jass-Eval: 2-Spieler-Vergleich.

Spielt N Bodensee-Partien zwischen zwei Spielern und liefert pro Spieler
aggregierte Statistiken.

Paired-Eval: pro Kartenverteilung zwei Partien -- einmal Spieler A auf Sitz 0,
einmal auf Sitz 1, jeweils mit IDENTISCHER Kartenverteilung (gleicher
Deal-Seed). Damit faellt das Karten-Glueck als Rauschquelle weg.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from evaluation.solo_stats import PlayerStats, _round_variant_id
from jass_engine.bodensee.game import BodenseeGameResult
from jass_engine.variants.bodensee_jass import play_bodensee_jass
from players.bodensee_player import BodenseePlayer


# Factory: (seat, rng) -> BodenseePlayer
BodenseePlayerFactory = Callable[[int, random.Random], BodenseePlayer]


@dataclass
class BodenseeEvalResult:
    label_a: str
    label_b: str
    stats_a: PlayerStats
    stats_b: PlayerStats
    games_played: int


def update_stats_from_bodensee_game(
    stats_by_seat: dict[int, PlayerStats],
    game: BodenseeGameResult,
) -> None:
    """Schreibt die Ergebnisse einer Bodensee-Partie in die per-Sitz-Stats."""
    if not game.rounds:
        return

    winner = game.winner  # 0 oder 1

    for seat, stats in stats_by_seat.items():
        stats.games_played += 1
        stats.total_score += game.final_scores.get(seat, 0)
        stats.total_rounds += len(game.rounds)

    if winner in stats_by_seat:
        stats_by_seat[winner].games_won += 1

    # Pro-Variante: Ansage der ersten Runde
    variant_id = _round_variant_id(game.rounds[0])
    for seat, stats in stats_by_seat.items():
        stats.games_by_variant_id[variant_id] = (
            stats.games_by_variant_id.get(variant_id, 0) + 1
        )
    if winner in stats_by_seat:
        stats_by_seat[winner].wins_by_variant_id[variant_id] = (
            stats_by_seat[winner].wins_by_variant_id.get(variant_id, 0) + 1
        )

    # Matsch pro Spieler pro Runde
    for rnd in game.rounds:
        if rnd.matsch_player is not None and rnd.matsch_player in stats_by_seat:
            stats_by_seat[rnd.matsch_player].matsch_for += 1


def two_player_match(
    label_a: str,
    factory_a: BodenseePlayerFactory,
    label_b: str,
    factory_b: BodenseePlayerFactory,
    num_games: int,
    target_score: int = 500,
    seed: int = 0,
    paired_eval: bool = False,
) -> BodenseeEvalResult:
    """Spielt N Bodensee-Partien Spieler A gegen Spieler B.

    Args:
        label_a / label_b: Anzeigenamen
        factory_a / factory_b: Funktionen, die einen BodenseePlayer erzeugen
        num_games: Anzahl Partien
        target_score: Punkteziel pro Partie (Default 500)
        seed: RNG-Seed
        paired_eval: bei True werden Partien paarweise mit identischer
            Kartenverteilung gespielt -- einmal A auf Sitz 0, einmal auf
            Sitz 1. num_games muss dann gerade sein.

    Returns:
        BodenseeEvalResult mit Statistiken pro Spieler.
    """
    rng = random.Random(seed)
    stats_a = PlayerStats()
    stats_b = PlayerStats()

    # Plan: Liste von (swap, sub_seed)
    plan: list[tuple[bool, int]] = []
    if paired_eval:
        if num_games % 2 != 0:
            raise ValueError(
                "paired_eval=True braucht eine gerade Zahl an Partien "
                f"(num_games={num_games})."
            )
        for _ in range(num_games // 2):
            pair_seed = rng.randint(0, 10**9)
            plan.append((False, pair_seed))  # A auf Sitz 0
            plan.append((True, pair_seed))   # A auf Sitz 1, gleiche Karten
    else:
        plan = [(False, rng.randint(0, 10**9)) for _ in range(num_games)]

    for swap, sub_seed in plan:
        sub_rng = random.Random(sub_seed)
        if swap:
            players = [
                factory_b(0, random.Random(sub_rng.randint(0, 10**9))),
                factory_a(1, random.Random(sub_rng.randint(0, 10**9))),
            ]
            a_seat, b_seat = 1, 0
        else:
            players = [
                factory_a(0, random.Random(sub_rng.randint(0, 10**9))),
                factory_b(1, random.Random(sub_rng.randint(0, 10**9))),
            ]
            a_seat, b_seat = 0, 1

        game = play_bodensee_jass(
            players,
            target_score=target_score,
            rng=random.Random(sub_rng.randint(0, 10**9)),
        )

        update_stats_from_bodensee_game(
            {a_seat: stats_a, b_seat: stats_b},
            game,
        )

    return BodenseeEvalResult(
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a,
        stats_b=stats_b,
        games_played=num_games,
    )


def format_bodensee_eval_summary(res: BodenseeEvalResult) -> str:
    """Konsolen-Zusammenfassung."""
    lines = []
    lines.append(f"Bodensee-Tournament: {res.label_a}  vs.  {res.label_b}")
    lines.append(f"Partien: {res.games_played}")
    lines.append("")

    col_w = 18
    label_w = 32
    headers = [res.label_a, res.label_b]
    lines.append(f"{'Metrik':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in headers))
    lines.append("-" * (label_w + col_w * len(headers)))

    def row(name: str, vals: list[str]) -> str:
        return f"{name:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in vals)

    sa, sb = res.stats_a, res.stats_b
    lines.append(row("Spiele", [str(sa.games_played), str(sb.games_played)]))
    lines.append(row("Siege", [str(sa.games_won), str(sb.games_won)]))
    lines.append(row("Win-Rate", [f"{sa.win_rate * 100:.1f}%", f"{sb.win_rate * 100:.1f}%"]))
    lines.append(row("Avg-Score / Partie", [f"{sa.avg_score:.1f}", f"{sb.avg_score:.1f}"]))
    lines.append(row("Avg-Runden / Partie", [f"{sa.avg_rounds:.1f}", f"{sb.avg_rounds:.1f}"]))
    lines.append(row(
        "Matsch-Rate / Runde",
        [f"{sa.matsch_rate_per_round * 100:.2f}%", f"{sb.matsch_rate_per_round * 100:.2f}%"],
    ))

    # Win-Rate pro Variante
    all_variants = sorted(
        set(sa.games_by_variant_id) | set(sb.games_by_variant_id)
    )
    if all_variants:
        lines.append("")
        lines.append(
            f"{'Win-Rate pro Variante':<{label_w}}"
            + "".join(f"{h:>{col_w}}" for h in headers)
        )
        lines.append("-" * (label_w + col_w * len(headers)))
        for v in all_variants:
            cells = []
            for s in (sa, sb):
                wr = s.win_rate_for_variant(v)
                n = s.games_by_variant_id.get(v, 0)
                cells.append(f"{wr * 100:.1f}% (n={n})")
            lines.append(f"  {v:<{label_w - 2}}" + "".join(f"{c:>{col_w}}" for c in cells))

    return "\n".join(lines)
