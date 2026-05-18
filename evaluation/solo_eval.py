"""Solo-Jass-Eval: Vier-Wege-Vergleich.

Spielt Solo-Partien mit vier Spielern (Modell A, Modell B, 2 x Heuristik) und
liefert pro-Rolle aggregierte Statistiken.

Paired-Eval: pro Kartenverteilung werden 4 Partien gespielt. Die Spieler-
Rollen rotieren zyklisch durch die vier Sitze, sodass jede Rolle jeden Sitz
einmal mit derselben Kartenverteilung erlebt. Damit faellt das Karten-Glueck
und auch die Sitz-Asymmetrie als Rauschquelle weg.

Beispiel-Rotation fuer pair_idx=k:
    Game 0: A=Sitz 0, B=Sitz 1, H1=Sitz 2, H2=Sitz 3
    Game 1: A=Sitz 1, B=Sitz 2, H1=Sitz 3, H2=Sitz 0
    Game 2: A=Sitz 2, B=Sitz 3, H1=Sitz 0, H2=Sitz 1
    Game 3: A=Sitz 3, B=Sitz 0, H1=Sitz 1, H2=Sitz 2
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from jass_engine.player import Player
from jass_engine.variants.solo_jass import play_solo_jass
from evaluation.solo_stats import (
    PlayerStats,
    update_stats_from_solo_game,
)


# Rollen: A und B sind die zu vergleichenden Modelle/Spieler, H1 und H2 sind
# zwei (idente) Heuristik-Bots als Vergleichsanker.
ROLE_A = "A"
ROLE_B = "B"
ROLE_H1 = "H1"
ROLE_H2 = "H2"
ALL_ROLES = (ROLE_A, ROLE_B, ROLE_H1, ROLE_H2)


PlayerFactory = Callable[[int, random.Random], Player]


@dataclass
class SoloEvalResult:
    label_a: str
    label_b: str
    label_h: str
    stats_a: PlayerStats
    stats_b: PlayerStats
    stats_h: PlayerStats   # gemerged ueber beide Heuristik-Sitze
    games_played: int


def _seat_assignment(pair_offset: int) -> dict[int, str]:
    """Liefert mapping seat_idx (0..3) -> role ('A', 'B', 'H1', 'H2').

    pair_offset 0..3 verschiebt die Zuweisung zyklisch.
    """
    return {
        (pair_offset + 0) % 4: ROLE_A,
        (pair_offset + 1) % 4: ROLE_B,
        (pair_offset + 2) % 4: ROLE_H1,
        (pair_offset + 3) % 4: ROLE_H2,
    }


def _random_seat_assignment(rng: random.Random) -> dict[int, str]:
    """Zufaellige Sitz-Zuweisung der vier Rollen."""
    roles = list(ALL_ROLES)
    rng.shuffle(roles)
    return {seat: roles[seat] for seat in range(4)}


def _build_players(
    seat_to_role: dict[int, str],
    factory_a: PlayerFactory,
    factory_b: PlayerFactory,
    factory_h: PlayerFactory,
    sub_rng: random.Random,
) -> list[Player]:
    """Erstellt die 4 Player-Instanzen entsprechend der Sitz-Rollen-Zuweisung."""
    players: list[Player | None] = [None] * 4
    for seat, role in seat_to_role.items():
        if role == ROLE_A:
            factory = factory_a
        elif role == ROLE_B:
            factory = factory_b
        else:  # H1 oder H2 -- beide aus derselben Heuristik-Factory
            factory = factory_h
        players[seat] = factory(seat, random.Random(sub_rng.randint(0, 10**9)))
    return players  # type: ignore[return-value]


def four_way_match(
    label_a: str,
    factory_a: PlayerFactory,
    label_b: str,
    factory_b: PlayerFactory,
    label_h: str,
    factory_h: PlayerFactory,
    num_games: int,
    target_score: int = 500,
    seed: int = 0,
    paired_eval: bool = False,
) -> SoloEvalResult:
    """Spielt N Solo-Partien mit dem Setup A vs B vs H vs H.

    Args:
        label_a / label_b / label_h: Anzeigenamen der drei Rollen
        factory_a / factory_b / factory_h: Factories (factory_h wird zweimal
            instanziiert, einmal fuer H1, einmal fuer H2)
        num_games: Gesamtanzahl Partien
        target_score: Punkteziel pro Partie (Default 500)
        seed: RNG-Seed fuer Reproduzierbarkeit
        paired_eval: bei True werden Partien in 4er-Gruppen mit identischer
            Kartenverteilung gespielt; die Rollen rotieren zyklisch durch die
            Sitze. Verlangt num_games % 4 == 0.

    Returns:
        SoloEvalResult mit aggregierten Stats pro Rolle.
    """
    rng = random.Random(seed)
    stats_a = PlayerStats()
    stats_b = PlayerStats()
    stats_h = PlayerStats()  # gemerged ueber beide Heuristik-Sitze

    # Plan: Liste von (seat_to_role, sub_seed)
    plan: list[tuple[dict[int, str], int]] = []
    if paired_eval:
        if num_games % 4 != 0:
            raise ValueError(
                "paired_eval=True braucht num_games als Vielfaches von 4 "
                f"(uebergeben: {num_games})."
            )
        num_pairs = num_games // 4
        for _ in range(num_pairs):
            pair_seed = rng.randint(0, 10**9)
            for pair_offset in range(4):
                plan.append((_seat_assignment(pair_offset), pair_seed))
    else:
        for _ in range(num_games):
            plan.append((_random_seat_assignment(rng), rng.randint(0, 10**9)))

    for seat_to_role, sub_seed in plan:
        sub_rng = random.Random(sub_seed)
        players = _build_players(
            seat_to_role, factory_a, factory_b, factory_h, sub_rng,
        )
        game = play_solo_jass(
            players,
            target_score=target_score,
            rng=random.Random(sub_rng.randint(0, 10**9)),
        )

        # Pro Sitz die zugehoerige Rolle ermitteln und Stats aktualisieren
        per_seat_stats: dict[int, PlayerStats] = {}
        for seat in range(4):
            role = seat_to_role[seat]
            if role == ROLE_A:
                per_seat_stats[seat] = stats_a
            elif role == ROLE_B:
                per_seat_stats[seat] = stats_b
            else:  # H1 oder H2 -- beide in dieselbe gemerged Heuristik-Stat
                per_seat_stats[seat] = stats_h

        # Wichtig: stats_h darf nicht doppelt aktualisiert werden pro Spiel,
        # wenn beide Heuristik-Sitze auf dasselbe PlayerStats-Objekt zeigen.
        # update_stats_from_solo_game macht aber per_seat-Aufrufe -- d.h. wenn
        # zwei Sitze auf stats_h zeigen, werden games_played, total_score etc.
        # zweimal aufaddiert. Das WOLLEN wir bei einer aggregierten Heuristik-
        # Statistik: 2 Heuristik-Sitze = 2 Spieler-Spiel-Eintraege pro Partie.
        update_stats_from_solo_game(per_seat_stats, game)

    return SoloEvalResult(
        label_a=label_a,
        label_b=label_b,
        label_h=label_h,
        stats_a=stats_a,
        stats_b=stats_b,
        stats_h=stats_h,
        games_played=num_games,
    )


def format_solo_eval_summary(res: SoloEvalResult) -> str:
    from evaluation.solo_stats import format_solo_stats_table

    lines = []
    lines.append(
        f"Solo-Tournament: {res.label_a}  vs.  {res.label_b}  "
        f"(+ 2x {res.label_h})"
    )
    lines.append(f"Partien: {res.games_played}")
    lines.append("")
    lines.append(format_solo_stats_table(
        res.label_a, res.stats_a,
        res.label_b, res.stats_b,
        res.label_h, res.stats_h,
    ))
    return "\n".join(lines)
