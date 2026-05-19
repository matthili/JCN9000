"""Bodensee-Jass-Partie: mehrere Runden bis ein Spieler das Punkteziel erreicht."""

from __future__ import annotations

import random
from dataclasses import dataclass

from jass_engine.bodensee.round import BodenseeRoundResult, play_bodensee_round
from players.bodensee_player import BodenseePlayer


DEFAULT_BODENSEE_TARGET_SCORE = 500
MIN_BODENSEE_TARGET_SCORE = 500


@dataclass
class BodenseeGameResult:
    """Ergebnis einer kompletten Bodensee-Partie."""

    rounds: list[BodenseeRoundResult]
    final_scores: dict[int, int]
    winner: int


def play_bodensee_game(
    players: list[BodenseePlayer],
    target_score: int = DEFAULT_BODENSEE_TARGET_SCORE,
    rng: random.Random | None = None,
    max_rounds: int = 200,
) -> BodenseeGameResult:
    """Spielt eine Bodensee-Partie bis zum Punkteziel.

    Args:
        players: zwei BodenseePlayer (Spieler 0, 1)
        target_score: mindestens 500 (typisches Solo-/Bodensee-Ziel)
        rng: optionaler RNG fuer Reproduzierbarkeit
        max_rounds: harter Stopper, falls die Punkte aus irgendeinem Grund nicht
            erreicht werden (sollte praktisch nie passieren)

    Returns:
        BodenseeGameResult mit allen Runden und finalem Score.
    """
    if len(players) != 2:
        raise ValueError("Bodensee-Jass braucht genau 2 Spieler.")
    if target_score < MIN_BODENSEE_TARGET_SCORE:
        raise ValueError(
            f"Bodensee-Mindestziel ist {MIN_BODENSEE_TARGET_SCORE} Punkte "
            f"(uebergeben: {target_score})."
        )
    if rng is None:
        rng = random.Random()

    rounds: list[BodenseeRoundResult] = []
    cumulative: dict[int, int] = {0: 0, 1: 0}
    last_announcer: int | None = None

    for round_idx in range(max_rounds):
        forced_announcer = None
        if round_idx > 0 and last_announcer is not None:
            # Nach Runde 1 wechselt der Ansager (im Uhrzeigersinn bei 2 Spielern = abwechselnd)
            forced_announcer = 1 - last_announcer

        result = play_bodensee_round(
            players=players,
            rng=rng,
            forced_announcer_idx=forced_announcer,
            initial_scores=(cumulative[0], cumulative[1]),
            round_idx=round_idx,
        )
        rounds.append(result)
        last_announcer = result.announcer_idx

        for pid in (0, 1):
            cumulative[pid] += result.player_total_points[pid]

        if any(score >= target_score for score in cumulative.values()):
            break

    winner = max(cumulative, key=lambda k: cumulative[k])
    return BodenseeGameResult(rounds=rounds, final_scores=cumulative, winner=winner)
