"""Kreuz-Jass: 4 Spieler, Partner sitzen gegenüber (über Kreuz)."""

from __future__ import annotations

import random

from jass_engine.game import DEFAULT_TARGET_SCORE, GameResult, play_game
from jass_engine.player import Player

# Sitzordnung: Spieler 0+2 = Team 0, Spieler 1+3 = Team 1 (Partner über Kreuz)
KREUZ_JASS_TEAMS: tuple[int, int, int, int] = (0, 1, 0, 1)


def play_kreuz_jass(
    players: list[Player],
    target_score: int = DEFAULT_TARGET_SCORE,
    rng: random.Random | None = None,
) -> GameResult:
    if len(players) != 4:
        raise ValueError("Kreuz-Jass benötigt genau 4 Spieler.")
    return play_game(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        target_score=target_score,
        rng=rng,
    )
