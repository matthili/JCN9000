"""Bodensee-Jass: 2 Spieler, mit Tisch-Mechanik.

Top-Level-API analog zu `kreuz_jass.py` und `solo_jass.py`. Delegiert an die
Bodensee-spezifische Engine in `jass_engine/bodensee/`.

Spielmechanik im Ueberblick:
- 2 Spieler
- 18 Karten pro Spieler: 6 Hand, 6 sichtbare Tisch-Karten, 6 verdeckte darunter
- 18 Stiche pro Runde
- Keine Weisen, keine Stoecke. Letzter Stich +5, Matsch +100
- Schieben gibt es nicht
- Default-Ziel 500 Punkte
"""

from __future__ import annotations

import random

from jass_engine.bodensee.game import (
    DEFAULT_BODENSEE_TARGET_SCORE,
    BodenseeGameResult,
    play_bodensee_game,
)
from players.bodensee_player import BodenseePlayer


def play_bodensee_jass(
    players: list[BodenseePlayer],
    target_score: int = DEFAULT_BODENSEE_TARGET_SCORE,
    rng: random.Random | None = None,
) -> BodenseeGameResult:
    """Spielt eine Bodensee-Jass-Partie bis target_score."""
    return play_bodensee_game(
        players=players,
        target_score=target_score,
        rng=rng,
    )
