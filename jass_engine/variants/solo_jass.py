"""Solo-Jass: 4 Spieler, jeder gegen jeden, kein Partner.

Spielmechanik wie beim Kreuz-Jass (Stiche, Trumpf, Weisen, Stoecke, Matsch),
aber:
- Punkte werden pro Spieler statt pro Team gefuehrt
- Schieben entfaellt (es gibt keinen Partner)
- Default-Spielziel ist 500 statt 1000 (anpassbar, mindestens 500)
- Weisen: nur der Spieler mit dem hoechsten Weis bekommt die Punkte
- Matsch: +100 fuer den einzelnen Spieler, der alle 9 Stiche macht
- Stoecke: +20 fuer den Stockhalter persoenlich

Implementiert ueber teams=[0,1,2,3] (jeder Spieler eigenes "Team") plus
`allow_push=False`. Die bestehende Engine-Logik fuer Weisen, Matsch und Stoecke
funktioniert dadurch automatisch korrekt: alle Aggregationen "pro Team" werden
in dieser Konfiguration zu "pro Spieler".
"""

from __future__ import annotations

import random

from jass_engine.game import GameResult, play_game
from jass_engine.player import Player


# Sitzordnung: jeder Spieler eigenes "Team" (= eigener Punkte-Topf)
SOLO_JASS_TEAMS: tuple[int, int, int, int] = (0, 1, 2, 3)

# Default-Spielziel beim Solo-Jass. Mindestziel 500 ist die uebliche Untergrenze
# beim Preis-Jassen; in der Web-App konfigurierbar bis hoch zu 1000+.
DEFAULT_SOLO_TARGET_SCORE = 500


def play_solo_jass(
    players: list[Player],
    target_score: int = DEFAULT_SOLO_TARGET_SCORE,
    rng: random.Random | None = None,
) -> GameResult:
    """Spielt eine Solo-Jass-Partie bis target_score.

    Args:
        players: Liste der 4 Spieler.
        target_score: Punkteziel pro Spieler (Default 500).
        rng: optionaler RNG fuer Reproduzierbarkeit.

    Returns:
        GameResult, bei dem `final_scores` und `winning_team` als Spieler-IDs
        zu lesen sind (0-3), nicht als Team-IDs.
    """
    if len(players) != 4:
        raise ValueError("Solo-Jass benoetigt genau 4 Spieler.")
    if target_score < 500:
        raise ValueError(
            f"Solo-Jass-Mindestziel ist 500 Punkte (uebergeben: {target_score})."
        )
    return play_game(
        players=players,
        teams=list(SOLO_JASS_TEAMS),
        target_score=target_score,
        rng=rng,
        allow_push=False,
    )
