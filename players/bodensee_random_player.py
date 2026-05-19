"""Zufalls-Spieler fuer Bodensee-Jass.

Dient als Smoke-Test der Engine. Spielt zufaellige legale Karten und sagt
zufaellig eine der 12 Spielarten an.
"""

from __future__ import annotations

import random

from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import ALL_SUITS, Card
from jass_engine.variant import Announcement, Variant
from players.bodensee_player import BodenseePlayer


def _possible_announcements() -> list[Announcement]:
    """Alle 12 erlaubten Ansagen im Bodensee-Jass."""
    out: list[Announcement] = []
    for s in ALL_SUITS:
        out.append(Announcement(variant=Variant.trumpf(s)))
    for s in ALL_SUITS:
        out.append(Announcement(variant=Variant.gumpf(s)))
    out.append(Announcement(variant=Variant.oben()))
    out.append(Announcement(variant=Variant.unten()))
    out.append(Announcement(variant=Variant.oben(), slalom=True))
    out.append(Announcement(variant=Variant.unten(), slalom=True))
    return out


class BodenseeRandomPlayer(BodenseePlayer):
    """Zufalls-Spieler: random Ansage, random legale Karte."""

    def __init__(self, name: str, rng: random.Random | None = None):
        super().__init__(name)
        self.rng = rng if rng is not None else random.Random()

    def choose_announcement(
        self,
        hand: list[Card],
        visible_table: list[Card],
        round_idx: int,
    ) -> Announcement:
        return self.rng.choice(_possible_announcements())

    def choose_card(
        self,
        hand: list[Card],
        visible_table: list[Card],
        state: BodenseeGameState,
    ) -> Card:
        # legale Karten ueber Hand + sichtbarem Tisch
        ps = BodenseePlayerState(hand=list(hand))
        # Wir bauen einen "leeren" PlayerState ohne echte Stapel, weil der Random-
        # Player nichts ueber die Stapel-Struktur weiss -- nur die Karten. Fuer
        # legal_moves brauchen wir aber den vollen Pool.
        # Trick: wir simulieren visible_table als unabhaengige Stapel mit nur
        # sichtbaren Karten -- legal_moves_bodensee schaut nur auf available_cards,
        # nicht auf die Stapel-Struktur.
        from jass_engine.bodensee.player_state import TableStack
        ps.table = [TableStack(visible=c, hidden=None) for c in visible_table]

        legal = legal_moves_bodensee(ps, state.current_trick_cards, state.variant)
        return self.rng.choice(legal)
