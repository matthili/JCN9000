"""Zufalls-Spieler: wählt aus den legalen Karten zufällig.

Dient als Smoke-Test der Engine und als Datengenerator-Baseline.
"""

from __future__ import annotations

import random

from jass_engine.card import ALL_SUITS, Card
from jass_engine.player import GameState, Player
from jass_engine.rules import legal_moves
from jass_engine.variant import Announcement, Variant
from jass_engine.weis import Weis


# Anteil der möglichen Ansagen, die der RandomPlayer zufällig wählt.
# (Vergabe: Trumpf-Farben + OBEN + UNTEN + SLALOM-OBEN + SLALOM-UNTEN)
def _possible_announcements() -> list[Announcement]:
    out: list[Announcement] = []
    for s in ALL_SUITS:
        out.append(Announcement(variant=Variant.trumpf(s)))
    out.append(Announcement(variant=Variant.oben()))
    out.append(Announcement(variant=Variant.unten()))
    out.append(Announcement(variant=Variant.oben(), slalom=True))
    out.append(Announcement(variant=Variant.unten(), slalom=True))
    return out


class RandomPlayer(Player):
    """Spielt zufällige legale Karten und sagt alle Weisen an."""

    def __init__(
        self,
        name: str,
        rng: random.Random | None = None,
        push_probability: float = 0.0,
    ):
        super().__init__(name)
        self.rng = rng if rng is not None else random.Random()
        self.push_probability = push_probability

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        if can_push and self.rng.random() < self.push_probability:
            return None
        return self.rng.choice(_possible_announcements())

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        legal = legal_moves(hand, state.current_trick_cards, state.variant)
        return self.rng.choice(legal)

    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        return list(possible_weise)
