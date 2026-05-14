"""Wrapper-Player, der jede Karten-Entscheidung als (state, mask, action) mitschneidet.

Der Wrapper delegiert alle Entscheidungen an einen "inneren" Player (z.B.
`HeuristicPlayer`), zeichnet aber bei jedem `choose_card`-Aufruf den Featurevektor,
die Aktionsmaske und die gewählte Karte auf.
"""

from __future__ import annotations

from jass_engine.card import Card
from jass_engine.player import GameState, Player
from jass_engine.variant import Announcement, Variant
from jass_engine.weis import Weis
from training.encoder import (
    action_index,
    encode_state,
    legal_action_mask,
)


class RecordingPlayer(Player):
    """Schreibt pro choose_card-Aufruf einen Trainingsdatensatz mit."""

    def __init__(self, inner: Player):
        super().__init__(inner.name)
        self.inner = inner
        # Buffer pro Spiel-Instanz; werden vom Datengenerator nach jeder Partie ausgelesen
        self.states: list = []
        self.masks: list = []
        self.actions: list = []

    def reset(self) -> None:
        self.states.clear()
        self.masks.clear()
        self.actions.clear()

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        return self.inner.choose_announcement(hand, round_idx, can_push)

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        x = encode_state(hand, state)
        mask = legal_action_mask(hand, state)
        chosen = self.inner.choose_card(hand, state)
        # Mitschreiben — erst nach der Wahl, damit ausschließlich gültige Aktionen drinstehen
        self.states.append(x)
        self.masks.append(mask)
        self.actions.append(action_index(chosen))
        return chosen

    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        return self.inner.announce_weise(hand, variant, possible_weise)
