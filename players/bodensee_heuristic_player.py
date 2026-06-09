"""Regelbasierter Spieler fuer Bodensee-Jass.

Wiederverwendet die Score-Funktionen aus `players/heuristic_player.py`, aber
mit folgenden Anpassungen:
- Bewertung der Hand-Staerke ueber **12 sichtbare Karten** (Hand + sichtbarer
  Tisch) statt 9. Die verdeckten 6 Karten kennt der Spieler nicht; eine
  bessere Heuristik koennte sie als "Durchschnitt der noch nicht gespielten
  Karten" schaetzen, das ist hier (noch) nicht implementiert.
- Kein Schmieren -- bei 2 Spielern gibt es keinen Partner.
- Stech-/Spar-Logik wie beim Solo-Heuristik-Spieler.
"""

from __future__ import annotations

import random

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import ALL_SUITS, Card, Rank, Suit
from jass_engine.rules import card_strength, card_value
from jass_engine.variant import Announcement, PlayMode, Variant
from players.bodensee_player import BodenseePlayer
from players.heuristic_player import (
    GUMPF_NON_TRUMP_VALUES,
    NON_TRUMP_HAND_VALUES,
    OBEN_HAND_VALUES,
    TRUMP_HAND_VALUES,
    UNTEN_HAND_VALUES,
)


class BodenseeHeuristicPlayer(BodenseePlayer):
    """Regelbasierter Spieler fuer 2-Spieler-Bodensee."""

    def __init__(
        self,
        name: str,
        rng: random.Random | None = None,
        slalom_base_factor: float = 0.85,
        # Relative Skalen der Ansage-Familien (Trumpf = Anker, immer 1.0).
        # Tunebar via scripts/tune_bodensee_announce.py.
        gumpf_scale: float = 1.0,
        oben_scale: float = 1.0,
        unten_scale: float = 1.0,
    ):
        super().__init__(name)
        self.rng = rng if rng is not None else random.Random()
        self.slalom_base_factor = slalom_base_factor
        self.gumpf_scale = gumpf_scale
        self.oben_scale = oben_scale
        self.unten_scale = unten_scale

    # ---------- Ansage ----------

    def choose_announcement(
        self,
        hand: list[Card],
        visible_table: list[Card],
        round_idx: int,
    ) -> Announcement:
        pool = list(hand) + list(visible_table)

        scores: dict[Announcement, int] = {}
        for suit in ALL_SUITS:
            scores[Announcement(variant=Variant.trumpf(suit))] = self._score_trumpf(pool, suit)
            scores[Announcement(variant=Variant.gumpf(suit))] = int(
                self._score_gumpf(pool, suit) * self.gumpf_scale
            )
        oben_score = int(self._score_oben(pool) * self.oben_scale)
        unten_score = int(self._score_unten(pool) * self.unten_scale)
        scores[Announcement(variant=Variant.oben())] = oben_score
        scores[Announcement(variant=Variant.unten())] = unten_score

        # Slalom: konservativ ueber max(oben, unten) * Faktor
        slalom_score = int(max(oben_score, unten_score) * self.slalom_base_factor)
        slalom_start = (
            Variant.oben() if oben_score >= unten_score else Variant.unten()
        )
        scores[Announcement(variant=slalom_start, slalom=True)] = slalom_score

        return max(scores, key=lambda a: scores[a])

    @staticmethod
    def _score_trumpf(pool: list[Card], trumpf: Suit) -> int:
        score = 0
        trump_count = sum(1 for c in pool if c.suit == trumpf)
        # Pool ist groesser als 9 -- entsprechend hoeherer Mengen-Bonus-Schwellwert
        score += max(0, trump_count - 4) * 6
        for c in pool:
            if c.suit == trumpf:
                score += TRUMP_HAND_VALUES[c.rank]
            else:
                score += NON_TRUMP_HAND_VALUES.get(c.rank, 0)
        return score

    @staticmethod
    def _score_gumpf(pool: list[Card], trumpf: Suit) -> int:
        score = 0
        trump_count = sum(1 for c in pool if c.suit == trumpf)
        score += max(0, trump_count - 4) * 6
        for c in pool:
            if c.suit == trumpf:
                score += TRUMP_HAND_VALUES[c.rank]
            else:
                score += GUMPF_NON_TRUMP_VALUES.get(c.rank, 0)
        return score

    @staticmethod
    def _score_oben(pool: list[Card]) -> int:
        return sum(OBEN_HAND_VALUES.get(c.rank, 0) for c in pool)

    @staticmethod
    def _score_unten(pool: list[Card]) -> int:
        return sum(UNTEN_HAND_VALUES.get(c.rank, 0) for c in pool)

    # ---------- Karten-Wahl ----------

    def choose_card(
        self,
        hand: list[Card],
        visible_table: list[Card],
        state: BodenseeGameState,
    ) -> Card:
        # PlayerState fuer legal_moves-Aufruf zusammensetzen
        ps = BodenseePlayerState(hand=list(hand))
        ps.table = [TableStack(visible=c, hidden=None) for c in visible_table]
        legal = legal_moves_bodensee(ps, state.current_trick_cards, state.variant)

        if not state.current_trick_cards:
            # Anspielen: hohe Karte fuer Stich-Potenzial
            return self._choose_opening(legal, state.variant)

        # Antwort: Stich uebernehmbar?
        lead_suit = state.current_trick_cards[0].suit
        opp_strength = card_strength(
            state.current_trick_cards[0], lead_suit, state.variant
        )
        winning_cards = [
            c for c in legal
            if card_strength(c, lead_suit, state.variant) > opp_strength
        ]

        if winning_cards:
            # Mit niedrigster reichender Karte uebernehmen -- bei 2 Spielern
            # ist der Stich danach zu Ende, hohe Karten fuer spaeter sparen
            return min(
                winning_cards,
                key=lambda c: card_strength(c, lead_suit, state.variant),
            )

        # Nicht uebernehmbar: niedrigste Karte mit niedrigstem Wert abwerfen
        return min(
            legal,
            key=lambda c: (card_value(c, state.variant), card_strength(c, c.suit, state.variant)),
        )

    def _choose_opening(self, legal: list[Card], variant: Variant) -> Card:
        """Erste Karte eines Stichs. Strategisch: starke Karten ziehen."""
        if variant.mode == PlayMode.TRUMPF or variant.mode == PlayMode.GUMPF:
            assert variant.trump_suit is not None
            trumpf = variant.trump_suit
            # Hohe Truempfe zuerst (Buur, Nell, Ass)
            for special in (Rank.UNTER, Rank.NEUN, Rank.ASS):
                c = Card(trumpf, special)
                if c in legal:
                    return c
            # Sonst: nichttrumpf-Ass falls vorhanden (sicherer Stich, solange Gegner bedienen muss)
            non_trump_aces = [
                c for c in legal if c.suit != trumpf and c.rank == Rank.ASS
            ]
            if non_trump_aces:
                return non_trump_aces[0]
            # Sonst: niedrigste Karte mit niedrigstem Wert
            return min(
                legal,
                key=lambda c: (card_value(c, variant), int(c.rank)),
            )
        if variant.mode == PlayMode.OBEN:
            for rank in (Rank.ASS, Rank.ZEHN, Rank.KOENIG):
                cards = [c for c in legal if c.rank == rank]
                if cards:
                    return cards[0]
            return max(legal, key=lambda c: int(c.rank))
        # UNTEN
        for rank in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT):
            cards = [c for c in legal if c.rank == rank]
            if cards:
                return cards[0]
        return min(legal, key=lambda c: int(c.rank))
