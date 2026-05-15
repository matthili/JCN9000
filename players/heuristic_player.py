"""Regelbasierter Heuristik-Spieler.

Spielstärke deutlich über RandomPlayer. Dient als Lehrer für das spätere NN
(Behavioral Cloning) und als Baseline für die Spielstärke-Evaluation.

Strategien:
  - Kartenwahl:
    * Anspielen: bei Trumpf hohe Trümpfe ziehen (Buur/Nell/Ass), bei Bock
      Asse, bei Geiss niedrige Karten.
    * Stich übernehmen wenn möglich: bei letztem Spieler so knapp wie nötig,
      sonst so hoch wie sinnvoll.
    * Schmieren: wenn der Partner führt, hohe Punktkarte legen, ohne ihn zu
      übertrumpfen.
    * Sparen: wenn nicht gewinnbar und Partner nicht führt, niedrigste Karte
      mit niedrigstem Wert abwerfen.
  - Ansage: Score je Variante; höchster Score gewinnt. Schiebe-Schwelle unten.
"""

from __future__ import annotations

import random

from jass_engine.card import ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState, Player
from jass_engine.rules import card_strength, card_value, legal_moves
from jass_engine.variant import Announcement, PlayMode, Variant
from jass_engine.weis import Weis


# Stärke einer Karte in einer Trumpf-Hand (für die Ansage-Bewertung)
TRUMP_HAND_VALUES: dict[Rank, int] = {
    Rank.UNTER: 25,   # Buur
    Rank.NEUN: 18,    # Nell
    Rank.ASS: 12,
    Rank.ZEHN: 7,
    Rank.KOENIG: 6,
    Rank.OBER: 5,
    Rank.ACHT: 3,
    Rank.SIEBEN: 2,
    Rank.SECHS: 1,
}

# Wert einer Nicht-Trumpf-Karte in einer Trumpf-Hand
NON_TRUMP_HAND_VALUES: dict[Rank, int] = {
    Rank.ASS: 9,
    Rank.ZEHN: 5,
    Rank.KOENIG: 3,
    Rank.OBER: 1,
}

# Bock: hohe Karten sind stark (sie können nicht gestochen werden, wenn man die Farbe anspielt)
OBEN_HAND_VALUES: dict[Rank, int] = {
    Rank.ASS: 13,
    Rank.KOENIG: 8,
    Rank.ZEHN: 7,
    Rank.OBER: 5,
    Rank.ACHT: 4,   # 8 Punkte
    Rank.UNTER: 2,
    Rank.NEUN: 1,
}

# Geiss: niedrige Karten sind stark
UNTEN_HAND_VALUES: dict[Rank, int] = {
    Rank.SECHS: 13,
    Rank.SIEBEN: 9,
    Rank.ACHT: 8,   # stark UND 8 Punkte
    Rank.NEUN: 5,
    Rank.ZEHN: 2,
}


class HeuristicPlayer(Player):
    """Regelbasierter Spieler mit Schmier-/Stech-/Spar-Strategie."""

    def __init__(
        self,
        name: str,
        push_threshold: int = 55,
        slalom_base_factor: float = 0.95,
        slalom_concentration_factor: int = 2,
        slalom_spread_factor: int = 1,
        rng: random.Random | None = None,
    ):
        super().__init__(name)
        self.push_threshold = push_threshold
        # Slalom-Score = max(oben, unten) * base_factor
        #              + min(max_oben_pro_farbe, max_unten_pro_farbe) * concentration_factor
        #              + min(n_oben_total, n_unten_total) * spread_factor
        # Konzentrations-Term: belohnt Karten am selben Ende in derselben Farbe (echte Dominanz).
        # Spread-Term: kleiner Zuschlag pro Balance-Karte irgendwo in der Hand.
        self.slalom_base_factor = slalom_base_factor
        self.slalom_concentration_factor = slalom_concentration_factor
        self.slalom_spread_factor = slalom_spread_factor
        self.rng = rng if rng is not None else random.Random()

    # ---------- Ansage ----------

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        scores: dict[Announcement, int] = {}

        for suit in ALL_SUITS:
            scores[Announcement(variant=Variant.trumpf(suit))] = self._score_trumpf(hand, suit)

        # Gumpf bewerten: gleicher Trumpf-Anteil wie Trumpf, aber Non-Trumpf-Asse
        # sind nichts mehr wert (6er übernehmen), darum eigene Bewertungsfunktion.
        for suit in ALL_SUITS:
            scores[Announcement(variant=Variant.gumpf(suit))] = self._score_gumpf(hand, suit)

        oben_score = self._score_oben(hand)
        unten_score = self._score_unten(hand)
        scores[Announcement(variant=Variant.oben())] = oben_score
        scores[Announcement(variant=Variant.unten())] = unten_score

        # Slalom: Basis ist die stärkere Single-Variante × Faktor, plus Konzentrations-
        # und Spread-Bonus für Hände mit Karten an beiden Enden des Spektrums.
        oben_ranks = (Rank.ASS, Rank.KOENIG, Rank.OBER)
        unten_ranks = (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)
        # Pro Farbe: wie viele oben-/unten-starke Karten?
        per_suit_oben = {s: 0 for s in ALL_SUITS}
        per_suit_unten = {s: 0 for s in ALL_SUITS}
        for c in hand:
            if c.rank in oben_ranks:
                per_suit_oben[c.suit] += 1
            elif c.rank in unten_ranks:
                per_suit_unten[c.suit] += 1
        max_oben_per_suit = max(per_suit_oben.values())
        max_unten_per_suit = max(per_suit_unten.values())
        n_oben_total = sum(per_suit_oben.values())
        n_unten_total = sum(per_suit_unten.values())
        konzentrations_bonus = (
            min(max_oben_per_suit, max_unten_per_suit) * self.slalom_concentration_factor
        )
        spread_bonus = min(n_oben_total, n_unten_total) * self.slalom_spread_factor
        balance_bonus = konzentrations_bonus + spread_bonus
        slalom_score = int(
            max(oben_score, unten_score) * self.slalom_base_factor + balance_bonus
        )
        # Slalom kann mit oben oder unten beginnen — Bot wählt den Anfang nach
        # der stärkeren Single-Score (Idee: zuerst der "sichere" Modus).
        slalom_start_variant = (
            Variant.oben() if oben_score >= unten_score else Variant.unten()
        )
        scores[Announcement(variant=slalom_start_variant, slalom=True)] = slalom_score

        best_ann = max(scores, key=lambda a: scores[a])
        best_score = scores[best_ann]

        if can_push and best_score < self.push_threshold:
            return None

        return best_ann

    @staticmethod
    def _score_trumpf(hand: list[Card], trumpf: Suit) -> int:
        score = 0
        trump_count = sum(1 for c in hand if c.suit == trumpf)
        # Mengen-Bonus: mehr Trumpf = wertvoller (länger durchhalten)
        score += max(0, trump_count - 3) * 6
        for c in hand:
            if c.suit == trumpf:
                score += TRUMP_HAND_VALUES[c.rank]
            else:
                score += NON_TRUMP_HAND_VALUES.get(c.rank, 0)
        return score

    @staticmethod
    def _score_gumpf(hand: list[Card], trumpf: Suit) -> int:
        """Gumpf konservativ bewerten.

        Die Variante wird in der echten Praxis seltener gespielt, und die
        Heuristik soll Trumpf bevorzugen, wenn beides möglich ist. Konkrete
        Konstruktion: Trumpf-Score als Basis (gleiche Trump-Bewertung) plus
        ein moderater Bonus pro 6er in Nicht-Trumpf-Farben (die in Gumpf
        sticht-relevant sind). Pauschaler Discount-Faktor von 0.85, damit
        Gumpf nur bei klar passender Hand gewählt wird.

        Bei balancierter Datengen wird die Verteilung der Varianten ohnehin
        durch ForcedAnnouncementPlayer erzwungen, deshalb braucht die
        Heuristik selbst keine "perfekte" Gumpf-Erkennung.
        """
        base = HeuristicPlayer._score_trumpf(hand, trumpf)
        sixes_non_trump = sum(
            1 for c in hand
            if c.rank == Rank.SECHS and c.suit != trumpf
        )
        return int((base + sixes_non_trump * 3) * 0.85)

    @staticmethod
    def _score_oben(hand: list[Card]) -> int:
        score = 0
        for c in hand:
            score += OBEN_HAND_VALUES.get(c.rank, 0)
        return score

    @staticmethod
    def _score_unten(hand: list[Card]) -> int:
        score = 0
        for c in hand:
            score += UNTEN_HAND_VALUES.get(c.rank, 0)
        return score

    # ---------- Weisen ansagen: immer alles ----------

    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        return list(possible_weise)

    # ---------- Kartenwahl ----------

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        legal = legal_moves(hand, state.current_trick_cards, state.variant)

        # Erste Karte im Stich
        if not state.current_trick_cards:
            return self._choose_opening(legal, state)

        # Stand im Stich analysieren
        lead_suit = state.current_trick_cards[0].suit
        winning_cards = self._winning_cards(legal, lead_suit, state)
        partner_winning = self._is_partner_winning(state)
        after_me = state.players_after_me_in_trick()

        # Partner führt → schmieren
        if partner_winning:
            return self._schmieren(legal, winning_cards, state)

        # Stich übernehmbar?
        if winning_cards:
            if after_me == 0:
                # Ich bin der letzte → übernehme mit der niedrigsten reichenden Karte
                # (Reservierung von Hochkarten für später)
                return min(
                    winning_cards,
                    key=lambda c: card_strength(c, lead_suit, state.variant),
                )
            # Es kommen noch Gegner nach mir → mit hoher Karte sichern
            # (sonst übertrumpfen sie wieder)
            return max(
                winning_cards,
                key=lambda c: card_strength(c, lead_suit, state.variant),
            )

        # Nicht gewinnbar → sparen
        return self._sparen(legal, state)

    # ---------- Helpers ----------

    @staticmethod
    def _winning_cards(
        legal: list[Card],
        lead_suit: Suit,
        state: GameState,
    ) -> list[Card]:
        current_best = max(
            card_strength(c, lead_suit, state.variant) for c in state.current_trick_cards
        )
        return [
            c for c in legal
            if card_strength(c, lead_suit, state.variant) > current_best
        ]

    @staticmethod
    def _is_partner_winning(state: GameState) -> bool:
        if not state.current_trick_cards:
            return False
        # Wer führt aktuell?
        lead_suit = state.current_trick_cards[0].suit
        strengths = [
            card_strength(c, lead_suit, state.variant)
            for c in state.current_trick_cards
        ]
        winning_pos = max(range(len(strengths)), key=lambda i: strengths[i])
        winning_player = (
            state.current_trick_starter + winning_pos
        ) % state.num_players
        if winning_player == state.player_idx:
            return False
        return state.teams[winning_player] == state.teams[state.player_idx]

    def _schmieren(
        self,
        legal: list[Card],
        winning_cards: list[Card],
        state: GameState,
    ) -> Card:
        """Hohe Punkt-Karte legen, ohne den Partner zu übertrumpfen."""
        # Karten, die NICHT den Partner übertrumpfen würden
        non_overtrumping = [c for c in legal if c not in winning_cards]
        if non_overtrumping:
            return max(non_overtrumping, key=lambda c: card_value(c, state.variant))
        # Zwangslage: ich muss übertrumpfen (alle legalen Karten gewinnen)
        # → niedrigste übertrumpfende Karte
        return min(legal, key=lambda c: (
            card_value(c, state.variant),
            card_strength(c, state.current_trick_cards[0].suit, state.variant),
        ))

    def _sparen(self, legal: list[Card], state: GameState) -> Card:
        """Niedrigste Karte mit niedrigstem Wert abwerfen."""
        # Zuerst die niedrigste Punktzahl, dann die niedrigste Spielstärke
        # in ihrer eigenen Farbe (= "schwächste Karte zuerst loswerden")
        return min(legal, key=lambda c: (
            card_value(c, state.variant),
            card_strength(c, c.suit, state.variant),
        ))

    def _choose_opening(self, legal: list[Card], state: GameState) -> Card:
        """Erste Karte des Stichs."""
        if state.variant.mode == PlayMode.TRUMPF:
            assert state.variant.trump_suit is not None
            trumpf = state.variant.trump_suit
            # Hohe Trümpfe ziehen, damit Gegner ihre Trümpfe verschwenden
            for special in (Rank.UNTER, Rank.NEUN, Rank.ASS):
                c = Card(trumpf, special)
                if c in legal:
                    return c
            # Sonst: Ass einer Nicht-Trumpf-Farbe (sicherer Stich, wenn keiner trumpfen muss)
            non_trump_aces = [
                c for c in legal
                if c.suit != trumpf and c.rank == Rank.ASS
            ]
            if non_trump_aces:
                return non_trump_aces[0]
            # Sonst: niedrigste Nicht-Trumpf-Karte
            non_trumps = [c for c in legal if c.suit != trumpf]
            if non_trumps:
                return min(non_trumps, key=lambda c: (
                    card_value(c, state.variant),
                    int(c.rank),
                ))
            return min(legal, key=lambda c: card_value(c, state.variant))

        if state.variant.mode == PlayMode.GUMPF:
            assert state.variant.trump_suit is not None
            trumpf = state.variant.trump_suit
            # In der Trumpf-Farbe: hohe Trümpfe ziehen wie bei Trumpf.
            for special in (Rank.UNTER, Rank.NEUN, Rank.ASS):
                c = Card(trumpf, special)
                if c in legal:
                    return c
            # In Nicht-Trumpf-Farben: 6er sind die sicheren Sticher (Geiss-Logik).
            non_trump_6 = [
                c for c in legal
                if c.suit != trumpf and c.rank == Rank.SECHS
            ]
            if non_trump_6:
                return non_trump_6[0]
            # Sonst: niedrigster Wert, niedriger Rang in Nicht-Trumpf.
            non_trumps = [c for c in legal if c.suit != trumpf]
            if non_trumps:
                return min(non_trumps, key=lambda c: (
                    card_value(c, state.variant),
                    int(c.rank),
                ))
            return min(legal, key=lambda c: card_value(c, state.variant))

        if state.variant.mode == PlayMode.OBEN:
            # Asse (unschlagbar), dann Zehner, dann König
            for rank in (Rank.ASS, Rank.ZEHN, Rank.KOENIG):
                cards = [c for c in legal if c.rank == rank]
                if cards:
                    return cards[0]
            return max(legal, key=lambda c: int(c.rank))

        # PlayMode.UNTEN
        # 6er (unschlagbar in der Farbe), dann 7er, 8er
        for rank in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT):
            cards = [c for c in legal if c.rank == rank]
            if cards:
                # 8er auch wegen 8 Punkten ein nettes "Schmieren beim Anspielen"
                return cards[0]
        return min(legal, key=lambda c: int(c.rank))
