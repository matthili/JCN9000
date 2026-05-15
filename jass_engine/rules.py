"""Kern-Regelmodul: Punktewerte, Stichgewinner, legale Züge.

Reine Funktionen — keine Spielzustands-Mutation. Alle Funktionen arbeiten mit einer
`Variant` (TRUMPF / OBEN / UNTEN). Bei Slalom liefert `Announcement.variant_for_trick()`
pro Stich die richtige Variant.
"""

from __future__ import annotations

from jass_engine.card import Card, Rank, Suit
from jass_engine.variant import PlayMode, Variant

# Kartenwerte wenn die Karte NICHT Trumpf ist und Variant TRUMPF gewählt wurde.
POINT_VALUES_NORMAL: dict[Rank, int] = {
    Rank.ASS: 11,
    Rank.ZEHN: 10,
    Rank.KOENIG: 4,
    Rank.OBER: 3,
    Rank.UNTER: 2,
    Rank.NEUN: 0,
    Rank.ACHT: 0,
    Rank.SIEBEN: 0,
    Rank.SECHS: 0,
}

# Kartenwerte für die Trumpf-Farbe (Buur=20, Nell=14).
POINT_VALUES_TRUMP: dict[Rank, int] = {
    **POINT_VALUES_NORMAL,
    Rank.UNTER: 20,
    Rank.NEUN: 14,
}

# Kartenwerte bei Bock/Geiss/Slalom: kein Buur/Nell-Bonus, aber 8er=8.
POINT_VALUES_OBEN_UNTEN: dict[Rank, int] = {
    Rank.ASS: 11,
    Rank.ZEHN: 10,
    Rank.KOENIG: 4,
    Rank.OBER: 3,
    Rank.UNTER: 2,
    Rank.ACHT: 8,
    Rank.NEUN: 0,
    Rank.SIEBEN: 0,
    Rank.SECHS: 0,
}

# Reihenfolge innerhalb der Trumpf-Farbe (hoch → niedrig durch hohen Wert).
TRUMP_RANK_ORDER: dict[Rank, int] = {
    Rank.UNTER: 8,
    Rank.NEUN: 7,
    Rank.ASS: 6,
    Rank.KOENIG: 5,
    Rank.OBER: 4,
    Rank.ZEHN: 3,
    Rank.ACHT: 2,
    Rank.SIEBEN: 1,
    Rank.SECHS: 0,
}

# Gumpf-Wertpunkte: identisch mit der Trumpf-Variante. Trumpf-Farbe = Trumpf-Werte
# (Buur=20, Nell=14); Nicht-Trumpf = normale Werte (8er=0, keine Aufwertung).
# Nur die Stärke-Reihenfolge ist in Nicht-Trumpf-Farben invertiert.
POINT_VALUES_GUMPF_TRUMP: dict[Rank, int] = POINT_VALUES_TRUMP
POINT_VALUES_GUMPF_NON_TRUMP: dict[Rank, int] = POINT_VALUES_NORMAL

LAST_TRICK_BONUS = 5
MATCH_BONUS = 100
# Trumpf: 3×30 + (11+10+4+3+20+14) = 90 + 62 = 152 Stichpunkte
# Bock/Geiss: 4×(11+10+4+3+2+8) = 4×38 = 152 Stichpunkte (Buur/Nell-Wegfall = +8er-Aufwertung)
TOTAL_POINTS_PER_ROUND = 157  # 152 + 5 letzter Stich, gilt in allen Varianten


def total_points_per_round(variant: Variant) -> int:
    """Erwartete Punktesumme pro Runde (vor Matsch-Bonus). In allen Varianten 157."""
    return TOTAL_POINTS_PER_ROUND


def card_value(card: Card, variant: Variant) -> int:
    """Punktewert einer Karte unter Berücksichtigung der gewählten Variante."""
    if variant.mode == PlayMode.TRUMPF:
        if card.suit == variant.trump_suit:
            return POINT_VALUES_TRUMP[card.rank]
        return POINT_VALUES_NORMAL[card.rank]
    if variant.mode == PlayMode.GUMPF:
        # Wertpunkte identisch mit Trumpf: 8er=0, Buur=20, Nell=14.
        if card.suit == variant.trump_suit:
            return POINT_VALUES_GUMPF_TRUMP[card.rank]
        return POINT_VALUES_GUMPF_NON_TRUMP[card.rank]
    # OBEN oder UNTEN
    return POINT_VALUES_OBEN_UNTEN[card.rank]


def _strength_oben(card: Card, lead_suit: Suit) -> int:
    """Bock: normale Reihenfolge, nur Lead-Farbe sticht."""
    if card.suit == lead_suit:
        return 100 + int(card.rank)
    return -1


def _strength_unten(card: Card, lead_suit: Suit) -> int:
    """Geiss: umgekehrte Reihenfolge (6 sticht alles), nur Lead-Farbe sticht."""
    if card.suit == lead_suit:
        # invertierte Rangstärke: 6 (rank=0) wird zu 8, Ass (rank=8) wird zu 0
        return 100 + (8 - int(card.rank))
    return -1


def card_strength(card: Card, lead_suit: Suit, variant: Variant) -> int:
    """Stärke einer Karte im aktuellen Stich. Höher = sticht."""
    if variant.mode == PlayMode.TRUMPF:
        assert variant.trump_suit is not None
        if card.suit == variant.trump_suit:
            return 1000 + TRUMP_RANK_ORDER[card.rank]
        if card.suit == lead_suit:
            return 100 + int(card.rank)
        return -1
    if variant.mode == PlayMode.GUMPF:
        # Trumpf-Farbe wie bei normalem Trumpf; Nicht-Trumpf-Farbe in der Lead-
        # Suit-Position sticht invertiert (6 stärkste, Ass schwächste).
        assert variant.trump_suit is not None
        if card.suit == variant.trump_suit:
            return 1000 + TRUMP_RANK_ORDER[card.rank]
        if card.suit == lead_suit:
            return 100 + (8 - int(card.rank))
        return -1
    if variant.mode == PlayMode.OBEN:
        return _strength_oben(card, lead_suit)
    return _strength_unten(card, lead_suit)


def _highest_trump_in(cards: list[Card], trumpf: Suit) -> Card | None:
    trumps = [c for c in cards if c.suit == trumpf]
    if not trumps:
        return None
    return max(trumps, key=lambda c: TRUMP_RANK_ORDER[c.rank])


def legal_moves(
    hand: list[Card],
    current_trick: list[Card],
    variant: Variant,
) -> list[Card]:
    """Liste der Karten, die der Spieler legal ausspielen darf.

    TRUMPF- und GUMPF-Modus (beide mit Trumpf-Farbe):
      - Farbzwang; Buur (Trumpf-Unter) immer spielbar; bei einzigem Trumpf=Buur darf
        beliebige Karte; bei Trumpf-Lead muss Trumpf bedient werden (außer Buur einzig);
        kein Untertrumpfen außer es bleibt keine andere Wahl.

    OBEN/UNTEN-Modus:
      - Reiner Farbzwang. Wer Lead-Farbe nicht bedienen kann, darf frei abwerfen.
        Kein Trumpf, kein Buur, kein Untertrumpfen-Konzept.
    """
    if not current_trick:
        return list(hand)

    lead_suit = current_trick[0].suit

    if variant.mode not in (PlayMode.TRUMPF, PlayMode.GUMPF):
        # Bock/Geiss: einfacher Farbzwang
        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit:
            return same_suit
        return list(hand)

    # Trumpf- oder Gumpf-Modus (beide haben eine trump_suit)
    assert variant.trump_suit is not None
    trumpf = variant.trump_suit
    buur = Card(trumpf, Rank.UNTER)
    has_buur = buur in hand

    # Trumpf wurde angespielt
    if lead_suit == trumpf:
        trumps_in_hand = [c for c in hand if c.suit == trumpf]
        non_buur_trumps = [c for c in trumps_in_hand if c.rank != Rank.UNTER]
        if non_buur_trumps:
            return trumps_in_hand
        if trumps_in_hand:
            return list(hand)
        return list(hand)

    # Nicht-Trumpf wurde angespielt
    same_suit = [c for c in hand if c.suit == lead_suit]
    if same_suit:
        if has_buur and buur not in same_suit:
            return same_suit + [buur]
        return same_suit

    highest_trump_in_trick = _highest_trump_in(current_trick, trumpf)
    if highest_trump_in_trick is None:
        return list(hand)

    highest_strength = TRUMP_RANK_ORDER[highest_trump_in_trick.rank]
    higher_trumps = [
        c for c in hand
        if c.suit == trumpf and TRUMP_RANK_ORDER[c.rank] > highest_strength
    ]
    non_trumps = [c for c in hand if c.suit != trumpf]
    legal = higher_trumps + non_trumps
    if legal:
        return legal
    return list(hand)


def trick_winner(trick: list[Card], variant: Variant) -> int:
    """Index des Gewinners im Stich."""
    if not trick:
        raise ValueError("Leerer Stich hat keinen Gewinner.")
    lead_suit = trick[0].suit
    strengths = [card_strength(c, lead_suit, variant) for c in trick]
    return max(range(len(trick)), key=lambda i: strengths[i])


def trick_points(trick: list[Card], variant: Variant, is_last_trick: bool = False) -> int:
    """Summe der Kartenwerte im Stich plus +5 für den letzten Stich."""
    pts = sum(card_value(c, variant) for c in trick)
    if is_last_trick:
        pts += LAST_TRICK_BONUS
    return pts
