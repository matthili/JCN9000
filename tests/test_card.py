"""Tests für Card, Suit, Rank, Deck."""

from __future__ import annotations

import random

from jass_engine.card import ALL_RANKS, ALL_SUITS, WELI, Card, Rank, Suit
from jass_engine.deck import deal, find_weli_holder, make_deck


def test_deck_hat_36_karten():
    deck = make_deck()
    assert len(deck) == 36
    assert len(set(deck)) == 36


def test_deck_alle_kombinationen():
    deck = make_deck()
    expected = {Card(s, r) for s in ALL_SUITS for r in ALL_RANKS}
    assert set(deck) == expected


def test_weli_identifikation():
    assert WELI.is_weli is True
    assert WELI == Card(Suit.SCHELLE, Rank.SECHS)
    # Andere 6er sind keine Welis
    assert Card(Suit.EICHEL, Rank.SECHS).is_weli is False
    assert Card(Suit.HERZ, Rank.SECHS).is_weli is False
    assert Card(Suit.LAUB, Rank.SECHS).is_weli is False


def test_deal_verteilt_alle_karten():
    rng = random.Random(42)
    hands = deal(num_players=4, rng=rng)
    assert len(hands) == 4
    for hand in hands:
        assert len(hand) == 9
    all_cards = [c for h in hands for c in h]
    assert len(set(all_cards)) == 36


def test_find_weli_holder_immer_eindeutig():
    rng = random.Random(0)
    for _ in range(100):
        hands = deal(num_players=4, rng=rng)
        idx = find_weli_holder(hands)
        assert any(c.is_weli for c in hands[idx])
        # Genau ein Spieler hat den Weli
        anzahl = sum(any(c.is_weli for c in h) for h in hands)
        assert anzahl == 1


def test_rank_reihenfolge():
    # Nicht-Trumpf-Reihenfolge: 6 < 7 < 8 < 9 < 10 < U < O < K < A
    assert (
        Rank.SECHS < Rank.SIEBEN < Rank.ACHT < Rank.NEUN < Rank.ZEHN
        < Rank.UNTER < Rank.OBER < Rank.KOENIG < Rank.ASS
    )
