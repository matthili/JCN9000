"""Tests fuer die void-aware (constraint-treue) Determinisierung."""

from __future__ import annotations

import random

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.trick import CompletedTrick
from training.data.determinization import _assign_constrained, determinize_hands


def _all_of(suit: Suit) -> set[Card]:
    return {Card(suit, r) for r in ALL_RANKS}


# ---------------------------------------------------------------------------
# Kern: _assign_constrained
# ---------------------------------------------------------------------------
def test_assign_respects_forbidden():
    rng = random.Random(0)
    unknown = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS][:24]
    sizes = {1: 8, 2: 8, 3: 8}
    forbidden = {1: _all_of(Suit.SCHELLE)}  # Sitz 1 darf keine Schelle
    assign = _assign_constrained(unknown, sizes, forbidden, rng)
    assert assign is not None
    assert all(c.suit != Suit.SCHELLE for c in assign[1])
    for s in (1, 2, 3):
        assert len(assign[s]) == sizes[s]
    # Alle Karten verteilt, keine doppelt
    flat = [c for s in assign for c in assign[s]]
    assert sorted(flat, key=lambda c: (int(c.suit), int(c.rank))) == \
        sorted(unknown, key=lambda c: (int(c.suit), int(c.rank)))


def test_assign_forces_concentration():
    # 9 Schelle muessen alle zu Sitz 1, weil 2 und 3 sie verbieten.
    rng = random.Random(1)
    schelle = list(_all_of(Suit.SCHELLE))           # 9 Karten
    others = [Card(Suit.EICHEL, r) for r in ALL_RANKS]  # 9 Karten
    unknown = schelle + others                       # 18
    sizes = {1: 9, 2: 5, 3: 4}                        # Summe 18
    forbidden = {2: _all_of(Suit.SCHELLE), 3: _all_of(Suit.SCHELLE)}
    assign = _assign_constrained(unknown, sizes, forbidden, rng)
    assert assign is not None
    assert set(assign[1]) >= set(schelle), "Alle Schelle muessen zu Sitz 1."


def test_assign_infeasible_returns_none():
    rng = random.Random(2)
    unknown = [Card(Suit.HERZ, Rank.ASS)]
    sizes = {1: 1}
    forbidden = {1: {Card(Suit.HERZ, Rank.ASS)}}  # einzige Karte fuer einzigen Sitz verboten
    assert _assign_constrained(unknown, sizes, forbidden, rng) is None


# ---------------------------------------------------------------------------
# Integration: determinize_hands mit forbidden_by_seat
# ---------------------------------------------------------------------------
def _consistent_state_after_one_trick():
    """Baut einen konsistenten Zustand: voller Deck-Split, 1 Stich gespielt."""
    deck = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
    seat_cards = [deck[i * 9:(i + 1) * 9] for i in range(4)]
    # Trick 0: jeder Sitz spielt seine erste Karte (Anspieler = Sitz 0)
    trick = CompletedTrick(
        starter=0,
        cards=tuple(seat_cards[s][0] for s in range(4)),
    )
    own_seat = 0
    own_hand = seat_cards[0][1:]  # 8 Karten (erste ist gespielt)
    return own_seat, own_hand, [trick]


def test_determinize_respects_forbidden_and_sizes():
    own_seat, own_hand, completed = _consistent_state_after_one_trick()
    # Verbiete Sitz 2 alle Laub-Karten (unter den Unbekannten).
    forbidden = {2: _all_of(Suit.LAUB)}
    rng = random.Random(7)
    hands = determinize_hands(
        own_seat=own_seat,
        own_hand=own_hand,
        completed_tricks=completed,
        current_trick_cards=[],
        current_trick_starter=0,
        num_players=4,
        rng=rng,
        forbidden_by_seat=forbidden,
    )
    assert hands[own_seat] == own_hand, "Eigene Hand unveraendert."
    assert all(c.suit != Suit.LAUB for c in hands[2]), "Sitz 2 darf kein Laub haben."
    # Handgroessen: jeder Mitspieler 8 (eine Karte gespielt)
    for s in (1, 2, 3):
        assert len(hands[s]) == 8
    # Keine Karte doppelt / keine schon gespielte. 4 Haende = 36 - 4 gespielte = 32.
    played = set(completed[0].cards)
    flat = [c for s in range(4) for c in hands[s]]
    assert len(flat) == len(set(flat)) == 32
    assert not (set(flat) & played), "Gespielte Karten duerfen nicht verteilt sein."


def test_determinize_backward_compatible_without_forbidden():
    own_seat, own_hand, completed = _consistent_state_after_one_trick()
    rng = random.Random(3)
    hands = determinize_hands(
        own_seat=own_seat,
        own_hand=own_hand,
        completed_tricks=completed,
        current_trick_cards=[],
        current_trick_starter=0,
        num_players=4,
        rng=rng,
    )
    assert hands[own_seat] == own_hand
    for s in (1, 2, 3):
        assert len(hands[s]) == 8
