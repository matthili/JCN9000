"""Tests fuer die Bodensee-Jass-Karten-Verteilung und Datenstrukturen."""

from __future__ import annotations

import random

import pytest

from jass_engine.bodensee.deal import (
    HAND_SIZE,
    NUM_PLAYERS,
    TABLE_STACKS,
    TOTAL_CARDS_PER_PLAYER,
    deal_bodensee,
    find_weli_holder_bodensee,
)
from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.card import Card, Rank, Suit
from jass_engine.deck import make_deck


# --- TableStack ---


def test_tablestack_anfangs_zwei_karten():
    s = TableStack(visible=Card(Suit.EICHEL, Rank.ASS), hidden=Card(Suit.HERZ, Rank.SECHS))
    assert s.card_count == 2
    assert s.has_visible
    assert s.has_hidden
    assert not s.is_empty


def test_tablestack_play_visible_deckt_hidden_auf():
    s = TableStack(
        visible=Card(Suit.EICHEL, Rank.ASS),
        hidden=Card(Suit.HERZ, Rank.SECHS),
    )
    played, new_visible = s.play_visible()
    assert played == Card(Suit.EICHEL, Rank.ASS)
    assert new_visible == Card(Suit.HERZ, Rank.SECHS)
    assert s.visible == Card(Suit.HERZ, Rank.SECHS)
    assert s.hidden is None
    assert s.card_count == 1


def test_tablestack_play_visible_leert_stapel_wenn_kein_hidden():
    s = TableStack(visible=Card(Suit.EICHEL, Rank.ASS), hidden=None)
    played, new_visible = s.play_visible()
    assert played == Card(Suit.EICHEL, Rank.ASS)
    assert new_visible is None
    assert s.is_empty


def test_tablestack_play_visible_ohne_sichtbar_wirft_fehler():
    s = TableStack(visible=None, hidden=None)
    with pytest.raises(RuntimeError):
        s.play_visible()


# --- BodenseePlayerState ---


def test_player_state_available_cards_kombiniert_hand_und_sichtbar():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.KOENIG)],
        table=[
            TableStack(
                visible=Card(Suit.LAUB, Rank.OBER),
                hidden=Card(Suit.SCHELLE, Rank.NEUN),
            ),
            TableStack(visible=Card(Suit.HERZ, Rank.SIEBEN), hidden=None),
            TableStack(visible=None, hidden=None),  # leerer Stapel
        ],
    )
    available = ps.available_cards
    assert Card(Suit.EICHEL, Rank.ASS) in available
    assert Card(Suit.HERZ, Rank.KOENIG) in available
    assert Card(Suit.LAUB, Rank.OBER) in available
    assert Card(Suit.HERZ, Rank.SIEBEN) in available
    # Verdeckte SCHELLE.NEUN darf NICHT in available sein
    assert Card(Suit.SCHELLE, Rank.NEUN) not in available
    assert len(available) == 4


def test_player_state_remove_from_hand():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.KOENIG)],
    )
    ps.remove_from_hand(Card(Suit.EICHEL, Rank.ASS))
    assert len(ps.hand) == 1
    assert Card(Suit.HERZ, Rank.KOENIG) in ps.hand


def test_player_state_remove_from_hand_unbekannte_karte():
    ps = BodenseePlayerState(hand=[Card(Suit.EICHEL, Rank.ASS)])
    with pytest.raises(ValueError):
        ps.remove_from_hand(Card(Suit.HERZ, Rank.SECHS))


def test_player_state_play_from_table_deckt_auf():
    ps = BodenseePlayerState(
        table=[
            TableStack(
                visible=Card(Suit.LAUB, Rank.OBER),
                hidden=Card(Suit.SCHELLE, Rank.NEUN),
            ),
        ],
    )
    new_visible = ps.play_from_table(Card(Suit.LAUB, Rank.OBER))
    assert new_visible == Card(Suit.SCHELLE, Rank.NEUN)
    assert ps.table[0].visible == Card(Suit.SCHELLE, Rank.NEUN)


def test_player_state_play_from_table_letzte_karte():
    ps = BodenseePlayerState(
        table=[TableStack(visible=Card(Suit.LAUB, Rank.OBER), hidden=None)],
    )
    new_visible = ps.play_from_table(Card(Suit.LAUB, Rank.OBER))
    assert new_visible is None
    assert ps.table[0].is_empty


def test_player_state_play_from_table_nicht_sichtbar():
    ps = BodenseePlayerState(
        table=[TableStack(visible=Card(Suit.LAUB, Rank.OBER), hidden=None)],
    )
    with pytest.raises(ValueError):
        ps.play_from_table(Card(Suit.HERZ, Rank.ASS))


# --- Deal ---


def test_deal_bodensee_anzahl_und_struktur():
    rng = random.Random(42)
    states = deal_bodensee(rng)
    assert len(states) == NUM_PLAYERS
    for ps in states:
        assert len(ps.hand) == HAND_SIZE
        assert len(ps.table) == TABLE_STACKS
        for stack in ps.table:
            assert stack.has_visible, "Jeder Stapel muss eine sichtbare Karte haben"
            assert stack.has_hidden, "Jeder Stapel muss eine verdeckte Karte haben"
        assert ps.total_cards_remaining == TOTAL_CARDS_PER_PLAYER


def test_deal_bodensee_alle_karten_verteilt_keine_doppelten():
    rng = random.Random(7)
    states = deal_bodensee(rng)
    all_cards: set[Card] = set()
    for ps in states:
        for c in ps.hand:
            assert c not in all_cards, f"Doppelte Karte: {c}"
            all_cards.add(c)
        for stack in ps.table:
            if stack.visible:
                assert stack.visible not in all_cards
                all_cards.add(stack.visible)
            if stack.hidden:
                assert stack.hidden not in all_cards
                all_cards.add(stack.hidden)
    assert len(all_cards) == 36, f"Expected 36 cards, got {len(all_cards)}"
    assert all_cards == set(make_deck())


def test_deal_bodensee_reproduzierbar_mit_seed():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    s1 = deal_bodensee(rng1)
    s2 = deal_bodensee(rng2)
    for ps1, ps2 in zip(s1, s2):
        assert ps1.hand == ps2.hand
        for stack1, stack2 in zip(ps1.table, ps2.table):
            assert stack1.visible == stack2.visible
            assert stack1.hidden == stack2.hidden


def test_find_weli_holder_findet_immer_einen():
    """Ueber viele zufaellige Verteilungen muss der Weli-Halter findbar sein."""
    for seed in range(50):
        rng = random.Random(seed)
        states = deal_bodensee(rng)
        holder = find_weli_holder_bodensee(states)
        assert holder in (0, 1)


def test_find_weli_holder_grobe_verteilung():
    """Ueber viele Deals sollte der Weli ungefaehr 50/50 zwischen den Spielern landen."""
    counts = {0: 0, 1: 0}
    for seed in range(1000):
        rng = random.Random(seed)
        states = deal_bodensee(rng)
        counts[find_weli_holder_bodensee(states)] += 1
    # Bei 1000 Deals und 50/50 ist SD ~16. Toleranz: 400-600.
    assert 400 <= counts[0] <= 600
    assert 400 <= counts[1] <= 600
