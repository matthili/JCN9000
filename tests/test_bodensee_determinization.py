"""Tests fuer die Bodensee-Determinization."""

from __future__ import annotations

import random

import pytest

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.card import Card, Rank, Suit
from jass_engine.deck import make_deck
from jass_engine.trick import CompletedTrick
from training.data.bodensee_determinization import (
    determinize_bodensee,
    determinize_bodensee_states,
)


def test_determinize_anfangs_alle_36_karten_im_spiel():
    """Direkt nach dem Austeilen: 6 own_hand + 6 own_visible + 6 opp_visible +
    6 own_hidden + 6 opp_hand + 6 opp_hidden = 36 Karten."""
    own_hand = [Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)]
    own_visible = [Card(Suit.EICHEL, r) for r in (Rank.OBER, Rank.KOENIG, Rank.ASS)] + \
                  [Card(Suit.SCHELLE, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]
    opp_visible = [Card(Suit.SCHELLE, r) for r in (Rank.NEUN, Rank.ZEHN, Rank.UNTER)] + \
                  [Card(Suit.HERZ, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]

    own_hidden, opp_hand, opp_hidden = determinize_bodensee(
        own_hand=own_hand,
        own_visible_table=own_visible,
        own_hidden_count=6,
        opp_visible_table=opp_visible,
        opp_hand_count=6,
        opp_hidden_count=6,
        completed_tricks=[],
        current_trick_cards=[],
        rng=random.Random(42),
    )
    assert len(own_hidden) == 6
    assert len(opp_hand) == 6
    assert len(opp_hidden) == 6


def test_determinize_keine_doppelten_karten():
    """Die determinisierten Karten + bekannte Karten ergeben das volle Deck."""
    own_hand = [Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)]
    own_visible = [Card(Suit.EICHEL, r) for r in (Rank.OBER, Rank.KOENIG, Rank.ASS)] + \
                  [Card(Suit.SCHELLE, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]
    opp_visible = [Card(Suit.SCHELLE, r) for r in (Rank.NEUN, Rank.ZEHN, Rank.UNTER)] + \
                  [Card(Suit.HERZ, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]

    own_hidden, opp_hand, opp_hidden = determinize_bodensee(
        own_hand=own_hand,
        own_visible_table=own_visible,
        own_hidden_count=6,
        opp_visible_table=opp_visible,
        opp_hand_count=6,
        opp_hidden_count=6,
        completed_tricks=[],
        current_trick_cards=[],
        rng=random.Random(42),
    )
    all_cards = set(own_hand) | set(own_visible) | set(opp_visible) | set(own_hidden) | set(opp_hand) | set(opp_hidden)
    assert len(all_cards) == 36
    assert all_cards == set(make_deck())


def test_determinize_inkonsistente_counts_wirft():
    """Wenn die counts nicht zu den verbleibenden unbekannten Karten passen, Fehler."""
    own_hand = [Card(Suit.EICHEL, Rank.SECHS)]
    with pytest.raises(ValueError, match="Inkonsistente Karten-Buchhaltung"):
        determinize_bodensee(
            own_hand=own_hand,
            own_visible_table=[],
            own_hidden_count=10,  # zu viele unbekannte erwartet
            opp_visible_table=[],
            opp_hand_count=10,
            opp_hidden_count=10,
            completed_tricks=[],
            current_trick_cards=[],
            rng=random.Random(0),
        )


def test_determinize_nach_einigen_stichen():
    """Mit einigen gespielten Stichen: weniger unbekannte Karten."""
    own_hand = [Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN)]
    own_visible = [Card(Suit.EICHEL, Rank.UNTER), Card(Suit.EICHEL, Rank.OBER),
                   Card(Suit.SCHELLE, Rank.SECHS), Card(Suit.SCHELLE, Rank.SIEBEN),
                   Card(Suit.SCHELLE, Rank.ACHT), Card(Suit.SCHELLE, Rank.NEUN)]
    opp_visible = [Card(Suit.HERZ, Rank.SECHS), Card(Suit.HERZ, Rank.SIEBEN)]
    # 4 Karten wurden in completed_tricks gespielt
    completed = [
        CompletedTrick(starter=0, cards=(Card(Suit.LAUB, Rank.SECHS), Card(Suit.LAUB, Rank.SIEBEN))),
        CompletedTrick(starter=1, cards=(Card(Suit.LAUB, Rank.ACHT), Card(Suit.LAUB, Rank.NEUN))),
    ]

    # bekannt: 5 (own_hand) + 6 (own_vis) + 2 (opp_vis) + 4 (completed) = 17
    # unbekannt: 36 - 17 = 19
    # Verteilung: own_hidden + opp_hand + opp_hidden = 19
    own_hidden, opp_hand, opp_hidden = determinize_bodensee(
        own_hand=own_hand,
        own_visible_table=own_visible,
        own_hidden_count=5,
        opp_visible_table=opp_visible,
        opp_hand_count=10,  # bisschen unrealistisch hoch, aber stimmt mathematisch
        opp_hidden_count=4,
        completed_tricks=completed,
        current_trick_cards=[],
        rng=random.Random(42),
    )
    assert len(own_hidden) == 5
    assert len(opp_hand) == 10
    assert len(opp_hidden) == 4


def test_determinize_reproduzierbar_mit_seed():
    own_hand = [Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)]
    own_visible = [Card(Suit.EICHEL, r) for r in (Rank.OBER, Rank.KOENIG, Rank.ASS)] + \
                  [Card(Suit.SCHELLE, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]
    opp_visible = [Card(Suit.SCHELLE, r) for r in (Rank.NEUN, Rank.ZEHN, Rank.UNTER)] + \
                  [Card(Suit.HERZ, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT)]

    r1 = determinize_bodensee(
        own_hand=own_hand, own_visible_table=own_visible, own_hidden_count=6,
        opp_visible_table=opp_visible, opp_hand_count=6, opp_hidden_count=6,
        completed_tricks=[], current_trick_cards=[],
        rng=random.Random(123),
    )
    r2 = determinize_bodensee(
        own_hand=own_hand, own_visible_table=own_visible, own_hidden_count=6,
        opp_visible_table=opp_visible, opp_hand_count=6, opp_hidden_count=6,
        completed_tricks=[], current_trick_cards=[],
        rng=random.Random(123),
    )
    assert r1 == r2


def test_determinize_states_baut_zwei_player_states():
    """determinize_bodensee_states liefert 2 vollstaendige BodenseePlayerStates.
    Setup mit vollem 36-Karten-Deck im Anfangs-Zustand."""
    own_state = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        table=[
            TableStack(visible=Card(Suit.EICHEL, Rank.OBER), hidden=Card(Suit.SCHELLE, Rank.SECHS)),
            TableStack(visible=Card(Suit.EICHEL, Rank.KOENIG), hidden=Card(Suit.SCHELLE, Rank.SIEBEN)),
            TableStack(visible=Card(Suit.EICHEL, Rank.ASS), hidden=Card(Suit.SCHELLE, Rank.ACHT)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.NEUN), hidden=Card(Suit.SCHELLE, Rank.ZEHN)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.UNTER), hidden=Card(Suit.SCHELLE, Rank.OBER)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.KOENIG), hidden=Card(Suit.SCHELLE, Rank.ASS)),
        ],
    )
    # Gegner: 6 visible auf dem Tisch (alle Herz)
    opp_visible = [Card(Suit.HERZ, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)]

    states = determinize_bodensee_states(
        own_state=own_state,
        opp_visible_table=opp_visible,
        opp_hand_count=6,
        opp_hidden_count=6,
        completed_tricks=[],
        current_trick_cards=[],
        own_seat=0,
        rng=random.Random(7),
    )
    assert len(states) == 2
    s0, s1 = states
    # Spieler 0 (own): Hand unveraendert
    assert s0.hand == own_state.hand
    # Tisch-Stapel-Struktur erhalten (6 Stapel, alle mit hidden)
    assert len(s0.table) == 6
    for st in s0.table:
        assert st.has_hidden is True

    # Spieler 1 (opp): Hand hat 6 Karten
    assert len(s1.hand) == 6
    # Tisch-Karten richtig
    assert s1.visible_table_cards == opp_visible
    # Hidden-Karten beim Gegner sollten 6 sein
    assert sum(1 for st in s1.table if st.has_hidden) == 6


def test_determinize_states_eigene_hidden_karten_random():
    """Aufruf mit demselben own_state, aber anderem RNG -> verschiedene hidden-Karten."""
    own_state = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        table=[
            TableStack(visible=Card(Suit.EICHEL, Rank.OBER), hidden=Card(Suit.HERZ, Rank.SECHS)),
            TableStack(visible=Card(Suit.EICHEL, Rank.KOENIG), hidden=Card(Suit.HERZ, Rank.SIEBEN)),
            TableStack(visible=Card(Suit.EICHEL, Rank.ASS), hidden=Card(Suit.HERZ, Rank.ACHT)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.SECHS), hidden=Card(Suit.HERZ, Rank.NEUN)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.SIEBEN), hidden=Card(Suit.HERZ, Rank.ZEHN)),
            TableStack(visible=Card(Suit.SCHELLE, Rank.ACHT), hidden=Card(Suit.HERZ, Rank.UNTER)),
        ],
    )
    opp_visible = [Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)]

    s_a = determinize_bodensee_states(
        own_state=own_state, opp_visible_table=opp_visible,
        opp_hand_count=6, opp_hidden_count=6,
        completed_tricks=[], current_trick_cards=[],
        rng=random.Random(1),
    )
    s_b = determinize_bodensee_states(
        own_state=own_state, opp_visible_table=opp_visible,
        opp_hand_count=6, opp_hidden_count=6,
        completed_tricks=[], current_trick_cards=[],
        rng=random.Random(2),
    )
    # Hidden-Karten der eigenen Stapel: bei verschiedenen RNGs sollten sie verschieden sein
    hidden_a = [s.hidden for s in s_a[0].table]
    hidden_b = [s.hidden for s in s_b[0].table]
    assert hidden_a != hidden_b
