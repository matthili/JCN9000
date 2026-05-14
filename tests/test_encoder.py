"""Tests für den State-Encoder."""

from __future__ import annotations

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from training.encoder import (
    ACTION_DIM,
    INPUT_DIM,
    NUM_CARDS,
    SECTION_OFFSETS,
    action_index,
    card_index,
    encode_state,
    index_to_card,
    legal_action_mask,
)


def test_card_index_eindeutig():
    indices = set()
    for s in ALL_SUITS:
        for r in ALL_RANKS:
            indices.add(card_index(Card(s, r)))
    assert len(indices) == NUM_CARDS == 36
    assert min(indices) == 0
    assert max(indices) == 35


def test_card_index_invertierbar():
    for s in ALL_SUITS:
        for r in ALL_RANKS:
            c = Card(s, r)
            assert index_to_card(card_index(c)) == c


def test_input_dim_konstante():
    # 3*36 + 5*4 + 4 (scalars) = 108 + 20 + 4 = 132
    assert INPUT_DIM == 132
    assert ACTION_DIM == 36


def test_encode_state_shape_und_typ():
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.EICHEL),
        announcement=Announcement(variant=Variant.trumpf(Suit.EICHEL)),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    hand = [Card(Suit.EICHEL, Rank.UNTER), Card(Suit.HERZ, Rank.ASS)]
    vec = encode_state(hand, state)
    assert vec.shape == (INPUT_DIM,)
    assert vec.dtype == np.float32


def test_encode_own_hand():
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.EICHEL),
        announcement=Announcement(variant=Variant.trumpf(Suit.EICHEL)),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    hand = [Card(Suit.EICHEL, Rank.UNTER), Card(Suit.HERZ, Rank.ASS)]
    vec = encode_state(hand, state)
    off, end = SECTION_OFFSETS["own_hand"]
    # Exakt 2 Karten sollen 1.0 sein
    assert vec[off:end].sum() == 2.0
    assert vec[off + card_index(Card(Suit.EICHEL, Rank.UNTER))] == 1.0
    assert vec[off + card_index(Card(Suit.HERZ, Rank.ASS))] == 1.0


def test_encode_played_history():
    completed = [
        [Card(Suit.EICHEL, Rank.ASS), Card(Suit.EICHEL, Rank.ZEHN),
         Card(Suit.EICHEL, Rank.KOENIG), Card(Suit.EICHEL, Rank.OBER)],
    ]
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.EICHEL),
        announcement=Announcement(variant=Variant.trumpf(Suit.EICHEL)),
        current_trick_cards=[],
        current_trick_starter=0,
        completed_tricks=completed,
        teams=[0, 1, 0, 1],
        trick_idx=1,
    )
    vec = encode_state([], state)
    off, end = SECTION_OFFSETS["played_history"]
    assert vec[off:end].sum() == 4.0


def test_encode_current_trick_und_lead_suit():
    state = GameState(
        player_idx=2,
        variant=Variant.trumpf(Suit.HERZ),
        announcement=Announcement(variant=Variant.trumpf(Suit.HERZ)),
        current_trick_cards=[Card(Suit.LAUB, Rank.ASS), Card(Suit.LAUB, Rank.SIEBEN)],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off_t, _ = SECTION_OFFSETS["current_trick"]
    assert vec[off_t + card_index(Card(Suit.LAUB, Rank.ASS))] == 1.0
    assert vec[off_t + card_index(Card(Suit.LAUB, Rank.SIEBEN))] == 1.0
    # Lead-Farbe = Laub
    off_l, _ = SECTION_OFFSETS["lead_suit"]
    assert vec[off_l + int(Suit.LAUB)] == 1.0
    # Trumpf-Farbe = Herz
    off_tr, _ = SECTION_OFFSETS["trump_suit"]
    assert vec[off_tr + int(Suit.HERZ)] == 1.0


def test_encode_mode_oben():
    state = GameState(
        player_idx=0,
        variant=Variant.oben(),
        announcement=Announcement(variant=Variant.oben()),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off, _ = SECTION_OFFSETS["mode"]
    assert vec[off + 0] == 0.0  # not trumpf
    assert vec[off + 1] == 1.0  # is oben
    assert vec[off + 2] == 0.0  # not unten
    assert vec[off + 3] == 0.0  # not slalom
    # Trumpf-Farbe muss leer sein
    off_tr, end_tr = SECTION_OFFSETS["trump_suit"]
    assert vec[off_tr:end_tr].sum() == 0.0


def test_encode_mode_slalom():
    state = GameState(
        player_idx=0,
        variant=Variant.unten(),  # aktueller Stich ist unten
        announcement=Announcement(variant=Variant.oben(), slalom=True),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off, _ = SECTION_OFFSETS["mode"]
    assert vec[off + 2] == 1.0  # aktueller Stich ist unten
    assert vec[off + 3] == 1.0  # Ansage war Slalom


def test_encode_my_seat_und_starter():
    state = GameState(
        player_idx=2,
        variant=Variant.oben(),
        announcement=Announcement(variant=Variant.oben()),
        current_trick_cards=[],
        current_trick_starter=1,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off_s, _ = SECTION_OFFSETS["my_seat"]
    assert vec[off_s + 2] == 1.0
    off_st, _ = SECTION_OFFSETS["starter_seat"]
    assert vec[off_st + 1] == 1.0


def test_legal_action_mask_erste_karte_alles_erlaubt():
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.EICHEL),
        announcement=Announcement(variant=Variant.trumpf(Suit.EICHEL)),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    hand = [
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.HERZ, Rank.ASS),
        Card(Suit.LAUB, Rank.SECHS),
    ]
    mask = legal_action_mask(hand, state)
    assert mask.shape == (NUM_CARDS,)
    # Genau die 3 Karten in der Hand sind erlaubt
    assert mask.sum() == 3
    for c in hand:
        assert mask[card_index(c)] == 1


def test_legal_action_mask_farbzwang():
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.EICHEL),
        announcement=Announcement(variant=Variant.trumpf(Suit.EICHEL)),
        current_trick_cards=[Card(Suit.HERZ, Rank.KOENIG)],
        current_trick_starter=3,
        teams=[0, 1, 0, 1],
    )
    hand = [
        Card(Suit.HERZ, Rank.ASS),    # legal: bedient
        Card(Suit.HERZ, Rank.SECHS),  # legal: bedient
        Card(Suit.LAUB, Rank.ASS),    # illegal: bricht Farbzwang
    ]
    mask = legal_action_mask(hand, state)
    assert mask[card_index(Card(Suit.HERZ, Rank.ASS))] == 1
    assert mask[card_index(Card(Suit.HERZ, Rank.SECHS))] == 1
    assert mask[card_index(Card(Suit.LAUB, Rank.ASS))] == 0


def test_action_index_konsistent_mit_card_index():
    c = Card(Suit.SCHELLE, Rank.OBER)
    assert action_index(c) == card_index(c)
