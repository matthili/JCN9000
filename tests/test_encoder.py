"""Tests für den State-Encoder (Version 2.0.0)."""

from __future__ import annotations

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant
from training.encoder import (
    ACTION_DIM,
    ENCODING_VERSION,
    INPUT_DIM,
    NUM_CARDS,
    SECTION_OFFSETS,
    action_index,
    card_index,
    encode_state,
    index_to_card,
    legal_action_mask,
)


def test_encoding_version():
    assert ENCODING_VERSION == "2.0.0"


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
    # 9*36 (eigene Hand + 4 played-by-* + 4 current-trick-by-*) + 5*4 + 4 (skalare) = 348
    assert INPUT_DIM == 348
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
    assert vec[off:end].sum() == 2.0
    assert vec[off + card_index(Card(Suit.EICHEL, Rank.UNTER))] == 1.0
    assert vec[off + card_index(Card(Suit.HERZ, Rank.ASS))] == 1.0


def test_encode_played_history_pro_spieler():
    """Ein abgeschlossener Stich mit Starter=Spieler 1, 4 Karten in Reihenfolge.
    Aus Sicht von Spieler 0:
      Position 0 in Stich = Spieler 1 = mein "linker Gegner" (rel=1)
      Position 1 in Stich = Spieler 2 = mein "Partner" (rel=2)
      Position 2 in Stich = Spieler 3 = mein "rechter Gegner" (rel=3)
      Position 3 in Stich = Spieler 0 = ich (rel=0)
    """
    completed = [
        CompletedTrick(
            starter=1,
            cards=(
                Card(Suit.EICHEL, Rank.ASS),     # Spieler 1 (links)
                Card(Suit.EICHEL, Rank.ZEHN),    # Spieler 2 (Partner)
                Card(Suit.EICHEL, Rank.KOENIG),  # Spieler 3 (rechts)
                Card(Suit.EICHEL, Rank.OBER),    # Spieler 0 (ich)
            ),
        ),
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
    # Karte ASS sollte in "played_by_left" sein
    off_l, _ = SECTION_OFFSETS["played_by_left"]
    assert vec[off_l + card_index(Card(Suit.EICHEL, Rank.ASS))] == 1.0
    # ZEHN in "played_by_partner"
    off_p, _ = SECTION_OFFSETS["played_by_partner"]
    assert vec[off_p + card_index(Card(Suit.EICHEL, Rank.ZEHN))] == 1.0
    # KOENIG in "played_by_right"
    off_r, _ = SECTION_OFFSETS["played_by_right"]
    assert vec[off_r + card_index(Card(Suit.EICHEL, Rank.KOENIG))] == 1.0
    # OBER in "played_by_me"
    off_m, _ = SECTION_OFFSETS["played_by_me"]
    assert vec[off_m + card_index(Card(Suit.EICHEL, Rank.OBER))] == 1.0


def test_encode_current_trick_pro_spieler_und_lead_suit():
    """Aktueller Stich: starter=0 (ich), 2 Karten gespielt:
    - Position 0: Spieler 0 (ich) -> current_trick_by_me
    - Position 1: Spieler 1 (links) -> current_trick_by_left
    """
    state = GameState(
        player_idx=0,
        variant=Variant.trumpf(Suit.HERZ),
        announcement=Announcement(variant=Variant.trumpf(Suit.HERZ)),
        current_trick_cards=[
            Card(Suit.LAUB, Rank.ASS),       # ich
            Card(Suit.LAUB, Rank.SIEBEN),    # links
        ],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off_m, _ = SECTION_OFFSETS["current_trick_by_me"]
    off_l, _ = SECTION_OFFSETS["current_trick_by_left"]
    assert vec[off_m + card_index(Card(Suit.LAUB, Rank.ASS))] == 1.0
    assert vec[off_l + card_index(Card(Suit.LAUB, Rank.SIEBEN))] == 1.0
    # Lead-Farbe = Laub
    off_lead, _ = SECTION_OFFSETS["lead_suit"]
    assert vec[off_lead + int(Suit.LAUB)] == 1.0
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
    assert vec[off + 0] == 0.0
    assert vec[off + 1] == 1.0
    assert vec[off + 2] == 0.0
    assert vec[off + 3] == 0.0
    off_tr, end_tr = SECTION_OFFSETS["trump_suit"]
    assert vec[off_tr:end_tr].sum() == 0.0


def test_encode_mode_slalom():
    state = GameState(
        player_idx=0,
        variant=Variant.unten(),
        announcement=Announcement(variant=Variant.oben(), slalom=True),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off, _ = SECTION_OFFSETS["mode"]
    assert vec[off + 2] == 1.0
    assert vec[off + 3] == 1.0


def test_encode_my_seat_und_starter_relativ():
    """Eigener Sitz absolut, Starter relativ zu mir."""
    state = GameState(
        player_idx=2,
        variant=Variant.oben(),
        announcement=Announcement(variant=Variant.oben()),
        current_trick_cards=[],
        current_trick_starter=1,  # Spieler 1 = mein "rechter Gegner" (rel=3 von Sicht 2)
        teams=[0, 1, 0, 1],
    )
    vec = encode_state([], state)
    off_s, _ = SECTION_OFFSETS["my_seat"]
    assert vec[off_s + 2] == 1.0   # absoluter Sitz 2
    off_st, _ = SECTION_OFFSETS["starter_seat_relative"]
    # Starter (Spieler 1) ist von Spieler 2 aus gesehen rel=(1-2)%4 = 3
    assert vec[off_st + 3] == 1.0


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
        Card(Suit.HERZ, Rank.ASS),
        Card(Suit.HERZ, Rank.SECHS),
        Card(Suit.LAUB, Rank.ASS),
    ]
    mask = legal_action_mask(hand, state)
    assert mask[card_index(Card(Suit.HERZ, Rank.ASS))] == 1
    assert mask[card_index(Card(Suit.HERZ, Rank.SECHS))] == 1
    assert mask[card_index(Card(Suit.LAUB, Rank.ASS))] == 0


def test_action_index_konsistent_mit_card_index():
    c = Card(Suit.SCHELLE, Rank.OBER)
    assert action_index(c) == card_index(c)
