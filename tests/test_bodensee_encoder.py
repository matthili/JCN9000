"""Tests fuer den Bodensee-Encoder (bodensee_1.0.0)."""

from __future__ import annotations

import numpy as np
import pytest

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card, Rank, Suit
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant
from training.bodensee_encoder import (
    ACTION_DIM,
    ENCODING_VERSION,
    INPUT_DIM,
    SECTION_OFFSETS,
    SECTIONS,
    card_index,
    encode_state_bodensee,
    legal_action_mask_bodensee,
)


# --- Layout-Konsistenz ---


def test_encoding_version_string():
    assert ENCODING_VERSION == "bodensee_1.0.0"


def test_input_dim_ist_291():
    assert INPUT_DIM == 291


def test_action_dim_ist_36():
    assert ACTION_DIM == 36


def test_section_offsets_decken_input_dim_komplett_ab():
    """Die summierten Section-Groessen muessen exakt INPUT_DIM ergeben."""
    total = sum(end - start for start, end in SECTION_OFFSETS.values())
    assert total == INPUT_DIM


def test_alle_sections_im_dict():
    expected = {name for name, _ in SECTIONS}
    assert set(SECTION_OFFSETS.keys()) == expected


# --- encode_state_bodensee: Basis-Verhalten ---


def _make_basic_state(
    variant: Variant = Variant.trumpf(Suit.EICHEL),
    current_trick_cards: list[Card] | None = None,
    current_trick_starter: int = 0,
    player_idx: int = 0,
    opp_visible_table: list[Card] | None = None,
    opp_hand_count: int = 6,
    opp_hidden_table_count: int = 6,
    own_score: int = 0,
    opp_score: int = 0,
    round_idx: int = 0,
    trick_idx: int = 0,
    slalom: bool = False,
) -> BodenseeGameState:
    return BodenseeGameState(
        player_idx=player_idx,
        variant=variant,
        announcement=Announcement(variant=variant, slalom=slalom),
        current_trick_cards=current_trick_cards or [],
        current_trick_starter=current_trick_starter,
        completed_tricks=[],
        opponent_visible_table=opp_visible_table or [],
        opponent_hand_count=opp_hand_count,
        opponent_hidden_table_count=opp_hidden_table_count,
        own_score=own_score,
        opp_score=opp_score,
        round_idx=round_idx,
        trick_idx=trick_idx,
    )


def _make_table(visible_cards: list[Card], hidden_cards: list[Card | None]) -> list[TableStack]:
    """Hilfsfunktion: erstellt 6 Tisch-Stapel."""
    assert len(visible_cards) == 6
    assert len(hidden_cards) == 6
    return [
        TableStack(visible=v, hidden=h)
        for v, h in zip(visible_cards, hidden_cards)
    ]


def test_encode_state_shape_und_dtype():
    state = _make_basic_state()
    hand = [Card(Suit.EICHEL, Rank.ASS)]
    table = _make_table(
        visible_cards=[Card(Suit.HERZ, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=True)
    assert vec.shape == (INPUT_DIM,)
    assert vec.dtype == np.float32


def test_encode_state_own_hand_bits_gesetzt():
    state = _make_basic_state()
    hand = [Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.SECHS)]
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    off, _ = SECTION_OFFSETS["own_hand"]
    assert vec[off + card_index(Card(Suit.EICHEL, Rank.ASS))] == 1.0
    assert vec[off + card_index(Card(Suit.HERZ, Rank.SECHS))] == 1.0
    # Keine anderen Hand-Bits gesetzt
    hand_section = vec[off:off + 36]
    assert hand_section.sum() == 2.0


def test_encode_state_visible_table_und_hidden_mask():
    state = _make_basic_state()
    hand: list[Card] = []
    visible_cards = [
        Card(Suit.LAUB, Rank.SECHS),
        Card(Suit.LAUB, Rank.SIEBEN),
        Card(Suit.LAUB, Rank.ACHT),
        Card(Suit.LAUB, Rank.NEUN),
        Card(Suit.LAUB, Rank.ZEHN),
        Card(Suit.LAUB, Rank.UNTER),
    ]
    hidden_cards = [
        Card(Suit.HERZ, Rank.SECHS),  # Stapel 0: hidden vorhanden
        None,                          # Stapel 1: kein hidden
        Card(Suit.HERZ, Rank.ACHT),
        None,
        Card(Suit.HERZ, Rank.UNTER),
        None,
    ]
    table = _make_table(visible_cards, hidden_cards)
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)

    # own_visible_table: alle 6 Laub-Karten gesetzt
    vis_off, _ = SECTION_OFFSETS["own_visible_table"]
    assert vec[vis_off:vis_off + 36].sum() == 6.0

    # own_hidden_table_mask: Stapel 0, 2, 4 haben hidden
    mask_off, _ = SECTION_OFFSETS["own_hidden_table_mask"]
    assert vec[mask_off + 0] == 1.0
    assert vec[mask_off + 1] == 0.0
    assert vec[mask_off + 2] == 1.0
    assert vec[mask_off + 3] == 0.0
    assert vec[mask_off + 4] == 1.0
    assert vec[mask_off + 5] == 0.0


def test_encode_state_opp_hand_count_one_hot():
    state = _make_basic_state(opp_hand_count=3)
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    off, _ = SECTION_OFFSETS["opp_hand_count"]
    # Position 3 sollte gesetzt sein
    assert vec[off + 3] == 1.0
    # Andere nicht
    section = vec[off:off + 7]
    assert section.sum() == 1.0


def test_encode_state_opp_hand_count_gekappt():
    """Wenn opp_hand_count > MAX_HAND_SIZE, wird auf MAX gekappt."""
    state = _make_basic_state(opp_hand_count=100)
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    off, _ = SECTION_OFFSETS["opp_hand_count"]
    assert vec[off + 6] == 1.0  # MAX_HAND_SIZE


# --- i_am_leading + opp_lead_card ---


def test_encode_state_ich_lead_kein_opp_lead_card():
    state = _make_basic_state(
        player_idx=0,
        current_trick_starter=0,
        current_trick_cards=[],
    )
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    off, _ = SECTION_OFFSETS["i_am_leading"]
    assert vec[off] == 1.0
    # opp_lead_card sollte leer sein
    lead_off, _ = SECTION_OFFSETS["opp_lead_card"]
    assert vec[lead_off:lead_off + 36].sum() == 0.0


def test_encode_state_opp_hat_geleadet():
    state = _make_basic_state(
        player_idx=1,
        current_trick_starter=0,
        current_trick_cards=[Card(Suit.EICHEL, Rank.ASS)],
    )
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)

    # Nicht ich leade
    off, _ = SECTION_OFFSETS["i_am_leading"]
    assert vec[off] == 0.0

    # opp_lead_card hat Eichel-Ass
    lead_off, _ = SECTION_OFFSETS["opp_lead_card"]
    assert vec[lead_off + card_index(Card(Suit.EICHEL, Rank.ASS))] == 1.0
    assert vec[lead_off:lead_off + 36].sum() == 1.0


# --- value_per_card und strength_per_card ---


def test_encode_state_value_per_card_buur_ist_20():
    state = _make_basic_state(variant=Variant.trumpf(Suit.EICHEL))
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    val_off, _ = SECTION_OFFSETS["value_per_card"]
    # Eichel-Unter = Trumpf-Buur = 20, normalisiert = 1.0
    buur_idx = card_index(Card(Suit.EICHEL, Rank.UNTER))
    assert vec[val_off + buur_idx] == pytest.approx(1.0)


def test_encode_state_mode_one_hot():
    """Pro Variante muss genau ein Mode-Bit (oder zwei bei Slalom) gesetzt sein."""
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    hand: list[Card] = []

    # Trumpf
    vec = encode_state_bodensee(hand, table, _make_basic_state(variant=Variant.trumpf(Suit.EICHEL)), i_am_announcer=False)
    mode_off = SECTION_OFFSETS["mode"][0]
    assert vec[mode_off + 0] == 1.0  # is_trumpf

    # Slalom + Oben
    state = _make_basic_state(variant=Variant.oben(), slalom=True)
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    assert vec[mode_off + 2] == 1.0  # is_oben
    assert vec[mode_off + 4] == 1.0  # is_slalom


# --- played_cards_this_round ---


def test_encode_state_played_cards_aus_completed_tricks():
    state = _make_basic_state()
    state.completed_tricks = [
        CompletedTrick(
            starter=0,
            cards=(Card(Suit.EICHEL, Rank.ASS), Card(Suit.EICHEL, Rank.SIEBEN)),
        ),
    ]
    hand: list[Card] = []
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )
    vec = encode_state_bodensee(hand, table, state, i_am_announcer=False)
    off, _ = SECTION_OFFSETS["played_cards_this_round"]
    assert vec[off + card_index(Card(Suit.EICHEL, Rank.ASS))] == 1.0
    assert vec[off + card_index(Card(Suit.EICHEL, Rank.SIEBEN))] == 1.0
    assert vec[off:off + 36].sum() == 2.0


# --- legal_action_mask_bodensee ---


def test_legal_action_mask_leerer_stich():
    """Bei leerem Stich sind alle verfuegbaren Karten legal."""
    state = _make_basic_state(variant=Variant.oben(), current_trick_cards=[])
    hand = [Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.SECHS)]
    visible_table = [Card(Suit.LAUB, Rank.NEUN), Card(Suit.SCHELLE, Rank.OBER)]

    mask = legal_action_mask_bodensee(hand, visible_table, state)
    assert mask.shape == (36,)
    assert mask.dtype == np.uint8
    assert mask.sum() == 4  # 2 hand + 2 tisch
    for c in hand + visible_table:
        assert mask[card_index(c)] == 1


def test_legal_action_mask_bedienzwang_inklusive_tisch():
    """Bei Bedienzwang muessen sowohl Hand- als auch Tisch-Karten der Lead-Farbe legal sein."""
    state = _make_basic_state(
        variant=Variant.oben(),
        current_trick_cards=[Card(Suit.EICHEL, Rank.SIEBEN)],
        current_trick_starter=1,
        player_idx=0,
    )
    hand = [Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.SECHS)]
    visible_table = [Card(Suit.EICHEL, Rank.NEUN), Card(Suit.LAUB, Rank.KOENIG)]

    mask = legal_action_mask_bodensee(hand, visible_table, state)
    # Eichel-Ass (Hand) + Eichel-Neun (Tisch) sind legal
    assert mask[card_index(Card(Suit.EICHEL, Rank.ASS))] == 1
    assert mask[card_index(Card(Suit.EICHEL, Rank.NEUN))] == 1
    # Herz-Sechs und Laub-Koenig sind NICHT legal
    assert mask[card_index(Card(Suit.HERZ, Rank.SECHS))] == 0
    assert mask[card_index(Card(Suit.LAUB, Rank.KOENIG))] == 0


# --- Reproduzierbarkeit ---


def test_encode_state_deterministisch():
    """Zweimaliges Encoden desselben Zustands liefert identische Vektoren."""
    state = _make_basic_state(variant=Variant.trumpf(Suit.HERZ))
    hand = [Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.UNTER)]
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[Card(Suit.SCHELLE, Rank.SECHS)] * 3 + [None] * 3,
    )
    v1 = encode_state_bodensee(hand, table, state, i_am_announcer=True)
    v2 = encode_state_bodensee(hand, table, state, i_am_announcer=True)
    assert np.array_equal(v1, v2)


def test_encode_state_unterschiedlich_bei_anderer_hand():
    """Aenderung der Hand muss zu unterschiedlichem Vektor fuehren."""
    state = _make_basic_state()
    table = _make_table(
        visible_cards=[Card(Suit.LAUB, r) for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN, Rank.UNTER)],
        hidden_cards=[None] * 6,
    )

    v1 = encode_state_bodensee([Card(Suit.EICHEL, Rank.ASS)], table, state, False)
    v2 = encode_state_bodensee([Card(Suit.HERZ, Rank.SECHS)], table, state, False)
    assert not np.array_equal(v1, v2)
