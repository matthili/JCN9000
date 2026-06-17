"""Tests fuer training.sample_weights -- Anspiel-Erkennung aus dem State-Vektor.

TF-frei (nutzt nur den NumPy-Encoder), laeuft also auch ohne TensorFlow.
"""

from __future__ import annotations

import numpy as np

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from training.encoder import encode_state
from training.sample_weights import is_lead_position, lead_sample_weights

HAND = [
    Card(Suit.EICHEL, Rank.SECHS),
    Card(Suit.EICHEL, Rank.SIEBEN),
    Card(Suit.SCHELLE, Rank.ASS),
    Card(Suit.HERZ, Rank.KOENIG),
    Card(Suit.LAUB, Rank.NEUN),
]


def _state(current_trick_cards: list[Card]) -> GameState:
    return GameState(
        player_idx=0,
        variant=Variant.unten(),
        announcement=Announcement(variant=Variant.unten()),
        current_trick_cards=current_trick_cards,
        current_trick_starter=0 if not current_trick_cards else 3,
        teams=[0, 1, 0, 1],
        completed_tricks=[],
        round_idx=0,
        trick_idx=0,
        num_players=4,
    )


def _stack(*states: GameState) -> np.ndarray:
    return np.stack([encode_state(HAND, s) for s in states]).astype(np.float32)


def test_lead_vs_nonlead_detection() -> None:
    x = _stack(_state([]), _state([Card(Suit.LAUB, Rank.ASS)]))
    assert is_lead_position(x).tolist() == [True, False]


def test_weights_apply_only_to_leads() -> None:
    x = _stack(_state([]), _state([Card(Suit.LAUB, Rank.ASS)]))
    w = lead_sample_weights(x, lead_weight=3.0)
    assert w.dtype == np.float32
    assert w.tolist() == [3.0, 1.0]


def test_weight_one_is_neutral() -> None:
    x = _stack(_state([]))
    assert lead_sample_weights(x, lead_weight=1.0).tolist() == [1.0]
