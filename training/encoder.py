"""State-Encoder für das neuronale Netz.

Wandelt einen `GameState` + die eigene Hand in einen Featurevektor um, plus
eine Aktionsmaske, die illegale Karten ausblendet.

Karten-Index: `suit * 9 + rank` ergibt einen eindeutigen Index 0..35.
"""

from __future__ import annotations

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.rules import legal_moves
from jass_engine.variant import PlayMode


# ----- Karten-Index -----

NUM_CARDS = 36


def card_index(card: Card) -> int:
    """Eindeutiger Index 0..35 pro Karte (suit * 9 + rank)."""
    return int(card.suit) * 9 + int(card.rank)


def index_to_card(idx: int) -> Card:
    """Umkehrung von card_index."""
    suit = Suit(idx // 9)
    rank = Rank(idx % 9)
    return Card(suit, rank)


# ----- Feature-Layout -----

# Sections: own hand (36) | played history (36) | current trick (36)
#           | lead suit (4) | trump suit (4) | mode (4) | my seat (4) | starter seat (4)
#           | score own | score opp | trick_idx | round_idx
SECTIONS = [
    ("own_hand", 36),
    ("played_history", 36),
    ("current_trick", 36),
    ("lead_suit", 4),
    ("trump_suit", 4),
    ("mode", 4),            # [is_trumpf, is_oben, is_unten, is_slalom]
    ("my_seat", 4),
    ("starter_seat", 4),
    ("score_own_norm", 1),
    ("score_opp_norm", 1),
    ("trick_idx_norm", 1),
    ("round_idx_norm", 1),
]

INPUT_DIM = sum(size for _, size in SECTIONS)  # 136
ACTION_DIM = NUM_CARDS  # 36

# Offsets pro Section, für Tests und debug
SECTION_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _size in SECTIONS:
    SECTION_OFFSETS[_name] = (_offset, _offset + _size)
    _offset += _size


def encode_state(hand: list[Card], state: GameState) -> np.ndarray:
    """Wandelt Spielzustand in einen Featurevektor (float32, Shape (INPUT_DIM,))."""
    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    def write_one_hot_cards(cards: list[Card], section: str) -> None:
        offset = SECTION_OFFSETS[section][0]
        for c in cards:
            vec[offset + card_index(c)] = 1.0

    write_one_hot_cards(hand, "own_hand")

    played_history = [c for trick in state.completed_tricks for c in trick]
    write_one_hot_cards(played_history, "played_history")

    write_one_hot_cards(state.current_trick_cards, "current_trick")

    # Lead-Farbe
    if state.current_trick_cards:
        lead = state.current_trick_cards[0].suit
        vec[SECTION_OFFSETS["lead_suit"][0] + int(lead)] = 1.0

    # Trumpf-Farbe
    if state.variant.mode == PlayMode.TRUMPF:
        assert state.variant.trump_suit is not None
        vec[SECTION_OFFSETS["trump_suit"][0] + int(state.variant.trump_suit)] = 1.0

    # Modus-Flags: [is_trumpf, is_oben, is_unten, is_slalom]
    mode_off = SECTION_OFFSETS["mode"][0]
    if state.variant.mode == PlayMode.TRUMPF:
        vec[mode_off + 0] = 1.0
    elif state.variant.mode == PlayMode.OBEN:
        vec[mode_off + 1] = 1.0
    else:  # UNTEN
        vec[mode_off + 2] = 1.0
    if state.announcement.slalom:
        vec[mode_off + 3] = 1.0

    # Sitze (one-hot, absolute Position 0..3)
    vec[SECTION_OFFSETS["my_seat"][0] + state.player_idx] = 1.0
    vec[SECTION_OFFSETS["starter_seat"][0] + state.current_trick_starter] = 1.0

    # Punkte normalisieren
    vec[SECTION_OFFSETS["score_own_norm"][0]] = min(state.own_team_score / 1000.0, 1.0)
    vec[SECTION_OFFSETS["score_opp_norm"][0]] = min(state.opp_team_score / 1000.0, 1.0)

    # Stich-Index 0..8
    vec[SECTION_OFFSETS["trick_idx_norm"][0]] = state.trick_idx / 9.0

    # Rundenindex grob normalisiert
    vec[SECTION_OFFSETS["round_idx_norm"][0]] = min(state.round_idx / 20.0, 1.0)

    return vec


def legal_action_mask(hand: list[Card], state: GameState) -> np.ndarray:
    """Binärmaske der Form (36,): 1 für legale Karten, 0 sonst."""
    mask = np.zeros(NUM_CARDS, dtype=np.uint8)
    for c in legal_moves(hand, state.current_trick_cards, state.variant):
        mask[card_index(c)] = 1
    return mask


def action_index(card: Card) -> int:
    """Gibt den Aktionsindex für eine gespielte Karte zurück (= card_index)."""
    return card_index(card)


def all_card_indices() -> dict[Card, int]:
    """Hilfsfunktion für Tests/Debug."""
    return {Card(s, r): card_index(Card(s, r)) for s in ALL_SUITS for r in ALL_RANKS}
