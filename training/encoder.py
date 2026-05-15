"""State-Encoder fuer das neuronale Netz (Version 2.0.0).

Wandelt einen `GameState` + die eigene Hand in einen Featurevektor um, plus
eine Aktionsmaske, die illegale Karten ausblendet.

Neuerungen gegenueber v1.0.0 (siehe spec/state_encoding.md):
- Played-History ist jetzt **spieler-zugeordnet** (4 × 36 Bits, je relative
  Position). Damit kann das NN inferieren, wer welche Karte gespielt hat --
  Voraussetzung fuer Opponent Modeling (klassisches "der Gegner hat keinen
  Trumpf, sonst haette er gestochen"-Schluss).
- Current-Trick ebenfalls spieler-positioniert (4 × 36 Bits).

Karten-Index: `suit * 9 + rank` ergibt einen eindeutigen Index 0..35.
"""

from __future__ import annotations

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.rules import legal_moves
from jass_engine.variant import PlayMode


ENCODING_VERSION = "2.0.0"

NUM_CARDS = 36
NUM_PLAYERS = 4
NUM_RELATIVE_POSITIONS = 4  # 0 = me, 1 = left, 2 = partner, 3 = right


def card_index(card: Card) -> int:
    """Eindeutiger Index 0..35 pro Karte (suit * 9 + rank)."""
    return int(card.suit) * 9 + int(card.rank)


def index_to_card(idx: int) -> Card:
    """Umkehrung von card_index."""
    suit = Suit(idx // 9)
    rank = Rank(idx % 9)
    return Card(suit, rank)


# ----- Feature-Layout v2 -----

# played_by_<relpos>: pro Spielerposition (relativ zum eigenen Sitz), welche
# Karten dieser Spieler in *abgeschlossenen* Stichen gespielt hat. Damit kann
# das NN inferieren, wer was hat (und nicht hat).
#
# current_trick_by_<relpos>: pro Spielerposition, welche Karte dieser Spieler
# im aktuellen, laufenden Stich gespielt hat (kann auch leer sein, wenn der
# Spieler noch nicht dran war).
#
# Relative Position:
#   0 = ich
#   1 = links neben mir (naechster im Uhrzeigersinn)
#   2 = gegenueber (mein Partner im Kreuz-Jass)
#   3 = rechts neben mir (vorletzter im Uhrzeigersinn)
SECTIONS = [
    ("own_hand", 36),
    ("played_by_me", 36),
    ("played_by_left", 36),
    ("played_by_partner", 36),
    ("played_by_right", 36),
    ("current_trick_by_me", 36),
    ("current_trick_by_left", 36),
    ("current_trick_by_partner", 36),
    ("current_trick_by_right", 36),
    ("lead_suit", 4),
    ("trump_suit", 4),
    ("mode", 4),            # [is_trumpf, is_oben, is_unten, is_slalom]
    ("my_seat", 4),
    ("starter_seat_relative", 4),  # one-hot: 0..3 = wer ist Anspieler relativ zu mir
    ("score_own_norm", 1),
    ("score_opp_norm", 1),
    ("trick_idx_norm", 1),
    ("round_idx_norm", 1),
]

INPUT_DIM = sum(size for _, size in SECTIONS)  # 9*36 + 5*4 + 4 = 324 + 20 + 4 = 348
ACTION_DIM = NUM_CARDS  # 36

SECTION_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _size in SECTIONS:
    SECTION_OFFSETS[_name] = (_offset, _offset + _size)
    _offset += _size


def _relative_position(seat: int, my_seat: int, num_players: int = NUM_PLAYERS) -> int:
    """Wandelt einen absoluten Sitz in eine relative Position (0..num_players-1) um.

    0 = ich selbst, 1 = links (naechster im Uhrzeigersinn), usw.
    """
    return (seat - my_seat) % num_players


def encode_state(hand: list[Card], state: GameState) -> np.ndarray:
    """Wandelt Spielzustand in einen Featurevektor (float32, Shape (INPUT_DIM,)).

    Encoder-Version: 2.0.0
    """
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    my_seat = state.player_idx

    # 1) Eigene Hand
    off, _ = SECTION_OFFSETS["own_hand"]
    for c in hand:
        vec[off + card_index(c)] = 1.0

    # 2) Played history pro Spieler-Position (relativ zu mir)
    played_section_names = {
        0: "played_by_me",
        1: "played_by_left",
        2: "played_by_partner",
        3: "played_by_right",
    }
    for completed in state.completed_tricks:
        for pos_in_trick, c in enumerate(completed.cards):
            seat_who_played = (completed.starter + pos_in_trick) % state.num_players
            rel = _relative_position(seat_who_played, my_seat, state.num_players)
            section = played_section_names[rel]
            off, _ = SECTION_OFFSETS[section]
            vec[off + card_index(c)] = 1.0

    # 3) Current-Trick pro Spielerposition
    # Hier haben wir current_trick_starter und kennen die Reihenfolge.
    trick_section_names = {
        0: "current_trick_by_me",
        1: "current_trick_by_left",
        2: "current_trick_by_partner",
        3: "current_trick_by_right",
    }
    for pos_in_trick, c in enumerate(state.current_trick_cards):
        seat_who_played = (state.current_trick_starter + pos_in_trick) % state.num_players
        rel = _relative_position(seat_who_played, my_seat, state.num_players)
        section = trick_section_names.get(rel, "current_trick_by_me")
        off, _ = SECTION_OFFSETS[section]
        vec[off + card_index(c)] = 1.0

    # 4) Lead-Suit
    if state.current_trick_cards:
        lead = state.current_trick_cards[0].suit
        vec[SECTION_OFFSETS["lead_suit"][0] + int(lead)] = 1.0

    # 5) Trump-Suit (one-hot wenn Trumpf-Modus)
    if state.variant.mode == PlayMode.TRUMPF:
        assert state.variant.trump_suit is not None
        vec[SECTION_OFFSETS["trump_suit"][0] + int(state.variant.trump_suit)] = 1.0

    # 6) Mode-Flags: [is_trumpf, is_oben, is_unten, is_slalom]
    mode_off = SECTION_OFFSETS["mode"][0]
    if state.variant.mode == PlayMode.TRUMPF:
        vec[mode_off + 0] = 1.0
    elif state.variant.mode == PlayMode.OBEN:
        vec[mode_off + 1] = 1.0
    else:  # UNTEN
        vec[mode_off + 2] = 1.0
    if state.announcement.slalom:
        vec[mode_off + 3] = 1.0

    # 7) Eigener Sitz (absolut, 0..3)
    vec[SECTION_OFFSETS["my_seat"][0] + my_seat] = 1.0

    # 8) Starter-Sitz relativ zu mir (0 = ich, 1 = links, etc.)
    starter_rel = _relative_position(state.current_trick_starter, my_seat, state.num_players)
    vec[SECTION_OFFSETS["starter_seat_relative"][0] + starter_rel] = 1.0

    # 9) Punkte normalisiert
    vec[SECTION_OFFSETS["score_own_norm"][0]] = min(state.own_team_score / 1000.0, 1.0)
    vec[SECTION_OFFSETS["score_opp_norm"][0]] = min(state.opp_team_score / 1000.0, 1.0)

    # 10) Stich-Index und Runden-Index normalisiert
    vec[SECTION_OFFSETS["trick_idx_norm"][0]] = state.trick_idx / 9.0
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
