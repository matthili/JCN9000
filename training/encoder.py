"""State-Encoder fuer das neuronale Netz (Version 3.0.0).

Wandelt einen `GameState` + die eigene Hand in einen Featurevektor um, plus
eine Aktionsmaske, die illegale Karten ausblendet.

Neuerungen gegenueber v2.0.0 (siehe spec/state_encoding.md):
- **Pre-computed Karten-Semantik**: Zwei zusaetzliche Sections, die pro Karte
  den **Wertpunkt** und die **Kraftpunkt**-Wert unter der aktuellen Variante
  und Lead-Suit-Lage liefern. Damit muss das NN die multiplikative Interaktion
  Karte × Variante nicht mehr selbst lernen — das ist insbesondere fuer Bock,
  Geiss und Gumpf entscheidend (invertierte/exotische Stärke-Reihenfolgen).
- **Mode-Feld auf 5 Dims erweitert**: zusaetzlich `is_gumpf`. Damit ist auch
  die neue Gumpf-Variante (Trumpf-Farbe normal, Nicht-Trumpf invertiert)
  korrekt kodiert.

Karten-Index: `suit * 9 + rank` ergibt einen eindeutigen Index 0..35.
"""

from __future__ import annotations

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.rules import (
    POINT_VALUES_NORMAL,
    POINT_VALUES_OBEN_UNTEN,
    POINT_VALUES_TRUMP,
    TRUMP_RANK_ORDER,
    card_value,
    legal_moves,
)
from jass_engine.variant import PlayMode, Variant


ENCODING_VERSION = "3.0.0"

NUM_CARDS = 36
NUM_PLAYERS = 4
NUM_RELATIVE_POSITIONS = 4  # 0 = me, 1 = left, 2 = partner, 3 = right

# Normalisierungs-Konstanten fuer die Pre-Computed-Features.
# Maximaler Punktwert pro Karte = 20 (Buur unter Trumpf/Gumpf). Wir teilen
# durch 20.0, damit value_per_card in [0, 1] bleibt.
MAX_CARD_VALUE = 20.0
# Maximaler Kraftwert pro Karte = 18 (Buur in Trumpf-Farbe). Wir teilen
# durch 18.0, damit strength_per_card in [0, 1] bleibt.
MAX_CARD_STRENGTH = 18.0


def card_index(card: Card) -> int:
    """Eindeutiger Index 0..35 pro Karte (suit * 9 + rank)."""
    return int(card.suit) * 9 + int(card.rank)


def index_to_card(idx: int) -> Card:
    """Umkehrung von card_index."""
    suit = Suit(idx // 9)
    rank = Rank(idx % 9)
    return Card(suit, rank)


# ----- Feature-Layout v3 -----
#
# played_by_<relpos>: pro Spielerposition (relativ zum eigenen Sitz), welche
# Karten dieser Spieler in *abgeschlossenen* Stichen gespielt hat.
#
# current_trick_by_<relpos>: pro Spielerposition, welche Karte dieser Spieler
# im aktuellen, laufenden Stich gespielt hat.
#
# value_per_card / strength_per_card: pro Karte (Index 0..35) der unter der
# aktuellen Variante gueltige Wert-/Kraftwert, normalisiert.
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
    ("value_per_card", 36),     # NEU v3: normalisierter Wertpunkt pro Karte
    ("strength_per_card", 36),  # NEU v3: normalisierter Kraftpunkt pro Karte
    ("lead_suit", 4),
    ("trump_suit", 4),
    ("mode", 5),                # v3: [is_trumpf, is_gumpf, is_oben, is_unten, is_slalom]
    ("my_seat", 4),
    ("starter_seat_relative", 4),
    ("score_own_norm", 1),
    ("score_opp_norm", 1),
    ("trick_idx_norm", 1),
    ("round_idx_norm", 1),
]

INPUT_DIM = sum(size for _, size in SECTIONS)  # 11*36 + 4+4+5+4+4 + 4*1 = 396 + 21 + 4 = 421
ACTION_DIM = NUM_CARDS  # 36

SECTION_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _size in SECTIONS:
    SECTION_OFFSETS[_name] = (_offset, _offset + _size)
    _offset += _size


def _relative_position(seat: int, my_seat: int, num_players: int = NUM_PLAYERS) -> int:
    """Wandelt einen absoluten Sitz in eine relative Position (0..num_players-1) um."""
    return (seat - my_seat) % num_players


def _card_strength_feature(card: Card, variant: Variant, lead_suit: Suit | None) -> int:
    """Liefert den 'Kraftpunkt' der Karte unter der aktuellen Variante.

    Werte liegen in 1..18. Hoehere Werte stehen fuer staerkere Karten. Die
    Reihenfolge entspricht exakt der vom Domaenenexperten gepflegten Tabelle:

    TRUMPF / GUMPF: keine Lead-Suit-Abhaengigkeit fuer Nicht-Trumpf.
      - Trumpf-Farbe: 10..18 nach TRUMP_RANK_ORDER (Buur=18, Nell=17, Ass=16, ..., 6=10).
      - Nicht-Trumpf:
        - TRUMPF: 1..9 aufsteigend nach Rang (6=1, 7=2, ..., Ass=9).
        - GUMPF: 1..9 absteigend nach Rang (Ass=1, ..., 6=9), entspricht invertierter Logik.

    OBEN / UNTEN: Lead-Suit-Abhaengigkeit.
      - Karte in Lead-Suit: 10..18 (OBEN aufsteigend, UNTEN absteigend nach Rang).
      - Karte in Nicht-Lead-Suit: 1..9 (gleiche Reihenfolge).
      - Wenn kein Lead aktiv: die eigene Suit der Karte wird als hypothetischer
        Lead angesehen, d.h. jede Karte bekommt den 10..18-Boost. Das spiegelt
        die 'Anspiel-Kraft' wider.
    """
    rank_i = int(card.rank)  # 0..8

    if variant.mode == PlayMode.TRUMPF:
        if card.suit == variant.trump_suit:
            return 10 + TRUMP_RANK_ORDER[card.rank]
        return 1 + rank_i

    if variant.mode == PlayMode.GUMPF:
        if card.suit == variant.trump_suit:
            return 10 + TRUMP_RANK_ORDER[card.rank]
        # Nicht-Trumpf: invertiert (6 stark, Ass schwach)
        return 1 + (8 - rank_i)

    # OBEN oder UNTEN
    if variant.mode == PlayMode.OBEN:
        base_rank_strength = rank_i
    else:  # UNTEN
        base_rank_strength = 8 - rank_i

    if lead_suit is None:
        # Anspielmoment: jede Karte bekommt 'Lead-Boost' fuer ihre eigene Suit.
        return 10 + base_rank_strength
    if card.suit == lead_suit:
        return 10 + base_rank_strength
    return 1 + base_rank_strength


def encode_state(hand: list[Card], state: GameState) -> np.ndarray:
    """Wandelt Spielzustand in einen Featurevektor (float32, Shape (INPUT_DIM,)).

    Encoder-Version: 3.0.0
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

    # 4) NEU v3: value_per_card und strength_per_card vorberechnen
    lead_suit: Suit | None = (
        state.current_trick_cards[0].suit if state.current_trick_cards else None
    )
    val_off, _ = SECTION_OFFSETS["value_per_card"]
    str_off, _ = SECTION_OFFSETS["strength_per_card"]
    for suit in ALL_SUITS:
        for rank in ALL_RANKS:
            c = Card(suit, rank)
            idx = card_index(c)
            vec[val_off + idx] = card_value(c, state.variant) / MAX_CARD_VALUE
            vec[str_off + idx] = (
                _card_strength_feature(c, state.variant, lead_suit) / MAX_CARD_STRENGTH
            )

    # 5) Lead-Suit
    if lead_suit is not None:
        vec[SECTION_OFFSETS["lead_suit"][0] + int(lead_suit)] = 1.0

    # 6) Trump-Suit (one-hot wenn Trumpf-Modus ODER Gumpf-Modus)
    if state.variant.mode in (PlayMode.TRUMPF, PlayMode.GUMPF):
        assert state.variant.trump_suit is not None
        vec[SECTION_OFFSETS["trump_suit"][0] + int(state.variant.trump_suit)] = 1.0

    # 7) Mode-Flags: [is_trumpf, is_gumpf, is_oben, is_unten, is_slalom_flag]
    mode_off = SECTION_OFFSETS["mode"][0]
    if state.variant.mode == PlayMode.TRUMPF:
        vec[mode_off + 0] = 1.0
    elif state.variant.mode == PlayMode.GUMPF:
        vec[mode_off + 1] = 1.0
    elif state.variant.mode == PlayMode.OBEN:
        vec[mode_off + 2] = 1.0
    else:  # UNTEN
        vec[mode_off + 3] = 1.0
    if state.announcement.slalom:
        vec[mode_off + 4] = 1.0

    # 8) Eigener Sitz (absolut, 0..3)
    vec[SECTION_OFFSETS["my_seat"][0] + my_seat] = 1.0

    # 9) Starter-Sitz relativ zu mir (0 = ich, 1 = links, etc.)
    starter_rel = _relative_position(state.current_trick_starter, my_seat, state.num_players)
    vec[SECTION_OFFSETS["starter_seat_relative"][0] + starter_rel] = 1.0

    # 10) Punkte normalisiert
    vec[SECTION_OFFSETS["score_own_norm"][0]] = min(state.own_team_score / 1000.0, 1.0)
    vec[SECTION_OFFSETS["score_opp_norm"][0]] = min(state.opp_team_score / 1000.0, 1.0)

    # 11) Stich-Index und Runden-Index normalisiert
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


# Backward-compatible export (POINT_VALUES_NORMAL etc. werden von externen
# Skripten ggf. importiert)
__all__ = [
    "ENCODING_VERSION",
    "INPUT_DIM",
    "ACTION_DIM",
    "NUM_CARDS",
    "SECTIONS",
    "SECTION_OFFSETS",
    "MAX_CARD_VALUE",
    "MAX_CARD_STRENGTH",
    "POINT_VALUES_NORMAL",
    "POINT_VALUES_OBEN_UNTEN",
    "POINT_VALUES_TRUMP",
    "card_index",
    "index_to_card",
    "encode_state",
    "legal_action_mask",
    "action_index",
    "all_card_indices",
]
