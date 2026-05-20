"""State-Encoder fuer Bodensee-Jass (Version bodensee_1.0.0).

Bodensee unterscheidet sich strukturell vom 4-Spieler-Engine:
- 2 Spieler statt 4
- Eigene Karten verteilt auf Hand (privat) + sichtbaren Tisch (oeffentlich) +
  verdeckten Tisch (selbst dem Besitzer unbekannt)
- 18 Stiche pro Runde statt 9

Deshalb hat dieser Encoder ein eigenes Layout. Karten-Indizes 0..35 bleiben
identisch zur v3.0.0-Codierung (`suit * 9 + rank`), damit das TF.js-Encoder-
Mapping fuer Karten konsistent bleibt.

Featurevektor-Layout (291 Dimensionen):

| Section                      | Bits | Bedeutung                                       |
|------------------------------|------|-------------------------------------------------|
| own_hand                     | 36   | Eigene private Handkarten (one-hot)            |
| own_visible_table            | 36   | Eigene sichtbare Tischkarten                   |
| own_hidden_table_mask        | 6    | Pro Stapel-Position: hat noch eine verdeckte?  |
| opp_visible_table            | 36   | Sichtbare Tischkarten des Gegners              |
| opp_hand_count               | 7    | One-hot fuer 0..6 Karten in Gegner-Hand        |
| opp_hidden_table_count       | 7    | One-hot fuer 0..6 verdeckte Karten beim Gegner |
| played_cards_this_round      | 36   | Alle Karten, die in dieser Runde gespielt sind |
| opp_lead_card                | 36   | Lead-Karte des Gegners (leer wenn ich leade)   |
| i_am_leading                 | 1    | 1 wenn ich Anspieler bin                       |
| value_per_card               | 36   | Karten-Wert unter aktueller Variante (0..1)    |
| strength_per_card            | 36   | Karten-Kraft unter Variante + Lead-Suit (0..1) |
| lead_suit                    | 4    | One-hot Lead-Farbe (oder alle 0)               |
| trump_suit                   | 4    | One-hot Trumpf-Farbe (bei Trumpf/Gumpf)        |
| mode                         | 5    | [is_trumpf, is_gumpf, is_oben, is_unten, is_slalom] |
| i_am_announcer               | 1    | 1 wenn ich diese Runde angesagt habe           |
| score_own_norm               | 1    | Eigener Punktestand / 1000                     |
| score_opp_norm               | 1    | Gegner-Punktestand / 1000                      |
| trick_idx_norm               | 1    | Aktueller Stich / 18                           |
| round_idx_norm               | 1    | Aktuelle Runde / 20 (gekappt)                  |
| **Summe**                    | **291** |                                              |

Aktionsraum: 36 (eine Karte). Die Wahl "Hand oder Tisch" ist nicht explizit
Teil der Aktion, weil jede Karte zu einem Zeitpunkt eindeutig an genau einer
Stelle liegt -- der Spieler findet die Quelle automatisch via
`jass_engine.bodensee.rules.card_source`.
"""

from __future__ import annotations

import numpy as np

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.rules import card_value
from jass_engine.variant import PlayMode, Variant
from training.encoder import (
    MAX_CARD_STRENGTH,
    MAX_CARD_VALUE,
    _card_strength_feature,
    _value_strength_arrays,
    card_index,
    index_to_card,
)


ENCODING_VERSION = "bodensee_1.0.0"

NUM_CARDS = 36
NUM_TABLE_STACKS = 6
MAX_HAND_SIZE = 6
MAX_HIDDEN_COUNT = 6  # max. 6 verdeckte Karten pro Spieler

ACTION_DIM = NUM_CARDS  # 36


SECTIONS = [
    ("own_hand", 36),
    ("own_visible_table", 36),
    ("own_hidden_table_mask", NUM_TABLE_STACKS),
    ("opp_visible_table", 36),
    ("opp_hand_count", MAX_HAND_SIZE + 1),         # one-hot 0..6
    ("opp_hidden_table_count", MAX_HIDDEN_COUNT + 1),  # one-hot 0..6
    ("played_cards_this_round", 36),
    ("opp_lead_card", 36),
    ("i_am_leading", 1),
    ("value_per_card", 36),
    ("strength_per_card", 36),
    ("lead_suit", 4),
    ("trump_suit", 4),
    ("mode", 5),
    ("i_am_announcer", 1),
    ("score_own_norm", 1),
    ("score_opp_norm", 1),
    ("trick_idx_norm", 1),
    ("round_idx_norm", 1),
]

INPUT_DIM = sum(size for _, size in SECTIONS)

SECTION_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _size in SECTIONS:
    SECTION_OFFSETS[_name] = (_offset, _offset + _size)
    _offset += _size


def _set_card_bits(vec: np.ndarray, section: str, cards: list[Card]) -> None:
    """Setzt fuer jede Karte das entsprechende One-Hot-Bit in der Section."""
    off, _ = SECTION_OFFSETS[section]
    for c in cards:
        vec[off + card_index(c)] = 1.0


def encode_state_bodensee(
    hand: list[Card],
    own_table_stacks: list[TableStack],
    state: BodenseeGameState,
    i_am_announcer: bool,
) -> np.ndarray:
    """Wandelt den Bodensee-Spielzustand in einen Featurevektor.

    Args:
        hand: aktuelle private Handkarten des Spielers
        own_table_stacks: die 6 Tisch-Stapel des Spielers (sichtbare + verdeckte
            Positionen). Wichtig: die `hidden`-Karten selbst werden NICHT
            kodiert, nur die Information, dass sie noch da sind (= Maske).
        state: BodenseeGameState mit Gegner-Sicht, Spielhistorie usw.
        i_am_announcer: True, wenn der aktuelle Spieler die Ansage gemacht hat
            (= Weli-Halter dieser Runde)

    Returns:
        Float32-Array der Form (INPUT_DIM,) = (291,)
    """
    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    # 1) Eigene Hand
    _set_card_bits(vec, "own_hand", hand)

    # 2) Eigene sichtbare Tischkarten
    visible_table = [s.visible for s in own_table_stacks if s.visible is not None]
    _set_card_bits(vec, "own_visible_table", visible_table)

    # 3) Eigene Hidden-Stapel-Maske
    off, _ = SECTION_OFFSETS["own_hidden_table_mask"]
    for stack_idx, stack in enumerate(own_table_stacks):
        if stack.has_hidden:
            vec[off + stack_idx] = 1.0

    # 4) Gegner sichtbare Tischkarten
    _set_card_bits(vec, "opp_visible_table", state.opponent_visible_table)

    # 5) Gegner Hand-Count (one-hot, gekappt bei MAX_HAND_SIZE)
    opp_hc = min(state.opponent_hand_count, MAX_HAND_SIZE)
    off, _ = SECTION_OFFSETS["opp_hand_count"]
    vec[off + opp_hc] = 1.0

    # 6) Gegner Hidden-Count (one-hot)
    opp_hidden = min(state.opponent_hidden_table_count, MAX_HIDDEN_COUNT)
    off, _ = SECTION_OFFSETS["opp_hidden_table_count"]
    vec[off + opp_hidden] = 1.0

    # 7) Bereits gespielte Karten dieser Runde (alle Stiche zusammen + laufender Stich)
    played: list[Card] = []
    for ct in state.completed_tricks:
        played.extend(ct.cards)
    played.extend(state.current_trick_cards)
    _set_card_bits(vec, "played_cards_this_round", played)

    # 8) Lead-Karte des Gegners, wenn ich nicht selbst leade
    i_am_leading = state.current_trick_starter == state.player_idx
    if not i_am_leading and state.current_trick_cards:
        _set_card_bits(vec, "opp_lead_card", [state.current_trick_cards[0]])

    # 9) i_am_leading-Bit
    if i_am_leading:
        off, _ = SECTION_OFFSETS["i_am_leading"]
        vec[off] = 1.0

    # 10) value_per_card und strength_per_card (gecached ueber variant+lead)
    lead_suit: Suit | None = (
        state.current_trick_cards[0].suit if state.current_trick_cards else None
    )
    val_off, _ = SECTION_OFFSETS["value_per_card"]
    str_off, _ = SECTION_OFFSETS["strength_per_card"]
    val_arr, str_arr = _value_strength_arrays(state.variant, lead_suit)
    vec[val_off:val_off + NUM_CARDS] = val_arr
    vec[str_off:str_off + NUM_CARDS] = str_arr

    # 11) Lead-Suit
    if lead_suit is not None:
        vec[SECTION_OFFSETS["lead_suit"][0] + int(lead_suit)] = 1.0

    # 12) Trump-Suit
    if state.variant.mode in (PlayMode.TRUMPF, PlayMode.GUMPF):
        assert state.variant.trump_suit is not None
        vec[SECTION_OFFSETS["trump_suit"][0] + int(state.variant.trump_suit)] = 1.0

    # 13) Mode-Flags
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

    # 14) i_am_announcer
    if i_am_announcer:
        off, _ = SECTION_OFFSETS["i_am_announcer"]
        vec[off] = 1.0

    # 15) Score normalisiert (gekappt bei 1.0 fuer Target = 1000)
    vec[SECTION_OFFSETS["score_own_norm"][0]] = min(state.own_score / 1000.0, 1.0)
    vec[SECTION_OFFSETS["score_opp_norm"][0]] = min(state.opp_score / 1000.0, 1.0)

    # 16) Stich- und Runden-Index normalisiert
    vec[SECTION_OFFSETS["trick_idx_norm"][0]] = state.trick_idx / 18.0
    vec[SECTION_OFFSETS["round_idx_norm"][0]] = min(state.round_idx / 20.0, 1.0)

    return vec


def legal_action_mask_bodensee(
    hand: list[Card],
    visible_table: list[Card],
    state: BodenseeGameState,
) -> np.ndarray:
    """Binaermaske der Form (36,): 1 fuer legale Karten, 0 sonst.

    Die Maske beruecksichtigt Hand und sichtbare Tisch-Karten zusammen
    (Bodensee-Bedienzwang gilt ueber beiden).
    """
    mask = np.zeros(NUM_CARDS, dtype=np.uint8)
    # Wir bauen einen voruebergehenden BodenseePlayerState fuer legal_moves_bodensee
    ps = BodenseePlayerState(hand=list(hand))
    ps.table = [TableStack(visible=c, hidden=None) for c in visible_table]
    for c in legal_moves_bodensee(ps, state.current_trick_cards, state.variant):
        mask[card_index(c)] = 1
    return mask


__all__ = [
    "ENCODING_VERSION",
    "INPUT_DIM",
    "ACTION_DIM",
    "NUM_CARDS",
    "SECTIONS",
    "SECTION_OFFSETS",
    "encode_state_bodensee",
    "legal_action_mask_bodensee",
    "card_index",
    "index_to_card",
]
