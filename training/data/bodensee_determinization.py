"""Determinization fuer Bodensee-Jass-MCTS-Rollouts.

Im Bodensee-Jass sind dem Spieler **mehr** Karten unbekannt als beim Kreuz-Jass:
- Die Karten der Gegnerhand
- Die verdeckten Karten unter dem eigenen sichtbaren Tisch (selbst die eigenen
  weiss man nicht, sie liegen ja zugedeckt)
- Die verdeckten Karten unter dem Tisch des Gegners

Bekannt sind nur: eigene Hand, eigene sichtbare Tisch-Karten, sichtbare
Tisch-Karten des Gegners, bereits in dieser Runde gespielte Karten.

Beim Determinisieren werden die unbekannten Karten zufaellig auf die drei
Slot-Gruppen verteilt:
1. Eigene verdeckte Tisch-Karten (so viele wie `own_hidden_count`)
2. Gegnerhand (so viele wie `opp_hand_count`)
3. Gegnerische verdeckte Tisch-Karten (so viele wie `opp_hidden_count`)
"""

from __future__ import annotations

import random

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.card import ALL_RANKS, ALL_SUITS, Card
from jass_engine.trick import CompletedTrick


def _full_deck() -> set[Card]:
    return {Card(s, r) for s in ALL_SUITS for r in ALL_RANKS}


def determinize_bodensee(
    own_hand: list[Card],
    own_visible_table: list[Card],
    own_hidden_count: int,
    opp_visible_table: list[Card],
    opp_hand_count: int,
    opp_hidden_count: int,
    completed_tricks: list[CompletedTrick],
    current_trick_cards: list[Card],
    rng: random.Random | None = None,
) -> tuple[list[Card], list[Card], list[Card]]:
    """Verteilt die unbekannten Karten zufaellig auf drei Slot-Gruppen.

    Returns:
        Tupel (own_hidden_cards, opp_hand, opp_hidden_cards). Reihenfolgen
        sind arbitrary -- die Aufrufer-Logik entscheidet, wie sie die Karten
        weiterverteilt.
    """
    if rng is None:
        rng = random.Random()

    seen: set[Card] = set(own_hand)
    seen.update(own_visible_table)
    seen.update(opp_visible_table)
    for trick in completed_tricks:
        seen.update(trick.cards)
    seen.update(current_trick_cards)

    # Sortierte Basis fuer Reproduzierbarkeit (Shuffle danach mit RNG)
    unknown = sorted(_full_deck() - seen, key=lambda c: (int(c.suit), int(c.rank)))

    required = own_hidden_count + opp_hand_count + opp_hidden_count
    if len(unknown) != required:
        raise ValueError(
            f"Inkonsistente Karten-Buchhaltung: erwarte {required} unbekannte "
            f"Karten, habe {len(unknown)}. "
            f"(own_hidden={own_hidden_count}, opp_hand={opp_hand_count}, "
            f"opp_hidden={opp_hidden_count})"
        )

    rng.shuffle(unknown)
    own_hidden_cards = unknown[:own_hidden_count]
    cursor = own_hidden_count
    opp_hand = unknown[cursor:cursor + opp_hand_count]
    cursor += opp_hand_count
    opp_hidden_cards = unknown[cursor:cursor + opp_hidden_count]

    return own_hidden_cards, opp_hand, opp_hidden_cards


def determinize_bodensee_states(
    own_state: BodenseePlayerState,
    opp_visible_table: list[Card],
    opp_hand_count: int,
    opp_hidden_count: int,
    completed_tricks: list[CompletedTrick],
    current_trick_cards: list[Card],
    own_seat: int = 0,
    rng: random.Random | None = None,
) -> list[BodenseePlayerState]:
    """Erzeugt zwei vollstaendig determinisierte BodenseePlayerStates.

    Die eigene Hand und die eigenen sichtbaren Tisch-Karten werden 1:1
    uebernommen. Die eigenen verdeckten Tisch-Karten werden zufaellig befuellt
    (selbst falls das Quell-State sie schon kannte -- der Algorithmus tut so,
    als wuerde der Spieler sie nicht sehen).

    Args:
        own_state: eigene Sicht (Hand + Tisch-Stapel mit `has_hidden`-Info,
            aber die `hidden`-Werte werden bewusst nicht uebernommen).
        opp_visible_table: sichtbare Karten des Gegners (Reihenfolge bestimmt
            die Stapel-Positionen im rekonstruierten Gegner-Zustand).
        opp_hand_count: Anzahl Karten in Gegnerhand.
        opp_hidden_count: Anzahl verdeckter Karten beim Gegner.
        own_seat: 0 oder 1 -- bestimmt Reihenfolge im Output.
        rng: optionaler RNG.

    Returns:
        Liste mit 2 BodenseePlayerStates [seat0, seat1].
    """
    own_hidden_positions = [
        i for i, s in enumerate(own_state.table) if s.has_hidden
    ]
    own_hidden_count = len(own_hidden_positions)

    own_hidden_cards, opp_hand, opp_hidden_cards = determinize_bodensee(
        own_hand=own_state.hand,
        own_visible_table=own_state.visible_table_cards,
        own_hidden_count=own_hidden_count,
        opp_visible_table=opp_visible_table,
        opp_hand_count=opp_hand_count,
        opp_hidden_count=opp_hidden_count,
        completed_tricks=completed_tricks,
        current_trick_cards=current_trick_cards,
        rng=rng,
    )

    # Eigenen State rekonstruieren: Tisch-Struktur erhalten, hidden-Karten
    # zufaellig befuellt
    own_rebuilt = BodenseePlayerState(hand=list(own_state.hand))
    hidden_cursor = 0
    for src_stack in own_state.table:
        new_stack = TableStack(visible=src_stack.visible, hidden=None)
        if src_stack.has_hidden:
            new_stack.hidden = own_hidden_cards[hidden_cursor]
            hidden_cursor += 1
        own_rebuilt.table.append(new_stack)

    # Gegner-State zusammensetzen. Konvention:
    # - Wir konstruieren genau max(6, ...) Stapel
    # - Sichtbare Karten landen zuerst auf den Stapeln 0, 1, 2, ... entsprechend
    #   ihrer Reihenfolge in opp_visible_table
    # - Verdeckte Karten verteilen wir gleichmaessig hinten: Stapel
    #   `len(opp_visible_table) - 1` abwaerts bekommen verdeckte
    # Diese Konvention ist eine Vereinfachung -- die echte Stapel-Zuordnung
    # ist dem MCTS-Aufrufer nicht bekannt. Fuer die Rollouts ist es genug,
    # dass die Karten in IRGENDeiner gueltigen Anordnung vorliegen.
    opp_rebuilt = BodenseePlayerState(hand=list(opp_hand))
    # Wir bauen genau so viele Stapel wie noetig: max(visible_count, total_with_hidden)
    n_stacks = max(len(opp_visible_table), opp_hidden_count, 1)
    # Mindestens len(opp_visible_table) + (opp_hidden_count - was bereits visible-Stapel ist)
    # Einfacher: opp_hidden_count Hidden-Karten verteilen wir auf den letzten
    # opp_hidden_count Stapel-Positionen (mit oder ohne sichtbare).
    # Falls visible_count + hidden_count > 6, kein Problem -- wir machen mehr Stapel.
    total_stacks = max(len(opp_visible_table), opp_hidden_count)
    hidden_cursor_opp = 0
    for i in range(total_stacks):
        visible = opp_visible_table[i] if i < len(opp_visible_table) else None
        # Die letzten `opp_hidden_count` Stapel bekommen eine Hidden-Karte
        has_hidden = i >= (total_stacks - opp_hidden_count)
        hidden: Card | None = None
        if has_hidden:
            hidden = opp_hidden_cards[hidden_cursor_opp]
            hidden_cursor_opp += 1
        opp_rebuilt.table.append(TableStack(visible=visible, hidden=hidden))

    states: list[BodenseePlayerState | None] = [None, None]
    states[own_seat] = own_rebuilt
    states[1 - own_seat] = opp_rebuilt
    return states  # type: ignore[return-value]
