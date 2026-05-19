"""Karten-Verteilung fuer Bodensee-Jass.

Abfolge:
1. Pro Spieler 6 Karten **verdeckt** auf den Tisch (in 6 Stapeln, je 1 Karte).
2. Pro Spieler 6 Karten als **Hand**.
3. Pro Spieler 6 Karten **sichtbar** auf die verdeckten 6 oben drauf (Stapel
   haben jetzt 2 Karten: oben sichtbar, unten verdeckt).

Insgesamt 36 Karten = volles Vorarlberger Deck.

Realer Tisch-Ablauf laut Regel: Karten werden zunaechst offen ausgeteilt, um
den Weli-Halter zu identifizieren. Der nicht-Weli-Halter mischt und gibt
dann richtig (= dieses Schema oben). Fuer die Simulation reicht ein einziges
Mischen + Verteilen, weil das Resultat statistisch identisch ist.
"""

from __future__ import annotations

import random

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.card import Card, Rank, Suit
from jass_engine.deck import make_deck


# Konstanten fuer den 2-Spieler-Aufbau
NUM_PLAYERS = 2
HAND_SIZE = 6
TABLE_STACKS = 6
TOTAL_CARDS_PER_PLAYER = HAND_SIZE + 2 * TABLE_STACKS  # 6 + 12 = 18
TRICKS_PER_ROUND = TOTAL_CARDS_PER_PLAYER  # 18


def deal_bodensee(
    rng: random.Random | None = None,
) -> list[BodenseePlayerState]:
    """Mischt das Deck und verteilt die Karten im Bodensee-Schema.

    Returns:
        Liste mit zwei BodenseePlayerState-Objekten (Spieler 0 und 1).
    """
    if rng is None:
        rng = random.Random()

    deck = make_deck()
    rng.shuffle(deck)

    p0 = BodenseePlayerState()
    p1 = BodenseePlayerState()
    states = [p0, p1]

    idx = 0
    # Stufe 1: 6 Stapel pro Spieler, jeder bekommt zunaechst eine verdeckte Karte
    for stack_idx in range(TABLE_STACKS):
        for player in states:
            player.table.append(TableStack(visible=None, hidden=deck[idx]))
            idx += 1

    # Stufe 2: 6 Hand-Karten pro Spieler
    for _ in range(HAND_SIZE):
        for player in states:
            player.hand.append(deck[idx])
            idx += 1

    # Stufe 3: 6 sichtbare Karten pro Spieler oben drauf
    for stack_idx in range(TABLE_STACKS):
        for player in states:
            assert player.table[stack_idx].visible is None
            player.table[stack_idx].visible = deck[idx]
            idx += 1

    assert idx == 36, f"Erwartet 36 Karten verteilt, tatsaechlich {idx}"

    # Sanity: jeder hat genau TOTAL_CARDS_PER_PLAYER Karten
    for i, ps in enumerate(states):
        assert ps.total_cards_remaining == TOTAL_CARDS_PER_PLAYER, (
            f"Spieler {i} hat {ps.total_cards_remaining} Karten "
            f"(erwartet {TOTAL_CARDS_PER_PLAYER})"
        )

    return states


def find_weli_holder_bodensee(states: list[BodenseePlayerState]) -> int:
    """Index des Spielers, dessen Karten den Weli (Schelle-6) enthalten.

    Sucht in Hand, sichtbaren Tisch-Karten UND verdeckten Tisch-Karten.
    """
    weli = Card(Suit.SCHELLE, Rank.SECHS)
    for idx, ps in enumerate(states):
        if weli in ps.hand:
            return idx
        for stack in ps.table:
            if stack.visible == weli or stack.hidden == weli:
                return idx
    raise RuntimeError("Weli nicht im Deck gefunden -- das sollte unmoeglich sein.")
