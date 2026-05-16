"""Determinization fuer Imperfect-Information-Lookahead.

Beim Jass kennt ein Spieler nur die eigene Hand und alle bereits gespielten
Karten. Um Rollouts zu machen, muessen wir die Karten der Mitspieler
*hypothetisch* befuellen ("determinisieren"). Mehrere zufaellige
Determinizations + Mittelwert = Monte-Carlo-Schaetzung des Spielwerts.

Aktuelle Variante: gleichverteilt zufaellig, mit Korrekt-Anzahl-Constraint.
Spaetere Erweiterung moeglich: Wahrscheinlichkeits-Modellierung anhand der
bisher gespielten Karten (z.B. "Spieler X konnte Herz nicht bedienen ->
hat sicher keine Herz-Karten mehr").
"""

from __future__ import annotations

import random

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card
from jass_engine.trick import CompletedTrick


def determinize_hands(
    own_seat: int,
    own_hand: list[Card],
    completed_tricks: list[CompletedTrick],
    current_trick_cards: list[Card],
    current_trick_starter: int,
    num_players: int = 4,
    rng: random.Random | None = None,
) -> list[list[Card]]:
    """Verteilt die unsichtbaren Karten zufaellig auf die Mitspieler.

    Args:
        own_seat: eigener Sitzplatz 0..3.
        own_hand: eigene Hand (wird unveraendert weitergegeben).
        completed_tricks: alle in dieser Runde schon abgeschlossenen Stiche.
        current_trick_cards: Karten im aktuellen, laufenden Stich.
        current_trick_starter: Sitz des Anspielers des aktuellen Stichs.
        num_players: typisch 4 fuer Kreuz-Jass.
        rng: optionaler RNG fuer Reproduzierbarkeit.

    Returns:
        Liste `hands` mit `hands[seat]` = Liste der Karten dieses Spielers.
        Eigene Hand bleibt unveraendert; die der Mitspieler ist eine
        zufaellige Verteilung der unsichtbaren Karten.
    """
    if rng is None:
        rng = random.Random()

    # 1) Wieviele Karten hat jeder noch?
    cards_played_per_seat = [0] * num_players
    for trick in completed_tricks:
        for i in range(len(trick.cards)):
            cards_played_per_seat[(trick.starter + i) % num_players] += 1
    for i in range(len(current_trick_cards)):
        cards_played_per_seat[(current_trick_starter + i) % num_players] += 1

    hand_sizes = [9 - cards_played_per_seat[s] for s in range(num_players)]

    if hand_sizes[own_seat] != len(own_hand):
        raise ValueError(
            f"Inkonsistenter Zustand: erwarte {hand_sizes[own_seat]} Karten "
            f"in own_hand, habe {len(own_hand)}. own_seat={own_seat}."
        )

    # 2) Welche Karten sind unbekannt (= noch im Spiel, aber nicht in eigener Hand)?
    seen: set[Card] = set(own_hand)
    for trick in completed_tricks:
        seen.update(trick.cards)
    seen.update(current_trick_cards)

    all_cards = {Card(s, r) for s in ALL_SUITS for r in ALL_RANKS}
    unknown = list(all_cards - seen)

    expected_unknown = sum(hand_sizes) - hand_sizes[own_seat]
    if len(unknown) != expected_unknown:
        raise ValueError(
            f"Inkonsistente Karten-Buchhaltung: erwarte {expected_unknown} "
            f"unbekannte Karten, habe {len(unknown)}."
        )

    # 3) Mischen und Verteilen
    rng.shuffle(unknown)
    hands: list[list[Card]] = [list(own_hand) if s == own_seat else [] for s in range(num_players)]

    idx = 0
    for seat in range(num_players):
        if seat == own_seat:
            continue
        size = hand_sizes[seat]
        hands[seat] = unknown[idx : idx + size]
        idx += size

    return hands
