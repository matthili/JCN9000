"""Deck-Erzeugung und Austeilen."""

from __future__ import annotations

import random

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card


def make_deck() -> list[Card]:
    """Liefert die 36 Karten des Vorarlberger Jass-Decks."""
    return [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]


def deal(
    num_players: int = 4,
    rng: random.Random | None = None,
) -> list[list[Card]]:
    """Mischt das Deck und verteilt die Karten gleichmäßig auf die Spieler."""
    if rng is None:
        rng = random.Random()
    deck = make_deck()
    rng.shuffle(deck)
    if len(deck) % num_players != 0:
        raise ValueError(
            f"36 Karten lassen sich nicht gleichmäßig auf {num_players} Spieler verteilen."
        )
    cards_per_player = len(deck) // num_players
    return [
        deck[i * cards_per_player : (i + 1) * cards_per_player]
        for i in range(num_players)
    ]


def find_weli_holder(hands: list[list[Card]]) -> int:
    """Index des Spielers, der den Weli (Schelle-6) auf der Hand hat."""
    for idx, hand in enumerate(hands):
        if any(c.is_weli for c in hand):
            return idx
    raise RuntimeError("Weli fehlt im Deck — das sollte unmöglich sein.")
