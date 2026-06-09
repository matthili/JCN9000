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


def _assign_constrained(
    unknown: list[Card],
    sizes: dict[int, int],
    forbidden_by_seat: dict[int, set[Card]],
    rng: random.Random,
    max_attempts: int = 200,
) -> dict[int, list[Card]] | None:
    """Verteilt `unknown` auf die Sitze unter Beachtung verbotener Karten.

    Most-constrained-card-first + randomisierte Sitzwahl, mit Neustarts. Gibt
    None zurueck, wenn keine gueltige Zuteilung gefunden wird (ueber-
    eingeschraenkt).
    """
    seats = [s for s in sizes if sizes[s] > 0]

    def feasible(card: Card) -> list[int]:
        return [s for s in seats if card not in forbidden_by_seat.get(s, ())]

    # Schnell-Abbruch: eine Karte ohne moeglichen Sitz -> unloesbar.
    for c in unknown:
        if not feasible(c):
            return None

    for _ in range(max_attempts):
        rng.shuffle(unknown)  # randomisiert Gleichstaende in der Sortierung
        order = sorted(unknown, key=lambda c: len(feasible(c)))  # knappste zuerst
        remaining = dict(sizes)
        assign: dict[int, list[Card]] = {s: [] for s in seats}
        ok = True
        for c in order:
            opts = [s for s in feasible(c) if remaining[s] > 0]
            if not opts:
                ok = False
                break
            choice = rng.choice(opts)
            assign[choice].append(c)
            remaining[choice] -= 1
        if ok and all(v == 0 for v in remaining.values()):
            return assign
    return None


def determinize_hands(
    own_seat: int,
    own_hand: list[Card],
    completed_tricks: list[CompletedTrick],
    current_trick_cards: list[Card],
    current_trick_starter: int,
    num_players: int = 4,
    rng: random.Random | None = None,
    forbidden_by_seat: dict[int, set[Card]] | None = None,
) -> list[list[Card]]:
    """Verteilt die unsichtbaren Karten auf die Mitspieler.

    Args:
        own_seat: eigener Sitzplatz 0..3.
        own_hand: eigene Hand (wird unveraendert weitergegeben).
        completed_tricks: alle in dieser Runde schon abgeschlossenen Stiche.
        current_trick_cards: Karten im aktuellen, laufenden Stich.
        current_trick_starter: Sitz des Anspielers des aktuellen Stichs.
        num_players: typisch 4 fuer Kreuz-Jass.
        rng: optionaler RNG fuer Reproduzierbarkeit.
        forbidden_by_seat: optional `seat -> Menge verbotener Karten` (z.B. aus
            `jass_engine.void_inference.infer_forbidden_cards`). Karten werden
            dann nur auf Sitze verteilt, die sie auch halten koennten. Faellt
            auf die zufaellige Verteilung zurueck, falls keine gueltige Zuteilung
            existiert (sehr selten; ein langer Datengen-Lauf soll daran nicht
            scheitern).

    Returns:
        Liste `hands` mit `hands[seat]` = Liste der Karten dieses Spielers.
        Eigene Hand bleibt unveraendert.
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

    # 3) Verteilen -- optional unter Beachtung verbotener Karten (Voids).
    other_seats = [s for s in range(num_players) if s != own_seat]
    sizes = {s: hand_sizes[s] for s in other_seats}
    hands: list[list[Card]] = [list(own_hand) if s == own_seat else [] for s in range(num_players)]

    assignment: dict[int, list[Card]] | None = None
    if forbidden_by_seat:
        assignment = _assign_constrained(list(unknown), sizes, forbidden_by_seat, rng)

    if assignment is None:
        # Unconstrained (Default oder Fallback, wenn die Constraints unloesbar sind).
        rng.shuffle(unknown)
        idx = 0
        for seat in other_seats:
            hands[seat] = unknown[idx : idx + sizes[seat]]
            idx += sizes[seat]
    else:
        for seat in other_seats:
            hands[seat] = assignment[seat]

    return hands
