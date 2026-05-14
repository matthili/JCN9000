"""Weis-Erkennung und Vergleich.

Sequenzen (3..9 Blatt gleicher Farbe), Vierlinge (4× Unter / Neuner / Zehner / Ober /
König / Ass) und Stöcke (Trumpf-Ober + Trumpf-König).

Vergleich: höchste Punktzahl gewinnt, bei Gleichstand höchste Spitzen-Karte, bei
weiterem Gleichstand wer zuerst gewiesen hat (Spielreihenfolge).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from jass_engine.card import Card, Rank, Suit
from jass_engine.variant import PlayMode, Variant


class WeisKind(Enum):
    SEQUENCE = "Sequenz"
    FOUR_OF_KIND = "Vierling"
    STOECKE = "Stöcke"


# Punktewerte Sequenzen nach Länge.
SEQUENCE_POINTS: dict[int, int] = {
    3: 20,
    4: 50,
    5: 100,
    6: 120,
    7: 140,
    8: 160,
    9: 180,
}

# Punktewerte Vierlinge nach Rang.
FOUR_OF_KIND_POINTS: dict[Rank, int] = {
    Rank.UNTER: 200,
    Rank.NEUN: 150,
    Rank.ZEHN: 100,
    Rank.OBER: 100,
    Rank.KOENIG: 100,
    Rank.ASS: 100,
}

STOECKE_POINTS = 20


@dataclass(frozen=True)
class Weis:
    kind: WeisKind
    cards: tuple[Card, ...]
    points: int
    top_rank: Rank  # höchste Karte für Tie-Break

    def __repr__(self) -> str:
        if self.kind == WeisKind.SEQUENCE:
            return f"{len(self.cards)}-Blatt {self.cards[0].suit.german_name} ({self.points})"
        if self.kind == WeisKind.FOUR_OF_KIND:
            return f"4× {self.top_rank.full_name} ({self.points})"
        return f"Stöcke ({self.points})"


def _find_sequences(hand: list[Card]) -> list[Weis]:
    by_suit: dict[Suit, list[Card]] = defaultdict(list)
    for card in hand:
        by_suit[card.suit].append(card)

    sequences: list[Weis] = []
    for suit, cards in by_suit.items():
        cards_sorted = sorted(cards, key=lambda c: c.rank)
        runs: list[list[Card]] = []
        current: list[Card] = [cards_sorted[0]]
        for prev, curr in zip(cards_sorted, cards_sorted[1:]):
            if int(curr.rank) == int(prev.rank) + 1:
                current.append(curr)
            else:
                if len(current) >= 3:
                    runs.append(current)
                current = [curr]
        if len(current) >= 3:
            runs.append(current)

        for run in runs:
            length = min(len(run), 9)
            points = SEQUENCE_POINTS[length]
            sequences.append(
                Weis(
                    kind=WeisKind.SEQUENCE,
                    cards=tuple(run),
                    points=points,
                    top_rank=run[-1].rank,
                )
            )
    return sequences


def _find_four_of_a_kind(hand: list[Card]) -> list[Weis]:
    by_rank: dict[Rank, list[Card]] = defaultdict(list)
    for card in hand:
        by_rank[card.rank].append(card)

    result: list[Weis] = []
    for rank, cards in by_rank.items():
        if len(cards) == 4 and rank in FOUR_OF_KIND_POINTS:
            result.append(
                Weis(
                    kind=WeisKind.FOUR_OF_KIND,
                    cards=tuple(cards),
                    points=FOUR_OF_KIND_POINTS[rank],
                    top_rank=rank,
                )
            )
    return result


def find_weise(hand: list[Card]) -> list[Weis]:
    """Findet alle Sequenzen (≥3 Blatt) und Vierlinge in einer Hand."""
    return _find_sequences(hand) + _find_four_of_a_kind(hand)


def has_stoecke(hand: list[Card], trumpf: Suit) -> bool:
    """Hat der Spieler Trumpf-Ober UND Trumpf-König?"""
    return (
        Card(trumpf, Rank.OBER) in hand
        and Card(trumpf, Rank.KOENIG) in hand
    )


def stoecke_weis(trumpf: Suit) -> Weis:
    """Erzeugt den Stöcke-Weis für die gegebene Trumpffarbe."""
    return Weis(
        kind=WeisKind.STOECKE,
        cards=(Card(trumpf, Rank.OBER), Card(trumpf, Rank.KOENIG)),
        points=STOECKE_POINTS,
        top_rank=Rank.KOENIG,
    )


def stoecke_apply(variant: Variant) -> bool:
    """Stöcke gibt es nur, wenn ein Trumpf existiert (also nicht bei Bock/Geiss/Slalom)."""
    return variant.mode == PlayMode.TRUMPF


def _weis_sort_key(weis: Weis) -> tuple[int, int]:
    """Sortier-Key für Weisen: (Punkte, top_rank)."""
    return (weis.points, int(weis.top_rank))


@dataclass
class TeamWeisResult:
    points: int
    winning_weisen: list[Weis] = field(default_factory=list)


def compare_team_weise(
    weise_per_player: list[list[Weis]],
    teams: list[int],
    announcement_order: list[int],
) -> dict[int, TeamWeisResult]:
    """Vergleicht die Weisen der Teams und vergibt Punkte.

    Nur das Team mit dem höchsten Einzel-Weis bekommt seine gesamten Weis-Punkte;
    das andere Team bekommt 0. Bei Gleichstand entscheidet die Spielreihenfolge
    (`announcement_order` listet Spieler-Indizes in Reihenfolge der ersten Karte).

    Args:
        weise_per_player: Liste der Weisen pro Spieler-Index.
        teams: Team-Zuordnung pro Spieler-Index (z.B. [0, 1, 0, 1]).
        announcement_order: Spielreihenfolge (Spieler-Indizes) für Gleichstand-Auflösung.

    Returns:
        Dict von Team-ID → TeamWeisResult.
    """
    team_ids = sorted(set(teams))
    result: dict[int, TeamWeisResult] = {tid: TeamWeisResult(points=0) for tid in team_ids}

    # Höchster Weis je Spieler, ergänzt um Position in announcement_order
    best_per_player: list[tuple[int, Weis] | None] = []
    for player_idx, weise in enumerate(weise_per_player):
        if not weise:
            best_per_player.append(None)
        else:
            best = max(weise, key=_weis_sort_key)
            best_per_player.append((player_idx, best))

    if all(b is None for b in best_per_player):
        return result

    # Ermittle Gewinner-Spieler über (Punkte, top_rank, früheste Position)
    def player_priority(entry: tuple[int, Weis]) -> tuple[int, int, int]:
        player_idx, weis = entry
        # negativer announcement-Index, damit "früher" höher gewichtet wird
        play_order_pos = announcement_order.index(player_idx)
        return (weis.points, int(weis.top_rank), -play_order_pos)

    candidates = [b for b in best_per_player if b is not None]
    winner_idx, _ = max(candidates, key=player_priority)
    winning_team = teams[winner_idx]

    # Gewinner-Team bekommt alle eigenen Weis-Punkte aufsummiert
    total = 0
    winning_weisen: list[Weis] = []
    for player_idx, weise in enumerate(weise_per_player):
        if teams[player_idx] == winning_team:
            for w in weise:
                total += w.points
                winning_weisen.append(w)
    result[winning_team] = TeamWeisResult(points=total, winning_weisen=winning_weisen)
    return result
