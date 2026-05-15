"""Einzelner Stich (bis zu N Karten, eine pro Spieler)."""

from __future__ import annotations

from dataclasses import dataclass, field

from jass_engine.card import Card, Suit
from jass_engine.rules import trick_points, trick_winner
from jass_engine.variant import Variant


@dataclass(frozen=True)
class CompletedTrick:
    """Abgeschlossener Stich mit Anspieler-Info -- wird in GameState.completed_tricks
    gespeichert, damit der Encoder spaeter weiss, wer welche Karte gespielt hat."""

    starter: int
    cards: tuple[Card, ...]

    def player_idx_for_card(self, card_pos: int, num_players: int = 4) -> int:
        """Spieler-Index fuer die Karte an Position `card_pos` (0..3)."""
        return (self.starter + card_pos) % num_players


@dataclass
class Trick:
    """Ein einzelner Stich.

    `starting_player_idx` ist der Spieler, der die erste Karte gespielt hat.
    `cards` enthält die Karten in der Reihenfolge, in der sie gespielt wurden.
    """

    starting_player_idx: int
    cards: list[Card] = field(default_factory=list)
    num_players: int = 4

    def is_complete(self) -> bool:
        return len(self.cards) == self.num_players

    def lead_suit(self) -> Suit | None:
        return self.cards[0].suit if self.cards else None

    def next_player_idx(self) -> int:
        return (self.starting_player_idx + len(self.cards)) % self.num_players

    def player_idx_for_card(self, card_pos: int) -> int:
        return (self.starting_player_idx + card_pos) % self.num_players

    def add_card(self, card: Card) -> None:
        if self.is_complete():
            raise RuntimeError("Stich ist bereits voll.")
        self.cards.append(card)

    def winner_idx(self, variant: Variant) -> int:
        winning_pos = trick_winner(self.cards, variant)
        return self.player_idx_for_card(winning_pos)

    def points(self, variant: Variant, is_last: bool = False) -> int:
        return trick_points(self.cards, variant, is_last_trick=is_last)
