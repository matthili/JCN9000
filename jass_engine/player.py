"""Abstrakte Spieler-Schnittstelle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jass_engine.card import Card
from jass_engine.variant import Announcement, Variant
from jass_engine.weis import Weis


@dataclass
class GameState:
    """Information, die einem Spieler beim Zug-Entscheid zur Verfügung steht."""

    player_idx: int
    variant: Variant  # effektive Variant für diesen Stich (Slalom bereits aufgelöst)
    announcement: Announcement  # was insgesamt angesagt wurde
    current_trick_cards: list[Card]
    current_trick_starter: int
    teams: list[int] = field(default_factory=lambda: [0, 1, 0, 1])
    completed_tricks: list[list[Card]] = field(default_factory=list)
    own_team_score: int = 0
    opp_team_score: int = 0
    round_idx: int = 0
    trick_idx: int = 0
    num_players: int = 4

    def partner_idx(self) -> int:
        for idx, t in enumerate(self.teams):
            if t == self.teams[self.player_idx] and idx != self.player_idx:
                return idx
        return -1

    def players_after_me_in_trick(self) -> int:
        return self.num_players - len(self.current_trick_cards) - 1


class Player(ABC):
    """Abstrakte Basisklasse für alle Spielertypen."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        """Wählt die Ansage (Trumpf, Bock/Oben, Geiss/Unten, Slalom).

        `None` bedeutet schieben (nur wenn `can_push=True`).
        """

    @abstractmethod
    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        """Wählt eine Karte aus der Hand (muss legaler Zug sein)."""

    @abstractmethod
    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        """Entscheidet, welche der möglichen Weisen angesagt werden."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"
