"""Basis-Klasse fuer Bodensee-Jass-Spieler.

Die Methoden sehen explizit nur **das, was der Spieler in echt sehen wuerde**:
- Eigene Hand
- Eigene sichtbare Tisch-Karten
- Per `state`: Sichtbares vom Gegner, Spielhistorie, Punktestand, etc.

Die verdeckten Karten (auch die eigenen!) werden bewusst NICHT durchgereicht --
das modelliert die echte Spielsituation, in der man die eigenen verdeckten
Karten noch nicht kennt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card
from jass_engine.variant import Announcement


class BodenseePlayer(ABC):
    """Abstrakte Basis-Klasse fuer alle Bodensee-Spieler."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_announcement(
        self,
        hand: list[Card],
        visible_table: list[Card],
        round_idx: int,
    ) -> Announcement:
        """Waehlt die Spielart fuer diese Runde.

        Args:
            hand: 6 private Karten
            visible_table: 6 sichtbare Tisch-Karten (die der Gegner auch sieht)
            round_idx: Nummer der aktuellen Runde

        Returns:
            Die gewaehlte Ansage. Schieben gibt es im Bodensee nicht; daher
            ist eine None-Rueckgabe nicht erlaubt.
        """
        ...

    @abstractmethod
    def choose_card(
        self,
        hand: list[Card],
        visible_table: list[Card],
        state: BodenseeGameState,
    ) -> Card:
        """Waehlt die naechste Karte. Muss in der Menge der legalen Karten liegen.

        Args:
            hand: aktuelle private Hand-Karten
            visible_table: aktuelle sichtbare Tisch-Karten
            state: weitere Spielinformationen (Gegner-Sicht, Spielhistorie, etc.)

        Returns:
            Eine Karte aus hand+visible_table, die legal ist.
        """
        ...
