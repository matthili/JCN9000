"""Datenstrukturen fuer den Bodensee-Jass-Spielzustand eines Spielers.

Jeder Spieler hat:
- Hand: 0-6 private Karten, nur er selbst sieht sie
- Tisch: 6 Stapel, jeder anfangs mit 1 verdeckter + 1 sichtbarer Karte
  - Die sichtbare Karte sehen beide Spieler
  - Die verdeckte sieht nur der Besitzer (oder niemand, wenn man die Engine
    streng nach Realitaet modelliert -- bei uns sieht der Engine alle Karten,
    aber die Spieler-Schnittstelle gibt nur die fuer den Zug-Spieler oeffentlichen
    weiter)

Wenn ein Spieler eine sichtbare Tisch-Karte spielt, wird die verdeckte
darunter (falls vorhanden) zur neuen sichtbaren. Beim naechsten Zug ist sie
dann verfuegbar.

Stapel-Lebenszyklus:
1. Anfangs:    visible=Card_A, hidden=Card_B  (2 Karten im Stapel)
2. Nach Spiel: visible=Card_B, hidden=None    (1 Karte im Stapel)
3. Nach Spiel: visible=None,   hidden=None    (Stapel leer)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jass_engine.card import Card


@dataclass
class TableStack:
    """Ein Tisch-Stapel mit bis zu zwei Karten.

    Erfindung der Engine:
    - visible: aktuell sichtbare Karte (oder None, wenn schon gespielt)
    - hidden:  darunter liegende verdeckte Karte (oder None, wenn aufgedeckt/leer)

    Konvention: ein "frischer" Stapel hat visible UND hidden gesetzt. Nach dem
    Spielen der sichtbaren wandert hidden auf die visible-Position und hidden
    wird None. Beim naechsten Spielen der visible ist der Stapel leer.
    """

    visible: Card | None = None
    hidden: Card | None = None

    @property
    def has_visible(self) -> bool:
        return self.visible is not None

    @property
    def has_hidden(self) -> bool:
        return self.hidden is not None

    @property
    def is_empty(self) -> bool:
        return self.visible is None and self.hidden is None

    @property
    def card_count(self) -> int:
        return int(self.visible is not None) + int(self.hidden is not None)

    def play_visible(self) -> tuple[Card, Card | None]:
        """Spielt die sichtbare Karte. Wenn eine verdeckte vorhanden ist, wird
        sie zur neuen sichtbaren -- der Spieler darf sie ab dem naechsten Zug
        verwenden.

        Returns:
            Tupel (gespielte_Karte, neu_aufgedeckte_Karte_oder_None).
        """
        if self.visible is None:
            raise RuntimeError("Stapel hat keine sichtbare Karte mehr.")
        played = self.visible
        if self.hidden is not None:
            self.visible = self.hidden
            self.hidden = None
        else:
            self.visible = None
        return played, self.visible


@dataclass
class BodenseePlayerState:
    """Vollstaendiger Bodensee-Spielzustand eines Spielers."""

    hand: list[Card] = field(default_factory=list)
    table: list[TableStack] = field(default_factory=list)

    @property
    def visible_table_cards(self) -> list[Card]:
        """Liste der aktuell sichtbaren Tisch-Karten (in Stapel-Reihenfolge)."""
        return [s.visible for s in self.table if s.visible is not None]

    @property
    def hidden_table_count(self) -> int:
        """Anzahl noch verdeckter Karten auf dem Tisch."""
        return sum(1 for s in self.table if s.hidden is not None)

    @property
    def total_cards_remaining(self) -> int:
        """Gesamtzahl der noch nicht gespielten Karten (Hand + alle Stapel)."""
        return len(self.hand) + sum(s.card_count for s in self.table)

    @property
    def available_cards(self) -> list[Card]:
        """Karten, die der Spieler aktuell spielen koennte (Hand + sichtbare Tisch).

        Wichtig: das ist die "Pool" fuer den Bedienzwang und die Karten-Wahl.
        Verdeckte Tisch-Karten zaehlen nicht.
        """
        return list(self.hand) + self.visible_table_cards

    def has_card_in_hand(self, card: Card) -> bool:
        return card in self.hand

    def has_card_on_visible_table(self, card: Card) -> bool:
        return any(s.visible == card for s in self.table)

    def remove_from_hand(self, card: Card) -> None:
        """Entfernt eine Karte aus der Hand."""
        if card not in self.hand:
            raise ValueError(f"Karte {card} nicht in Hand {self.hand}")
        self.hand.remove(card)

    def play_from_table(self, card: Card) -> Card | None:
        """Spielt eine sichtbare Tisch-Karte und deckt evtl. die darunterliegende auf.

        Returns:
            Die neu aufgedeckte Karte (oder None, falls keine mehr da war).
        """
        for stack in self.table:
            if stack.visible == card:
                _, new_visible = stack.play_visible()
                return new_visible
        raise ValueError(
            f"Karte {card} liegt nicht sichtbar auf dem Tisch (sichtbar: "
            f"{self.visible_table_cards})"
        )
