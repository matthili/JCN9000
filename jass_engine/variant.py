"""Spielvarianten: Trumpf, Bock (oben), Geiss (unten), Slalom.

Diese Modelle steuern die Regel-Funktionen in `rules.py` zentral. Eine `Variant` ist
der pro-Stich-Effektivmodus; eine `Announcement` ist das, was der Ansager gewählt hat
(inkl. Slalom-Information, die pro Stich aufgelöst werden muss).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jass_engine.card import Suit


class PlayMode(Enum):
    TRUMPF = "trumpf"   # Mit Trumpf-Farbe (Buur=20, Nell=14, klassisch)
    GUMPF = "gumpf"     # G(eiss) + Tr(umpf): Trumpf-Farbe wie Trumpf,
                         # aber Nicht-Trumpf-Farben mit invertierter Stärke
                         # (6 stärkste Karte in der Lead-Farbe).
                         # Wertpunkte: identisch mit Trumpf-Variante (8er=0).
    OBEN = "oben"        # Bock: kein Trumpf, normale Reihenfolge, 8er=8
    UNTEN = "unten"      # Geiss: kein Trumpf, umgekehrte Reihenfolge, 8er=8


_HAS_TRUMP_MODES = (PlayMode.TRUMPF, PlayMode.GUMPF)


@dataclass(frozen=True)
class Variant:
    """Effektiver Modus für einen einzelnen Stich."""

    mode: PlayMode
    trump_suit: Suit | None = None  # nur bei mode in (TRUMPF, GUMPF)

    def __post_init__(self):
        if self.mode in _HAS_TRUMP_MODES and self.trump_suit is None:
            raise ValueError(f"Variant.{self.mode.name} benötigt eine trump_suit.")
        if self.mode not in _HAS_TRUMP_MODES and self.trump_suit is not None:
            raise ValueError(
                "Nur Variant.TRUMPF und Variant.GUMPF dürfen eine trump_suit haben."
            )

    @classmethod
    def trumpf(cls, suit: Suit) -> "Variant":
        return cls(mode=PlayMode.TRUMPF, trump_suit=suit)

    @classmethod
    def gumpf(cls, suit: Suit) -> "Variant":
        return cls(mode=PlayMode.GUMPF, trump_suit=suit)

    @classmethod
    def oben(cls) -> "Variant":
        return cls(mode=PlayMode.OBEN)

    @classmethod
    def unten(cls) -> "Variant":
        return cls(mode=PlayMode.UNTEN)

    @property
    def has_trump(self) -> bool:
        return self.mode in _HAS_TRUMP_MODES

    def __repr__(self) -> str:
        if self.mode == PlayMode.TRUMPF:
            assert self.trump_suit is not None
            return f"Trumpf {self.trump_suit.german_name}"
        if self.mode == PlayMode.GUMPF:
            assert self.trump_suit is not None
            return f"Gumpf {self.trump_suit.german_name}"
        return self.mode.value.capitalize()


@dataclass(frozen=True)
class Announcement:
    """Was der Ansager gewählt hat.

    Für TRUMPF/OBEN/UNTEN ist `variant` der ganze Runden-Modus.
    Für SLALOM ist `slalom=True`; `variant` gibt den Anfangs-Modus (OBEN oder UNTEN) an,
    und ab dann wechselt der Modus pro Stich.
    """

    variant: Variant
    slalom: bool = False

    def __post_init__(self):
        if self.slalom and self.variant.mode in _HAS_TRUMP_MODES:
            raise ValueError("Slalom kann nicht mit Trumpf oder Gumpf kombiniert werden.")

    def variant_for_trick(self, trick_idx: int) -> Variant:
        """Effektiver Modus für den Stich `trick_idx` (0-basiert)."""
        if not self.slalom:
            return self.variant
        # Slalom: pro Stich wechseln
        if trick_idx % 2 == 0:
            return self.variant
        # Toggle OBEN <-> UNTEN
        if self.variant.mode == PlayMode.OBEN:
            return Variant.unten()
        return Variant.oben()

    def __repr__(self) -> str:
        if self.slalom:
            return f"Slalom (Start: {self.variant})"
        return repr(self.variant)
