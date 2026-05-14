"""Karten, Farben (Suits) und Ränge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    EICHEL = 0
    SCHELLE = 1
    HERZ = 2
    LAUB = 3

    @property
    def german_name(self) -> str:
        return {
            Suit.EICHEL: "Eichel",
            Suit.SCHELLE: "Schelle",
            Suit.HERZ: "Herz",
            Suit.LAUB: "Laub",
        }[self]

    @property
    def symbol(self) -> str:
        return {
            Suit.EICHEL: "🌰",
            Suit.SCHELLE: "🔔",
            Suit.HERZ: "♥",
            Suit.LAUB: "🍀",
        }[self]


class Rank(IntEnum):
    """Ränge in aufsteigender Nicht-Trumpf-Stärke."""

    SECHS = 0
    SIEBEN = 1
    ACHT = 2
    NEUN = 3
    ZEHN = 4
    UNTER = 5
    OBER = 6
    KOENIG = 7
    ASS = 8

    @property
    def german_name(self) -> str:
        return {
            Rank.SECHS: "6",
            Rank.SIEBEN: "7",
            Rank.ACHT: "8",
            Rank.NEUN: "9",
            Rank.ZEHN: "10",
            Rank.UNTER: "U",
            Rank.OBER: "O",
            Rank.KOENIG: "K",
            Rank.ASS: "A",
        }[self]

    @property
    def full_name(self) -> str:
        return {
            Rank.SECHS: "Sechs",
            Rank.SIEBEN: "Sieben",
            Rank.ACHT: "Acht",
            Rank.NEUN: "Neun",
            Rank.ZEHN: "Zehn",
            Rank.UNTER: "Unter",
            Rank.OBER: "Ober",
            Rank.KOENIG: "König",
            Rank.ASS: "Ass",
        }[self]


@dataclass(frozen=True, order=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def is_weli(self) -> bool:
        return self.suit == Suit.SCHELLE and self.rank == Rank.SECHS

    def __repr__(self) -> str:
        return f"{self.suit.german_name}-{self.rank.german_name}"

    def short(self) -> str:
        """Kompakte Darstellung, z.B. 'E-A' für Eichel-Ass."""
        suit_letter = {
            Suit.EICHEL: "E",
            Suit.SCHELLE: "S",
            Suit.HERZ: "H",
            Suit.LAUB: "L",
        }[self.suit]
        return f"{suit_letter}-{self.rank.german_name}"


ALL_SUITS: tuple[Suit, ...] = (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)
ALL_RANKS: tuple[Rank, ...] = (
    Rank.SECHS,
    Rank.SIEBEN,
    Rank.ACHT,
    Rank.NEUN,
    Rank.ZEHN,
    Rank.UNTER,
    Rank.OBER,
    Rank.KOENIG,
    Rank.ASS,
)

WELI = Card(Suit.SCHELLE, Rank.SECHS)
