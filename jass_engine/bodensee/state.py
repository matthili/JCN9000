"""Spielzustand-Objekt fuer Bodensee-Spieler.

Anders als beim 4-Spieler-Engine: hier gibt es deutlich asymmetrische Sicht --
ein Spieler sieht seine Hand (privat), seine sichtbaren Tischkarten (oeffentlich)
und die sichtbaren Tischkarten des Gegners. Die verdeckten Karten beider Spieler
sind unbekannt (selbst die eigenen -- die sind in der echten Spielsituation
unter den sichtbaren versteckt).

Diese Klasse buendelt alles, was ein Spieler-Algorithmus in `choose_card`
sehen darf -- ohne die verdeckten Karten zu enthalten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jass_engine.card import Card
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant


@dataclass
class BodenseeGameState:
    """Sicht eines Spielers auf den aktuellen Bodensee-Spielzustand."""

    player_idx: int                            # 0 oder 1 -- ich
    variant: Variant                           # aktuelle Stich-Variante (bei Slalom pro Stich anders)
    announcement: Announcement                 # vollstaendige Ansage der Runde
    current_trick_cards: list[Card]            # 0 oder 1 Karten im laufenden Stich
    current_trick_starter: int                 # wer hat angespielt (0 oder 1)
    completed_tricks: list[CompletedTrick]     # alle bereits abgeschlossenen Stiche

    # Was ich vom Gegner sehe
    opponent_visible_table: list[Card] = field(default_factory=list)
    opponent_hand_count: int = 0
    opponent_hidden_table_count: int = 0

    # Was ich ueber meinen eigenen Tisch weiss (= an welchen Positionen noch
    # eine verdeckte Karte unter der sichtbaren liegt). Wert: count der
    # eigenen noch nicht aufgedeckten Stapel-Plaetze.
    own_hidden_table_count: int = 0

    # Punktestand der laufenden Partie
    own_score: int = 0
    opp_score: int = 0

    # Indizes
    round_idx: int = 0
    trick_idx: int = 0
