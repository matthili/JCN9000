"""Void-Inferenz aus der Spielhistorie (welche Farben kann ein Sitz nicht haben).

Gemeinsamer Baustein fuer zwei Konsumenten:
- die **NN-Determinisierung** (Rollouts verteilen unbekannte Karten nur auf
  Sitze, die sie auch wirklich halten koennten)
- die **Heuristik** (z.B. "hoer auf Trumpf zu ziehen, wenn die Gegner blank sind")

Grundlage ist die Bedien-Regel (`jass_engine.rules.legal_moves`):

  * Nicht-Trumpf-Farbe L angespielt (oder Oben/Unten/Slalom ohne Trumpf):
    Wer eine Karte spielt, die WEDER L NOCH Trumpf ist, hatte kein L
    -> **blank in L** (sicher; man haette sonst L bedienen oder trumpfen muessen).
    Eine Trumpf-Karte auf einen Nicht-Trumpf-Lead erlaubt KEINEN Schluss
    (Stechen ist auch mit L auf der Hand erlaubt: "bedienen ODER stechen").

  * Trumpf angespielt: Wer eine Nicht-Trumpf-Karte spielt, hatte keinen
    Nicht-Buur-Trumpf (sonst Trumpf-Zwang). Der Buur kann er aber noch halten
    (Buur-Ausnahme: ist der Buur der einzige Trumpf, darf frei gespielt werden).
    -> **blank in allen Truempfen AUSSER dem Buur**.

Das Ergebnis ist pro Sitz eine Menge **verbotener Karten** (Cards, die der Sitz
sicher nicht halten kann). Das ist allgemeiner als "blank in Farbe X" und
modelliert die Buur-Ausnahme exakt: Bei Trumpf-Lead verbieten wir alle Truempfe
ausser dem Buur -- nie eine Karte, die der Sitz legal noch haben koennte
(soundness: wir schliessen nur aus, was beweisbar unmoeglich ist).
"""

from __future__ import annotations

from jass_engine.card import ALL_RANKS, Card, Rank, Suit
from jass_engine.trick import CompletedTrick
from jass_engine.variant import PlayMode


def _trump_suit_for(variant) -> Suit | None:
    """Trumpf-Farbe der Variante, oder None (Oben/Unten/Slalom)."""
    if variant.mode in (PlayMode.TRUMPF, PlayMode.GUMPF):
        return variant.trump_suit
    return None


def infer_forbidden_cards(
    completed_tricks: list[CompletedTrick],
    announcement,
    num_players: int = 4,
    start_trick_idx: int = 0,
) -> dict[int, set[Card]]:
    """Leitet pro Sitz die Menge der Karten ab, die er sicher NICHT halten kann.

    Args:
        completed_tricks: abgeschlossene Stiche der laufenden Runde, in Reihenfolge.
            Position i entspricht Runden-Stich `start_trick_idx + i` (relevant nur
            fuer Slalom, wo die Variante pro Stich wechselt).
        announcement: Ansage der Runde (liefert `variant_for_trick(idx)`).
        num_players: Spielerzahl (4 bei Kreuz/Solo).
        start_trick_idx: Runden-Index des ersten Stichs in `completed_tricks`
            (Default 0 -- die Engine sammelt Stiche ab Rundenbeginn).

    Returns:
        `forbidden[seat]` = Menge von Karten, die `seat` beweisbar nicht hat.
        Bereits gespielte Karten koennen enthalten sein (harmlos -- sie liegen
        ohnehin nicht mehr im unbekannten Pool).
    """
    forbidden: dict[int, set[Card]] = {s: set() for s in range(num_players)}

    for offset, trick in enumerate(completed_tricks):
        cards = list(trick.cards)
        if len(cards) < 2:
            continue
        led_suit = cards[0].suit
        variant = announcement.variant_for_trick(start_trick_idx + offset)
        trump = _trump_suit_for(variant)

        for j in range(1, len(cards)):
            seat = (trick.starter + j) % num_players
            played = cards[j]

            if trump is not None and led_suit == trump:
                # Trumpf angespielt: Nicht-Trumpf gespielt -> blank in Trumpf
                # AUSSER moeglicherweise dem Buur (Trumpf-Unter).
                if played.suit != trump:
                    for r in ALL_RANKS:
                        if r != Rank.UNTER:
                            forbidden[seat].add(Card(trump, r))
                # Trumpf gespielt (Folgen) -> kein Schluss
            else:
                # Nicht-Trumpf-Lead (oder Variante ohne Trumpf):
                # Karte gespielt, die weder Lead-Farbe noch Trumpf ist -> blank
                # in der Lead-Farbe. (Trumpf auf Nicht-Trumpf-Lead = Stechen,
                # erlaubt auch mit Lead-Farbe -> kein Schluss.)
                is_trump_card = trump is not None and played.suit == trump
                if played.suit != led_suit and not is_trump_card:
                    for r in ALL_RANKS:
                        forbidden[seat].add(Card(led_suit, r))

    return forbidden


def seat_is_void_in_trump(forbidden_for_seat: set[Card], trump: Suit) -> bool:
    """True, wenn der Sitz beweisbar keinen Trumpf ausser evtl. dem Buur hat.

    Praktisch fuer die Heuristik: "Lohnt es sich noch, Trumpf zu ziehen?"
    Der Buur wird bewusst ignoriert (Buur-Ausnahme) -- ein einzelner Buur beim
    Gegner aendert an der "ausgetrumpft"-Lage nichts Wesentliches.
    """
    return all(
        Card(trump, r) in forbidden_for_seat
        for r in ALL_RANKS
        if r != Rank.UNTER
    )


__all__ = ["infer_forbidden_cards", "seat_is_void_in_trump"]
