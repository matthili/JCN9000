"""Bodensee-Jass-Spielregeln.

Stich-Regeln (Farbzwang, Buur-Ausnahme) sind dieselben wie bei Kreuz-/Solo-Jass.
Der einzige strukturelle Unterschied: der "Karten-Pool" pro Spieler besteht nicht
aus 9 Hand-Karten, sondern aus bis zu 12 verfuegbaren Karten = Hand +
sichtbare Tisch-Karten. Verdeckte Tisch-Karten zaehlen weder fuer den
Bedienzwang noch sind sie spielbar.

Die "kein-Untertrumpfen"-Regel ist im 2-Spieler-Bodensee strukturell
irrelevant: ein Stich besteht aus genau 2 Karten, einem Anspiel und einer
Antwort -- es gibt keinen dritten Spieler, der untertrumpfen koennte. Die
bestehende `legal_moves`-Logik aus `jass_engine/rules.py` greift diese Regel
korrekt nur dann, wenn schon ein Trumpf im Stich liegt UND der Lead
Nicht-Trumpf ist -- bei 2 Spielern eine unmoegliche Konstellation, weil der
Lead ja gerade die erste Karte ist.

Deshalb koennen wir die bestehende `legal_moves(hand, current_trick, variant)`
1:1 wiederverwenden und ihr einfach den vollen Verfuegbar-Pool als "hand"
mitgeben.
"""

from __future__ import annotations

from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.card import Card
from jass_engine.rules import legal_moves
from jass_engine.variant import Variant


def legal_moves_bodensee(
    player_state: BodenseePlayerState,
    current_trick: list[Card],
    variant: Variant,
) -> list[Card]:
    """Karten, die der Spieler legal ausspielen darf.

    Pool ist Hand + sichtbare Tisch-Karten. Verdeckte Tisch-Karten sind nicht
    spielbar.

    Args:
        player_state: Bodensee-Spielzustand mit hand und table.
        current_trick: aktuell offene Karten im Stich (0 oder 1 Karte).
        variant: aktuelle Spielart (Trumpf, Gumpf, Oben, Unten).

    Returns:
        Liste der spielbaren Karten. Jede zurueckgegebene Karte liegt
        eindeutig entweder in der Hand oder sichtbar auf dem Tisch -- der
        Aufrufer kann ueber `player_state.has_card_in_hand` bzw.
        `has_card_on_visible_table` rausfinden, woher er sie nimmt.
    """
    pool = player_state.available_cards
    return legal_moves(pool, current_trick, variant)


def card_source(player_state: BodenseePlayerState, card: Card) -> str:
    """Liefert "hand" oder "table", je nachdem wo die Karte liegt.

    Wirft ValueError, wenn die Karte weder in der Hand noch sichtbar auf dem
    Tisch ist. Hilfsfunktion fuer die Trick-Engine, die nach einer
    Karten-Wahl wissen muss, welche Datenstruktur zu modifizieren ist.
    """
    if player_state.has_card_in_hand(card):
        return "hand"
    if player_state.has_card_on_visible_table(card):
        return "table"
    raise ValueError(
        f"Karte {card} liegt weder in Hand {player_state.hand} noch sichtbar "
        f"auf dem Tisch {player_state.visible_table_cards}"
    )
