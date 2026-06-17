"""Stich-Logik fuer Bodensee-Jass.

Pro Stich gibt's 2 Karten (eine pro Spieler). Spielablauf eines Stichs:
1. Anspieler waehlt eine Karte aus seinem Verfuegbar-Pool (Hand + sichtbare Tisch-Karten).
2. Karte wird aus Hand bzw. Tisch entfernt; bei Tisch-Karte wird ggf. die
   darunterliegende verdeckte Karte zur neuen sichtbaren.
3. Der andere Spieler waehlt unter Bedienzwang (= aus seinem Verfuegbar-Pool
   die legalen).
4. Karte wird wie oben angewandt.
5. Stich-Gewinner und -Punkte werden berechnet.
"""

from __future__ import annotations

from typing import Callable

from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.rules import card_source, legal_moves_bodensee
from jass_engine.card import Card
from jass_engine.trick import Trick
from jass_engine.variant import Variant


# Callback-Signatur fuer "Spieler waehlt eine Karte"
# (eigentliches Player-Interface kommt in players/bodensee_player.py)
ChooseCardFn = Callable[
    [BodenseePlayerState, list[Card], list[Card], Variant],
    Card,
]


def play_card_from_state(
    player_state: BodenseePlayerState,
    card: Card,
) -> None:
    """Entfernt eine gewaehlte Karte aus dem Spielzustand.

    Wenn die Karte in der Hand liegt: einfaches Entfernen.
    Wenn sie auf dem Tisch sichtbar liegt: Tisch-Stapel aufdecken (verdeckte
    Karte wird neu sichtbar).
    """
    source = card_source(player_state, card)
    if source == "hand":
        player_state.remove_from_hand(card)
    else:
        player_state.play_from_table(card)


def play_bodensee_trick(
    starter_idx: int,
    player_states: list[BodenseePlayerState],
    choose_card_fns: list[ChooseCardFn],
    variant: Variant,
    is_last_trick: bool = False,
) -> tuple[Trick, int, int]:
    """Spielt einen einzelnen Bodensee-Stich (2 Karten).

    Args:
        starter_idx: 0 oder 1 -- Spieler, der den Stich anspielt
        player_states: zwei BodenseePlayerStates (Spieler 0, 1)
        choose_card_fns: Pro Spieler eine Callback-Funktion, die die naechste
            Karte waehlt (gegeben: eigener Spielzustand, legale Karten,
            aktueller Trick, Variant).
        variant: Spielart fuer diesen Stich (kann bei Slalom pro Stich wechseln)
        is_last_trick: True wenn dies der letzte Stich der Runde ist (+5)

    Returns:
        Tupel (Trick-Objekt, winner_idx (0 oder 1), Punkte fuer den Gewinner).
    """
    trick = Trick(starting_player_idx=starter_idx, num_players=2)

    for _ in range(2):
        cur_idx = trick.next_player_idx()
        ps = player_states[cur_idx]
        legal = legal_moves_bodensee(ps, list(trick.cards), variant)
        chosen = choose_card_fns[cur_idx](ps, legal, list(trick.cards), variant)
        if chosen not in legal:
            raise RuntimeError(
                f"Spieler {cur_idx} hat illegale Karte {chosen} gewaehlt "
                f"(legal: {legal}, Verfuegbar: {ps.available_cards}, Variant: {variant})"
            )
        play_card_from_state(ps, chosen)
        trick.add_card(chosen)

    winner_idx = trick.winner_idx(variant)
    points = trick.points(variant, is_last=is_last_trick)
    return trick, winner_idx, points
