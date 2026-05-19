"""Demo: zwei BodenseeHeuristic-Spieler spielen eine komplette Runde,
Stich fuer Stich auf der Konsole sichtbar.

Aufruf: python -m scripts.demo_bodensee
"""

from __future__ import annotations

import random

from jass_engine.bodensee.deal import (
    TRICKS_PER_ROUND,
    deal_bodensee,
    find_weli_holder_bodensee,
)
from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.bodensee.trick import play_bodensee_trick
from jass_engine.card import Card
from jass_engine.rules import MATCH_BONUS
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_player import BodenseePlayer


def fmt_card(card: Card) -> str:
    """Kurzform: z.B. 'Eichel-Ass' oder '* Schelle-6 (Weli)'."""
    base = f"{card.suit.german_name}-{card.rank.full_name}"
    if card.is_weli:
        base += " (Weli)"
    return base


def fmt_cards(cards: list[Card]) -> str:
    if not cards:
        return "(leer)"
    return ", ".join(fmt_card(c) for c in cards)


def fmt_table(ps: BodenseePlayerState) -> str:
    """Stapel als 'sichtbar(?)' Liste. ? = verdeckte Karte darunter, - = leer."""
    parts = []
    for s in ps.table:
        if s.visible is None and s.hidden is None:
            parts.append("[leer]")
        else:
            v = fmt_card(s.visible) if s.visible else "—"
            h = "?" if s.hidden else "—"
            parts.append(f"{v} ({h})")
    return "  |  ".join(parts)


def make_choose_card_fn(
    player: BodenseePlayer,
    seat: int,
    player_states: list[BodenseePlayerState],
    announcement: Announcement,
    completed_tricks: list[CompletedTrick],
    points: dict[int, int],
    starter_ref: list[int],
    round_idx: int,
    trick_idx_ref: list[int],
):
    def choose(ps, legal, current_trick, variant):
        opp_idx = 1 - seat
        opp_ps = player_states[opp_idx]
        state = BodenseeGameState(
            player_idx=seat,
            variant=variant,
            announcement=announcement,
            current_trick_cards=list(current_trick),
            current_trick_starter=starter_ref[0],
            completed_tricks=list(completed_tricks),
            opponent_visible_table=opp_ps.visible_table_cards,
            opponent_hand_count=len(opp_ps.hand),
            opponent_hidden_table_count=opp_ps.hidden_table_count,
            own_score=points[seat],
            opp_score=points[opp_idx],
            round_idx=round_idx,
            trick_idx=trick_idx_ref[0],
        )
        return player.choose_card(
            hand=list(ps.hand),
            visible_table=ps.visible_table_cards,
            state=state,
        )

    return choose


def main():
    rng = random.Random(42)
    p0 = BodenseeHeuristicPlayer("Anna", rng=random.Random(1))
    p1 = BodenseeHeuristicPlayer("Berni", rng=random.Random(2))
    names = ["Anna", "Berni"]

    print("\n" + "=" * 70)
    print("  Bodensee-Jass Demo:  Anna  vs  Berni")
    print("=" * 70)

    player_states = deal_bodensee(rng)
    ps0, ps1 = player_states

    print("\nAusgangslage (was beide Spieler sehen koennten):\n")
    for i, ps in enumerate(player_states):
        print(f"  {names[i]}:")
        print(f"    Hand (nur fuer {names[i]} sichtbar):")
        print(f"      {fmt_cards(ps.hand)}")
        print(f"    Tisch (beide Spieler sehen die sichtbaren, '?' = verdeckt):")
        print(f"      {fmt_table(ps)}")
        print()

    weli_idx = find_weli_holder_bodensee(player_states)
    print(f"Weli-Halter: {names[weli_idx]}  -- sagt an.")

    ann = [p0, p1][weli_idx].choose_announcement(
        hand=list(player_states[weli_idx].hand),
        visible_table=player_states[weli_idx].visible_table_cards,
        round_idx=0,
    )
    print(f"Ansage: {ann}\n")

    print("-" * 70)
    print("Stiche:")
    print("-" * 70)

    starter_ref = [weli_idx]
    trick_idx_ref = [0]
    points = {0: 0, 1: 0}
    completed_tricks: list[CompletedTrick] = []
    trick_winners: list[int] = []

    for trick_idx in range(TRICKS_PER_ROUND):
        trick_idx_ref[0] = trick_idx
        variant_now = ann.variant_for_trick(trick_idx)
        is_last = trick_idx == TRICKS_PER_ROUND - 1

        # Sicht VOR dem Stich (was hat wer)
        # (gedruckt nur fuer die ersten und letzten paar Stiche, um den Output kurz zu halten)
        verbose_state = (trick_idx < 3) or (trick_idx >= TRICKS_PER_ROUND - 3)
        if verbose_state:
            print(f"\nStich {trick_idx + 1:2d} ({variant_now}, "
                  f"Anspieler: {names[starter_ref[0]]})")
            for i, ps in enumerate(player_states):
                print(f"  {names[i]} Hand: {fmt_cards(ps.hand)}")
                print(f"  {names[i]} Tisch: {fmt_table(ps)}")

        choose_fns = [
            make_choose_card_fn(
                p0, 0, player_states, ann, completed_tricks,
                points, starter_ref, 0, trick_idx_ref,
            ),
            make_choose_card_fn(
                p1, 1, player_states, ann, completed_tricks,
                points, starter_ref, 0, trick_idx_ref,
            ),
        ]

        trick, winner, pts = play_bodensee_trick(
            starter_idx=starter_ref[0],
            player_states=player_states,
            choose_card_fns=choose_fns,
            variant=variant_now,
            is_last_trick=is_last,
        )
        trick_winners.append(winner)
        points[winner] += pts
        completed_tricks.append(CompletedTrick(
            starter=trick.starting_player_idx,
            cards=tuple(trick.cards),
        ))

        card1, card2 = trick.cards
        first_name = names[starter_ref[0]]
        second_name = names[1 - starter_ref[0]]
        line = (
            f"Stich {trick_idx + 1:2d} ({variant_now}): "
            f"{first_name} {fmt_card(card1)}  vs  "
            f"{second_name} {fmt_card(card2)}  "
            f"-> {names[winner]} +{pts}  "
            f"[Anna {points[0]}, Berni {points[1]}]"
        )
        print(line)

        starter_ref[0] = winner

    print("\n" + "=" * 70)
    print(f"Runde fertig. Anna: {points[0]}, Berni: {points[1]}")
    if trick_winners.count(0) == TRICKS_PER_ROUND:
        points[0] += MATCH_BONUS
        print(f"!!! MATSCH von Anna !!! +{MATCH_BONUS} Bonus -> Anna {points[0]}")
    elif trick_winners.count(1) == TRICKS_PER_ROUND:
        points[1] += MATCH_BONUS
        print(f"!!! MATSCH von Berni !!! +{MATCH_BONUS} Bonus -> Berni {points[1]}")
    print(
        f"Stich-Verteilung: Anna {trick_winners.count(0)}, "
        f"Berni {trick_winners.count(1)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
