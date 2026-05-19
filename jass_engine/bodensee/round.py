"""Eine Bodensee-Jass-Runde: Ansage, 18 Stiche, Matsch-Pruefung.

Anders als Kreuz/Solo:
- Keine Weisen, keine Stoecke
- 18 Stiche statt 9
- Matsch (alle 18 Stiche) = +100 Bonus
- Schieben gibt es nicht
"""

from __future__ import annotations

import random
from dataclasses import dataclass

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
from players.bodensee_player import BodenseePlayer


@dataclass
class BodenseeRoundResult:
    """Ergebnis einer Bodensee-Runde."""

    announcement: Announcement
    announcer_idx: int
    trick_winners: list[int]                    # 18 Eintraege, je 0 oder 1
    trick_points: list[int]                     # 18 Eintraege
    player_card_points: dict[int, int]          # Stich-Punkte pro Spieler (vor Matsch)
    matsch_player: int | None                   # 0 oder 1, wenn ein Spieler alle 18 Stiche gewann
    player_total_points: dict[int, int]         # final, inklusive Matsch-Bonus


def play_bodensee_round(
    players: list[BodenseePlayer],
    rng: random.Random | None = None,
    forced_announcer_idx: int | None = None,
    initial_scores: tuple[int, int] = (0, 0),
    round_idx: int = 0,
) -> BodenseeRoundResult:
    """Spielt eine vollstaendige Bodensee-Runde.

    Args:
        players: 2 BodenseePlayer-Instanzen
        rng: optionaler RNG
        forced_announcer_idx: wenn gesetzt, ueberschreibt den Weli-Halter (fuer Runde 2+)
        initial_scores: Punktestand der Partie vor dieser Runde
        round_idx: Nummer dieser Runde (0-basiert)

    Returns:
        BodenseeRoundResult mit Stichen, Punkten, evtl. Matsch.
    """
    if len(players) != 2:
        raise ValueError("Bodensee-Jass braucht genau 2 Spieler.")
    if rng is None:
        rng = random.Random()

    # 1. Karten verteilen
    player_states = deal_bodensee(rng)

    # 2. Ansager bestimmen
    if forced_announcer_idx is not None:
        announcer_idx = forced_announcer_idx
    else:
        announcer_idx = find_weli_holder_bodensee(player_states)

    # 3. Ansage durch announcer
    ps_announcer = player_states[announcer_idx]
    announcement = players[announcer_idx].choose_announcement(
        hand=list(ps_announcer.hand),
        visible_table=ps_announcer.visible_table_cards,
        round_idx=round_idx,
    )
    if announcement is None:
        raise RuntimeError("Schieben gibt es im Bodensee-Jass nicht.")

    # 4. 18 Stiche spielen
    trick_winners: list[int] = []
    trick_points: list[int] = []
    completed_tricks: list[CompletedTrick] = []
    player_card_points: dict[int, int] = {0: 0, 1: 0}

    starter = announcer_idx
    for trick_idx in range(TRICKS_PER_ROUND):
        variant_now = announcement.variant_for_trick(trick_idx)
        is_last = trick_idx == TRICKS_PER_ROUND - 1

        # Pro Spieler eine Choose-Card-Callback bauen
        def make_choose_fn(player: BodenseePlayer, seat: int):
            def choose(ps: BodenseePlayerState, legal: list[Card],
                       current_trick: list[Card], variant) -> Card:
                opp_idx = 1 - seat
                opp_ps = player_states[opp_idx]
                state = BodenseeGameState(
                    player_idx=seat,
                    variant=variant,
                    announcement=announcement,
                    current_trick_cards=list(current_trick),
                    current_trick_starter=starter,
                    completed_tricks=list(completed_tricks),
                    opponent_visible_table=opp_ps.visible_table_cards,
                    opponent_hand_count=len(opp_ps.hand),
                    opponent_hidden_table_count=opp_ps.hidden_table_count,
                    own_score=initial_scores[seat] + player_card_points[seat],
                    opp_score=initial_scores[opp_idx] + player_card_points[opp_idx],
                    round_idx=round_idx,
                    trick_idx=trick_idx,
                )
                return player.choose_card(
                    hand=list(ps.hand),
                    visible_table=ps.visible_table_cards,
                    state=state,
                )
            return choose

        choose_fns = [make_choose_fn(players[0], 0), make_choose_fn(players[1], 1)]

        trick, winner, points = play_bodensee_trick(
            starter_idx=starter,
            player_states=player_states,
            choose_card_fns=choose_fns,
            variant=variant_now,
            is_last_trick=is_last,
        )

        trick_winners.append(winner)
        trick_points.append(points)
        completed_tricks.append(CompletedTrick(
            starter=trick.starting_player_idx,
            cards=tuple(trick.cards),
        ))
        player_card_points[winner] += points
        starter = winner

    # 5. Matsch-Pruefung: ein Spieler hat alle 18 Stiche gewonnen
    matsch_player: int | None = None
    for pid in (0, 1):
        if trick_winners.count(pid) == TRICKS_PER_ROUND:
            matsch_player = pid
            player_card_points[pid] += MATCH_BONUS
            break

    # 6. Konsistenz: Summe der Karten-Punkte stimmt mit erwartetem Wert
    # Total Points = 152 + 5 (letzter Stich) = 157 oder, bei Trumpf/Gumpf,
    # auch +0 (keine Aenderung). Bei Oben/Unten: 150 + 5 = 155.
    # Da `variant_for_trick` bei Slalom pro Stich wechselt, summieren wir die
    # Punkte ueber alle Stiche.
    # Sanity-Check: kein Punktbetrag negativ, plausible Bereiche
    total_points = sum(player_card_points.values())
    expected_min = 155
    expected_max = 257  # 157 + 100 (Matsch)
    assert expected_min <= total_points <= expected_max, (
        f"Punkte-Summe {total_points} ausserhalb des Plausibilitaets-Bereichs"
    )

    player_total_points = dict(player_card_points)  # keine Weisen/Stoecke im Bodensee

    return BodenseeRoundResult(
        announcement=announcement,
        announcer_idx=announcer_idx,
        trick_winners=trick_winners,
        trick_points=trick_points,
        player_card_points=player_card_points,
        matsch_player=matsch_player,
        player_total_points=player_total_points,
    )
