"""Eine Runde: Ansage (mit Schieben), Weisen, 9 Stiche, Stöcke, Matsch."""

from __future__ import annotations

from dataclasses import dataclass

from jass_engine.card import Card, Rank
from jass_engine.deck import deal, find_weli_holder
from jass_engine.player import GameState, Player
from jass_engine.rules import (
    MATCH_BONUS,
    legal_moves,
    total_points_per_round,
)
from jass_engine.trick import CompletedTrick, Trick
from jass_engine.variant import Announcement, PlayMode
from jass_engine.weis import (
    TeamWeisResult,
    Weis,
    compare_team_weise,
    find_weise,
    has_stoecke,
    stoecke_apply,
    stoecke_weis,
)


@dataclass
class RoundResult:
    announcement: Announcement
    announcer_idx: int
    pushed_to: int | None
    trick_winners: list[int]
    trick_points: list[int]
    team_card_points: dict[int, int]
    team_weis_results: dict[int, TeamWeisResult]
    team_stoecke: dict[int, int]
    matsch_team: int | None
    team_total_points: dict[int, int]


def play_round(
    players: list[Player],
    teams: list[int],
    round_idx: int,
    rng=None,
    forced_announcer_idx: int | None = None,
    allow_push: bool = True,
) -> RoundResult:
    """Spielt eine vollständige Runde.

    In Runde 1 sagt der Weli-Halter an (außer `forced_announcer_idx` ist gesetzt).
    Ab Runde 2 darf der Ansager schieben — der Geschobene wählt dann die Ansage,
    aber der ursprüngliche Ansager spielt trotzdem als erster aus.

    Args:
        allow_push: erlaubt Schieben (Default True). Bei Solo-Jass wird das auf
            False gesetzt, weil es keinen Partner gibt, zu dem man schieben könnte.
    """
    num_players = len(players)
    if num_players != 4:
        raise NotImplementedError("Kreuz-Jass-Runde aktuell nur für 4 Spieler.")

    hands = deal(num_players=num_players, rng=rng)

    if forced_announcer_idx is not None:
        announcer_idx = forced_announcer_idx
    elif round_idx == 0:
        announcer_idx = find_weli_holder(hands)
    else:
        announcer_idx = round_idx % num_players

    can_push = allow_push and round_idx > 0
    announcement = players[announcer_idx].choose_announcement(
        hands[announcer_idx], round_idx, can_push
    )
    pushed_to: int | None = None
    if announcement is None:
        if not can_push:
            raise RuntimeError("Schieben ist in Runde 1 nicht erlaubt.")
        partner_idx = _partner_of(announcer_idx, teams)
        pushed_to = partner_idx
        announcement = players[partner_idx].choose_announcement(
            hands[partner_idx], round_idx, can_push=False
        )
        if announcement is None:
            raise RuntimeError("Mitspieler darf nach Schieben nicht erneut schieben.")

    # Weisen ansagen (vor erstem Stich)
    initial_variant = announcement.variant_for_trick(0)
    weise_per_player: list[list[Weis]] = []
    for p_idx, player in enumerate(players):
        possible = find_weise(hands[p_idx])
        announced = player.announce_weise(hands[p_idx], initial_variant, possible)
        for w in announced:
            if w not in possible:
                raise RuntimeError(f"Spieler {p_idx} hat ungültigen Weis angesagt: {w}")
        weise_per_player.append(announced)

    starter = announcer_idx
    trick_results: list[Trick] = []
    trick_winners: list[int] = []
    points_per_trick: list[int] = []
    team_card_points: dict[int, int] = {tid: 0 for tid in set(teams)}

    # Stock-Tracking (nur bei Trumpf-Modus)
    stock_player: int | None = None
    stock_seen_count = 0
    stoecke_announced_by_team: dict[int, bool] = {}
    stock_cards: set[Card] = set()
    if stoecke_apply(initial_variant):
        assert initial_variant.trump_suit is not None
        stock_cards = {
            Card(initial_variant.trump_suit, Rank.OBER),
            Card(initial_variant.trump_suit, Rank.KOENIG),
        }
        for p_idx in range(num_players):
            if has_stoecke(hands[p_idx], initial_variant.trump_suit):
                stock_player = p_idx
                break

    announcement_order = [(announcer_idx + i) % num_players for i in range(num_players)]

    for trick_idx in range(9):
        variant_for_this_trick = announcement.variant_for_trick(trick_idx)
        trick = Trick(starting_player_idx=starter, num_players=num_players)
        for _ in range(num_players):
            cur = trick.next_player_idx()
            state = GameState(
                player_idx=cur,
                variant=variant_for_this_trick,
                announcement=announcement,
                current_trick_cards=list(trick.cards),
                current_trick_starter=trick.starting_player_idx,
                teams=list(teams),
                completed_tricks=[
                    CompletedTrick(starter=t.starting_player_idx, cards=tuple(t.cards))
                    for t in trick_results
                ],
                round_idx=round_idx,
                trick_idx=trick_idx,
                num_players=num_players,
            )
            legal = legal_moves(hands[cur], trick.cards, variant_for_this_trick)
            chosen = players[cur].choose_card(hands[cur], state)
            if chosen not in legal:
                raise RuntimeError(
                    f"Illegaler Zug Spieler {cur}: {chosen} nicht in {legal} "
                    f"(Variant: {variant_for_this_trick})"
                )
            hands[cur].remove(chosen)
            trick.add_card(chosen)

            if stock_cards and chosen in stock_cards:
                stock_seen_count += 1
                if (
                    stock_seen_count == 2
                    and stock_player is not None
                    and not stoecke_announced_by_team.get(teams[stock_player], False)
                ):
                    stoecke_announced_by_team[teams[stock_player]] = True

        is_last = trick_idx == 8
        winner_idx = trick.winner_idx(variant_for_this_trick)
        pts = trick.points(variant_for_this_trick, is_last=is_last)
        trick_results.append(trick)
        trick_winners.append(winner_idx)
        points_per_trick.append(pts)
        team_card_points[teams[winner_idx]] += pts
        starter = winner_idx

    # Matsch
    matsch_team: int | None = None
    for tid in team_card_points:
        if sum(1 for w in trick_winners if teams[w] == tid) == 9:
            matsch_team = tid
            team_card_points[tid] += MATCH_BONUS
            break

    # Konsistenzprüfung: Summe der Kartenpunkte muss passen
    # Für Slalom: pro Stich kann der Modus wechseln, also rechnen wir den Erwartungswert
    # über die effektiven Stich-Varianten zusammen
    expected_base = _expected_card_points(announcement)
    expected = expected_base + (MATCH_BONUS if matsch_team is not None else 0)
    if sum(team_card_points.values()) != expected:
        raise RuntimeError(
            f"Punkte-Konsistenzfehler: erwartet {expected}, "
            f"bekommen {sum(team_card_points.values())} bei {announcement}"
        )

    team_weis_results = compare_team_weise(
        weise_per_player=weise_per_player,
        teams=teams,
        announcement_order=announcement_order,
    )

    team_stoecke: dict[int, int] = {tid: 0 for tid in set(teams)}
    if (
        stoecke_apply(initial_variant)
        and stock_player is not None
        and stoecke_announced_by_team.get(teams[stock_player], False)
    ):
        assert initial_variant.trump_suit is not None
        team_stoecke[teams[stock_player]] = stoecke_weis(initial_variant.trump_suit).points

    team_total_points: dict[int, int] = {}
    for tid in set(teams):
        team_total_points[tid] = (
            team_card_points[tid]
            + team_weis_results[tid].points
            + team_stoecke[tid]
        )

    return RoundResult(
        announcement=announcement,
        announcer_idx=announcer_idx,
        pushed_to=pushed_to,
        trick_winners=trick_winners,
        trick_points=points_per_trick,
        team_card_points=team_card_points,
        team_weis_results=team_weis_results,
        team_stoecke=team_stoecke,
        matsch_team=matsch_team,
        team_total_points=team_total_points,
    )


def _expected_card_points(announcement: Announcement) -> int:
    """Erwartete Stich-Punktesumme einer Runde inkl. letzter-Stich-Bonus.

    Bei Trumpf: 157 (152 + 5). Bei OBEN/UNTEN: 155 (150 + 5). Bei Slalom: ebenfalls 155,
    da pro Stich entweder OBEN oder UNTEN gilt — beide haben dieselbe Punktesumme.
    """
    # Bei jeder Slalom-Variant ist der Wertesatz identisch (OBEN_UNTEN), also reicht der erste
    return total_points_per_round(announcement.variant_for_trick(0))


def _partner_of(player_idx: int, teams: list[int]) -> int:
    own_team = teams[player_idx]
    for idx, t in enumerate(teams):
        if t == own_team and idx != player_idx:
            return idx
    raise RuntimeError("Kein Partner gefunden.")
