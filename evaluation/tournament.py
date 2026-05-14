"""Tournament-Runner: laesst mehrere Player-Typen gegeneinander spielen
und aggregiert Elo-Ratings und Team-Statistiken.

Modi:
- "two_teams": Spieler-A-Typ als Team 0+2, Spieler-B-Typ als Team 1+3 (Standard)
- "round_robin": Alle Player-Typen einer Liste paarweise gegeneinander (kommt spaeter)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from jass_engine.player import Player
from jass_engine.variants.kreuz_jass import play_kreuz_jass
from evaluation.elo import EloRating
from evaluation.stats import TeamStats, update_stats_from_game


PlayerFactory = Callable[[int, random.Random], Player]


@dataclass
class TournamentResult:
    label_a: str
    label_b: str
    stats_a: TeamStats
    stats_b: TeamStats
    elo: EloRating
    games_played: int


def two_team_match(
    label_a: str,
    factory_a: PlayerFactory,
    label_b: str,
    factory_b: PlayerFactory,
    num_games: int,
    target_score: int = 1000,
    seed: int = 0,
    swap_seats_each_half: bool = True,
    elo: EloRating | None = None,
) -> TournamentResult:
    """Spielt N Partien Team-A vs Team-B.

    Args:
        label_a / label_b: Anzeigenamen
        factory_a / factory_b: Funktion, die einen Player erzeugt
        num_games: Gesamtanzahl Partien
        target_score: Punkteziel pro Partie (Default 1000)
        seed: RNG-Seed fuer Reproduzierbarkeit
        swap_seats_each_half: Tauscht in der zweiten Haelfte die Sitzplaetze,
            damit moegliche Sitz-Boni egalisiert sind
        elo: optionales bestehendes Elo, wird mit den Resultaten aktualisiert
    """
    rng = random.Random(seed)
    stats_a = TeamStats()
    stats_b = TeamStats()
    if elo is None:
        elo = EloRating()

    half = num_games // 2 if swap_seats_each_half else num_games

    for game_idx in range(num_games):
        if swap_seats_each_half and game_idx >= half:
            # In der zweiten Haelfte: Team A sitzt auf 1+3 statt 0+2
            seat_to_factory = {0: factory_b, 1: factory_a, 2: factory_b, 3: factory_a}
            team_a_team_id = 1
            team_b_team_id = 0
        else:
            seat_to_factory = {0: factory_a, 1: factory_b, 2: factory_a, 3: factory_b}
            team_a_team_id = 0
            team_b_team_id = 1

        players = [
            seat_to_factory[i](i, random.Random(rng.randint(0, 10**9)))
            for i in range(4)
        ]
        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )

        update_stats_from_game(
            stats_a, stats_b, game,
            team_a_id=team_a_team_id, team_b_id=team_b_team_id,
        )

        # Elo-Update: zwei Instanzen pro Team
        score_a = game.final_scores.get(team_a_team_id, 0)
        score_b = game.final_scores.get(team_b_team_id, 0)
        if score_a > score_b:
            elo_score_a = 1.0
        elif score_b > score_a:
            elo_score_a = 0.0
        else:
            elo_score_a = 0.5
        elo.update_team_match(
            team_a_players=[label_a, label_a],
            team_b_players=[label_b, label_b],
            score_a=elo_score_a,
        )

    return TournamentResult(
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a,
        stats_b=stats_b,
        elo=elo,
        games_played=num_games,
    )


def format_tournament_summary(res: TournamentResult) -> str:
    """Kompakte Zusammenfassung als Konsolen-Text."""
    from evaluation.stats import format_stats_table

    lines = []
    lines.append(f"Tournament: {res.label_a}  vs.  {res.label_b}")
    lines.append(f"Partien: {res.games_played}")
    lines.append(format_stats_table(res.label_a, res.stats_a, res.label_b, res.stats_b))
    lines.append("")
    lines.append("Elo-Leaderboard:")
    for name, rating, games in res.elo.leaderboard():
        lines.append(f"  {name:<24} {rating:>7.1f}   (n={games})")
    return "\n".join(lines)
