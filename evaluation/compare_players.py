"""Spielstärke-Vergleich zwischen Player-Typen.

Aufruf:
    python -m evaluation.compare_players --games 1000

Setup: Team 0 (Sitze 0+2) = "A", Team 1 (Sitze 1+3) = "B".
Jedes Spiel wird bis 1000 Punkte gespielt; Sieger-Team wird gezählt.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

from jass_engine.player import Player
from jass_engine.variants.kreuz_jass import play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer


@dataclass
class MatchResult:
    games: int
    team_a_wins: int
    team_b_wins: int
    team_a_avg_score: float
    team_b_avg_score: float
    avg_rounds_per_game: float
    elapsed_seconds: float

    @property
    def team_a_winrate(self) -> float:
        return self.team_a_wins / self.games if self.games else 0.0


def run_match(
    team_a_factory,
    team_b_factory,
    num_games: int,
    target_score: int = 1000,
    seed: int = 0,
) -> MatchResult:
    rng = random.Random(seed)
    a_wins = 0
    b_wins = 0
    a_total = 0
    b_total = 0
    total_rounds = 0
    start = time.perf_counter()

    for g in range(num_games):
        players: list[Player] = [
            team_a_factory(0, rng),  # Sitz 0 = Team A
            team_b_factory(1, rng),  # Sitz 1 = Team B
            team_a_factory(2, rng),  # Sitz 2 = Team A
            team_b_factory(3, rng),  # Sitz 3 = Team B
        ]
        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        a = game.final_scores.get(0, 0)
        b = game.final_scores.get(1, 0)
        a_total += a
        b_total += b
        total_rounds += len(game.rounds)
        if a > b:
            a_wins += 1
        elif b > a:
            b_wins += 1

    elapsed = time.perf_counter() - start
    return MatchResult(
        games=num_games,
        team_a_wins=a_wins,
        team_b_wins=b_wins,
        team_a_avg_score=a_total / num_games,
        team_b_avg_score=b_total / num_games,
        avg_rounds_per_game=total_rounds / num_games,
        elapsed_seconds=elapsed,
    )


def _random_factory(seat: int, rng: random.Random) -> Player:
    return RandomPlayer(name=f"Random_{seat}", rng=random.Random(rng.randint(0, 10**9)))


def _heuristic_factory(seat: int, rng: random.Random) -> Player:
    return HeuristicPlayer(name=f"Heuristic_{seat}", rng=random.Random(rng.randint(0, 10**9)))


def format_result(label_a: str, label_b: str, r: MatchResult) -> str:
    return (
        f"\n{label_a:>14} vs {label_b:<14}\n"
        f"{'-' * 46}\n"
        f"Partien:           {r.games}\n"
        f"Siege {label_a:>10}:  {r.team_a_wins:6} ({r.team_a_winrate * 100:.1f} %)\n"
        f"Siege {label_b:>10}:  {r.team_b_wins:6} ({(1 - r.team_a_winrate) * 100:.1f} %)\n"
        f"Avg-Score {label_a:>6}:  {r.team_a_avg_score:7.1f}\n"
        f"Avg-Score {label_b:>6}:  {r.team_b_avg_score:7.1f}\n"
        f"Avg-Runden/Partie: {r.avg_rounds_per_game:.1f}\n"
        f"Zeit:              {r.elapsed_seconds:.1f} s "
        f"({r.elapsed_seconds / r.games * 1000:.1f} ms/Partie)\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Spielstärke-Vergleich")
    parser.add_argument("--games", type=int, default=500, help="Anzahl Partien")
    parser.add_argument("--target", type=int, default=1000, help="Punkte-Ziel pro Partie")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Vergleich ueber {args.games} Partien (Ziel {args.target} Punkte)...")

    # Heuristic vs Random
    res = run_match(
        _heuristic_factory,
        _random_factory,
        num_games=args.games,
        target_score=args.target,
        seed=args.seed,
    )
    print(format_result("Heuristic", "Random", res))

    # Random vs Random (Sanity-Check: sollte ~50/50 sein)
    res2 = run_match(
        _random_factory,
        _random_factory,
        num_games=args.games,
        target_score=args.target,
        seed=args.seed + 1,
    )
    print(format_result("Random", "Random", res2))

    # Heuristic vs Heuristic (Sanity-Check: sollte ~50/50 sein)
    res3 = run_match(
        _heuristic_factory,
        _heuristic_factory,
        num_games=args.games,
        target_score=args.target,
        seed=args.seed + 2,
    )
    print(format_result("Heuristic", "Heuristic", res3))


if __name__ == "__main__":
    main()
