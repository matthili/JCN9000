"""Performance-Benchmark + Profiling der Spielsimulation.

Misst Partien/s mit Random-vs-Random (worst case fuer die Engine, da am wenigsten
Bot-Overhead) und Heuristic-vs-Heuristic, optional mit cProfile-Output.

Aufruf:
    python -m scripts.benchmark --games 500
    python -m scripts.benchmark --games 200 --profile --top 25
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import time

from jass_engine.variants.kreuz_jass import play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer


def run_random(num_games: int, target: int, seed: int) -> int:
    rng = random.Random(seed)
    rounds = 0
    for _ in range(num_games):
        players = [
            RandomPlayer(f"R{i}", random.Random(rng.randint(0, 10**9)))
            for i in range(4)
        ]
        game = play_kreuz_jass(
            players, target_score=target, rng=random.Random(rng.randint(0, 10**9))
        )
        rounds += len(game.rounds)
    return rounds


def run_heuristic(num_games: int, target: int, seed: int) -> int:
    rng = random.Random(seed)
    rounds = 0
    for _ in range(num_games):
        players = [
            HeuristicPlayer(f"H{i}", rng=random.Random(rng.randint(0, 10**9)))
            for i in range(4)
        ]
        game = play_kreuz_jass(
            players, target_score=target, rng=random.Random(rng.randint(0, 10**9))
        )
        rounds += len(game.rounds)
    return rounds


def benchmark(label: str, fn, num_games: int, target: int, seed: int) -> None:
    start = time.perf_counter()
    rounds = fn(num_games=num_games, target=target, seed=seed)
    elapsed = time.perf_counter() - start
    print(
        f"  {label:<28} {num_games:>6} Partien in {elapsed:6.2f}s "
        f"=> {num_games / elapsed:7.1f} Partien/s, "
        f"{rounds / elapsed:7.0f} Runden/s "
        f"({elapsed / num_games * 1000:5.2f} ms/Partie)"
    )


def profile(fn, num_games: int, target: int, seed: int, top: int) -> None:
    prof = cProfile.Profile()
    prof.enable()
    fn(num_games=num_games, target=target, seed=seed)
    prof.disable()
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(top)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true",
                        help="Zusaetzlich cProfile mit Top-N Aufruferfilter")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--scenario", choices=["random", "heuristic", "both"],
                        default="both")
    args = parser.parse_args()

    print(f"Benchmark: {args.games} Partien, Ziel {args.target}, Seed {args.seed}\n")

    if args.scenario in ("random", "both"):
        benchmark("Random vs Random", run_random, args.games, args.target, args.seed)
    if args.scenario in ("heuristic", "both"):
        benchmark("Heuristic vs Heuristic", run_heuristic, args.games, args.target, args.seed)

    if args.profile:
        # Profiling lieber auf reduzierter Partienzahl, sonst zu langsam
        prof_games = max(50, args.games // 5)
        fn = run_random if args.scenario != "heuristic" else run_heuristic
        print(f"\n--- cProfile auf {prof_games} {args.scenario}-Partien, Top {args.top} ---\n")
        profile(fn, prof_games, args.target, args.seed, args.top)


if __name__ == "__main__":
    main()
