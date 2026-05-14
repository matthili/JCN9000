"""Echter Multi-Process-Benchmark der Engine.

Misst, wie gut die Engine auf parallele Worker skaliert -- OHNE den IPC-/IO-Overhead
des Datengenerators, der die Daten zurueck zum Master schickt. Hier produziert
jeder Worker nur eine Zahl ("Anzahl gespielter Runden"), damit der Pickling-
Overhead vernachlaessigbar ist.

So sehen wir die theoretische Engine-Performance bei N parallelen Kernen,
unabhaengig von Datentransfer-Overhead.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time


def _worker(args: tuple[int, int, int]) -> int:
    """Spielt N Partien, gibt Anzahl gespielter Runden zurueck."""
    num_games, seed, target = args
    # Lazy import damit Worker den State nicht doppelt initialisiert
    from jass_engine.variants.kreuz_jass import play_kreuz_jass
    from players.random_player import RandomPlayer

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


def benchmark(workers: int, games_per_worker: int, target: int, seed: int) -> tuple[float, int, int]:
    base_rng = random.Random(seed)
    tasks = [(games_per_worker, base_rng.randint(0, 10**9), target) for _ in range(workers)]
    start = time.perf_counter()
    if workers == 1:
        rounds = _worker(tasks[0])
    else:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks)
        rounds = sum(results)
    elapsed = time.perf_counter() - start
    return elapsed, workers * games_per_worker, rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-worker", type=int, default=200)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-counts", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12, 16, 20, 28])
    args = parser.parse_args()

    print(f"MP-Benchmark: {args.games_per_worker} Partien je Worker, Ziel {args.target}")
    print(f"{'Workers':>8}  {'Partien':>8}  {'Zeit (s)':>9}  {'Partien/s':>11}  {'Runden/s':>10}  {'Speed-Up':>10}")
    print("-" * 70)

    baseline_speed = None
    for n in args.worker_counts:
        elapsed, games, rounds = benchmark(n, args.games_per_worker, args.target, args.seed)
        speed = games / elapsed
        rounds_per_s = rounds / elapsed
        if baseline_speed is None:
            baseline_speed = speed
            speedup = 1.0
        else:
            speedup = speed / baseline_speed
        print(
            f"{n:>8}  {games:>8}  {elapsed:>9.2f}  {speed:>11.1f}  {rounds_per_s:>10.0f}  {speedup:>9.2f}x"
        )


if __name__ == "__main__":
    main()
