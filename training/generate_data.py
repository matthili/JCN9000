"""Datengenerator: simuliert Partien mit dem HeuristicPlayer und speichert (state, mask, action).

Aufruf:
    python -m training.generate_data --games 10000 --output data/heuristic_train
    python -m training.generate_data --games 100000 --shard-size 10000 --workers 8

Output: ein oder mehrere .npz-Dateien im angegebenen Verzeichnis mit den Arrays
    X      : (N, INPUT_DIM)  float32  — Featurevektoren
    masks  : (N, ACTION_DIM) uint8    — legale Aktionsmasken (1=erlaubt)
    actions: (N,)            uint8    — gewählter Aktionsindex 0..35
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jass_engine.variants.kreuz_jass import play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from training.encoder import ACTION_DIM, INPUT_DIM
from training.recording_player import RecordingPlayer


@dataclass
class ShardData:
    X: np.ndarray
    masks: np.ndarray
    actions: np.ndarray

    @property
    def size(self) -> int:
        return len(self.X)


def _simulate_games(num_games: int, seed: int, target_score: int = 1000) -> ShardData:
    """Spielt `num_games` Partien und sammelt alle (state, mask, action)-Tupel."""
    rng = random.Random(seed)
    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []

    for _ in range(num_games):
        players = [
            RecordingPlayer(
                HeuristicPlayer(
                    name=f"H{i}",
                    rng=random.Random(rng.randint(0, 10**9)),
                )
            )
            for i in range(4)
        ]
        play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        for p in players:
            all_states.extend(p.states)
            all_masks.extend(p.masks)
            all_actions.extend(p.actions)

    X = np.stack(all_states).astype(np.float32) if all_states else np.empty((0, INPUT_DIM), dtype=np.float32)
    masks = np.stack(all_masks).astype(np.uint8) if all_masks else np.empty((0, ACTION_DIM), dtype=np.uint8)
    actions = np.array(all_actions, dtype=np.uint8) if all_actions else np.empty((0,), dtype=np.uint8)
    return ShardData(X=X, masks=masks, actions=actions)


def _worker(args: tuple[int, int, int]) -> ShardData:
    """Worker für multiprocessing.Pool."""
    games, seed, target = args
    return _simulate_games(games, seed=seed, target_score=target)


def generate(
    output_dir: Path,
    num_games: int,
    shard_size: int,
    workers: int,
    target_score: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = (num_games + shard_size - 1) // shard_size
    base_rng = random.Random(seed)
    tasks = []
    for shard_idx in range(num_shards):
        games_in_shard = min(shard_size, num_games - shard_idx * shard_size)
        tasks.append((games_in_shard, base_rng.randint(0, 10**9), target_score))

    total_samples = 0
    start = time.perf_counter()

    if workers <= 1:
        # Sequenziell
        with tqdm(total=num_games, desc="Spiele") as pbar:
            for shard_idx, task in enumerate(tasks):
                shard = _worker(task)
                _save_shard(output_dir, shard_idx, shard)
                total_samples += shard.size
                pbar.update(task[0])
    else:
        with mp.Pool(processes=workers) as pool:
            with tqdm(total=num_games, desc="Spiele") as pbar:
                for shard_idx, shard in enumerate(pool.imap_unordered(_worker, tasks)):
                    _save_shard(output_dir, shard_idx, shard)
                    total_samples += shard.size
                    pbar.update(tasks[shard_idx][0])

    elapsed = time.perf_counter() - start
    print(
        f"\nFertig: {num_games} Partien, {total_samples} Samples in {num_shards} Shards. "
        f"Dauer {elapsed:.1f} s, {num_games / elapsed:.0f} Partien/s, "
        f"{total_samples / elapsed:.0f} Samples/s."
    )


def _save_shard(output_dir: Path, shard_idx: int, shard: ShardData) -> None:
    path = output_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(path, X=shard.X, masks=shard.masks, actions=shard.actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000, help="Anzahl Partien insgesamt")
    parser.add_argument("--output", type=str, default="data/heuristic_train", help="Output-Verzeichnis")
    parser.add_argument("--shard-size", type=int, default=2_000, help="Partien pro Shard-Datei")
    parser.add_argument("--workers", type=int, default=1, help="Parallele Prozesse (1 = sequenziell)")
    parser.add_argument("--target", type=int, default=1000, help="Punktziel pro Partie")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(
        f"Generiere {args.games} Partien -> {output_dir} "
        f"(Shards a {args.shard_size}, Workers {args.workers})"
    )
    generate(
        output_dir=output_dir,
        num_games=args.games,
        shard_size=args.shard_size,
        workers=args.workers,
        target_score=args.target,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
