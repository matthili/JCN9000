"""Datengenerator: simuliert Partien mit dem HeuristicPlayer und speichert (state, mask, action).

Aufruf:
    python -m training.generate_data --games 10000 --output data/heuristic_train
    python -m training.generate_data --games 100000 --shard-size 1000 --workers 16

Output: ein oder mehrere .npz-Dateien im angegebenen Verzeichnis mit den Arrays
    X      : (N, INPUT_DIM)  float32  — Featurevektoren
    masks  : (N, ACTION_DIM) uint8    — legale Aktionsmasken (1=erlaubt)
    actions: (N,)            uint8    — gewählter Aktionsindex 0..35

Wichtig fuer Performance: jeder Worker schreibt seinen Shard **direkt zur Disk**.
Der Master sammelt nur Dateipfade + Sample-Count, kein Massendaten-Transfer ueber
IPC. Damit verschwindet der bisherige Pickling-Overhead von hunderten MB pro Shard.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
class ShardResult:
    """Was ein Worker dem Master zurueckmeldet -- nur Metadaten, keine Daten."""

    path: Path
    num_samples: int
    num_games: int


def _simulate_and_save(
    num_games: int,
    seed: int,
    target_score: int,
    output_path: Path,
) -> ShardResult:
    """Spielt `num_games` Partien, speichert Shard direkt zur Disk."""
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

    if all_states:
        X = np.stack(all_states).astype(np.float32, copy=False)
        masks = np.stack(all_masks).astype(np.uint8, copy=False)
        actions = np.array(all_actions, dtype=np.uint8)
    else:
        X = np.empty((0, INPUT_DIM), dtype=np.float32)
        masks = np.empty((0, ACTION_DIM), dtype=np.uint8)
        actions = np.empty((0,), dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, masks=masks, actions=actions)
    return ShardResult(path=output_path, num_samples=len(actions), num_games=num_games)


def _worker(args: tuple[int, int, int, Path]) -> ShardResult:
    games, seed, target, output_path = args
    return _simulate_and_save(games, seed=seed, target_score=target, output_path=output_path)


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

    tasks: list[tuple[int, int, int, Path]] = []
    for shard_idx in range(num_shards):
        games_in_shard = min(shard_size, num_games - shard_idx * shard_size)
        out_path = output_dir / f"shard_{shard_idx:05d}.npz"
        tasks.append((games_in_shard, base_rng.randint(0, 10**9), target_score, out_path))

    total_samples = 0
    start = time.perf_counter()

    if workers <= 1:
        with tqdm(total=num_games, desc="Spiele") as pbar:
            for task in tasks:
                result = _worker(task)
                total_samples += result.num_samples
                pbar.update(result.num_games)
    else:
        # mp.Pool mit chunksize=1, damit Worker einzelne Shards holen
        # (jeder Shard ist gross genug, dass die Worker selten leer laufen)
        with mp.Pool(processes=workers) as pool:
            with tqdm(total=num_games, desc="Spiele") as pbar:
                for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                    total_samples += result.num_samples
                    pbar.update(result.num_games)

    elapsed = time.perf_counter() - start
    print(
        f"\nFertig: {num_games} Partien, {total_samples:,} Samples in {num_shards} Shards. "
        f"Dauer {elapsed:.1f} s, {num_games / elapsed:.0f} Partien/s, "
        f"{total_samples / elapsed:.0f} Samples/s."
    )


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
