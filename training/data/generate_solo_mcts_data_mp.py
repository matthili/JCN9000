"""Multiprocessing-Variante der Solo-MCTS-Datengen mit Chunk-Queue.

Analog zu `generate_mcts_data_mp.py`, aber fuer Solo-Jass:
- play_solo_jass statt play_kreuz_jass
- Solo-Reward (own - max(others)) / 200
- Variable Target-Score-Verteilung pro Partie
- Solo-Heuristik als Fallback

Architektur:
- N Worker-Prozesse, jeder mit eigenem TF-Modell auf GPU
- Geteilte multiprocessing.Manager.Queue mit Chunk-Tasks
- Worker pullt Chunks dynamisch -> sehr gleichmaessige GPU-Auslastung bis zum Ende

Aufruf (empfohlen mit 8 Workern x 32 Threads, siehe Throughput-Tests):
    python -u -m training.data.generate_solo_mcts_data_mp \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 500 \\
        --games-per-chunk 50 \\
        --rollouts-per-card 30 \\
        --target-distribution "500:0.5,1000:0.5" \\
        --workers 8 \\
        --parallel-threads-per-worker 32 \\
        --inference-batch-size 1024 \\
        --output data/solo_mcts/phase1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from training.data.generate_solo_mcts_data import (
    ALL_VARIANTS,
    VariantSpec,
    _parse_target_distribution,
)


def _worker_process(
    worker_id: int,
    task_queue,
    args_dict: dict,
) -> None:
    """Wird in einem separaten Prozess gestartet (spawn-Context).

    Laedt eigenes TensorFlow + Modell, faehrt einen eigenen InferenceServer
    und arbeitet Chunks aus der geteilten Queue ab.
    """
    import os
    import sys

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    from training.model import MaskBias  # noqa: F401

    print(f"[SoloWorker {worker_id}] Lade Modell auf GPU: {args_dict['warm_start']}")
    model = keras.models.load_model(args_dict["warm_start"])

    from training.rl.batched_selfplay import InferenceServer

    server = InferenceServer(
        model=model,
        max_batch_size=args_dict["inference_batch_size"],
    )

    from training.data.generate_solo_mcts_data import generate_for_variant

    target_distribution = _parse_target_distribution(args_dict["target_distribution"])
    output_dir = Path(args_dict["output"])
    skip_existing = args_dict.get("skip_existing", False)

    chunks_done = 0
    t0 = time.perf_counter()

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            variant_spec, chunk_idx, games_in_chunk, seed = task
            print(
                f"\n[SoloWorker {worker_id}] === {variant_spec.label} chunk {chunk_idx} "
                f"({games_in_chunk} Spiele) ==="
            )
            generate_for_variant(
                output_dir=output_dir,
                variant_spec=variant_spec,
                games_per_variant=games_in_chunk,
                rollouts_per_card=args_dict["rollouts_per_card"],
                target_distribution=target_distribution,
                inference_server=server,
                parallel_threads=args_dict["parallel_threads_per_worker"],
                seed=seed,
                chunk_idx=chunk_idx,
                skip_existing=skip_existing,
            )
            chunks_done += 1

        elapsed = time.perf_counter() - t0
        print(
            f"\n[SoloWorker {worker_id}] {chunks_done} Chunks abgearbeitet in "
            f"{elapsed / 60:.1f} min."
        )
    finally:
        server.shutdown()


def _build_chunk_tasks(
    selected: list[VariantSpec],
    games_per_variant: int,
    games_per_chunk: int,
    base_seed: int,
) -> list[tuple]:
    """Baut die Chunk-Tasks (variant, chunk_idx, games_in_chunk, seed)."""
    tasks = []
    for vs in selected:
        n_chunks = (games_per_variant + games_per_chunk - 1) // games_per_chunk
        for chunk_idx in range(n_chunks):
            games_in_chunk = min(
                games_per_chunk,
                games_per_variant - chunk_idx * games_per_chunk,
            )
            seed = base_seed + hash(vs.label) % 10_000 + chunk_idx * 17
            tasks.append((vs, chunk_idx, games_in_chunk, seed))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm-start", type=str, required=True,
        help="Pfad zum NN-Modell (z.B. models/v5/best.keras).",
    )
    parser.add_argument(
        "--games-per-variant", type=int, default=72,
        help="Gesamtanzahl Spiele pro Variante.",
    )
    parser.add_argument(
        "--games-per-chunk", type=int, default=50,
        help=(
            "Spiele pro Chunk. Kleiner = mehr Tasks, besseres Load-Balancing, "
            "aber mehr Shard-Dateien. Default 50."
        ),
    )
    parser.add_argument(
        "--rollouts-per-card", type=int, default=30,
        help="Wieviele Rollouts pro legaler Karte.",
    )
    parser.add_argument(
        "--target-distribution",
        type=str,
        default="500:0.5,1000:0.5",
        help="Verteilung der Spielziele pro Partie (Format 'TARGET:PROB,...').",
    )
    parser.add_argument("--output", type=str, default="data/solo_mcts/phase1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Anzahl Worker-Prozesse. Empfehlung auf RTX 3060 12GB: 8.",
    )
    parser.add_argument(
        "--parallel-threads-per-worker", type=int, default=32,
        help=(
            "Wieviele Game-Threads pro Worker. Empfehlung aus dem Throughput-Test "
            "fuer RTX 3060: 32."
        ),
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=1024,
        help="Max. Batch-Groesse im InferenceServer pro Worker.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Nur bestimmte Varianten (Label). Default: alle 12.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Wenn gesetzt: Chunks, deren Shard-Datei schon existiert, werden "
            "uebersprungen (zum Wiederaufnehmen nach Crash)."
        ),
    )
    args = parser.parse_args()

    # Vorab-Validierung der Distribution
    _parse_target_distribution(args.target_distribution)

    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants}.")
            return

    tasks = _build_chunk_tasks(
        selected=selected,
        games_per_variant=args.games_per_variant,
        games_per_chunk=args.games_per_chunk,
        base_seed=args.seed,
    )

    n_workers = min(args.workers, len(tasks))

    print(
        f"Solo-Multiprocessing-MCTS-Datengen (Chunk-Queue):\n"
        f"  - {n_workers} Worker-Prozesse\n"
        f"  - je {args.parallel_threads_per_worker} Game-Threads pro Worker\n"
        f"  - {len(selected)} Varianten gesamt, {args.games_per_variant} Spiele pro Variante\n"
        f"  - Chunk-Groesse: {args.games_per_chunk} Spiele\n"
        f"  - Gesamt: {len(tasks)} Chunks in der Queue\n"
    )

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()

    with ctx.Manager() as manager:
        task_queue = manager.Queue()
        for task in tasks:
            task_queue.put(task)
        for _ in range(n_workers):
            task_queue.put(None)

        processes: list[mp.Process] = []
        for i in range(n_workers):
            p = ctx.Process(
                target=_worker_process,
                args=(i, task_queue, vars(args)),
                name=f"SoloMCTSWorker-{i}",
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    elapsed = time.perf_counter() - t0
    print(f"\n=== Alle Solo-Worker fertig in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()
