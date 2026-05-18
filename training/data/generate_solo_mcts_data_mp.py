"""Multiprocessing-Variante der Solo-MCTS-Datengen.

Analog zu `generate_mcts_data_mp.py`, aber fuer Solo-Jass mit eigener
Reward-Berechnung. Jeder Worker laedt das Modell auf der GPU, betreibt
einen InferenceServer-Thread und generiert die ihm zugeteilten Varianten.

Aufruf (empfohlen mit 8 Workern x 32 Threads, siehe Throughput-Tests):
    python -u -m training.data.generate_solo_mcts_data_mp \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 500 \\
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
    variant_specs: list,
    args_dict: dict,
) -> None:
    """Wird in einem separaten Prozess gestartet (spawn-Context).

    Laedt eigenes TensorFlow + Modell, faehrt einen eigenen InferenceServer
    und generiert die ihm zugeteilten Varianten.
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

    print(
        f"[SoloWorker {worker_id}] Bereit. {len(variant_specs)} Varianten zugeteilt: "
        f"{[vs.label for vs in variant_specs]}"
    )

    output_dir = Path(args_dict["output"])
    seed_base = args_dict["seed"] + worker_id * 1_000_003

    try:
        t0 = time.perf_counter()
        for vs in variant_specs:
            print(f"\n[SoloWorker {worker_id}] === Variante {vs.label} ===")
            generate_for_variant(
                output_dir=output_dir,
                variant_spec=vs,
                games_per_variant=args_dict["games_per_variant"],
                rollouts_per_card=args_dict["rollouts_per_card"],
                target_distribution=target_distribution,
                inference_server=server,
                parallel_threads=args_dict["parallel_threads_per_worker"],
                seed=seed_base + hash(vs.label) % 10_000,
            )
        elapsed = time.perf_counter() - t0
        print(
            f"\n[SoloWorker {worker_id}] Alle {len(variant_specs)} Varianten fertig "
            f"in {elapsed / 60:.1f} min."
        )
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm-start", type=str, required=True,
        help="Pfad zum NN-Modell (z.B. models/v5/best.keras).",
    )
    parser.add_argument(
        "--games-per-variant", type=int, default=72,
        help="Wieviele Partien pro Variante.",
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
            "fuer RTX 3060: 32. Default 32."
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
    args = parser.parse_args()

    # Vorab-Validierung der Distribution, damit ein Fehler nicht erst im Worker auftaucht
    _parse_target_distribution(args.target_distribution)

    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants}.")
            return

    n_workers = min(args.workers, len(selected))
    chunks: list[list[VariantSpec]] = [[] for _ in range(n_workers)]
    for i, vs in enumerate(selected):
        chunks[i % n_workers].append(vs)

    print(
        f"Solo-Multiprocessing-MCTS-Datengen:\n"
        f"  - {n_workers} Worker-Prozesse\n"
        f"  - je {args.parallel_threads_per_worker} Game-Threads pro Worker\n"
        f"  - {len(selected)} Varianten gesamt, aufgeteilt:\n"
    )
    for i, c in enumerate(chunks):
        print(f"    SoloWorker {i}: {[vs.label for vs in c]}")

    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    t0 = time.perf_counter()

    for i in range(n_workers):
        if not chunks[i]:
            continue
        p = ctx.Process(
            target=_worker_process,
            args=(i, chunks[i], vars(args)),
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
