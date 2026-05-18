"""Multiprocessing-Variante der MCTS-Datengen.

Architektur (anders als generate_mcts_data.py):
- Der Hauptprozess teilt die Varianten auf N Worker-Prozesse auf.
- Jeder Worker-Prozess:
    * laedt **selbst** TensorFlow und das Modell auf der GPU
    * faehrt einen eigenen InferenceServer-Thread fuer Batch-Inferenz
    * laeuft `parallel-threads-per-worker` Game-Threads parallel
- Multiprocessing umgeht das GIL (Global Interpreter Lock = Pythons Sperre,
  die im selben Prozess echte parallele Threads verhindert). Damit
  konkurrieren die Worker NICHT um CPU-Zeit, jeder hat seine eigene Python-VM.
- Alle Worker teilen sich die GPU: jeder allokiert per Memory-Growth so viel
  VRAM wie er braucht (typisch 1-1.5 GB pro Worker). Auf einer RTX 3060 mit
  12 GB sind 4-6 Worker problemlos.
- Pro Worker schickt der InferenceServer-Thread seine Batches direkt zur GPU.
  Die GPU bekommt damit gleichzeitig mehrere Inferenz-Stroeme; das CUDA-
  Scheduling staffelt sie sauber.

Aufruf:
    python -u -m training.data.generate_mcts_data_mp \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 72 \\
        --rollouts-per-card 30 \\
        --target 1000 \\
        --workers 4 \\
        --parallel-threads-per-worker 8 \\
        --inference-batch-size 1024 \\
        --lookahead-mode full-round-vec \\
        --output data/mcts/phase1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from training.data.generate_mcts_data import (
    ALL_VARIANTS,
    VariantSpec,
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
    # TF-Logs leiser
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Unbuffered stdout/stderr -- damit Worker-print-Ausgaben sofort im
    # Hauptterminal sichtbar werden statt im Pipe-Buffer zu haengen.
    # `python -u` wirkt nur fuer den Hauptprozess; Worker brauchen das selbst.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    import tensorflow as tf
    from tensorflow import keras

    # Memory-Growth: Worker allokieren GPU-Speicher inkrementell, sodass
    # alle Worker sich die GPU teilen koennen.
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    # Modell + Custom-Layer-Registrierung
    from training.model import MaskBias  # noqa: F401
    print(f"[Worker {worker_id}] Lade Modell auf GPU: {args_dict['warm_start']}")
    model = keras.models.load_model(args_dict["warm_start"])

    # Inferenz-Server (eigen pro Worker)
    from training.rl.batched_selfplay import InferenceServer
    server = InferenceServer(
        model=model,
        max_batch_size=args_dict["inference_batch_size"],
    )

    # Importe nach TF-Setup
    from training.data.generate_mcts_data import generate_for_variant

    print(
        f"[Worker {worker_id}] Bereit. {len(variant_specs)} Varianten zugeteilt: "
        f"{[vs.label for vs in variant_specs]}"
    )

    output_dir = Path(args_dict["output"])
    seed_base = args_dict["seed"] + worker_id * 1_000_003

    try:
        t0 = time.perf_counter()
        for vs in variant_specs:
            print(f"\n[Worker {worker_id}] === Variante {vs.label} ===")
            generate_for_variant(
                output_dir=output_dir,
                variant_spec=vs,
                games_per_variant=args_dict["games_per_variant"],
                rollouts_per_card=args_dict["rollouts_per_card"],
                target_score=args_dict["target"],
                inference_server=server,
                parallel_threads=args_dict["parallel_threads_per_worker"],
                seed=seed_base + hash(vs.label) % 10_000,
                lookahead_mode=args_dict["lookahead_mode"],
            )
        elapsed = time.perf_counter() - t0
        print(
            f"\n[Worker {worker_id}] Alle {len(variant_specs)} Varianten fertig "
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
        help="Wieviele Rollouts pro legaler Karte (mehr = stabilerer Lehrer, mehr Compute).",
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/mcts/phase1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4,
        help=(
            "Anzahl Worker-Prozesse. Jeder laedt das Modell auf die GPU. "
            "Bei 12 GB VRAM und unserem 1.25M-Modell sind 4-6 Worker sicher. "
            "Default 4."
        ),
    )
    parser.add_argument(
        "--parallel-threads-per-worker", type=int, default=8,
        help=(
            "Wieviele Game-Threads pro Worker (innerhalb des Prozesses). "
            "Default 8. Mit --workers 4 ergibt das insgesamt 32 parallel "
            "laufende Spiele -- ohne GIL-Konflikt zwischen den Workern."
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
        "--lookahead-mode",
        choices=["single-trick", "full-round-vec"],
        default="full-round-vec",
    )
    args = parser.parse_args()

    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants}.")
            return

    # Varianten auf Worker aufteilen (Round-Robin)
    n_workers = min(args.workers, len(selected))
    chunks: list[list[VariantSpec]] = [[] for _ in range(n_workers)]
    for i, vs in enumerate(selected):
        chunks[i % n_workers].append(vs)

    print(
        f"Multiprocessing-MCTS-Datengen:\n"
        f"  - {n_workers} Worker-Prozesse\n"
        f"  - je {args.parallel_threads_per_worker} Game-Threads pro Worker\n"
        f"  - {len(selected)} Varianten gesamt, aufgeteilt:\n"
    )
    for i, c in enumerate(chunks):
        print(f"    Worker {i}: {[vs.label for vs in c]}")

    # Spawn-Context: TF ist nicht fork-safe, deshalb fresh Python pro Worker.
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    t0 = time.perf_counter()

    for i in range(n_workers):
        if not chunks[i]:
            continue
        p = ctx.Process(
            target=_worker_process,
            args=(i, chunks[i], vars(args)),
            name=f"MCTSWorker-{i}",
        )
        p.start()
        processes.append(p)

    # Auf alle Worker warten
    for p in processes:
        p.join()

    elapsed = time.perf_counter() - t0
    print(f"\n=== Alle Worker fertig in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()
