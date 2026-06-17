"""Parallel-Self-Play: persistent worker pool mit weight-sync via Queue.

[INAKTIV] Teil des archivierten PPO/RL-Experiments -- siehe
training/rl/train_rl.py. Nicht in der aktuellen MCTS-BC-Pipeline; bleibt
getestet erhalten.

Architektur:
    Hauptprozess (GPU, PPO-Update)              Worker 0..N-1 (CPU, Self-Play)
    -----------------------------------          -----------------------------------
    1. Initialisierung:
       - Modell laden (auf GPU)
       - ParallelSelfPlayPool starten ----------> jeder Worker:
                                                    - CUDA_VISIBLE_DEVICES=-1
                                                    - Initial-Modell laden (Datei)
                                                    - Loop: warte auf Job

    2. Pro Iteration:
       - weights = model.get_weights()
       - pool.collect(weights, num_games, ...)
       - Job in Queue          --------------->  Worker bekommt Job:
                                                    - model.set_weights(weights)
                                                    - collect_trajectories(...)
                                                    - Trajektorien zurueck
       <----- Trajektorien ----- result_queue
       - GAE + PPO-Update
       - weiter mit naechster Iteration

    3. Shutdown:
       - sentinel (None) in Job-Queue   ------>  Worker beendet sich sauber

Vorteile gegenueber dem sequentiellen Self-Play:
- 16 Spiele in 16 Workers parallel statt sequentiell im Hauptprozess
- Worker-Lebenszeit ueber alle Iterationen -> kein Spawn-Overhead je Iter
- Gewichte via Queue (NumPy-pickle) statt Disk-IO

Limitationen:
- Workers nutzen CPU-Inference (CUDA pro Worker -> Konflikte). RTX 3060 wird
  nur fuer PPO-Update genutzt. Das ist ok, weil PPO-Update mit Batch 512+
  ohnehin der einzige GPU-intensive Schritt ist.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


# Sentinel-Wert fuer "Worker bitte beenden"
_SHUTDOWN = object()


def _worker_loop(
    job_queue,
    result_queue,
    initial_model_path: str,
    worker_id: int,
) -> None:
    """Worker-Hauptschleife. Laeuft in einem separaten Process.

    Args:
        job_queue: Queue, ueber die der Hauptprozess Jobs sendet.
            Jeder Job ist ein Tupel
                (weights, num_games, seed, target_score, heuristic_mix_rate)
            oder das Sentinel-String "__SHUTDOWN__".
        result_queue: Queue, ueber die der Worker Trajektorien zurueckschickt.
            Result-Format: (worker_id, list[Trajectory]).
        initial_model_path: Pfad zu .keras-Datei, die der Worker initial laedt
            (fuer die Modell-Architektur). Spaeter werden nur noch Gewichte
            ueber die Queue synchronisiert.
        worker_id: 0..N-1, nur fuer Debug-Output.
    """
    # CUDA komplett ausschalten BEVOR TF importiert wird. Bei N parallelen
    # Workers wuerden sonst N CUDA-Contexte gleichzeitig VRAM allokieren ->
    # entweder OOM oder Sharing-Konflikte.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # TF/Keras erst nach dem CUDA-Disable importieren
    from tensorflow import keras
    from training.model import MaskBias  # noqa: F401 -- Custom-Layer registrieren
    from training.rl.selfplay import collect_trajectories

    # Initial-Modell laden (gleiche Architektur wie im Hauptprozess)
    model = keras.models.load_model(initial_model_path)

    while True:
        job = job_queue.get()
        if job == "__SHUTDOWN__":
            break

        weights, num_games, seed, target_score, mix_rate = job
        if weights is not None:
            model.set_weights(weights)

        trajs = collect_trajectories(
            model=model,
            num_games=num_games,
            target_score=target_score,
            seed=seed,
            heuristic_mix_rate=mix_rate,
        )
        result_queue.put((worker_id, trajs))


class ParallelSelfPlayPool:
    """Persistent Worker-Pool fuer Self-Play. Workers leben ueber alle
    Iterationen; nur die Modell-Gewichte werden pro Iter ueber die Queue
    synchronisiert.

    Verwendung:
        pool = ParallelSelfPlayPool(
            num_workers=16,
            initial_model_path="models/v5/best.keras",
        )
        try:
            for iteration in range(num_iterations):
                trajs = pool.collect(
                    weights=model.get_weights(),
                    num_games_total=64,
                    seed=42 + iteration,
                    target_score=1000,
                    heuristic_mix_rate=0.3,
                )
                # ... GAE + PPO-Update ...
        finally:
            pool.close()
    """

    def __init__(
        self,
        num_workers: int,
        initial_model_path: Path | str,
    ):
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.num_workers = num_workers
        self._job_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._workers: list[mp.Process] = []

        initial_model_str = str(initial_model_path)
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    self._job_queue,
                    self._result_queue,
                    initial_model_str,
                    i,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def collect(
        self,
        weights: list[np.ndarray],
        num_games_total: int,
        seed: int,
        target_score: int,
        heuristic_mix_rate: float,
    ) -> list:
        """Verteilt num_games_total auf die Workers und sammelt Trajektorien.

        Args:
            weights: Liste von numpy-Arrays (`model.get_weights()`).
            num_games_total: Gesamt-Spiele dieser Iteration; wird gleichmaessig
                auf die Workers aufgeteilt.
            seed: Master-Seed; pro Worker wird ein abgeleiteter Sub-Seed benutzt.
            target_score: Punkteziel pro Partie.
            heuristic_mix_rate: 0..1, Anti-Drift-Anker fuer Self-Play.

        Returns:
            Liste aller Trajektorien (Reihenfolge nicht deterministisch).
        """
        # Spiele aufteilen
        base = num_games_total // self.num_workers
        remainder = num_games_total % self.num_workers
        batches = [base + (1 if i < remainder else 0) for i in range(self.num_workers)]

        # Jobs verteilen
        num_active = 0
        for i, batch in enumerate(batches):
            if batch == 0:
                continue
            sub_seed = seed + i * 10_000_003
            self._job_queue.put(
                (weights, batch, sub_seed, target_score, heuristic_mix_rate)
            )
            num_active += 1

        # Ergebnisse einsammeln (Reihenfolge unwichtig)
        all_trajs: list = []
        for _ in range(num_active):
            _worker_id, trajs = self._result_queue.get()
            all_trajs.extend(trajs)
        return all_trajs

    def close(self) -> None:
        """Schickt Shutdown-Signal an alle Workers und wartet auf Beendigung."""
        for _ in self._workers:
            self._job_queue.put("__SHUTDOWN__")
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers = []

    def __enter__(self) -> "ParallelSelfPlayPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
