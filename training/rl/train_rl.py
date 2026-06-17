"""RL-Hauptschleife: Self-Play -> Trajektorien-Sammlung -> PPO-Update -> Snapshot.

Aufruf:
    # Erster Lauf -- startet aus dem BC-Warmstart-Modell (Default: v5)
    python -m training.rl.train_rl --warm-start models/v5/best.keras \
        --iterations 50 --games-per-iter 16 --target 500 \
        --output models/rl_v2

    # Spaeter weitermachen -- haengt automatisch an den bestehenden Stand an
    # Das output_dir existiert schon mit state.json drin -> Resume passiert automatisch
    python -m training.rl.train_rl --iterations 50 --games-per-iter 16 \
        --target 500 --output models/rl_v2

Resume:
  Wenn `<output>/state.json` existiert, wird der bestehende Stand aus
  `<output>/final.keras` geladen, der Iterations-Counter laeuft weiter, und
  log.csv wird angefuegt. `--warm-start` wird in diesem Fall IGNORIERT
  (das laufende Training hat Vorrang).

Ein Run produziert:
    output_dir/
        ├── iter_00000.keras   Snapshot nach Iteration 0
        ├── iter_00010.keras   ...alle 10 Iterationen
        ├── final.keras        Aktuellster Stand (wird bei Resume wieder geladen)
        ├── state.json         Iterations-Counter fuer Resume
        ├── log.csv            Loss/Entropy/KL pro Iteration (wird bei Resume erweitert)
        └── elo.json           Elo-Verlauf gegen frueheren Snapshots (kommt spaeter)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from training.model import MaskBias, build_model  # noqa: F401 (MaskBias-Registrierung)
from training.rl.batched_selfplay import collect_trajectories_batched
from training.rl.parallel_selfplay import ParallelSelfPlayPool
from training.rl.ppo import ppo_train_step
from training.rl.selfplay import collect_trajectories
from training.rl.trajectory import compute_gae, stack_trajectories


STATE_FILENAME = "state.json"
LOG_FILENAME = "log.csv"


def _load_state(output_dir: Path) -> dict | None:
    state_path = output_dir / STATE_FILENAME
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(output_dir: Path, state: dict) -> None:
    state_path = output_dir / STATE_FILENAME
    state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def configure_gpu_memory() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Keine GPU erkannt - RL-Training laeuft auf CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"RL-Training auf {len(gpus)} GPU(s) mit Memory-Growth aktiviert.")


def _load_or_build_model(model_path: Path | None) -> tf.keras.Model:
    if model_path is not None and model_path.exists():
        print(f"Lade Modell: {model_path}")
        return keras.models.load_model(str(model_path))
    print("Kein Modell-Pfad gegeben oder Datei existiert nicht -- baue frisches Modell.")
    return build_model(with_value_head=True)


def _save_snapshot(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def run(
    output_dir: Path,
    warm_start: Path | None,
    iterations: int,
    games_per_iter: int,
    target_score: int,
    epochs_per_update: int,
    batch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    learning_rate: float,
    gamma: float,
    lam: float,
    snapshot_every: int,
    seed: int,
    verbose: int,
    heuristic_mix_rate: float = 0.3,
    workers: int = 1,
    inference_mode: str = "sequential",
    inference_batch_size: int = 64,
) -> None:
    configure_gpu_memory()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume-Logik
    existing_state = _load_state(output_dir)
    if existing_state is not None:
        last_iter = int(existing_state.get("last_completed_iter", -1))
        total_partien_before = int(existing_state.get("total_partien", 0))
        start_iter = last_iter + 1
        resume_model = output_dir / "final.keras"
        if not resume_model.exists():
            raise FileNotFoundError(
                f"state.json gefunden in {output_dir}, aber final.keras fehlt. "
                f"Kann nicht resumen."
            )
        print(
            f"\n=== Resume erkannt: starte bei Iteration {start_iter} "
            f"(bisher {total_partien_before} Self-Play-Partien) ===\n"
        )
        if warm_start is not None:
            print(f"  Hinweis: --warm-start {warm_start} wird ignoriert; "
                  f"resume aus {resume_model}.")
        model = _load_or_build_model(resume_model)
    else:
        start_iter = 0
        total_partien_before = 0
        model = _load_or_build_model(warm_start)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Self-Play-Modus auswerten:
    #   - "sequential": klassisch sequentiell im Hauptprozess (langsam, einfach)
    #   - "cpu-workers": ParallelSelfPlayPool, Workers mit CPU-Inferenz
    #   - "batched-gpu": batched_selfplay.collect_trajectories_batched,
    #     N parallele Threads + Server mit GPU-Batch-Inferenz
    selfplay_pool: ParallelSelfPlayPool | None = None
    if inference_mode == "cpu-workers":
        if workers <= 1:
            raise ValueError(
                "inference_mode='cpu-workers' braucht workers > 1."
            )
        worker_model_path = output_dir / "final.keras"
        if not worker_model_path.exists():
            _save_snapshot(model, worker_model_path)
        print(
            f"\nInitialisiere ParallelSelfPlayPool ({workers} Workers, CPU-only)..."
        )
        selfplay_pool = ParallelSelfPlayPool(
            num_workers=workers,
            initial_model_path=worker_model_path,
        )
    elif inference_mode == "batched-gpu":
        print(
            f"\nSelf-Play-Modus: batched-gpu "
            f"(threading + InferenceServer, Batch <= {inference_batch_size})"
        )
    elif inference_mode == "sequential":
        pass  # Default-Pfad, nichts zu tun
    else:
        raise ValueError(
            f"Unbekannter inference_mode: {inference_mode!r}. "
            f"Erlaubt: 'sequential', 'cpu-workers', 'batched-gpu'."
        )

    log_path = output_dir / LOG_FILENAME
    log_exists = log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow([
            "iter", "games", "transitions", "policy_loss", "value_loss",
            "entropy", "approx_kl", "grad_norm", "mean_reward", "sec",
            "timestamp",
        ])

    end_iter = start_iter + iterations
    for it in range(start_iter, end_iter):
        t0 = time.perf_counter()
        if verbose >= 1:
            print(f"\n=== RL-Iteration {it + 1} (in dieser Session {it - start_iter + 1}/{iterations}) ===")

        # 1) Self-Play (ggf. mit Heuristik-Anker als Anti-Drift)
        if inference_mode == "cpu-workers" and selfplay_pool is not None:
            trajs = selfplay_pool.collect(
                weights=model.get_weights(),
                num_games_total=games_per_iter,
                seed=seed + it,
                target_score=target_score,
                heuristic_mix_rate=heuristic_mix_rate,
            )
        elif inference_mode == "batched-gpu":
            trajs = collect_trajectories_batched(
                model=model,
                num_games=games_per_iter,
                target_score=target_score,
                seed=seed + it,
                heuristic_mix_rate=heuristic_mix_rate,
                max_batch_size=inference_batch_size,
            )
        else:
            # sequential (Default)
            trajs = collect_trajectories(
                model=model,
                num_games=games_per_iter,
                target_score=target_score,
                seed=seed + it,
                heuristic_mix_rate=heuristic_mix_rate,
            )
        sp_secs = time.perf_counter() - t0
        n_transitions = sum(len(t) for t in trajs)
        mean_reward = float(np.mean([
            tr.reward for traj in trajs for tr in traj.transitions
        ])) if n_transitions else 0.0
        if verbose >= 1:
            print(
                f"  Self-Play: {len(trajs)} Trajektorien, {n_transitions} Transitions "
                f"in {sp_secs:.1f}s (mean_reward={mean_reward:+.3f})"
            )

        # 2) GAE
        for traj in trajs:
            compute_gae(traj, gamma=gamma, lam=lam)
        batch = stack_trajectories(trajs, normalize=True)
        if len(batch["actions"]) == 0:
            print("  Keine Transitions -- ueberspringe Update.")
            continue

        # 3) PPO-Update (mehrere Epochen ueber denselben Batch)
        upd_t0 = time.perf_counter()
        last_metrics = {}
        n = len(batch["actions"])
        for ep in range(epochs_per_update):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                mb_idx = idx[start : start + batch_size]
                mb = {k: v[mb_idx] for k, v in batch.items()}
                metrics = ppo_train_step(
                    model=model,
                    optimizer=optimizer,
                    batch=mb,
                    clip_ratio=clip_ratio,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
                last_metrics = {k: float(v.numpy()) for k, v in metrics.items()}
        upd_secs = time.perf_counter() - upd_t0
        total_secs = time.perf_counter() - t0

        if verbose >= 1:
            print(
                f"  PPO-Update: {epochs_per_update} Epochen x {n} Samples "
                f"in {upd_secs:.1f}s "
                f"(policy_loss={last_metrics.get('policy_loss', 0):+.4f}, "
                f"value_loss={last_metrics.get('value_loss', 0):.4f}, "
                f"entropy={last_metrics.get('entropy', 0):.3f}, "
                f"kl={last_metrics.get('approx_kl', 0):+.4f})"
            )

        writer.writerow([
            it,
            games_per_iter,
            n_transitions,
            last_metrics.get("policy_loss", 0),
            last_metrics.get("value_loss", 0),
            last_metrics.get("entropy", 0),
            last_metrics.get("approx_kl", 0),
            last_metrics.get("grad_norm", 0),
            mean_reward,
            total_secs,
            datetime.now(timezone.utc).isoformat(),
        ])
        log_file.flush()

        # 4) Snapshot + state.json + final.keras (immer letzten Stand persistieren)
        _save_snapshot(model, output_dir / "final.keras")
        if (it % snapshot_every == 0) or (it == end_iter - 1):
            snap = output_dir / f"iter_{it:05d}.keras"
            _save_snapshot(model, snap)
            if verbose >= 1:
                print(f"  Snapshot: {snap}")
        _save_state(output_dir, {
            "last_completed_iter": it,
            "total_partien": total_partien_before + (it - start_iter + 1) * games_per_iter,
            "last_session_timestamp": datetime.now(timezone.utc).isoformat(),
        })

    log_file.close()

    # Pool sauber beenden (sentinel + join)
    if selfplay_pool is not None:
        print("\nBeende Self-Play-Worker-Pool...")
        selfplay_pool.close()

    print(
        f"\nRL-Training-Session fertig. "
        f"Letzte Iteration: {end_iter - 1}. "
        f"Stand: {output_dir / 'final.keras'}"
    )
    print("  Weitermachen: nochmal denselben Aufruf -- Resume passiert automatisch.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/rl_v2"))
    parser.add_argument("--warm-start", type=Path, default=Path("models/v5/best.keras"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--target", type=int, default=500, help="Punkteziel pro Partie")
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--heuristic-mix-rate", type=float, default=0.3,
        help="Anteil Partien (0..1), in denen das RL-Team gegen 2 Heuristik-Gegner "
             "spielt. Als Anti-Drift-Anker bei Self-Play. Default 0.3.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Anzahl paralleler Self-Play-Worker (nur fuer --inference-mode "
            "cpu-workers). Default 1 = sequentielles Self-Play im Hauptprozess."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=["sequential", "cpu-workers", "batched-gpu"],
        default="sequential",
        help=(
            "Wie die Self-Play-Inferenzen gemacht werden:\n"
            "  sequential   = klassisch, eine Inferenz pro Stich im Hauptprozess\n"
            "  cpu-workers  = N Worker-Prozesse, CPU-only, GIL-frei (Variante C)\n"
            "  batched-gpu  = ein Process, N Game-Threads + GPU-Inferenz-Server,\n"
            "                 echte Batch-Inferenz auf der GPU (Variante D)"
        ),
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=64,
        help=(
            "Nur fuer --inference-mode batched-gpu: maximale Batch-Groesse, "
            "die der InferenceServer sammelt, bevor er das Modell aufruft. "
            "Default 64. Sinnvoller Wert: ~num_games oder ein Bisschen drunter."
        ),
    )
    args = parser.parse_args()

    run(
        output_dir=args.output,
        warm_start=args.warm_start,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        target_score=args.target,
        epochs_per_update=args.epochs_per_update,
        batch_size=args.batch_size,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lam=args.lam,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        verbose=args.verbose,
        heuristic_mix_rate=args.heuristic_mix_rate,
        workers=args.workers,
        inference_mode=args.inference_mode,
        inference_batch_size=args.inference_batch_size,
    )


if __name__ == "__main__":
    main()
