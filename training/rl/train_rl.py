"""RL-Hauptschleife: Self-Play -> Trajektorien-Sammlung -> PPO-Update -> Snapshot.

Aufruf:
    # Schneller Smoke-Test (1 Iteration, 4 Partien, kein vollwertiges Training)
    python -m training.rl.train_rl --warm-start models/v3/best.keras \
        --iterations 1 --games-per-iter 4 --target 200

    # Echter Run (Iteration heisst hier: ein Sammel/Update-Zyklus)
    python -m training.rl.train_rl --warm-start models/v3/best.keras \
        --iterations 100 --games-per-iter 64 --target 500 \
        --output models/rl_v1

Ein Run produziert:
    output_dir/
        ├── iter_00000.keras   Snapshot nach Iteration 0
        ├── iter_00010.keras   ...alle 10 Iterationen
        ├── final.keras        Letzter Stand
        ├── log.csv            Loss/Entropy/KL pro Iteration
        └── elo.json           Elo-Verlauf gegen frueheren Snapshots
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from training.encoder import ACTION_DIM, INPUT_DIM
from training.model import MaskBias, build_model  # noqa: F401 (MaskBias-Registrierung)
from training.rl.ppo import ppo_train_step
from training.rl.selfplay import collect_trajectories
from training.rl.trajectory import compute_gae, stack_trajectories


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


def _load_or_build_model(warm_start: Path | None) -> tf.keras.Model:
    if warm_start is not None and warm_start.exists():
        print(f"Lade Warmstart-Modell: {warm_start}")
        return keras.models.load_model(str(warm_start))
    print("Kein Warmstart -- baue frisches Modell.")
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
) -> None:
    configure_gpu_memory()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_or_build_model(warm_start)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    log_path = output_dir / "log.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow([
        "iter", "games", "transitions", "policy_loss", "value_loss",
        "entropy", "approx_kl", "grad_norm", "mean_reward", "sec",
    ])

    for it in range(iterations):
        t0 = time.perf_counter()
        if verbose >= 1:
            print(f"\n=== RL-Iteration {it + 1}/{iterations} ===")

        # 1) Self-Play
        trajs = collect_trajectories(
            model=model,
            num_games=games_per_iter,
            target_score=target_score,
            seed=seed + it,
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
        ])
        log_file.flush()

        # 4) Snapshot
        if (it % snapshot_every == 0) or (it == iterations - 1):
            snap = output_dir / f"iter_{it:05d}.keras"
            _save_snapshot(model, snap)
            if verbose >= 1:
                print(f"  Snapshot: {snap}")

    # Final-Modell
    _save_snapshot(model, output_dir / "final.keras")
    log_file.close()
    print(f"\nRL-Training fertig. Final-Modell: {output_dir / 'final.keras'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/rl_v1"))
    parser.add_argument("--warm-start", type=Path, default=Path("models/v3/best.keras"))
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
    )


if __name__ == "__main__":
    main()
