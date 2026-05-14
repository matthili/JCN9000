"""Trainings-Loop: lädt Shards, trainiert das Mask-Aware MLP, speichert Checkpoints.

Aufruf:
    python -m training.train --data data/heuristic_50k --output models/v1
    python -m training.train --data data/heuristic_50k --epochs 30 --batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from training.dataset import load_split
from training.model import build_model


def configure_gpu_memory() -> None:
    """Aktiviert "Memory Growth" auf allen erkannten GPUs.

    Standardmaessig versucht TF, fast den gesamten GPU-Speicher beim ersten Tensor
    zu allokieren. Bei grossen Datensaetzen scheitert das. Mit Memory-Growth wird
    Speicher inkrementell genommen.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Keine GPU erkannt - Training laeuft auf CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"  Memory-Growth-Setup fuer {gpu.name} gescheitert: {e}")
    print(f"GPU-Training auf {len(gpus)} GPU(s) mit Memory-Growth aktiviert.")


def _make_dataset(
    X: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Baut ein tf.data.Dataset, das den GPU-Speicher nicht ueberlaeuft.

    KRITISCH: from_tensor_slices muss innerhalb von tf.device('/CPU:0') aufgerufen
    werden. Sonst versucht TF, die kompletten Arrays als Konstanten auf die GPU
    zu schieben (mehrere GB), was bei groesseren Datasets immer scheitert
    ("Dst tensor is not initialized").

    Mit dem CPU-Scope bleiben die Daten im CPU-RAM; nur die Batches werden zur
    GPU prefetched.
    """
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            ({"state": X, "mask": masks}, actions)
        )
        if shuffle:
            # Shuffle-Buffer 100k: gute Mischung ohne RAM-Sprengung
            ds = ds.shuffle(
                buffer_size=100_000, seed=seed, reshuffle_each_iteration=True
            )
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 1024,
    val_fraction: float = 0.1,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> None:
    configure_gpu_memory()
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(data_dir, val_fraction=val_fraction, seed=seed)
    print(
        f"\nModell wird aufgebaut (Input {split.train.X.shape[1]}, "
        f"Aktionen {split.train.masks.shape[1]})…"
    )
    model = build_model(
        input_dim=split.train.X.shape[1],
        action_dim=split.train.masks.shape[1],
        learning_rate=learning_rate,
    )
    model.summary(print_fn=lambda s: print("  " + s))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"\nTraining startet: epochs={epochs}, batch_size={batch_size}")
    train_ds = _make_dataset(
        split.train.X, split.train.masks, split.train.actions,
        batch_size=batch_size, shuffle=True, seed=seed,
    )
    val_ds = _make_dataset(
        split.val.X, split.val.masks, split.val.actions,
        batch_size=batch_size, shuffle=False, seed=seed,
    )
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.perf_counter() - start
    print(f"\nTraining fertig in {elapsed / 60:.1f} min.")

    # Final-Checkpoint (überschreibt evtl. best, wenn besser)
    final_path = output_dir / "final.keras"
    model.save(final_path)
    print(f"Final-Modell: {final_path}")

    # Trainings-Historie als JSON
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    (output_dir / "history.json").write_text(json.dumps(hist_dict, indent=2))
    best_val_acc = max(hist_dict.get("val_accuracy", [0.0]))
    print(f"Beste val_accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/heuristic_50k")
    parser.add_argument("--output", type=str, default="models/v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
