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
from tensorflow import keras

from training.dataset import load_split
from training.model import build_model


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
    start = time.perf_counter()
    history = model.fit(
        x={"state": split.train.X, "mask": split.train.masks},
        y=split.train.actions,
        validation_data=(
            {"state": split.val.X, "mask": split.val.masks},
            split.val.actions,
        ),
        epochs=epochs,
        batch_size=batch_size,
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
