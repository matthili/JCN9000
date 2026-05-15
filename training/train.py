"""Trainings-Loop: streamt Shards via tf.data, trainiert das Mask-Aware MLP, speichert Checkpoints.

Streaming-Architektur (seit v0.5.0):
  Statt alle Shards in den RAM zu laden (Peak ~80 GB bei 12 Varianten x 50k Runden),
  laeuft die Pipeline ueber `tf.data.Dataset.interleave`: pro Trainings-Schritt
  werden nur ~4 Shards parallel offen gehalten (~1-5 GB Peak). Damit ist auch
  ein voller v3-Datensatz mit 21 M Samples in 64 GB WSL2 trainierbar.

Aufruf:
    python -m training.train --data data/balanced_v3 --output models/v5
    python -m training.train --data data/balanced_v3 --epochs 40 --batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from training.dataset import split_shards, total_sample_count
from training.encoder import ACTION_DIM, INPUT_DIM
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


def _load_shard_arrays(path_bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy-Loader, wird via tf.py_function pro Shard aufgerufen.

    Akzeptiert sowohl Python-str als auch ein 0-dim tf-bytes-Tensor (so kommt
    es aus tf.py_function rein).
    """
    if hasattr(path_bytes, "numpy"):
        path_bytes = path_bytes.numpy()
    if isinstance(path_bytes, bytes):
        path_str = path_bytes.decode("utf-8")
    else:
        path_str = str(path_bytes)
    with np.load(path_str) as d:
        X = np.asarray(d["X"], dtype=np.float32)
        # masks im Shard sind uint8 -> wir casten zu float32, weil das Modell
        # mit float32-Maske rechnet (mask_bias-Layer).
        masks = np.asarray(d["masks"], dtype=np.float32)
        actions = np.asarray(d["actions"], dtype=np.int32)
        if "rewards" in d.files:
            rewards = np.asarray(d["rewards"], dtype=np.float32)
        else:
            rewards = np.zeros(len(actions), dtype=np.float32)
    return X, masks, actions, rewards


def _shard_to_sample_dataset(path_tensor: tf.Tensor) -> tf.data.Dataset:
    """tf.data-Wrapper: laedt einen Shard und entpackt ihn in einen Sample-Stream."""
    X, masks, actions, rewards = tf.py_function(
        _load_shard_arrays,
        [path_tensor],
        [tf.float32, tf.float32, tf.int32, tf.float32],
    )
    # py_function liefert Shape (None, ...) zurueck -- wir muessen die zweite
    # Dimension explizit setzen, sonst weiss tf.data nicht, was kommt.
    X.set_shape([None, INPUT_DIM])
    masks.set_shape([None, ACTION_DIM])
    actions.set_shape([None])
    rewards.set_shape([None])
    return tf.data.Dataset.from_tensor_slices(
        (
            {"state": X, "mask": masks},
            {"policy": actions, "value": rewards},
        )
    )


def _make_streaming_dataset(
    shard_paths: list,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int = 50_000,
    interleave_cycle: int = 4,
) -> tf.data.Dataset:
    """Baut einen tf.data-Stream ueber Shard-Pfade.

    Architektur:
      Shard-Pfade -> (optional Shuffle) -> interleave(load_shard, cycle=4)
        -> unbatch zu Samples -> Sample-Shuffle-Buffer -> batch -> prefetch.

    `interleave_cycle=4` haelt vier Shards gleichzeitig offen (RAM-Peak je nach
    Shard-Groesse ~1-5 GB), und der Sample-Shuffle-Buffer mischt deren Inhalt
    durcheinander. Das reicht in der Praxis: jedes Shard enthaelt bereits eine
    Mischung aus mehreren Partien.

    Kein `with tf.device("/CPU:0")` noetig: bei Streaming gibt es keine
    Konstanten-Promotion-Falle (kein from_tensor_slices auf das volle Array).
    """
    paths = [str(p) for p in shard_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        # Pfad-Reihenfolge je Epoche neu mischen -> kein Bias durch Datei-Index
        path_ds = path_ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )

    ds = path_ds.interleave(
        _shard_to_sample_dataset,
        cycle_length=interleave_cycle,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
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
    verbose: int = 2,
    hidden_units: tuple[int, ...] | None = None,
) -> None:
    configure_gpu_memory()
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_split = split_shards(data_dir, val_fraction=val_fraction, seed=seed)
    # Sample-Count nur lesen, um den Fortschritt anzuzeigen + steps_per_epoch
    # exakt zu rechnen. Schnell (nur Metadaten-Header pro Shard).
    print("Zaehle Samples pro Shard (Metadata-Sweep)…")
    train_samples = total_sample_count(shard_split.train_shards)
    val_samples = total_sample_count(shard_split.val_shards)
    print(
        f"  Train: {train_samples:>12,} Samples in {len(shard_split.train_shards)} Shards"
    )
    print(
        f"  Val:   {val_samples:>12,} Samples in {len(shard_split.val_shards)} Shards"
    )

    print(
        f"\nModell wird aufgebaut (Input {INPUT_DIM}, Aktionen {ACTION_DIM}, "
        f"Hidden {hidden_units or 'Default'})…"
    )
    from training.model import DEFAULT_HIDDEN
    used_hidden = tuple(hidden_units) if hidden_units else DEFAULT_HIDDEN
    model = build_model(
        input_dim=INPUT_DIM,
        action_dim=ACTION_DIM,
        hidden_units=used_hidden,
        learning_rate=learning_rate,
    )
    model.summary(print_fn=lambda s: print("  " + s))

    # Bei Multi-Output-Modellen heisst die Metric "val_policy_accuracy" statt "val_accuracy"
    monitor_metric = "val_policy_accuracy"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
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

    print(
        f"\nTraining startet (Streaming): epochs={epochs}, batch_size={batch_size}"
    )
    train_ds = _make_streaming_dataset(
        shard_split.train_shards,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = _make_streaming_dataset(
        shard_split.val_shards,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    # steps_per_epoch/validation_steps explizit setzen -- bei Streaming-Datasets
    # weiss Keras sonst nicht, wann eine Epoche zu Ende ist.
    steps_per_epoch = (train_samples + batch_size - 1) // batch_size
    validation_steps = (val_samples + batch_size - 1) // batch_size
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
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
    # Multi-Output-Modell: val_policy_accuracy; Legacy-Modell: val_accuracy
    acc_key = "val_policy_accuracy" if "val_policy_accuracy" in hist_dict else "val_accuracy"
    best_val_acc = max(hist_dict.get(acc_key, [0.0]))
    print(f"Beste {acc_key}: {best_val_acc:.4f}")
    if "val_value_mae" in hist_dict:
        best_val_mae = min(hist_dict["val_value_mae"])
        print(f"Beste val_value_mae: {best_val_mae:.4f}")


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
    parser.add_argument(
        "--verbose", type=int, default=2, choices=[0, 1, 2],
        help="0=still, 1=Live-Progressbar (kann auf WSL2/non-TTY langsam sein), "
             "2=eine Zeile pro Epoche (Default, empfohlen)",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=None,
        help="Versteckte Layer-Groessen, z.B. --hidden 512 512 256 (Default aus model.py)",
    )
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
        verbose=args.verbose,
        hidden_units=tuple(args.hidden) if args.hidden else None,
    )


if __name__ == "__main__":
    main()
