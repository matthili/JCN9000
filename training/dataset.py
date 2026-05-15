"""Shard-Loader für Trainingsdaten.

Zwei Modi:

1. **In-Memory** (`load_split`): Lädt alle .npz-Shards aus einem Verzeichnis in
   den RAM (typisch ~10 GB bei 50k Partien) und splittet in Train/Val. Geeignet
   für kleine Datasets oder Tests.

2. **Streaming** (`split_shards`): Gibt nur die Shard-Pfade pro Train/Val-Split
   zurück; das eigentliche Laden uebernimmt der Trainings-Loop pro Shard
   (`tf.data.Dataset.interleave`). RAM-Spitze ~1-2 GB statt 40-80 GB. Pflicht
   fuer den balanced-v3-Datensatz (12 Varianten x 50k Runden = ~21 M Samples).

Unterstuetzt beide Datenformate:
  - Legacy (v0.1.0): X, masks, actions  -> reward = 0 als Fallback
  - Neu (v0.2.0+):   X, masks, actions, rewards
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray         # (N, 132) float32
    masks: np.ndarray     # (N, 36) float32 (für TF: float, nicht uint8)
    actions: np.ndarray   # (N,) int32
    rewards: np.ndarray   # (N,) float32 -- normalisierter Round-Outcome

    def __len__(self) -> int:
        return len(self.X)


@dataclass
class SplitDataset:
    train: Dataset
    val: Dataset


@dataclass
class ShardSplit:
    """Pfad-Splits fuer Streaming (kein RAM-Footprint."""
    train_shards: list[Path]
    val_shards: list[Path]

    def __repr__(self) -> str:
        return (
            f"ShardSplit(train={len(self.train_shards)}, val={len(self.val_shards)})"
        )


def load_shards(data_dir: str | Path) -> list[Path]:
    """Findet alle .npz-Shards rekursiv unter data_dir.

    Unterstuetzt sowohl flache Verzeichnisse (data/heuristic_50k/shard_*.npz)
    als auch verschachtelte Layouts (data/balanced/trumpf_eichel/shard_*.npz).
    """
    return sorted(Path(data_dir).rglob("*.npz"))


def _load_concat(shards: list[Path]) -> Dataset:
    Xs, ms, ys, rs = [], [], [], []
    for s in shards:
        d = np.load(s)
        Xs.append(d["X"])
        ms.append(d["masks"])
        ys.append(d["actions"])
        # Reward ist erst ab v0.2.0 vorhanden -- Legacy-Shards bekommen 0
        if "rewards" in d.files:
            rs.append(d["rewards"])
        else:
            rs.append(np.zeros(len(d["actions"]), dtype=np.float32))
    X = np.concatenate(Xs).astype(np.float32, copy=False)
    masks = np.concatenate(ms).astype(np.float32, copy=False)  # für TF
    actions = np.concatenate(ys).astype(np.int32, copy=False)
    rewards = np.concatenate(rs).astype(np.float32, copy=False)
    return Dataset(X=X, masks=masks, actions=actions, rewards=rewards)


def load_split(
    data_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> SplitDataset:
    """Lädt alle Shards und splittet shard-weise in Train/Val.

    Shard-weiser Split ist sauberer als sample-weiser: Partien werden nicht zerrissen,
    und das Modell sieht in Val Spielzustände aus für es komplett neuen Partien.
    """
    shards = load_shards(data_dir)
    if not shards:
        raise FileNotFoundError(f"Keine .npz-Shards in {data_dir} gefunden.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(shards))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(shards) * val_fraction)))
    val_idx = set(indices[:val_count].tolist())
    train_shards = [s for i, s in enumerate(shards) if i not in val_idx]
    val_shards = [s for i, s in enumerate(shards) if i in val_idx]

    print(f"Lade Daten: {len(train_shards)} Train-Shards, {len(val_shards)} Val-Shards…")
    train = _load_concat(train_shards)
    val = _load_concat(val_shards)
    print(
        f"  Train: {len(train):>10,} Samples "
        f"({train.X.nbytes / 2**30:.2f} GB X + {train.masks.nbytes / 2**30:.2f} GB Masks)"
    )
    print(f"  Val:   {len(val):>10,} Samples")
    return SplitDataset(train=train, val=val)


def split_shards(
    data_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> ShardSplit:
    """Wie `load_split`, aber gibt nur die Shard-Pfade zurueck.

    Train/Val-Split passiert auf Shard-Ebene (nicht Sample-Ebene), damit
    Partien nicht zerrissen werden. Das eigentliche Laden uebernimmt der
    Trainings-Loop pro Shard via tf.data — RAM-Spitze bleibt minimal.
    """
    shards = load_shards(data_dir)
    if not shards:
        raise FileNotFoundError(f"Keine .npz-Shards in {data_dir} gefunden.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(shards))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(shards) * val_fraction)))
    val_idx = set(indices[:val_count].tolist())
    train_shards = [s for i, s in enumerate(shards) if i not in val_idx]
    val_shards = [s for i, s in enumerate(shards) if i in val_idx]

    print(
        f"Shard-Streaming: {len(train_shards)} Train-Shards, "
        f"{len(val_shards)} Val-Shards (kein RAM-Preload)."
    )
    return ShardSplit(train_shards=train_shards, val_shards=val_shards)


def shard_sample_count(shard_path: Path) -> int:
    """Liefert die Anzahl der Samples in einem Shard, ohne die Daten zu laden.

    Nutzt das `actions`-Array (kleinster Datentyp, schnellster Zugriff). Wird
    fuer korrekte `steps_per_epoch`-Berechnung beim Streaming gebraucht.
    """
    with np.load(shard_path) as d:
        return int(len(d["actions"]))


def total_sample_count(shard_paths: list[Path]) -> int:
    """Summiert Sample-Counts ueber eine Shard-Liste (eine kurze Metadaten-Runde)."""
    return sum(shard_sample_count(p) for p in shard_paths)
