"""Smoke-Tests fuer das Modell mit Policy- + Value-Head."""

from __future__ import annotations

import numpy as np
import pytest


# TF nicht in jeder Testumgebung verfuegbar (CI ohne ML extras)
tf = pytest.importorskip("tensorflow")
from training.encoder import ACTION_DIM, INPUT_DIM  # noqa: E402
from training.model import build_model  # noqa: E402


def test_modell_baut_mit_value_head_und_policy_head():
    model = build_model(with_value_head=True)
    assert isinstance(model, tf.keras.Model)
    # Zwei Inputs (state + mask)
    assert len(model.inputs) == 2
    # Zwei Outputs (policy + value)
    assert len(model.outputs) == 2


def test_modell_forward_pass_shapes_und_legalitaet():
    """Forward auf 4 Samples: Policy hat richtige Shape, Value liegt in [-1,1]."""
    model = build_model(with_value_head=True)
    batch = 4
    x = np.random.rand(batch, INPUT_DIM).astype(np.float32)
    # Aktionsmaske: jeweils 3-5 zufaellige Karten erlaubt
    masks = np.zeros((batch, ACTION_DIM), dtype=np.float32)
    for i in range(batch):
        legal = np.random.choice(ACTION_DIM, size=4, replace=False)
        masks[i, legal] = 1.0

    out = model({"state": x, "mask": masks}, training=False)
    policy = out["policy"].numpy() if hasattr(out, "__getitem__") else None
    value = out["value"].numpy() if hasattr(out, "__getitem__") else None

    assert policy.shape == (batch, ACTION_DIM)
    assert value.shape == (batch, 1)
    # Value ist tanh-aktiviert -> [-1, +1]
    assert (value >= -1.0).all() and (value <= 1.0).all()
    # Policy-Wahrscheinlichkeiten summieren zu 1 pro Sample
    sums = policy.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)
    # Illegale Aktionen haben Wahrscheinlichkeit ~ 0
    for i in range(batch):
        illegal = masks[i] < 0.5
        assert (policy[i][illegal] < 1e-4).all()


def test_modell_save_und_load_roundtrip(tmp_path):
    """Modell speichern und wieder laden ergibt das gleiche Verhalten."""
    model = build_model(with_value_head=True)
    x = np.random.rand(2, INPUT_DIM).astype(np.float32)
    mask = np.zeros((2, ACTION_DIM), dtype=np.float32)
    mask[:, :5] = 1.0

    pred_before = model({"state": x, "mask": mask}, training=False)

    path = tmp_path / "model.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    pred_after = loaded({"state": x, "mask": mask}, training=False)

    assert np.allclose(pred_before["policy"].numpy(), pred_after["policy"].numpy(), atol=1e-5)
    assert np.allclose(pred_before["value"].numpy(), pred_after["value"].numpy(), atol=1e-5)


def test_legacy_modell_ohne_value_head():
    """Backward-Compat: with_value_head=False liefert das alte Single-Output-Modell."""
    model = build_model(with_value_head=False)
    assert len(model.outputs) == 1
    x = np.random.rand(2, INPUT_DIM).astype(np.float32)
    mask = np.zeros((2, ACTION_DIM), dtype=np.float32)
    mask[:, :5] = 1.0
    out = model({"state": x, "mask": mask}, training=False)
    # Single-Output: direkt Policy-Tensor
    arr = out.numpy() if hasattr(out, "numpy") else out
    assert arr.shape == (2, ACTION_DIM)
