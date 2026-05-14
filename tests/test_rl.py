"""Tests fuer das RL-Modul (Trajectory, GAE, PPO, Self-Play-Smoke)."""

from __future__ import annotations

import numpy as np
import pytest

# TF in CI moeglicherweise nicht installiert -> skippen statt fail
tf = pytest.importorskip("tensorflow")

from training.encoder import ACTION_DIM, INPUT_DIM  # noqa: E402
from training.model import build_model  # noqa: E402
from training.rl.ppo import ppo_train_step  # noqa: E402
from training.rl.trajectory import (  # noqa: E402
    Trajectory,
    Transition,
    compute_gae,
    normalize_advantages,
    stack_trajectories,
)


# ---------- Trajectory + GAE ----------

def _make_trajectory(rewards: list[float], values: list[float]) -> Trajectory:
    traj = Trajectory()
    for r, v in zip(rewards, values):
        traj.append(Transition(
            state=np.zeros(INPUT_DIM, dtype=np.float32),
            mask=np.ones(ACTION_DIM, dtype=np.uint8),
            action=0,
            log_prob=0.0,
            value=float(v),
            reward=float(r),
            done=False,
        ))
    if traj.transitions:
        traj.transitions[-1].done = True
    return traj


def test_gae_leere_trajectory():
    traj = Trajectory()
    compute_gae(traj)
    assert traj.advantages is not None and len(traj.advantages) == 0
    assert traj.returns is not None and len(traj.returns) == 0


def test_gae_einzel_transition():
    """Bei nur einem Zug ist advantage = reward - value (kein Bootstrap noetig)."""
    traj = _make_trajectory(rewards=[1.0], values=[0.3])
    compute_gae(traj, gamma=0.99, lam=0.95)
    # done=True, also kein Bootstrap. delta = 1.0 + 0 - 0.3 = 0.7. lambda*gamma*done==0.
    assert traj.advantages[0] == pytest.approx(0.7, abs=1e-5)
    assert traj.returns[0] == pytest.approx(0.7 + 0.3, abs=1e-5)


def test_gae_sparse_reward_propagiert_zurueck():
    """Sparse reward am Ende sollte mit gamma=1 auf alle Schritte gleichmaessig
    durchpropagiert werden, bei perfekt korrektem Value=0."""
    traj = _make_trajectory(rewards=[0, 0, 0, 1.0], values=[0, 0, 0, 0])
    compute_gae(traj, gamma=1.0, lam=1.0)
    # Bei lam=1 -> GAE = sum future rewards - values. Mit value=0 ueberall:
    # advantage[t] = sum of rewards from t onward = 1.0 fuer alle t
    np.testing.assert_array_almost_equal(traj.advantages, [1.0, 1.0, 1.0, 1.0])


def test_normalize_advantages():
    advs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    normed = normalize_advantages(advs)
    assert abs(normed.mean()) < 1e-5
    assert abs(normed.std() - 1.0) < 1e-3


def test_stack_trajectories_baut_richtiges_batch():
    traj1 = _make_trajectory(rewards=[0.5], values=[0.1])
    traj2 = _make_trajectory(rewards=[-0.3, 0.7], values=[0.0, 0.2])
    compute_gae(traj1)
    compute_gae(traj2)
    batch = stack_trajectories([traj1, traj2], normalize=False)
    assert batch["states"].shape == (3, INPUT_DIM)
    assert batch["masks"].shape == (3, ACTION_DIM)
    assert batch["actions"].shape == (3,)
    assert batch["advantages"].shape == (3,)
    assert batch["returns"].shape == (3,)


# ---------- PPO-Step ----------

def test_ppo_train_step_aendert_model_und_liefert_metriken():
    """Smoke-Test: ein PPO-Schritt aendert die Gewichte und produziert plausible Metriken."""
    model = build_model(with_value_head=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    batch_size = 8
    states = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)
    masks = np.zeros((batch_size, ACTION_DIM), dtype=np.float32)
    for i in range(batch_size):
        legal = np.random.choice(ACTION_DIM, size=5, replace=False)
        masks[i, legal] = 1.0
    actions = np.array([np.where(masks[i] > 0)[0][0] for i in range(batch_size)], dtype=np.int32)

    batch = {
        "states": states,
        "masks": masks,
        "actions": actions,
        "old_log_probs": np.zeros(batch_size, dtype=np.float32) - 1.0,  # log(1/e)
        "advantages": np.random.randn(batch_size).astype(np.float32),
        "returns": np.random.randn(batch_size).astype(np.float32) * 0.5,
    }

    weights_before = [w.numpy().copy() for w in model.trainable_variables]
    metrics = ppo_train_step(model, optimizer, batch)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    # Mindestens eine Gewichts-Variable hat sich veraendert
    weights_after = [w.numpy() for w in model.trainable_variables]
    changed = any(
        not np.allclose(b, a, atol=1e-6) for b, a in zip(weights_before, weights_after)
    )
    assert changed, "PPO-Schritt hat keine Gewichte veraendert"
    # Entropy ist eine nicht-negative Groesse (bei maskierten Policies)
    assert float(metrics["entropy"]) >= 0


# ---------- Self-Play-Smoke ----------

def test_selfplay_eine_partie_produziert_trajektorien():
    """Smoke-Test: eine Self-Play-Partie mit RLPlayer laeuft durch und produziert
    4 Trajektorien mit jeweils mindestens 9 Transitions (1 Runde minimum)."""
    from training.rl.selfplay import collect_trajectories

    model = build_model(with_value_head=True)
    trajs = collect_trajectories(model=model, num_games=1, target_score=100, seed=42)
    assert len(trajs) == 4  # 4 Spieler pro Partie
    for tr in trajs:
        # Mindestens 9 Karten-Entscheidungen pro Spieler (1 Runde)
        assert len(tr) >= 9
        # Reward in [-2, +2] (skaliert durch /200)
        for t in tr.transitions:
            assert -3.0 <= t.reward <= 3.0
        # Letzte Transition jeder Runde ist done
        # (Nicht zwingend die allerletzte der Trajektorie, aber mindestens eine done)
        assert any(t.done for t in tr.transitions)
