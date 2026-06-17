"""PPO-Update-Schritt fuer das Policy+Value-Modell.

[INAKTIV] Teil des archivierten PPO/RL-Experiments -- siehe
training/rl/train_rl.py. Nicht in der aktuellen MCTS-BC-Pipeline; bleibt
getestet erhalten.

Klassisches PPO mit Clipped-Surrogate-Objective:
    L_policy  = -E[ min( r * A, clip(r, 1-eps, 1+eps) * A ) ]
    L_value   = MSE( returns, V(state) )         [optional auch clipped]
    L_entropy = -E[ Entropy(pi(.|state)) ]       [als Bonus subtrahiert]

    L_total = L_policy + value_coef * L_value + entropy_coef * L_entropy

Referenz: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).

Wir verwenden das maskenbewusste Multi-Head-Modell aus training/model.py: der
Policy-Output ist bereits maskiert (illegale Aktionen haben Wahrscheinlichkeit
~0). Beim Logarithmieren clippen wir auf einen Mindestwert, um log(0) zu
vermeiden.
"""

from __future__ import annotations

import tensorflow as tf


LOG_EPS = 1e-9
MIN_LOG_PROB = -50.0  # falls die Policy fuer eine Aktion ~0 sagt


def _action_log_probs(policy: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    """Liest die Log-Wahrscheinlichkeit der jeweils gewaehlten Aktion aus.

    policy: (B, 36) Wahrscheinlichkeiten
    actions: (B,) int32 Aktions-Indizes
    return: (B,) Log-Wahrscheinlichkeiten
    """
    action_one_hot = tf.one_hot(actions, depth=tf.shape(policy)[-1], dtype=policy.dtype)
    action_probs = tf.reduce_sum(policy * action_one_hot, axis=-1)
    log_probs = tf.math.log(tf.maximum(action_probs, LOG_EPS))
    return tf.maximum(log_probs, MIN_LOG_PROB)


def _entropy(policy: tf.Tensor) -> tf.Tensor:
    """Mittlere Entropy der Policy-Verteilungen.

    Achtung: bei maskierten Policies sind viele Eintraege = 0; diese tragen
    nichts zur Entropy bei (0 * log 0 = 0). Wir behandeln das durch Clipping
    im Log.
    """
    log_policy = tf.math.log(tf.maximum(policy, LOG_EPS))
    return -tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=-1))


def ppo_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: dict[str, tf.Tensor],
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, tf.Tensor]:
    """Macht einen einzelnen PPO-Gradientenschritt.

    Args:
        model: Das Multi-Head-Modell (Inputs state+mask, Outputs policy+value).
        optimizer: Keras-Optimizer.
        batch: dict mit "states", "masks", "actions", "old_log_probs",
               "old_values", "advantages", "returns" (alles tf.Tensor oder
               numpy-Arrays).
        clip_ratio: PPO-Clipping-Epsilon (0.1-0.3 ueblich).
        value_coef: Gewichtung des Value-Losses.
        entropy_coef: Gewichtung des Entropy-Bonus (foerdert Exploration).
        max_grad_norm: Gradient-Clipping fuer Stabilitaet.

    Returns:
        Dict mit Loss-Komponenten als Skalar-Tensors.
    """
    states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
    masks = tf.convert_to_tensor(batch["masks"], dtype=tf.float32)
    actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
    old_log_probs = tf.convert_to_tensor(batch["old_log_probs"], dtype=tf.float32)
    advantages = tf.convert_to_tensor(batch["advantages"], dtype=tf.float32)
    returns = tf.convert_to_tensor(batch["returns"], dtype=tf.float32)

    with tf.GradientTape() as tape:
        outputs = model({"state": states, "mask": masks}, training=True)
        policy = outputs["policy"]
        value = tf.squeeze(outputs["value"], axis=-1)

        new_log_probs = _action_log_probs(policy, actions)
        ratio = tf.exp(new_log_probs - old_log_probs)

        # Clipped Surrogate
        unclipped = ratio * advantages
        clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

        # Value-Loss (einfache MSE; PPO-clipped-Value ist optional)
        value_loss = tf.reduce_mean(tf.square(returns - value))

        # Entropy-Bonus
        entropy = _entropy(policy)

        total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    if max_grad_norm is not None and max_grad_norm > 0:
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    else:
        grad_norm = tf.linalg.global_norm(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # KL-Divergenz als Diagnose (approximativ)
    approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "total_loss": total_loss,
        "approx_kl": approx_kl,
        "grad_norm": grad_norm,
    }
