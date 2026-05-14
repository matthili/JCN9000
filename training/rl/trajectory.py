"""Datenklassen fuer Trajektorien + GAE-Advantage-Berechnung.

Eine Trajektorie ist die Liste aller Karten-Entscheidungen, die ein Spieler
in einer Partie getroffen hat. Pro Entscheidung speichern wir:
    - state, mask          : die Eingabe ins NN
    - action               : welche Karte (Index 0..35)
    - log_prob             : Logarithmus der Wahrscheinlichkeit dieser Wahl
                             unter der "alten" Policy (zur Sample-Zeit)
    - value                : Value-Schaetzung des NN zur Sample-Zeit
    - reward               : Belohnung; bei Jass NUR am Rundenende != 0
                             (alle Zuege innerhalb einer Runde bekommen den
                             gleichen Round-Reward als gemeinsamen Endwert)

Aus den Trajektorien werden mit GAE die *advantages* und *returns* berechnet,
die im PPO-Update verwendet werden.

Referenz: Schulman et al., "High-Dimensional Continuous Control Using
Generalized Advantage Estimation" (2015).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transition:
    """Ein einzelner Spielzug eines Spielers."""

    state: np.ndarray         # shape (132,) float32
    mask: np.ndarray          # shape (36,) uint8 oder float32
    action: int               # 0..35
    log_prob: float           # log pi_old(action | state)
    value: float              # V_old(state)
    reward: float = 0.0       # wird nach der Runde gesetzt
    done: bool = False        # True wenn dies der letzte Zug der Runde war


@dataclass
class Trajectory:
    """Sequenz von Transitions eines Spielers in einer Partie."""

    transitions: list[Transition] = field(default_factory=list)
    # Nach GAE-Berechnung gefuellt:
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, t: Transition) -> None:
        self.transitions.append(t)


def compute_gae(
    trajectory: Trajectory,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
) -> None:
    """Generalized Advantage Estimation: berechnet advantages + returns in-place.

    Bei Sparse-Reward-Spielen (Reward nur am Rundenende) sorgt GAE dafuer,
    dass der Reward sinnvoll auf alle Zwischenzuege zurueckpropagiert wird,
    gemischt mit den Value-Schaetzungen des NN.

    Args:
        trajectory: Die Sequenz von Transitions.
        gamma: Diskontfaktor (~0.99 ueblich)
        lam:   Lambda-Parameter (~0.95 ueblich; Trade-off zwischen Bias und Varianz)
        bootstrap_value: V_old(s_T+1), Value-Schaetzung fuer den Folgezustand
                         nach dem letzten Zug. Ueblicherweise 0 wenn die Runde
                         am Ende der Trajektorie wirklich vorbei ist.
    """
    n = len(trajectory)
    if n == 0:
        trajectory.advantages = np.zeros(0, dtype=np.float32)
        trajectory.returns = np.zeros(0, dtype=np.float32)
        return

    rewards = np.array([t.reward for t in trajectory.transitions], dtype=np.float32)
    values = np.array([t.value for t in trajectory.transitions], dtype=np.float32)
    dones = np.array([t.done for t in trajectory.transitions], dtype=np.float32)

    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    last_value = bootstrap_value

    for t in reversed(range(n)):
        # Wenn done[t]=True, ist t der letzte Zug der Runde, danach Reset:
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
        last_value = values[t]

    returns = advantages + values

    trajectory.advantages = advantages
    trajectory.returns = returns


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Standardisiert Advantages auf Mittelwert 0 und Standardabweichung 1.

    Ueblicher Trick fuer stabileres PPO-Training.
    """
    if len(advantages) == 0:
        return advantages
    mean = advantages.mean()
    std = advantages.std() + 1e-8
    return (advantages - mean) / std


def stack_trajectories(
    trajectories: list[Trajectory],
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Bringt mehrere Trajektorien in flache Batch-Form fuer das PPO-Update.

    Returns:
        Dict mit den Keys "states", "masks", "actions", "old_log_probs",
        "old_values", "advantages", "returns" -- alles 1D bzw. 2D-Arrays.
    """
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    advs: list[np.ndarray] = []
    rets: list[np.ndarray] = []

    for traj in trajectories:
        for t in traj.transitions:
            states.append(t.state)
            masks.append(t.mask)
            actions.append(t.action)
            log_probs.append(t.log_prob)
            values.append(t.value)
        assert traj.advantages is not None and traj.returns is not None, (
            "compute_gae muss vor stack_trajectories aufgerufen werden"
        )
        advs.append(traj.advantages)
        rets.append(traj.returns)

    if not states:
        return {
            "states": np.empty((0, 0), dtype=np.float32),
            "masks": np.empty((0, 0), dtype=np.float32),
            "actions": np.empty(0, dtype=np.int32),
            "old_log_probs": np.empty(0, dtype=np.float32),
            "old_values": np.empty(0, dtype=np.float32),
            "advantages": np.empty(0, dtype=np.float32),
            "returns": np.empty(0, dtype=np.float32),
        }

    out_states = np.stack(states).astype(np.float32, copy=False)
    out_masks = np.stack(masks).astype(np.float32, copy=False)
    out_actions = np.array(actions, dtype=np.int32)
    out_log_probs = np.array(log_probs, dtype=np.float32)
    out_values = np.array(values, dtype=np.float32)
    out_advs = np.concatenate(advs).astype(np.float32, copy=False)
    out_rets = np.concatenate(rets).astype(np.float32, copy=False)

    if normalize:
        out_advs = normalize_advantages(out_advs)

    return {
        "states": out_states,
        "masks": out_masks,
        "actions": out_actions,
        "old_log_probs": out_log_probs,
        "old_values": out_values,
        "advantages": out_advs,
        "returns": out_rets,
    }
