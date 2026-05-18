"""Vektorisierter Full-Round-Lookahead fuer Solo-Jass.

Aufbau identisch zu `training/data/vectorized_lookahead.py`, aber mit
Solo-spezifischer Reward-Berechnung:

  team:  reward = (eigenes_team - gegner_team) / REWARD_SCALE
  solo:  reward = (eigene_punkte - max(gegner_punkte)) / REWARD_SCALE

Das `max(gegner_punkte)` modelliert die Solo-Strategie "ich muss vor dem
fuehrenden Gegner liegen" -- der hoechste Gegner ist die Bedrohung, nicht
der Durchschnitt.

Sonstige Verbesserungen, die hier nicht noetig sind aber Platz fuer
spaeteres Solo-Tuning lassen:
- Anderer Rollout-Count pro Karte
- Risiko-Bewusste-Reward (z.B. Bonus fuer Vermeidung von max(others) > eigener_score)
- Eigene Stoecke-/Weisen-Modellierung in Rollouts

Die `Rollout`-Klasse wird wiederverwendet (struktur-identisch); nur das
Reward-Extracten am Schluss ist anders.
"""

from __future__ import annotations

import random

import numpy as np

from jass_engine.card import Card
from jass_engine.player import GameState
from jass_engine.rules import legal_moves
from training.data.vectorized_lookahead import (
    REWARD_SCALE,
    Rollout,
    _make_rollout,
)
from training.encoder import encode_state, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


def _solo_reward(r: Rollout) -> float:
    """Reward fuer einen Solo-Rollout: eigene Punkte minus staerkster Gegner.

    Im Solo (teams=[0,1,2,3]) ist `r.teams[r.root_seat]` der Spieler-Index.
    `team_points` ist dann ein 4-Eintraege-Dict, eines pro Spieler.
    """
    own_team_id = r.teams[r.root_seat]
    own = r.team_points.get(own_team_id, 0)
    others = [pts for tid, pts in r.team_points.items() if tid != own_team_id]
    opp = max(others) if others else 0
    return (own - opp) / REWARD_SCALE


def compute_card_scores_solo_vectorized(
    hand: list[Card],
    state: GameState,
    inference_server: InferenceServer,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
    max_steps_safety: int = 200,
) -> dict[Card, float]:
    """Solo-Variante von `compute_card_scores_vectorized`.

    Returns:
        Mapping `card -> mean_solo_reward` ueber alle Rollouts dieser Karte.
    """
    if rng is None:
        rng = random.Random()

    legal = legal_moves(hand, state.current_trick_cards, state.variant)
    if len(legal) == 1:
        return {legal[0]: 0.0}

    # Initialisierung: pro Karte x Rollout je ein Rollout-Objekt
    all_rollouts: list[Rollout] = []
    for card_idx, card in enumerate(legal):
        for _ in range(rollouts_per_card):
            all_rollouts.append(_make_rollout(card_idx, card, hand, state, rng))

    # Tick all rollouts in lockstep, bis alle done sind.
    for _ in range(max_steps_safety):
        waiting = [r for r in all_rollouts if r.needs_inference()]
        if not waiting:
            break

        states_batch = []
        masks_batch = []
        for r in waiting:
            cur_hand, cur_state = r.get_state_for_inference()
            x = encode_state(cur_hand, cur_state).astype(np.float32)
            m = legal_action_mask(cur_hand, cur_state).astype(np.float32)
            states_batch.append(x)
            masks_batch.append(m)

        results = inference_server.request_many(states_batch, masks_batch)

        for r, mask, (policy, _value) in zip(waiting, masks_batch, results):
            legal_policy = policy * mask
            s = legal_policy.sum()
            if s <= 0:
                legal_indices = np.where(mask > 0.5)[0]
                action_idx = int(rng.choice(legal_indices))
            else:
                action_idx = int(np.argmax(legal_policy))
            r.apply_action(action_idx, rng)

    # Aggregiere Solo-Rewards pro first_move-Karte
    rewards_per_card: dict[Card, list[float]] = {c: [] for c in legal}
    for r in all_rollouts:
        rewards_per_card[legal[r.card_idx_of_first_move]].append(_solo_reward(r))

    return {c: float(np.mean(rs)) if rs else 0.0 for c, rs in rewards_per_card.items()}
