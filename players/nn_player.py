"""NN-Player: nutzt ein trainiertes Keras-Modell für die Kartenwahl.

Trumpfansage und Weisen werden weiterhin vom (deterministischen) HeuristicPlayer
übernommen — das NN lernt zunächst nur die Kartenwahl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from jass_engine.card import Card
from jass_engine.player import GameState, Player
from jass_engine.variant import Announcement, Variant
from jass_engine.weis import Weis
from players.heuristic_player import HeuristicPlayer
from training.encoder import encode_state, index_to_card, legal_action_mask
# Import sorgt dafür, dass MaskBias bei keras.models.load_model() registriert ist
from training.model import MaskBias  # noqa: F401


class NNPlayer(Player):
    """Spielt mit einem trainierten Mask-Aware MLP."""

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        fallback: Player | None = None,
        greedy: bool = True,
    ):
        super().__init__(name)
        self.model = keras.models.load_model(str(model_path))
        # Für Ansage/Weisen: Heuristik als Fallback (NN lernt das nicht in v1)
        self.fallback = fallback or HeuristicPlayer(name + "_fallback")
        # greedy: höchste Wahrscheinlichkeit; sonst Sampling aus der Verteilung
        self.greedy = greedy

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        return self.fallback.choose_announcement(hand, round_idx, can_push)

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        x = encode_state(hand, state)[np.newaxis, :].astype(np.float32)
        mask = legal_action_mask(hand, state)[np.newaxis, :].astype(np.float32)
        # Wichtig: NICHT model.predict() (hat ~30ms Python-Overhead pro Call wegen
        # interner Dataset-Konstruktion). Das __call__ ist direkter Tensor-Forward
        # und 10-50x schneller fuer Single-Sample-Inferenz.
        prediction = self.model({"state": x, "mask": mask}, training=False)

        # Multi-Output-Modell liefert dict mit "policy" und "value";
        # Legacy-Modell liefert direkt das Policy-Array.
        # Wenn aus model(...) statt model.predict(...): Tensors statt numpy -> .numpy()
        def _to_np(t):
            return t.numpy() if hasattr(t, "numpy") else t

        if isinstance(prediction, dict):
            probs = _to_np(prediction["policy"])[0]
        elif isinstance(prediction, (list, tuple)) and len(prediction) >= 1:
            probs = _to_np(prediction[0])[0]
        else:
            probs = _to_np(prediction)[0]

        if self.greedy:
            chosen_idx = int(np.argmax(probs))
        else:
            chosen_idx = int(np.random.choice(len(probs), p=probs))
        chosen = index_to_card(chosen_idx)
        # Safety: falls das NN durch numerische Rundungsfehler etwas Illegales wählt,
        # bricht der Engine-Check sowieso ab; aber zur Sicherheit auf legale fallback.
        if mask[0, chosen_idx] < 0.5:
            # Höchste Wahrscheinlichkeit unter legalen Karten
            legal_probs = probs * mask[0]
            chosen_idx = int(np.argmax(legal_probs))
            chosen = index_to_card(chosen_idx)
        return chosen

    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        return self.fallback.announce_weise(hand, variant, possible_weise)
