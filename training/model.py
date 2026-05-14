"""Keras-Modell: Mask-Aware MLP mit Policy- und Value-Head.

Eingaben:
    - state: Featurevektor (132 Floats)
    - mask:  Aktionsmaske (36 Bits)

Ausgaben:
    - policy: Wahrscheinlichkeitsverteilung ueber 36 Karten (Softmax mit Maske)
    - value:  Skalar in [-1, 1], "wie gut ist dieser Zustand fuer mein Team"

Architektur:
    state -> shared trunk (Dense-Layers) -> zwei Koepfe:
      - Policy-Head: Dense(action_dim) + Maskierung + Softmax
      - Value-Head:  Dense(1) + tanh

Maskierung: Auf die Policy-Logits wird ein grosser negativer Bias fuer illegale
Aktionen addiert, sodass nach Softmax die Wahrscheinlichkeit fuer illegale Karten
effektiv 0 ist.

Im Training gibt es zwei Losses:
    - policy_loss: sparse_categorical_crossentropy (Kartenwahl)
    - value_loss:  mean_squared_error             (Reward-Schaetzung)
"""

from __future__ import annotations

import keras
from tensorflow.keras import layers

from training.encoder import ACTION_DIM, INPUT_DIM


DEFAULT_HIDDEN = (256, 256, 128)
DEFAULT_VALUE_LOSS_WEIGHT = 0.5  # Value-Loss leichter gewichtet als Policy


@keras.saving.register_keras_serializable(package="jass")
class MaskBias(layers.Layer):
    """Wandelt eine Aktionsmaske (1=legal, 0=illegal) in einen Bias-Tensor um:
    illegale Aktionen bekommen einen sehr negativen Wert, sodass Softmax sie auf 0 setzt.
    """

    def __init__(self, neg_value: float = -1.0e9, **kwargs):
        super().__init__(**kwargs)
        self.neg_value = neg_value

    def call(self, mask):
        return (1.0 - mask) * self.neg_value

    def get_config(self):
        return {**super().get_config(), "neg_value": self.neg_value}


def build_model(
    input_dim: int = INPUT_DIM,
    action_dim: int = ACTION_DIM,
    hidden_units: tuple[int, ...] = DEFAULT_HIDDEN,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT,
    with_value_head: bool = True,
) -> keras.Model:
    """Baut das Mask-Aware MLP mit Policy- und (optional) Value-Head und kompiliert es.

    Args:
        with_value_head: Wenn False, wird nur der Policy-Head gebaut (Legacy-Modus,
            kompatibel mit dem v0.1.0-Release-Format).
    """

    state_in = keras.Input(shape=(input_dim,), name="state")
    mask_in = keras.Input(shape=(action_dim,), name="mask")

    # Shared trunk
    x = state_in
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    # Policy-Head
    logits = layers.Dense(action_dim, name="logits")(x)
    mask_bias = MaskBias(name="mask_bias")(mask_in)
    masked_logits = layers.Add(name="masked_logits")([logits, mask_bias])
    policy = layers.Softmax(name="policy")(masked_logits)

    if not with_value_head:
        # Legacy-Modell: nur Policy-Output (kompatibel mit fruehen Releases)
        model = keras.Model(inputs=[state_in, mask_in], outputs=policy, name="jass_policy")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Value-Head
    v = layers.Dense(64, activation="relu", name="value_dense")(x)
    value = layers.Dense(1, activation="tanh", name="value")(v)

    model = keras.Model(
        inputs={"state": state_in, "mask": mask_in},
        outputs={"policy": policy, "value": value},
        name="jass_policy_value",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "policy": "sparse_categorical_crossentropy",
            "value": "mean_squared_error",
        },
        loss_weights={
            "policy": 1.0,
            "value": value_loss_weight,
        },
        metrics={
            "policy": ["accuracy"],
            "value": ["mae"],
        },
    )
    return model
