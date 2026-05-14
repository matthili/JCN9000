"""Keras-Modell: Mask-Aware MLP für Kartenwahl.

Eingaben: Featurevektor (132 Floats) + Aktionsmaske (36 Bits).
Ausgabe: Wahrscheinlichkeitsverteilung über 36 Karten, mit illegalen Karten auf 0.

Maskierung: Auf die Logits wird eine grosse negative Konstante für illegale Aktionen
addiert (-1e9), sodass nach dem Softmax die Wahrscheinlichkeit für illegale Karten
effektiv auf 0 fällt. Das Netz lernt damit nur über legale Aktionen.
"""

from __future__ import annotations

import keras
from tensorflow.keras import layers

from training.encoder import ACTION_DIM, INPUT_DIM


DEFAULT_HIDDEN = (256, 256, 128)


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
) -> keras.Model:
    """Baut das Mask-Aware MLP und kompiliert es."""

    state_in = keras.Input(shape=(input_dim,), name="state")
    mask_in = keras.Input(shape=(action_dim,), name="mask")

    x = state_in
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    logits = layers.Dense(action_dim, name="logits")(x)

    # Maskierung: illegale Aktionen bekommen einen sehr grossen negativen Bias.
    # Nach dem Softmax sind die zugehörigen Wahrscheinlichkeiten ~0.
    mask_bias = MaskBias(name="mask_bias")(mask_in)
    masked_logits = layers.Add(name="masked_logits")([logits, mask_bias])

    probs = layers.Softmax(name="probs")(masked_logits)

    model = keras.Model(inputs=[state_in, mask_in], outputs=probs, name="jass_policy")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
