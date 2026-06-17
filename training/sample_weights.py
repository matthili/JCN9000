"""Sample-Gewichte fuers Training -- aktuell: Anspiel-Stellungen hoeher gewichten.

Hintergrund: Die Anspiel-Policy (erster Zug eines Stichs) ist im Kreuz-Modell
v0.7.2 unterfittet (siehe docs/next_training_round.md, Posten 1). Ein billiger
Hebel OHNE Daten-Re-Gen ist, Anspiel-Samples im Trainings-Loss hoeher zu
gewichten. Ob ein Sample ein Anspiel ist, laesst sich direkt aus dem
State-Vektor ablesen: bei einem Anspiel ist der aktuelle Stich leer, also sind
alle `current_trick_by_*`-Sektionen 0.

Reines NumPy, TF-frei und unit-getestet (tests/test_lead_weights.py). Der
eigentliche tf.data-Einbau in training/train.py ist als Rezept in
docs/next_training_round.md beschrieben (bewusst nicht ungetestet eingebaut).
"""

from __future__ import annotations

import numpy as np


def current_trick_span(section_offsets) -> tuple[int, int]:
    """(lo, hi)-Index-Spanne aller `current_trick_by_*`-Sektionen im State-Vektor."""
    keys = [k for k in section_offsets if k.startswith("current_trick_by_")]
    if not keys:
        raise ValueError("Keine 'current_trick_by_*'-Sektionen im Encoder-Layout gefunden.")
    lo = min(section_offsets[k][0] for k in keys)
    hi = max(section_offsets[k][1] for k in keys)
    return lo, hi


def is_lead_position(states: np.ndarray, section_offsets=None) -> np.ndarray:
    """Bool-Array (N,): True, wo der aktuelle Stich leer ist (= Anspiel/Lead)."""
    if section_offsets is None:
        from training.encoder import SECTION_OFFSETS as section_offsets
    lo, hi = current_trick_span(section_offsets)
    return states[:, lo:hi].sum(axis=1) == 0


def lead_sample_weights(
    states: np.ndarray, lead_weight: float, section_offsets=None
) -> np.ndarray:
    """Gewicht pro Sample: `lead_weight` fuer Anspiel-Stellungen, sonst 1.0.

    Args:
        states: (N, INPUT_DIM)-Float-Array (die `X`-Matrix eines Batches/Shards).
        lead_weight: Faktor fuer Anspiel-Samples (1.0 = neutral/aus).
        section_offsets: Encoder-Layout (Default: training.encoder.SECTION_OFFSETS).

    Returns:
        (N,)-Float32-Array mit `lead_weight` an Anspiel-Positionen, sonst 1.0.
    """
    lead = is_lead_position(states, section_offsets)
    return np.where(lead, np.float32(lead_weight), np.float32(1.0)).astype(np.float32)
