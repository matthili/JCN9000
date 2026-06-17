"""Probe: Wie bewertet das Kreuz-Modell den ANSPIEL-Zug in REINEM Unten (Geiss)?

Baut exakt die vom Domaenenexperten beobachtete Situation nach:
  - Reines Unten (Geiss), KEIN Slalom  -> die 6 ist die staerkste Karte einer
    Farbe, das Ass die schwaechste (und 11 Punkte schwer).
  - Das NN ist am Zug und spielt AN (leerer Stich -> alle Handkarten legal).
  - Die Hand enthaelt die Boss-Paarung 6+7 einer Farbe PLUS zwei Aesser
    (maximale "lead-high"-Verlockung, falls das Modell einen Oben-Reflex hat).

Erwartung bei korrektem Spiel: die 6 (oder 7) der Boss-Farbe oben in der Policy,
die Aesser weit unten. Spielt das Modell stattdessen ein Ass an, ist es ein
echter Lernfehler -- NICHT blosses Sampling, denn die App nimmt argmax
(apps/api/.../nn-player.ts: `return indexToCard(res.argmax)`).

Lauf (WSL2, conda-env jass-gpu):
    python -m scripts.probe_unten_lead
    python -m scripts.probe_unten_lead --model models/kreuz_mcts3/best.keras
    python -m scripts.probe_unten_lead --dry-run   # ohne TF/Modell: nur State+Encoding+Maske

--dry-run laeuft auch ohne TensorFlow und prueft nur, dass State, Encoding und
Maske sauber gebaut werden (Engine-/Encoder-API-Check).
"""

from __future__ import annotations

import argparse

import numpy as np

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from training.encoder import encode_state, index_to_card, legal_action_mask


def build_scenario() -> tuple[list[Card], GameState]:
    """Reines Unten, NN spielt an, Hand mit Boss-Paar 6+7 (Eichel) + zwei Aessern."""
    hand = [
        Card(Suit.EICHEL, Rank.SECHS),    # in Unten der BOSS dieser Farbe
        Card(Suit.EICHEL, Rank.SIEBEN),   # zweitstaerkste in der Farbe
        Card(Suit.EICHEL, Rank.ZEHN),     # 10 Punkte, in Unten schwach
        Card(Suit.SCHELLE, Rank.ASS),     # 11 Punkte, in Unten SCHWAECHSTE -> Verlockung
        Card(Suit.SCHELLE, Rank.KOENIG),
        Card(Suit.HERZ, Rank.ASS),        # zweites Ass
        Card(Suit.HERZ, Rank.ACHT),
        Card(Suit.LAUB, Rank.NEUN),
        Card(Suit.LAUB, Rank.UNTER),
    ]
    state = GameState(
        player_idx=0,
        variant=Variant.unten(),                                  # effektive Variante = reines Unten
        announcement=Announcement(variant=Variant.unten(), slalom=False),
        current_trick_cards=[],                                   # leer -> NN spielt an
        current_trick_starter=0,
        teams=[0, 1, 0, 1],                                       # Kreuz (Partner ueber Kreuz)
        completed_tricks=[],
        round_idx=0,
        trick_idx=0,
        num_players=4,
    )
    return hand, state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="models/kreuz_mcts3/best.keras",
        help="Pfad zum Kreuz-Modell (Default: v0.7.2 = kreuz_mcts3).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Ohne TF/Modell: nur State, Encoding und Maske pruefen.",
    )
    args = ap.parse_args()

    hand, state = build_scenario()
    vec = encode_state(hand, state).astype(np.float32)
    mask = legal_action_mask(hand, state).astype(np.float32)
    legal_idx = [i for i in range(len(mask)) if mask[i] > 0.5]

    print("Szenario: REINES Unten (Geiss), NN spielt an. Hand:")
    print("  " + ", ".join(str(c) for c in hand))
    print(f"Encoding-Dim: {vec.shape[0]}   legale Karten in Maske: {len(legal_idx)}")
    print("  legal: " + ", ".join(str(index_to_card(i)) for i in legal_idx))

    if args.dry_run:
        print("\n[--dry-run] State/Encoding/Maske sauber gebaut. Kein Modell geladen.")
        return

    # --- Modell laden + Policy abfragen (braucht TensorFlow) ---
    from tensorflow import keras  # noqa: PLC0415  (bewusst lazy, damit --dry-run TF-frei bleibt)

    from training.model import MaskBias  # noqa: F401,PLC0415  (registriert Custom-Layer fuer load_model)

    model = keras.models.load_model(args.model)
    pred = model({"state": vec[np.newaxis, :], "mask": mask[np.newaxis, :]}, training=False)

    def to_np(t):
        return t.numpy() if hasattr(t, "numpy") else np.asarray(t)

    if isinstance(pred, dict):
        probs = to_np(pred["policy"])[0]
    elif isinstance(pred, (list, tuple)):
        probs = to_np(pred[0])[0]
    else:
        probs = to_np(pred)[0]

    ranked = sorted(legal_idx, key=lambda i: float(probs[i]), reverse=True)
    print(f"\nModell: {args.model}")
    print("Policy ueber legale Karten (absteigend) -- die App spielt die oberste (masked argmax):")
    for pos, i in enumerate(ranked, 1):
        c = index_to_card(i)
        if c.suit == Suit.EICHEL and c.rank == Rank.SECHS:
            marker = "   <- die 6 (Boss in Unten = KORREKTES Anspiel)"
        elif c.rank == Rank.ASS:
            marker = "   <- Ass (in Unten schwaechste Karte)"
        else:
            marker = ""
        print(f"  {pos:2d}. {str(c):14s} p={float(probs[i]):.4f}{marker}")

    top = index_to_card(ranked[0])
    print()
    if top.suit == Suit.EICHEL and top.rank == Rank.SECHS:
        print("VERDIKT: Modell waehlt die 6 -> korrekt. Dann liegt der App-Fehler eher")
        print("         im Encoding-Skew (Fixtures fix_006-fix_010 pruefen) als am Modell.")
    elif top.rank == Rank.ASS:
        print("VERDIKT: Modell spielt das ASS an -> ECHTER Lernfehler in reinem Unten.")
        print("         Fix-Richtung: invertierte Modi (unten / gumpf-Seite / slalom_unten)")
        print("         oversamplen + nachtrainieren.")
    else:
        print(f"VERDIKT: Top-Karte ist {top} -- weder 6 noch Ass; Hand/Erwartung pruefen.")


if __name__ == "__main__":
    main()
