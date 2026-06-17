"""Batch-Sweep: Schaerfe & Korrektheit der ANSPIEL-Policy in Unten vs Oben.

Erweitert scripts/probe_unten_lead.py von n=1 auf eine Statistik. Zieht N
zufaellige 9-Karten-Haende und baut je eine "ich spiele Stich 0 an"-Stellung
einmal in UNTEN (Geiss) und einmal in OBEN (Bock), fragt das Modell batched ab
und misst pro Modus (Definitionen symmetrisch, damit Unten/Oben fair vergleichbar):

  blunder    % der Stellungen, in denen der (maskierte) argMax die im Modus
             SCHWAECHSTE Karte ist  (Unten: ein Ass, Oben: eine 6).
             -> der klarste "objektiv schlecht"-Indikator. Die App spielt argMax.
  strongmass mittlere Wahrscheinlichkeitsmasse auf den TOP-2-Raengen des Modus
             (Unten: 6/7, Oben: Ass/Koenig). Hoch = Policy konzentriert auf gute Karten.
  badmass    mittlere Masse auf dem schwaechsten Rang (Unten: Ass, Oben: 6). Niedrig = gut.
  mean_top1  mittlere Top-1-Wahrscheinlichkeit ueber legale Karten (Roh-Schaerfe).

Erwartung, falls die Vermutung stimmt: OBEN scharf/sauber, UNTEN weicher und mit
hoeherer blunder-/badmass-Rate  =>  Schwaeche in den invertierten Modi belegt.

Lauf (WSL2, conda-env jass-gpu):
    python -m scripts.probe_lead_sweep
    python -m scripts.probe_lead_sweep --n 1000 --seed 7
    python -m scripts.probe_lead_sweep --dry-run   # ohne TF: nur Stellungs-/Encoding-Check
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from training.encoder import encode_state, index_to_card, legal_action_mask

FULL_DECK = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
HAND_SIZE = 9

# Pro Modus: Variant-Factory, schwaechster Rang (= Blunder), die zwei staerksten Raenge.
MODES = {
    "unten": {"factory": Variant.unten, "worst": Rank.ASS, "strong": {Rank.SECHS, Rank.SIEBEN}},
    "oben": {"factory": Variant.oben, "worst": Rank.SECHS, "strong": {Rank.ASS, Rank.KOENIG}},
}


def lead_state(variant: Variant) -> GameState:
    """Stellung 'ich (Sitz 0) spiele Stich 0 an' -- leerer Stich, alle Handkarten legal."""
    return GameState(
        player_idx=0,
        variant=variant,
        announcement=Announcement(variant=variant, slalom=False),
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
        completed_tricks=[],
        round_idx=0,
        trick_idx=0,
        num_players=4,
    )


def policy_batch(model, x_batch, m_batch):
    pred = model({"state": x_batch, "mask": m_batch}, training=False)
    if isinstance(pred, dict):
        out = pred["policy"]
    elif isinstance(pred, (list, tuple)):
        out = pred[0]
    else:
        out = pred
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/kreuz_mcts3/best.keras")
    ap.add_argument("--n", type=int, default=500, help="Anzahl zufaelliger Haende.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Ohne TF/Modell.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hands = [rng.sample(FULL_DECK, HAND_SIZE) for _ in range(args.n)]

    # Pro Modus: alle States encoden (Batch-Arrays vorbereiten).
    encoded = {}
    for name, cfg in MODES.items():
        st = lead_state(cfg["factory"]())
        x = np.stack([encode_state(h, st).astype(np.float32) for h in hands])
        m = np.stack([legal_action_mask(h, st).astype(np.float32) for h in hands])
        encoded[name] = (x, m)

    print(f"N={args.n} zufaellige Anspiel-Stellungen (Stich 0), Seed={args.seed}.")
    print("Beispiel-Hand:", ", ".join(str(c) for c in sorted(hands[0])))

    if args.dry_run:
        for name in MODES:
            x, m = encoded[name]
            print(f"  {name}: X{tuple(x.shape)} M{tuple(m.shape)}  legale Karten Hand[0]: {int(m[0].sum())}")
        print("\n[--dry-run] States + Encoding gebaut. Kein Modell geladen.")
        return

    from tensorflow import keras  # noqa: PLC0415  (lazy -> --dry-run bleibt TF-frei)

    from training.model import MaskBias  # noqa: F401,PLC0415  (Custom-Layer fuer load_model)

    model = keras.models.load_model(args.model)
    print(f"Modell: {args.model}\n")

    # Einzelfall-Anschauung (Hand[0], Unten), damit man die Verteilung konkret sieht.
    x0, m0 = encoded["unten"]
    p0 = policy_batch(model, x0[:1], m0[:1])[0] * m0[0]
    legal0 = sorted((i for i in range(36) if m0[0, i] > 0.5), key=lambda i: p0[i], reverse=True)
    print("Beispiel (Hand[0], UNTEN) -- Policy ueber legale Karten, absteigend:")
    for i in legal0[:5]:
        print(f"    {str(index_to_card(i)):14s} p={float(p0[i]):.4f}")
    print()

    header = f"{'Modus':6s} {'blunder':>9s} {'strongmass':>11s} {'badmass':>9s} {'mean_top1':>10s}"
    print(header)
    print("-" * len(header))
    for name, cfg in MODES.items():
        x, m = encoded[name]
        worst_i = int(cfg["worst"])
        strong_ints = {int(r) for r in cfg["strong"]}
        is_strong = np.array([(i % 9) in strong_ints for i in range(36)], dtype=np.float32)
        is_worst = np.array([(i % 9) == worst_i for i in range(36)], dtype=np.float32)

        masked = policy_batch(model, x, m) * m                 # (N, 36), nur legale
        arg = masked.argmax(axis=1)                            # was die App spielt
        blunder = float(np.mean((arg % 9) == worst_i)) * 100
        strongmass = float((masked * is_strong).sum(axis=1).mean()) * 100
        badmass = float((masked * is_worst).sum(axis=1).mean()) * 100
        top1 = float(masked.max(axis=1).mean())
        print(f"{name:6s} {blunder:8.1f}% {strongmass:10.1f}% {badmass:8.1f}% {top1:10.4f}")

    print("\nLesehilfe: hohe blunder-/badmass-Rate + niedrige strongmass + niedriges")
    print("mean_top1 in UNTEN gegenueber OBEN  =>  invertierte-Modi-Schwaeche bestaetigt.")


if __name__ == "__main__":
    main()
