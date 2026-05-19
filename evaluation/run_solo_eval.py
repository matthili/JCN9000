"""CLI: Solo-Jass-Eval mit zwei NN-Modellen (A, B) gegen zwei Heuristik-Bots.

Beispiele:
    # Solo-Phase-1-Modell gegen Solo-Phase-2-Modell vergleichen
    python -m evaluation.run_solo_eval \\
        --model-a models/solo_phase1/best.keras \\
        --model-b models/solo_phase2/best.keras \\
        --games 800 \\
        --paired-eval

    # Heuristik gegen Heuristik (Sanity-Check: ~25% pro Rolle erwartet)
    python -m evaluation.run_solo_eval --a heuristic --b heuristic --games 400
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from evaluation.solo_eval import (
    format_solo_eval_summary,
    four_way_match,
)
from jass_engine.player import Player
from players.solo_heuristic_player import SoloHeuristicPlayer
from players.random_player import RandomPlayer


def _factory_random(seat: int, rng: random.Random) -> Player:
    return RandomPlayer(name=f"R{seat}", rng=rng)


def _factory_solo_heuristic(seat: int, rng: random.Random) -> Player:
    return SoloHeuristicPlayer(name=f"H{seat}", rng=rng)


def _make_factory(kind: str, model_path: Path | None):
    if kind == "random":
        return _factory_random
    if kind == "heuristic":
        return _factory_solo_heuristic
    if kind == "nn":
        if model_path is None or not model_path.exists():
            sys.exit(f"NN-Player braucht --model, Pfad existiert nicht: {model_path}")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        from players.nn_player import NNPlayer
        from jass_engine.player import Player as _P
        from players.solo_heuristic_player import SoloHeuristicPlayer as _SH

        shared_model = keras.models.load_model(str(model_path))

        def _factory(seat: int, rng: random.Random) -> Player:
            class _FastSoloNN(NNPlayer):
                def __init__(self, name: str):
                    _P.__init__(self, name)
                    self.model = shared_model
                    # Wichtig: Fallback ist die Solo-Heuristik, nicht die Team-Variante
                    self.fallback = _SH(name + "_fb")
                    self.greedy = True
            return _FastSoloNN(name=f"NN{seat}")
        return _factory
    sys.exit(f"Unbekannter Player-Typ: {kind} (random | heuristic | nn)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a", required=True, choices=["random", "heuristic", "nn"],
        help="Player-Typ fuer Rolle A",
    )
    parser.add_argument(
        "--b", required=True, choices=["random", "heuristic", "nn"],
        help="Player-Typ fuer Rolle B",
    )
    parser.add_argument(
        "--model-a", type=Path, default=None,
        help="NN-Modell fuer Rolle A (wenn --a nn)",
    )
    parser.add_argument(
        "--model-b", type=Path, default=None,
        help="NN-Modell fuer Rolle B (wenn --b nn)",
    )
    parser.add_argument(
        "--games", type=int, default=400,
        help="Anzahl Partien. Bei --paired-eval muss durch 4 teilbar sein.",
    )
    parser.add_argument(
        "--target", type=int, default=500,
        help="Spielziel pro Partie (Default 500).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paired-eval", action="store_true",
        help=(
            "Pro Kartenverteilung 4 Partien mit zyklischer Sitz-Rotation. "
            "Eliminiert Karten-Glueck und Sitz-Vorteile als Rauschquelle."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=["sequential", "batched-gpu"],
        default="sequential",
        help=(
            "sequential   = klassisch, ein Spiel nach dem anderen mit "
            "model(x)-Inferenz pro Karte (Default).\n"
            "batched-gpu  = viele Game-Threads + GPU-Inferenz-Server mit "
            "Batch-Inferenz. Empfohlen fuer >= 1000 Spiele."
        ),
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=64,
        help="Nur fuer batched-gpu: max. Server-Batch-Groesse.",
    )
    parser.add_argument(
        "--parallel-threads", type=int, default=64,
        help="Nur fuer batched-gpu: max. gleichzeitig spielende Game-Threads.",
    )
    args = parser.parse_args()

    # Bei paired-eval --games durch 4 teilbar?
    if args.paired_eval and args.games % 4 != 0:
        sys.exit(
            f"--paired-eval verlangt --games als Vielfaches von 4 (uebergeben: {args.games})."
        )

    def _label(kind: str, model_path: Path | None, suffix: str) -> str:
        if kind == "nn" and model_path is not None:
            return f"NN({model_path.parent.name})_{suffix}"
        return kind.capitalize() + f"_{suffix}"

    label_a = _label(args.a, args.model_a, "A")
    label_b = _label(args.b, args.model_b, "B")
    label_h = "SoloHeuristic"

    paired_str = ", paired-eval" if args.paired_eval else ""
    mode_str = (
        f", batched-gpu (batch <= {args.inference_batch_size}, "
        f"threads = {args.parallel_threads})"
        if args.inference_mode == "batched-gpu"
        else ""
    )
    print(
        f"Solo-Tournament: {label_a} vs. {label_b}  "
        f"(+ 2x {label_h}, {args.games} Partien{paired_str}{mode_str})"
    )

    start = time.perf_counter()
    if args.inference_mode == "batched-gpu":
        from evaluation.batched_solo_eval import four_way_match_batched_gpu
        result = four_way_match_batched_gpu(
            label_a=label_a,
            kind_a=args.a,
            model_a=args.model_a,
            label_b=label_b,
            kind_b=args.b,
            model_b=args.model_b,
            label_h=label_h,
            num_games=args.games,
            target_score=args.target,
            seed=args.seed,
            paired_eval=args.paired_eval,
            inference_batch_size=args.inference_batch_size,
            parallel_threads=args.parallel_threads,
        )
    else:
        factory_a = _make_factory(args.a, args.model_a)
        factory_b = _make_factory(args.b, args.model_b)
        factory_h = _factory_solo_heuristic
        result = four_way_match(
            label_a=label_a,
            factory_a=factory_a,
            label_b=label_b,
            factory_b=factory_b,
            label_h=label_h,
            factory_h=factory_h,
            num_games=args.games,
            target_score=args.target,
            seed=args.seed,
            paired_eval=args.paired_eval,
        )
    elapsed = time.perf_counter() - start

    print()
    print(format_solo_eval_summary(result))
    print()
    print(
        f"Dauer: {elapsed:.1f} s  "
        f"({elapsed / args.games * 1000:.1f} ms/Partie)"
    )


if __name__ == "__main__":
    main()
