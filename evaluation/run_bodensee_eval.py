"""CLI: Bodensee-Jass-Eval -- zwei Spieler gegeneinander.

Beispiele:
    # Bodensee-Phase-2-Modell gegen Phase-1-Modell
    python -m evaluation.run_bodensee_eval \\
        --a nn --model-a models/bodensee_phase2/best.keras \\
        --b nn --model-b models/bodensee_phase1/best.keras \\
        --games 2000 --paired-eval --inference-mode batched-gpu

    # NN gegen Heuristik
    python -m evaluation.run_bodensee_eval \\
        --a nn --model-a models/bodensee_phase2/best.keras \\
        --b heuristic --games 2000 --paired-eval --inference-mode batched-gpu

    # Heuristik vs Heuristik (Sanity-Check: ~50/50 erwartet)
    python -m evaluation.run_bodensee_eval --a heuristic --b heuristic --games 200
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from evaluation.bodensee_eval import (
    format_bodensee_eval_summary,
    two_player_match,
)
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_player import BodenseePlayer


def _factory_heuristic(seat: int, rng: random.Random) -> BodenseePlayer:
    return BodenseeHeuristicPlayer(name=f"H{seat}", rng=rng)


def _make_sequential_factory(kind: str, model_path: Path | None):
    """Factory fuer den sequentiellen Modus. NN wird hier nicht unterstuetzt --
    fuer NN-Eval bitte --inference-mode batched-gpu verwenden."""
    if kind == "heuristic":
        return _factory_heuristic
    if kind == "nn":
        sys.exit(
            "NN-Eval im sequentiellen Modus ist nicht implementiert. "
            "Bitte --inference-mode batched-gpu verwenden."
        )
    sys.exit(f"Unbekannter Player-Typ: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, choices=["heuristic", "nn"])
    parser.add_argument("--b", required=True, choices=["heuristic", "nn"])
    parser.add_argument("--model-a", type=Path, default=None)
    parser.add_argument("--model-b", type=Path, default=None)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paired-eval", action="store_true",
        help="Pro Kartenverteilung 2 Partien mit gespiegelten Sitzen.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["sequential", "batched-gpu"],
        default="sequential",
        help=(
            "sequential   = nur fuer Heuristik vs Heuristik.\n"
            "batched-gpu  = noetig sobald ein NN-Spieler beteiligt ist."
        ),
    )
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--parallel-threads", type=int, default=64)
    args = parser.parse_args()

    if args.paired_eval and args.games % 2 != 0:
        sys.exit(f"--paired-eval verlangt gerade --games (uebergeben: {args.games}).")

    # NN beteiligt -> batched-gpu erzwingen
    nn_involved = args.a == "nn" or args.b == "nn"
    if nn_involved and args.inference_mode != "batched-gpu":
        print(
            "[Hinweis] NN-Spieler beteiligt -> wechsle automatisch auf "
            "--inference-mode batched-gpu."
        )
        args.inference_mode = "batched-gpu"

    def _label(kind: str, model_path: Path | None, suffix: str) -> str:
        if kind == "nn" and model_path is not None:
            return f"NN({model_path.parent.name})_{suffix}"
        return kind.capitalize() + f"_{suffix}"

    label_a = _label(args.a, args.model_a, "A")
    label_b = _label(args.b, args.model_b, "B")

    paired_str = ", paired-eval" if args.paired_eval else ""
    print(
        f"Bodensee-Tournament: {label_a} vs. {label_b}  "
        f"({args.games} Partien{paired_str}, {args.inference_mode})"
    )

    start = time.perf_counter()
    if args.inference_mode == "batched-gpu":
        from evaluation.batched_bodensee_eval import two_player_match_batched_gpu
        result = two_player_match_batched_gpu(
            label_a=label_a, kind_a=args.a, model_a=args.model_a,
            label_b=label_b, kind_b=args.b, model_b=args.model_b,
            num_games=args.games,
            target_score=args.target,
            seed=args.seed,
            paired_eval=args.paired_eval,
            inference_batch_size=args.inference_batch_size,
            parallel_threads=args.parallel_threads,
        )
    else:
        factory_a = _make_sequential_factory(args.a, args.model_a)
        factory_b = _make_sequential_factory(args.b, args.model_b)
        result = two_player_match(
            label_a=label_a, factory_a=factory_a,
            label_b=label_b, factory_b=factory_b,
            num_games=args.games,
            target_score=args.target,
            seed=args.seed,
            paired_eval=args.paired_eval,
        )
    elapsed = time.perf_counter() - start

    print()
    print(format_bodensee_eval_summary(result))
    print()
    print(f"Dauer: {elapsed:.1f} s  ({elapsed / args.games * 1000:.1f} ms/Partie)")


if __name__ == "__main__":
    main()
