"""CLI: laesst Player-Typen gegeneinander spielen und gibt Elo + Statistiken aus.

Beispiele:
    # 100 Partien Heuristic vs Random
    python -m evaluation.run_eval --a heuristic --b random --games 100

    # 200 Partien Heuristic vs Heuristic (Sitz-Symmetrie-Check)
    python -m evaluation.run_eval --a heuristic --b heuristic --games 200

    # NN gegen Heuristic (Modell aus --model fuer Team A)
    python -m evaluation.run_eval --a nn --b heuristic --games 50 \
        --model models/v3/best.keras

    # Zwei NNs gegeneinander (alt vs neu) -- benoetigt --model-a und --model-b
    python -m evaluation.run_eval --a nn --b nn --games 100 \
        --model-a models/v1/best.keras --model-b models/v3/best.keras

Optional --save-elo PATH speichert das Elo-State, sodass spaetere
Tournaments die Ratings akkumulieren koennen.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from evaluation.batched_eval import two_team_match_batched_gpu
from evaluation.elo import EloRating
from evaluation.tournament import (
    format_tournament_summary,
    two_team_match,
    two_team_match_parallel,
)
from jass_engine.player import Player
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer


def _factory_random(seat: int, rng: random.Random) -> Player:
    return RandomPlayer(name=f"R{seat}", rng=rng)


def _factory_heuristic(seat: int, rng: random.Random) -> Player:
    return HeuristicPlayer(name=f"H{seat}", rng=rng)


def _make_factory(kind: str, model_path: Path | None):
    if kind == "random":
        return _factory_random
    if kind == "heuristic":
        return _factory_heuristic
    if kind == "nn":
        if model_path is None or not model_path.exists():
            sys.exit(f"NN-Player braucht --model, Pfad existiert nicht: {model_path}")
        # Lazy import - tensorflow ist eine schwere Abhaengigkeit
        from players.nn_player import NNPlayer

        # Einmal laden, mehrfach verwenden
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401

        shared_model = keras.models.load_model(str(model_path))

        def _factory(seat: int, rng: random.Random) -> Player:
            from jass_engine.player import Player as _P
            # Player-Subclass, der das geladene Modell wiederverwendet
            class _FastNN(NNPlayer):
                def __init__(self, name: str):
                    _P.__init__(self, name)
                    self.model = shared_model
                    self.fallback = HeuristicPlayer(name + "_fb")
                    self.greedy = True
            return _FastNN(name=f"NN{seat}")
        return _factory
    sys.exit(f"Unbekannter Player-Typ: {kind} (random | heuristic | nn)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, choices=["random", "heuristic", "nn"],
                        help="Team-A-Spielertyp")
    parser.add_argument("--b", required=True, choices=["random", "heuristic", "nn"],
                        help="Team-B-Spielertyp")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--target", type=int, default=1000, help="Punkteziel pro Partie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=None,
                        help="Gemeinsamer NN-Modell-Pfad (wenn nur ein NN benoetigt wird "
                             "oder beide NN-Teams dasselbe Modell verwenden)")
    parser.add_argument("--model-a", type=Path, default=None,
                        help="NN-Modell fuer Team A (ueberschreibt --model)")
    parser.add_argument("--model-b", type=Path, default=None,
                        help="NN-Modell fuer Team B (ueberschreibt --model)")
    parser.add_argument("--save-elo", type=Path, default=None)
    parser.add_argument("--load-elo", type=Path, default=None)
    parser.add_argument("--no-swap-seats", action="store_true",
                        help="Sitzplatz-Tausch in der zweiten Haelfte deaktivieren")
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Anzahl paralleler Subprocesses fuer die Spiele-Simulation. "
            "Default 1 = sequenziell mit Elo-Update. Bei >1 wird jeder Worker "
            "mit CPU-only-TF gestartet (kein GPU-Konflikt) und Elo deaktiviert. "
            "Empfehlung fuer NN-Eval auf >=16-Kern-CPU: --workers 8 oder 16."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=["sequential", "cpu-workers", "batched-gpu"],
        default="sequential",
        help=(
            "Wie die Inferenzen ablaufen:\n"
            "  sequential   = klassisch, ein Process, eine Inferenz pro Stich\n"
            "  cpu-workers  = N Subprocesses mit CPU-only-TF (entspricht --workers > 1)\n"
            "  batched-gpu  = ein Process, viele Game-Threads + GPU-Inferenz-Server\n"
            "                 mit Batch-Inferenz (analog zum RL-batched-Mode).\n"
            "                 Schnellster Modus fuer NN-Eval auf einer GPU."
        ),
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=64,
        help="Nur fuer --inference-mode batched-gpu: max. Server-Batch-Groesse.",
    )
    parser.add_argument(
        "--parallel-threads", type=int, default=128,
        help=(
            "Nur fuer --inference-mode batched-gpu: max. gleichzeitig spielende "
            "Game-Threads. Sollte ungefaehr inference_batch_size sein."
        ),
    )
    parser.add_argument(
        "--paired-eval", action="store_true",
        help=(
            "Gepaarte Bewertung: pro Paar zwei Spiele mit IDENTISCHER "
            "Kartenverteilung -- einmal Modell A auf Sitzen 0+2, einmal auf "
            "1+3. Eliminiert das Karten-Glueck als Rauschquelle und macht "
            "kleine Staerkenunterschiede zwischen Modellen sichtbar. "
            "Setzt voraus, dass --games gerade ist (sonst Fehler)."
        ),
    )
    args = parser.parse_args()

    model_a = args.model_a if args.model_a is not None else args.model
    model_b = args.model_b if args.model_b is not None else args.model

    # Bei zwei NN-Modellen das Modell-Name im Label kenntlich machen
    def _label(kind: str, model_path: Path | None, suffix: str) -> str:
        if kind == "nn" and model_path is not None:
            return f"NN({model_path.parent.name})_{suffix}"
        return kind.capitalize() + f"_{suffix}"
    label_a = _label(args.a, model_a, "A")
    label_b = _label(args.b, model_b, "B")

    # Inference-Mode-Verzweigung. --workers > 1 ist Abkuerzung fuer cpu-workers.
    effective_mode = args.inference_mode
    if effective_mode == "sequential" and args.workers > 1:
        # Rückwärtskompatibel: alte Aufrufe mit --workers > 1 ohne expliziten
        # inference_mode landen auf cpu-workers.
        effective_mode = "cpu-workers"

    if effective_mode == "batched-gpu":
        if args.save_elo or args.load_elo:
            print(
                "[warn] --save-elo / --load-elo werden im batched-gpu-Modus "
                "ignoriert."
            )
        paired_str = ", paired-eval" if args.paired_eval else ""
        print(
            f"Tournament: {label_a} vs. {label_b}  "
            f"({args.games} Partien, batched-gpu{paired_str}, "
            f"batch <= {args.inference_batch_size}, "
            f"threads = {args.parallel_threads})"
        )
        start = time.perf_counter()
        result = two_team_match_batched_gpu(
            label_a=label_a,
            kind_a=args.a,
            model_a=model_a,
            label_b=label_b,
            kind_b=args.b,
            model_b=model_b,
            num_games=args.games,
            target_score=args.target,
            seed=args.seed,
            swap_seats_each_half=not args.no_swap_seats,
            inference_batch_size=args.inference_batch_size,
            parallel_threads=args.parallel_threads,
            paired_eval=args.paired_eval,
        )
        elapsed = time.perf_counter() - start
        print(format_tournament_summary(result))
        print(
            f"\nDauer: {elapsed:.1f} s  "
            f"({elapsed / args.games * 1000:.1f} ms/Partie, batched-gpu)"
        )
        return

    if effective_mode == "cpu-workers":
        if args.save_elo or args.load_elo:
            print(
                "[warn] --save-elo / --load-elo werden im cpu-workers-Modus "
                "ignoriert."
            )
        paired_str = ", paired-eval" if args.paired_eval else ""
        print(
            f"Tournament: {label_a} vs. {label_b}  "
            f"({args.games} Partien, {args.workers} Worker{paired_str})"
        )
        start = time.perf_counter()
        result = two_team_match_parallel(
            label_a=label_a,
            kind_a=args.a,
            model_a=model_a,
            label_b=label_b,
            kind_b=args.b,
            model_b=model_b,
            num_games=args.games,
            workers=args.workers,
            target_score=args.target,
            seed=args.seed,
            swap_seats_each_half=not args.no_swap_seats,
            paired_eval=args.paired_eval,
        )
        elapsed = time.perf_counter() - start
        print(format_tournament_summary(result))
        print(
            f"\nDauer: {elapsed:.1f} s  "
            f"({elapsed / args.games * 1000:.1f} ms/Partie, {args.workers} Worker)"
        )
        return

    # Sequenzieller Pfad mit Elo
    factory_a = _make_factory(args.a, model_a)
    factory_b = _make_factory(args.b, model_b)
    elo = EloRating.load_json(args.load_elo) if args.load_elo else EloRating()

    paired_str = ", paired-eval" if args.paired_eval else ""
    print(f"Tournament: {label_a} vs. {label_b}  ({args.games} Partien{paired_str})")
    start = time.perf_counter()
    result = two_team_match(
        label_a=label_a,
        factory_a=factory_a,
        label_b=label_b,
        factory_b=factory_b,
        num_games=args.games,
        target_score=args.target,
        seed=args.seed,
        swap_seats_each_half=not args.no_swap_seats,
        elo=elo,
        paired_eval=args.paired_eval,
    )
    elapsed = time.perf_counter() - start

    print(format_tournament_summary(result))
    print(f"\nDauer: {elapsed:.1f} s  ({elapsed / args.games * 1000:.1f} ms/Partie)")

    if args.save_elo:
        elo.save_json(args.save_elo)
        print(f"Elo-State gespeichert: {args.save_elo}")


if __name__ == "__main__":
    main()
