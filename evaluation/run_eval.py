"""CLI: laesst Player-Typen gegeneinander spielen und gibt Elo + Statistiken aus.

Beispiele:
    # 100 Partien Heuristic vs Random
    python -m evaluation.run_eval --a heuristic --b random --games 100

    # 200 Partien Heuristic vs Heuristic (Sitz-Symmetrie-Check)
    python -m evaluation.run_eval --a heuristic --b heuristic --games 200

    # NN gegen Heuristic
    python -m evaluation.run_eval --a nn --b heuristic --games 50 \
        --model models/v2/best.keras

Optional --save-elo PATH speichert das Elo-State, sodass spaetere
Tournaments die Ratings akkumulieren koennen.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from evaluation.elo import EloRating
from evaluation.tournament import format_tournament_summary, two_team_match
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
                        help="NN-Modell-Pfad (nur bei --a nn oder --b nn)")
    parser.add_argument("--save-elo", type=Path, default=None)
    parser.add_argument("--load-elo", type=Path, default=None)
    parser.add_argument("--no-swap-seats", action="store_true",
                        help="Sitzplatz-Tausch in der zweiten Haelfte deaktivieren")
    args = parser.parse_args()

    factory_a = _make_factory(args.a, args.model)
    factory_b = _make_factory(args.b, args.model)
    label_a = args.a.capitalize() + "_A"
    label_b = args.b.capitalize() + "_B"

    elo = EloRating.load_json(args.load_elo) if args.load_elo else EloRating()

    print(f"Tournament: {label_a} vs. {label_b}  ({args.games} Partien)")
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
    )
    elapsed = time.perf_counter() - start

    print(format_tournament_summary(result))
    print(f"\nDauer: {elapsed:.1f} s  ({elapsed / args.games * 1000:.1f} ms/Partie)")

    if args.save_elo:
        elo.save_json(args.save_elo)
        print(f"Elo-State gespeichert: {args.save_elo}")


if __name__ == "__main__":
    main()
