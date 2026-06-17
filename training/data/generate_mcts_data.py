"""MCTS-augmentierte Trainings-Daten-Generierung.

Aehnlich wie training/generate_balanced_data.py, aber statt der naiven
HeuristicPlayer-Kartenwahl wird pro Entscheidung ein MCTS-Lite-Lookahead
mit NN-Rollouts gemacht. Das Trainings-Label ist die per Lookahead
ermittelte beste Karte (statt der direkten Heuristik-Karte).

Ansage-Wahl und Weisen-Logik bleiben Heuristik -- nur die Karten-
Entscheidungen werden augmentiert.

Architektur:
- 1 Hauptprozess mit GPU-Inferenz-Server (InferenceServer)
- 64 parallele Game-Threads (ThreadPoolExecutor)
- Pro Stich/Entscheidung: MCTS-Lookahead-Rollouts via Server
- Pro Spiel werden alle (state, mask, lookahead_card)-Samples als
  Transition aufgezeichnet und am Ende als Shard gespeichert.

Aufruf:
    python -m training.data.generate_mcts_data \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 500 \\
        --rollouts-per-card 10 \\
        --output data/mcts_v1 \\
        --parallel-threads 32 \\
        --inference-batch-size 256
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jass_engine.card import Card, Suit
from jass_engine.player import GameState, Player
from jass_engine.variant import Announcement, Variant
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS, play_kreuz_jass
from players.forced_announcement_player import ForcedAnnouncementPlayer
from players.heuristic_player import HeuristicPlayer
from training.data.mcts_lookahead import mcts_lookahead_best_card
from training.data.vectorized_lookahead import compute_card_scores_vectorized
from training.encoder import encode_state, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


REWARD_SCALE = 200.0


@dataclass
class VariantSpec:
    label: str
    announcement: Announcement

    @property
    def output_subdir(self) -> str:
        return self.label


ALL_VARIANTS: list[VariantSpec] = [
    VariantSpec("trumpf_eichel", Announcement(variant=Variant.trumpf(Suit.EICHEL))),
    VariantSpec("trumpf_schelle", Announcement(variant=Variant.trumpf(Suit.SCHELLE))),
    VariantSpec("trumpf_herz", Announcement(variant=Variant.trumpf(Suit.HERZ))),
    VariantSpec("trumpf_laub", Announcement(variant=Variant.trumpf(Suit.LAUB))),
    VariantSpec("gumpf_eichel", Announcement(variant=Variant.gumpf(Suit.EICHEL))),
    VariantSpec("gumpf_schelle", Announcement(variant=Variant.gumpf(Suit.SCHELLE))),
    VariantSpec("gumpf_herz", Announcement(variant=Variant.gumpf(Suit.HERZ))),
    VariantSpec("gumpf_laub", Announcement(variant=Variant.gumpf(Suit.LAUB))),
    VariantSpec("oben", Announcement(variant=Variant.oben())),
    VariantSpec("unten", Announcement(variant=Variant.unten())),
    VariantSpec("slalom_oben", Announcement(variant=Variant.oben(), slalom=True)),
    VariantSpec("slalom_unten", Announcement(variant=Variant.unten(), slalom=True)),
]


class MCTSAugmentedPlayer(Player):
    """Player, der seine Kartenwahl per MCTS-Lookahead trifft und alle
    (state, mask, chosen_card)-Tripel aufzeichnet.

    Ansage und Weisen werden an einen Heuristik-Fallback delegiert.
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        rollouts_per_card: int,
        rng: random.Random,
        fallback_for_announce: Player | None = None,
        lookahead_mode: str = "single-trick",
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.rollouts_per_card = rollouts_per_card
        self.rng = rng
        self.lookahead_mode = lookahead_mode
        self.fallback = fallback_for_announce or HeuristicPlayer(
            name + "_fb", rng=random.Random(rng.randint(0, 10**9))
        )
        # Aufzeichnung der Trainings-Samples
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.player_indices: list[int] = []
        self.round_indices: list[int] = []

    def choose_announcement(self, hand, round_idx, can_push):
        return self.fallback.choose_announcement(hand, round_idx, can_push)

    def announce_weise(self, hand, variant, possible_weise):
        return self.fallback.announce_weise(hand, variant, possible_weise)

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        x = encode_state(hand, state).astype(np.float32)
        mask = legal_action_mask(hand, state).astype(np.uint8)

        if self.lookahead_mode == "full-round-vec":
            scores = compute_card_scores_vectorized(
                hand=hand,
                state=state,
                inference_server=self.inference_server,
                rollouts_per_card=self.rollouts_per_card,
                rng=self.rng,
            )
            chosen = max(scores, key=lambda c: scores[c])
        else:
            # Default: single-trick-Lookahead
            result = mcts_lookahead_best_card(
                hand=hand,
                state=state,
                inference_server=self.inference_server,
                rollouts_per_card=self.rollouts_per_card,
                rng=self.rng,
            )
            chosen = result.best_card

        # Trainings-Sample aufzeichnen
        from training.encoder import card_index
        self.states.append(x)
        self.masks.append(mask)
        self.actions.append(card_index(chosen))
        self.player_indices.append(state.player_idx)
        self.round_indices.append(state.round_idx)
        return chosen


# ---- Game-Setup pro Partie ----


def _play_one_variant_game(
    variant_spec: VariantSpec,
    inference_server: InferenceServer,
    rollouts_per_card: int,
    target_score: int,
    seed: int,
    lookahead_mode: str = "single-trick",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float]]:
    """Spielt EINE Partie mit erzwungener Ansage, gibt
    (states, masks, actions, rewards) zurueck.

    Alle 4 Spieler sind MCTS-augmentiert und sagen `variant_spec.announcement`
    an (via ForcedAnnouncementPlayer im Fallback).
    """
    rng = random.Random(seed)
    teams = list(KREUZ_JASS_TEAMS)

    players: list[Player] = []
    mcts_players: list[MCTSAugmentedPlayer] = []
    for i in range(4):
        sub_rng = random.Random(rng.randint(0, 10**9))
        fallback = ForcedAnnouncementPlayer(
            name=f"F{i}",
            forced_announcement=variant_spec.announcement,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        p = MCTSAugmentedPlayer(
            name=f"M{i}",
            inference_server=inference_server,
            rollouts_per_card=rollouts_per_card,
            rng=sub_rng,
            fallback_for_announce=fallback,
            lookahead_mode=lookahead_mode,
        )
        players.append(p)
        mcts_players.append(p)

    game = play_kreuz_jass(
        players,
        target_score=target_score,
        rng=random.Random(rng.randint(0, 10**9)),
    )

    # Pro Spieler die Round-Rewards zuordnen
    round_rewards_per_round: list[dict[int, float]] = []
    for rnd in game.rounds:
        d: dict[int, float] = {}
        for tid in rnd.team_total_points:
            own_pts = rnd.team_total_points[tid]
            opp_pts = sum(p for t, p in rnd.team_total_points.items() if t != tid)
            d[tid] = (own_pts - opp_pts) / REWARD_SCALE
        round_rewards_per_round.append(d)

    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []

    for p_idx, p in enumerate(mcts_players):
        team_id = teams[p_idx]
        for st, mk, ac, _pi, r_idx in zip(
            p.states, p.masks, p.actions, p.player_indices, p.round_indices
        ):
            reward = (
                round_rewards_per_round[r_idx][team_id]
                if r_idx < len(round_rewards_per_round)
                else 0.0
            )
            all_states.append(st)
            all_masks.append(mk)
            all_actions.append(ac)
            all_rewards.append(reward)

    return all_states, all_masks, all_actions, all_rewards


# ---- Top-level: pro Variante einen Shard schreiben ----


def generate_for_variant(
    output_dir: Path,
    variant_spec: VariantSpec,
    games_per_variant: int,
    rollouts_per_card: int,
    target_score: int,
    inference_server: InferenceServer,
    parallel_threads: int,
    seed: int,
    lookahead_mode: str = "single-trick",
    chunk_idx: int = 0,
    skip_existing: bool = False,
) -> int:
    """Sammelt `games_per_variant` Partien dieser Variante und schreibt einen
    Shard nach output_dir/<variant>/shard_<chunk_idx:05d>.npz.

    Args:
        chunk_idx: Index des zu schreibenden Shards (Default 0). Im Chunk-Queue-
            Modus produziert ein Worker mehrere Shards pro Variante, jeweils mit
            unterschiedlichem chunk_idx.
        skip_existing: Wenn True und die Ziel-Shard-Datei schon existiert, wird
            uebersprungen (Returns 0). Praktisch zum Wiederaufnehmen nach Crash.

    Returns: Anzahl Samples im Shard (0 wenn keine Daten generiert).
    """
    sub_dir = output_dir / variant_spec.output_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

    out_path = sub_dir / f"shard_{chunk_idx:05d}.npz"
    if skip_existing and out_path.exists():
        print(f"    [{variant_spec.label}#{chunk_idx}] Shard existiert schon, ueberspringe.")
        return 0

    base_rng = random.Random(seed)
    seeds = [base_rng.randint(0, 10**9) for _ in range(games_per_variant)]

    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []

    start = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(
        max_workers=parallel_threads,
        thread_name_prefix=f"MCTS-{variant_spec.label}",
    ) as pool:
        futures = [
            pool.submit(
                _play_one_variant_game,
                variant_spec,
                inference_server,
                rollouts_per_card,
                target_score,
                s,
                lookahead_mode,
            )
            for s in seeds
        ]
        for fut in as_completed(futures):
            states, masks, actions, rewards = fut.result()
            all_states.extend(states)
            all_masks.extend(masks)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            completed += 1
            if completed % max(1, games_per_variant // 5) == 0 or completed == games_per_variant:
                elapsed = time.perf_counter() - start
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"    [{variant_spec.label}#{chunk_idx}] {completed}/{games_per_variant} "
                    f"Partien fertig ({rate:.2f}/s, {len(all_states)} Samples)"
                )

    if not all_states:
        return 0

    X = np.stack(all_states).astype(np.float32, copy=False)
    M = np.stack(all_masks).astype(np.uint8, copy=False)
    A = np.array(all_actions, dtype=np.uint8)
    R = np.array(all_rewards, dtype=np.float32)

    np.savez_compressed(out_path, X=X, masks=M, actions=A, rewards=R)
    print(
        f"    [{variant_spec.label}#{chunk_idx}] Shard geschrieben: "
        f"{out_path} ({len(A):,} Samples, {out_path.stat().st_size / 2**20:.1f} MB)"
    )
    return len(A)


# ---- CLI ----


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm-start", type=Path, required=True,
        help="Pfad zum NN-Modell (z.B. models/v5/best.keras), das fuer die "
             "Rollouts genutzt wird.",
    )
    parser.add_argument(
        "--games-per-variant", type=int, default=500,
        help="Wieviele Partien pro Variante. Default 500 (12 Var. = 6000).",
    )
    parser.add_argument(
        "--rollouts-per-card", type=int, default=10,
        help="Wieviele Determinizations pro legaler Karte. Default 10.",
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/mcts_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parallel-threads", type=int, default=32,
        help="Parallele Game-Threads. Sollte ungefaehr inference_batch_size sein.",
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=256,
        help="Max. Batch fuer den InferenceServer. Hoeher = bessere GPU-Auslastung.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Nur bestimmte Varianten (Label). Default: alle 12.",
    )
    parser.add_argument(
        "--lookahead-mode",
        choices=["single-trick", "full-round-vec"],
        default="full-round-vec",
        help=(
            "Lookahead-Tiefe und Vektorisierung:\n"
            "  single-trick    = nur den aktuellen Stich rollouten (schnell, "
            "wenig GPU-Last, schwaecherer Lehrer)\n"
            "  full-round-vec  = bis zum Rundenende rollouten, alle Rollouts "
            "pro Decision parallel als ein Batch (volle GPU-Last, "
            "strategisch besserer Lehrer). Default."
        ),
    )
    args = parser.parse_args()

    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants}.")
            return

    # Modell laden + Inferenz-Server starten (einmal, fuer alle Varianten)
    print(f"Lade Modell: {args.warm_start}")
    from tensorflow import keras
    from training.model import MaskBias  # noqa: F401
    model = keras.models.load_model(str(args.warm_start))
    server = InferenceServer(model, max_batch_size=args.inference_batch_size)
    print(
        f"InferenceServer gestartet (batch <= {args.inference_batch_size}). "
        f"{args.parallel_threads} parallele Game-Threads pro Variante.\n"
        f"Starte MCTS-augmentierte Datengen: {len(selected)} Varianten x "
        f"{args.games_per_variant} Partien = "
        f"{len(selected) * args.games_per_variant} Partien total.\n"
    )

    try:
        total_samples = 0
        t0 = time.perf_counter()
        for vs in selected:
            print(f"\n=== Variante {vs.label} ===")
            n = generate_for_variant(
                output_dir=args.output,
                variant_spec=vs,
                games_per_variant=args.games_per_variant,
                rollouts_per_card=args.rollouts_per_card,
                target_score=args.target,
                inference_server=server,
                parallel_threads=args.parallel_threads,
                seed=args.seed + hash(vs.label) % 10000,
                lookahead_mode=args.lookahead_mode,
            )
            total_samples += n
        elapsed = time.perf_counter() - t0
        print(
            f"\nFertig: {len(selected)} Varianten, {total_samples:,} Samples total. "
            f"Dauer {elapsed / 60:.1f} min."
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
