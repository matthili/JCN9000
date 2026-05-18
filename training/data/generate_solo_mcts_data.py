"""MCTS-augmentierte Trainingsdaten-Generierung fuer Solo-Jass.

Analoge Pipeline zu `generate_mcts_data.py`, aber:
- play_solo_jass statt play_kreuz_jass (teams=[0,1,2,3], kein Schieben)
- SoloHeuristicPlayer als Ansage-/Fallback-Heuristik
- Solo-Reward: (own_points - max(others_points)) / REWARD_SCALE
  (statt Team-Differenz)
- Variables Spielziel pro Partie (Default: 50/50 zwischen 500 und 1000)
- Solo-vektorisierter Lookahead (separater Modul, fuer Solo-Fine-Tuning)

Aufruf:
    python -m training.data.generate_solo_mcts_data \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 500 \\
        --rollouts-per-card 30 \\
        --output data/solo_mcts/phase1 \\
        --parallel-threads 32 \\
        --inference-batch-size 1024

Tipp:
    `--warm-start` darf das Kreuz-Jass-Modell v0.7.0 sein -- es kennt zwar
    die Solo-Belohnung nicht, gibt aber eine vernuenftige Karten-Policy ab.
    Das Solo-spezifische Verhalten lernt das resultierende NN durch die
    MCTS-Lehrer-Aktionen ueber tausende Spielsituationen.
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
from jass_engine.variants.solo_jass import SOLO_JASS_TEAMS, play_solo_jass
from players.forced_announcement_player import ForcedAnnouncementPlayer
from players.solo_heuristic_player import SoloHeuristicPlayer
from training.data.solo_vectorized_lookahead import (
    compute_card_scores_solo_vectorized,
)
from training.encoder import card_index, encode_state, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


REWARD_SCALE = 200.0


@dataclass
class VariantSpec:
    label: str
    announcement: Announcement

    @property
    def output_subdir(self) -> str:
        return self.label


# 12 Varianten wie bei Kreuz-Jass. Falls sich im echten Solo-Spiel zeigt, dass
# manche Varianten nie/sehr selten angesagt werden, kann der --variants-Filter
# spaeter benutzt werden, um Trainingszeit zu sparen.
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


class SoloMCTSAugmentedPlayer(Player):
    """Player, der seine Kartenwahl per Solo-MCTS-Lookahead trifft und alle
    (state, mask, chosen_card)-Tripel aufzeichnet.

    Ansage und Weisen werden an einen Solo-Heuristik-Fallback delegiert.
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        rollouts_per_card: int,
        rng: random.Random,
        fallback_for_announce: Player | None = None,
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.rollouts_per_card = rollouts_per_card
        self.rng = rng
        self.fallback = fallback_for_announce or SoloHeuristicPlayer(
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

        scores = compute_card_scores_solo_vectorized(
            hand=hand,
            state=state,
            inference_server=self.inference_server,
            rollouts_per_card=self.rollouts_per_card,
            rng=self.rng,
        )
        chosen = max(scores, key=lambda c: scores[c])

        self.states.append(x)
        self.masks.append(mask)
        self.actions.append(card_index(chosen))
        self.player_indices.append(state.player_idx)
        self.round_indices.append(state.round_idx)
        return chosen


def _parse_target_distribution(s: str) -> list[tuple[int, float]]:
    """Parsed "500:0.5,1000:0.5" zu [(500, 0.5), (1000, 0.5)].

    Wirft ValueError, wenn Format kaputt oder Wahrscheinlichkeiten nicht ~1.0
    summieren.
    """
    out: list[tuple[int, float]] = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Eintrag '{entry}' muss Form 'TARGET:WAHRSCHEINLICHKEIT' haben")
        t_str, p_str = entry.split(":", 1)
        out.append((int(t_str.strip()), float(p_str.strip())))
    total = sum(p for _, p in out)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            f"Wahrscheinlichkeiten in --target-distribution summieren zu {total:.3f}, erwartet 1.0"
        )
    return out


def _sample_target(distribution: list[tuple[int, float]], rng: random.Random) -> int:
    """Zieht ein Spielziel aus der gegebenen Verteilung."""
    r = rng.random()
    cum = 0.0
    for target, prob in distribution:
        cum += prob
        if r <= cum:
            return target
    return distribution[-1][0]


def _play_one_variant_game(
    variant_spec: VariantSpec,
    inference_server: InferenceServer,
    rollouts_per_card: int,
    target_distribution: list[tuple[int, float]],
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float]]:
    """Spielt EINE Solo-Partie mit erzwungener Ansage, gibt
    (states, masks, actions, rewards) zurueck.

    Alle 4 Spieler sind MCTS-augmentiert und sagen `variant_spec.announcement`
    an. Spielziel wird pro Partie aus `target_distribution` gezogen.
    """
    rng = random.Random(seed)
    target_score = _sample_target(target_distribution, rng)
    teams = list(SOLO_JASS_TEAMS)

    players: list[Player] = []
    mcts_players: list[SoloMCTSAugmentedPlayer] = []
    for i in range(4):
        sub_rng = random.Random(rng.randint(0, 10**9))
        fallback = ForcedAnnouncementPlayer(
            name=f"F{i}",
            forced_announcement=variant_spec.announcement,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        p = SoloMCTSAugmentedPlayer(
            name=f"M{i}",
            inference_server=inference_server,
            rollouts_per_card=rollouts_per_card,
            rng=sub_rng,
            fallback_for_announce=fallback,
        )
        players.append(p)
        mcts_players.append(p)

    game = play_solo_jass(
        players,
        target_score=target_score,
        rng=random.Random(rng.randint(0, 10**9)),
    )

    # Solo-Reward pro Runde: pro Spieler eigene Punkte minus staerkster Gegner
    round_rewards_per_round: list[dict[int, float]] = []
    for rnd in game.rounds:
        d: dict[int, float] = {}
        all_tids = list(rnd.team_total_points.keys())
        for tid in all_tids:
            own_pts = rnd.team_total_points[tid]
            others = [
                pts for other_tid, pts in rnd.team_total_points.items()
                if other_tid != tid
            ]
            opp_pts = max(others) if others else 0
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


def generate_for_variant(
    output_dir: Path,
    variant_spec: VariantSpec,
    games_per_variant: int,
    rollouts_per_card: int,
    target_distribution: list[tuple[int, float]],
    inference_server: InferenceServer,
    parallel_threads: int,
    seed: int,
) -> int:
    """Sammelt `games_per_variant` Solo-Partien dieser Variante und schreibt
    einen Shard nach output_dir/<variant>/shard_00000.npz.

    Returns: Anzahl Samples im Shard.
    """
    sub_dir = output_dir / variant_spec.output_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

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
        thread_name_prefix=f"SoloMCTS-{variant_spec.label}",
    ) as pool:
        futures = [
            pool.submit(
                _play_one_variant_game,
                variant_spec,
                inference_server,
                rollouts_per_card,
                target_distribution,
                s,
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
                    f"    [solo:{variant_spec.label}] {completed}/{games_per_variant} "
                    f"Partien fertig ({rate:.2f}/s, {len(all_states)} Samples)"
                )

    if not all_states:
        return 0

    X = np.stack(all_states).astype(np.float32, copy=False)
    M = np.stack(all_masks).astype(np.uint8, copy=False)
    A = np.array(all_actions, dtype=np.uint8)
    R = np.array(all_rewards, dtype=np.float32)

    out_path = sub_dir / "shard_00000.npz"
    np.savez_compressed(out_path, X=X, masks=M, actions=A, rewards=R)
    print(
        f"    [solo:{variant_spec.label}] Shard geschrieben: "
        f"{out_path} ({len(A):,} Samples, {out_path.stat().st_size / 2**20:.1f} MB)"
    )
    return len(A)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm-start", type=Path, required=True,
        help="Pfad zum NN-Modell (z.B. models/v5/best.keras) als Rollout-Initialisierung.",
    )
    parser.add_argument(
        "--games-per-variant", type=int, default=500,
        help="Wieviele Partien pro Variante. Default 500 (12 Var. = 6000).",
    )
    parser.add_argument(
        "--rollouts-per-card", type=int, default=30,
        help="Wieviele Determinizations pro legaler Karte. Default 30.",
    )
    parser.add_argument(
        "--target-distribution",
        type=str,
        default="500:0.5,1000:0.5",
        help=(
            "Verteilung der Spielziele pro Trainings-Partie. "
            "Format: 'TARGET:PROB,TARGET:PROB,...'. "
            "Default '500:0.5,1000:0.5' (50/50 zwischen den beiden gaengigen Zielen). "
            "Wahrscheinlichkeiten muessen zu 1.0 summieren."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/solo_mcts/phase1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parallel-threads", type=int, default=32,
        help="Parallele Game-Threads im Hauptprozess.",
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=1024,
        help="Max. Batch fuer den InferenceServer.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Nur bestimmte Varianten (Label). Default: alle 12.",
    )
    args = parser.parse_args()

    target_distribution = _parse_target_distribution(args.target_distribution)

    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants}.")
            return

    print(f"Lade Modell: {args.warm_start}")
    from tensorflow import keras
    from training.model import MaskBias  # noqa: F401

    model = keras.models.load_model(str(args.warm_start))
    server = InferenceServer(model, max_batch_size=args.inference_batch_size)
    print(
        f"InferenceServer gestartet (batch <= {args.inference_batch_size}). "
        f"{args.parallel_threads} parallele Game-Threads pro Variante.\n"
        f"Solo-MCTS-Datengen: {len(selected)} Varianten x "
        f"{args.games_per_variant} Partien = "
        f"{len(selected) * args.games_per_variant} Partien total.\n"
        f"Spielziel-Verteilung: {target_distribution}\n"
    )

    try:
        total_samples = 0
        t0 = time.perf_counter()
        for vs in selected:
            print(f"\n=== Solo-Variante {vs.label} ===")
            n = generate_for_variant(
                output_dir=args.output,
                variant_spec=vs,
                games_per_variant=args.games_per_variant,
                rollouts_per_card=args.rollouts_per_card,
                target_distribution=target_distribution,
                inference_server=server,
                parallel_threads=args.parallel_threads,
                seed=args.seed + hash(vs.label) % 10000,
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
