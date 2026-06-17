"""Heuristik-Bootstrap-Datengen fuer Bodensee-Jass.

Damit man eine Bodensee-MCTS-Pipeline starten kann, braucht man ein erstes
Bodensee-Modell (das v0.7.0/v0.8.0-Kreuz/Solo-Modell hat den falschen
Encoder-Dim 421 vs Bodensee 291). Dieses Skript liefert die "Phase 0"-Daten:

- Zwei BodenseeHeuristicPlayer spielen N Partien gegeneinander
- Vor jeder Karten-Entscheidung wird der Encoder-Input + die Maske aufgezeichnet
- Die vom Heuristik-Player gewaehlte Karte ist das Trainings-Label
- Reward = (eigene_runden_punkte - gegner_punkte) / 200

Aus den Daten kann man dann ein erstes Bodensee-NN trainieren, das spaeter
als Warm-Start fuer die MCTS-Phase 1 dient.

Parallelisierung: echtes Multiprocessing. Dieser Schritt hat keine
NN-Inferenz -- es ist reine Python-Spiellogik. Threading wuerde wegen des
Python-GIL nichts bringen (alle Threads liefen auf einem Kern). Mit
Multiprocessing laeuft jeder Worker in einem eigenen Interpreter und nutzt
einen echten CPU-Kern.

Aufruf:
    python -u -m training.data.generate_bodensee_heuristic_data \\
        --games 10000 \\
        --output data/bodensee_heuristic_bootstrap \\
        --target-distribution "500:0.5,1000:0.5" \\
        --workers 12
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from functools import partial
from pathlib import Path

import numpy as np

from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_player import BodenseePlayer
from training.bodensee_encoder import (
    card_index,
    encode_state_bodensee,
    legal_action_mask_bodensee,
)
from training.data.generate_bodensee_mcts_data import (
    _build_synthetic_own_state,
    _parse_target_distribution,
    _sample_target,
)


REWARD_SCALE = 200.0


class BodenseeRecordingHeuristicPlayer(BodenseePlayer):
    """Bodensee-Heuristik-Spieler, der zusaetzlich (state, mask, gewaehlte_karte)
    fuer jede Karten-Entscheidung aufzeichnet -- zum Trainieren des ersten
    Bodensee-NNs."""

    def __init__(
        self,
        name: str,
        rng: random.Random | None = None,
    ):
        super().__init__(name)
        self.heuristic = BodenseeHeuristicPlayer(
            name + "_h",
            rng=rng if rng is not None else random.Random(),
        )
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.round_indices: list[int] = []
        self._i_am_announcer_current_round: bool = False

    def set_announcer_flag(self, value: bool) -> None:
        self._i_am_announcer_current_round = value

    def choose_announcement(self, hand, visible_table, round_idx):
        return self.heuristic.choose_announcement(hand, visible_table, round_idx)

    def choose_card(self, hand, visible_table, state: BodenseeGameState) -> Card:
        # Encoder-Vektor + Maske aufzeichnen
        synth_state = _build_synthetic_own_state(
            hand=hand,
            visible_table=visible_table,
            own_hidden_count=state.own_hidden_table_count,
        )
        x = encode_state_bodensee(
            hand=hand,
            own_table_stacks=synth_state.table,
            state=state,
            i_am_announcer=self._i_am_announcer_current_round,
        ).astype(np.float32)
        mask = legal_action_mask_bodensee(hand, visible_table, state).astype(np.uint8)

        # Heuristik waehlt
        chosen = self.heuristic.choose_card(hand, visible_table, state)

        self.states.append(x)
        self.masks.append(mask)
        self.actions.append(card_index(chosen))
        self.round_indices.append(state.round_idx)
        return chosen


def _play_one_game(
    seed: int,
    target_distribution: list[tuple[int, float]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float]]:
    """Spielt EINE Bodensee-Heuristik-Partie und gibt die aufgezeichneten Samples
    plus berechnete Rewards zurueck."""
    from jass_engine.bodensee.round import play_bodensee_round

    rng = random.Random(seed)
    target_score = _sample_target(target_distribution, rng)

    p0 = BodenseeRecordingHeuristicPlayer(
        "P0", rng=random.Random(rng.randint(0, 10**9))
    )
    p1 = BodenseeRecordingHeuristicPlayer(
        "P1", rng=random.Random(rng.randint(0, 10**9))
    )
    players = [p0, p1]

    cumulative = {0: 0, 1: 0}
    last_announcer: int | None = None
    rounds_data: list[dict] = []
    max_rounds = 200

    for round_idx in range(max_rounds):
        forced_announcer = None
        if round_idx > 0 and last_announcer is not None:
            forced_announcer = 1 - last_announcer

        # Announcer-Flag pragmatisch auf False -- wird in der Heuristik nicht
        # genutzt; nur fuer den Encoder-Input ein zusaetzliches Bit
        for p in players:
            p.set_announcer_flag(False)

        result = play_bodensee_round(
            players=players,  # type: ignore[arg-type]
            rng=random.Random(rng.randint(0, 10**9)),
            forced_announcer_idx=forced_announcer,
            initial_scores=(cumulative[0], cumulative[1]),
            round_idx=round_idx,
        )
        last_announcer = result.announcer_idx
        for pid in (0, 1):
            cumulative[pid] += result.player_total_points[pid]
        rounds_data.append({
            "player_total_points": result.player_total_points,
        })

        if any(s >= target_score for s in cumulative.values()):
            break

    # Reward pro Runde aus Sicht jedes Spielers
    round_rewards: list[dict[int, float]] = []
    for r in rounds_data:
        p0_pts = r["player_total_points"][0]
        p1_pts = r["player_total_points"][1]
        round_rewards.append({
            0: (p0_pts - p1_pts) / REWARD_SCALE,
            1: (p1_pts - p0_pts) / REWARD_SCALE,
        })

    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []

    for p_idx, p in enumerate(players):
        for st, mk, ac, r_idx in zip(p.states, p.masks, p.actions, p.round_indices):
            reward = round_rewards[r_idx][p_idx] if r_idx < len(round_rewards) else 0.0
            all_states.append(st)
            all_masks.append(mk)
            all_actions.append(ac)
            all_rewards.append(reward)

    return all_states, all_masks, all_actions, all_rewards


def _write_shard(
    output_dir: Path,
    chunk_idx: int,
    states: list[np.ndarray],
    masks: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
) -> Path:
    X = np.stack(states).astype(np.float32, copy=False)
    M = np.stack(masks).astype(np.uint8, copy=False)
    A = np.array(actions, dtype=np.uint8)
    R = np.array(rewards, dtype=np.float32)
    out_path = output_dir / f"shard_{chunk_idx:05d}.npz"
    np.savez_compressed(out_path, X=X, masks=M, actions=A, rewards=R)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap-Datengen fuer Bodensee-Jass (Heuristik vs Heuristik).",
    )
    parser.add_argument("--games", type=int, default=5000,
                        help="Wieviele Partien. Default 5000.")
    parser.add_argument(
        "--games-per-shard", type=int, default=100,
        help=(
            "Wieviele Partien pro Output-Shard-Datei. Default 100. Kleinere "
            "Werte = mehr Shards (sauberere Train/Val-Aufteilung), aber mehr "
            "Dateien. Wichtig: muss > 0 sein, damit split_shards eine sinnvolle "
            "Train/Val-Aufteilung machen kann."
        ),
    )
    parser.add_argument(
        "--target-distribution", type=str, default="500:0.5,1000:0.5",
        help="Verteilung der Spielziele pro Partie.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/bodensee_heuristic_bootstrap"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=12,
        help=(
            "Anzahl paralleler Worker-Prozesse. Da keine NN-Inferenz noetig ist, "
            "skaliert das praktisch linear mit den CPU-Kernen. Default 12."
        ),
    )
    args = parser.parse_args()

    target_distribution = _parse_target_distribution(args.target_distribution)
    args.output.mkdir(parents=True, exist_ok=True)

    base_rng = random.Random(args.seed)
    seeds = [base_rng.randint(0, 10**9) for _ in range(args.games)]

    n_shards = (args.games + args.games_per_shard - 1) // args.games_per_shard

    print(
        f"Bodensee-Heuristik-Bootstrap-Datengen:\n"
        f"  - {args.games} Partien\n"
        f"  - {args.workers} Worker-Prozesse (echtes Multiprocessing, kein GIL-Limit)\n"
        f"  - {args.games_per_shard} Partien pro Shard -> {n_shards} Shards insgesamt\n"
        f"  - Output: {args.output}\n"
    )
    start = time.perf_counter()
    total_samples = 0
    total_size_bytes = 0

    # Spawn-Context: jeder Worker ein frischer Interpreter. Ohne TF-Abhaengigkeit
    # waere auch fork moeglich, aber spawn ist plattformunabhaengig.
    ctx = mp.get_context("spawn")
    worker_fn = partial(_play_one_game, target_distribution=target_distribution)

    # Pool einmal erzeugen und ueber alle Shards wiederverwenden.
    with ctx.Pool(processes=args.workers) as pool:
        for shard_idx in range(n_shards):
            shard_start = shard_idx * args.games_per_shard
            shard_end = min(shard_start + args.games_per_shard, args.games)
            shard_seeds = seeds[shard_start:shard_end]

            shard_states: list[np.ndarray] = []
            shard_masks: list[np.ndarray] = []
            shard_actions: list[int] = []
            shard_rewards: list[float] = []

            for states, masks, actions, rewards in pool.imap_unordered(
                worker_fn, shard_seeds
            ):
                shard_states.extend(states)
                shard_masks.extend(masks)
                shard_actions.extend(actions)
                shard_rewards.extend(rewards)

            if not shard_states:
                continue

            out_path = _write_shard(
                args.output, shard_idx,
                shard_states, shard_masks, shard_actions, shard_rewards,
            )
            total_samples += len(shard_states)
            total_size_bytes += out_path.stat().st_size

            elapsed = time.perf_counter() - start
            rate = (shard_end / elapsed) if elapsed > 0 else 0
            print(
                f"  Shard {shard_idx + 1}/{n_shards} fertig ({len(shard_states):,} Samples, "
                f"{out_path.stat().st_size / 2**20:.1f} MB) -- "
                f"{shard_end}/{args.games} Partien gesamt ({rate:.1f} Partien/s)"
            )

    elapsed = time.perf_counter() - start
    print(
        f"\nFertig: {total_samples:,} Samples in {n_shards} Shards "
        f"({total_size_bytes / 2**20:.1f} MB) -- Dauer {elapsed / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
