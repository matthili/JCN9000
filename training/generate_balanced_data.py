"""Balancierter Datengenerator: gleich viele Trainings-Runden pro Variante.

Das normale generate_data.py nutzt die HeuristicPlayer-Wahl. Die bevorzugt
Trumpf-Varianten stark; Bock und Geiss sind unterrepraesentiert -> das
trainierte NN ist dort dann schwach.

Dieses Skript erzwingt mit ForcedAnnouncementPlayer pro Sub-Run eine bestimmte
Variante und sammelt pro Variante ungefaehr gleich viele Runden.

Aufruf:
    python -m training.generate_balanced_data \
        --runs-per-variant 50000 --target 1000 --workers 20 \
        --output data/balanced

Output (im output-Verzeichnis):
    trumpf_eichel/   shard_00000.npz, ...
    trumpf_schelle/
    trumpf_herz/
    trumpf_laub/
    oben/
    unten/
    slalom_oben/
    slalom_unten/

Beim Trainings-Loading kann man alle Unterverzeichnisse zusammen laden oder
gezielt eine Auswahl treffen.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jass_engine.card import Suit
from jass_engine.variant import Announcement, Variant
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS, play_kreuz_jass
from players.forced_announcement_player import ForcedAnnouncementPlayer
from training.encoder import ACTION_DIM, INPUT_DIM
from training.recording_player import RecordingPlayer


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
    VariantSpec("oben", Announcement(variant=Variant.oben())),
    VariantSpec("unten", Announcement(variant=Variant.unten())),
    VariantSpec("slalom_oben", Announcement(variant=Variant.oben(), slalom=True)),
    VariantSpec("slalom_unten", Announcement(variant=Variant.unten(), slalom=True)),
]


@dataclass
class ShardResult:
    path: Path
    num_samples: int
    num_games: int


def _simulate_with_forced_announcement(
    num_games: int,
    seed: int,
    target_score: int,
    output_path: Path,
    forced_announcement: Announcement,
) -> ShardResult:
    """Simuliert `num_games` Partien, in denen alle Spieler die forced_announcement
    ansagen, und speichert die Trajektorien (state/mask/action/reward) als Shard."""
    rng = random.Random(seed)
    teams = list(KREUZ_JASS_TEAMS)
    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []

    for _ in range(num_games):
        players = [
            RecordingPlayer(
                ForcedAnnouncementPlayer(
                    name=f"F{i}",
                    forced_announcement=forced_announcement,
                    rng=random.Random(rng.randint(0, 10**9)),
                )
            )
            for i in range(4)
        ]
        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )

        # Pro Runde: Reward pro Team
        round_rewards: list[dict[int, float]] = []
        for rnd in game.rounds:
            team_ids = list(rnd.team_total_points.keys())
            d = {}
            for tid in team_ids:
                own_pts = rnd.team_total_points[tid]
                opp_pts = sum(p for t, p in rnd.team_total_points.items() if t != tid)
                d[tid] = (own_pts - opp_pts) / REWARD_SCALE
            round_rewards.append(d)

        for p in players:
            for st, mk, ac, p_idx, r_idx in zip(
                p.states, p.masks, p.actions, p.player_indices, p.round_indices
            ):
                team_id = teams[p_idx]
                reward = round_rewards[r_idx][team_id] if r_idx < len(round_rewards) else 0.0
                all_states.append(st)
                all_masks.append(mk)
                all_actions.append(ac)
                all_rewards.append(reward)

    if all_states:
        X = np.stack(all_states).astype(np.float32, copy=False)
        masks = np.stack(all_masks).astype(np.uint8, copy=False)
        actions = np.array(all_actions, dtype=np.uint8)
        rewards = np.array(all_rewards, dtype=np.float32)
    else:
        X = np.empty((0, INPUT_DIM), dtype=np.float32)
        masks = np.empty((0, ACTION_DIM), dtype=np.uint8)
        actions = np.empty((0,), dtype=np.uint8)
        rewards = np.empty((0,), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, masks=masks, actions=actions, rewards=rewards)
    return ShardResult(path=output_path, num_samples=len(actions), num_games=num_games)


def _worker(args: tuple[int, int, int, Path, Announcement]) -> ShardResult:
    games, seed, target, output_path, forced = args
    return _simulate_with_forced_announcement(
        num_games=games, seed=seed, target_score=target,
        output_path=output_path, forced_announcement=forced,
    )


def _estimate_games_for_target_rounds(target_rounds: int, target_score: int) -> int:
    """Grobe Schaetzung: Wieviele Partien brauchen wir, um target_rounds Runden
    abzudecken? Bei target_score=1000 sind das ~10 Runden pro Partie, mit etwas
    Sicherheitspuffer 12.

    Bei kleinerem target_score entsprechend mehr Partien.
    """
    avg_runden_pro_partie = max(2.0, target_score / 100.0)
    safety = 1.1  # 10% Sicherheitspuffer
    return int(target_rounds / avg_runden_pro_partie * safety) + 1


def generate_for_variant(
    output_dir: Path,
    variant_spec: VariantSpec,
    runs_per_variant: int,
    shard_size: int,
    workers: int,
    target_score: int,
    seed: int,
) -> int:
    """Generiert Daten fuer eine Variante. Gibt die Anzahl gespielter Partien zurueck."""
    sub_dir = output_dir / variant_spec.output_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

    num_games = _estimate_games_for_target_rounds(runs_per_variant, target_score)
    num_shards = (num_games + shard_size - 1) // shard_size

    base_rng = random.Random(seed)
    tasks = []
    for shard_idx in range(num_shards):
        games_in_shard = min(shard_size, num_games - shard_idx * shard_size)
        out_path = sub_dir / f"shard_{shard_idx:05d}.npz"
        tasks.append((
            games_in_shard,
            base_rng.randint(0, 10**9),
            target_score,
            out_path,
            variant_spec.announcement,
        ))

    desc = f"{variant_spec.label:>16}"
    if workers <= 1:
        with tqdm(total=num_games, desc=desc, leave=False) as pbar:
            for task in tasks:
                result = _worker(task)
                pbar.update(result.num_games)
    else:
        with mp.Pool(processes=workers) as pool:
            with tqdm(total=num_games, desc=desc, leave=False) as pbar:
                for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                    pbar.update(result.num_games)

    return num_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-per-variant", type=int, default=50_000,
        help="Anzahl Runden pro Variante (Default: 50k -> 400k gesamt)",
    )
    parser.add_argument(
        "--output", type=str, default="data/balanced",
        help="Output-Verzeichnis (jede Variante in einem Unterordner)",
    )
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target", type=int, default=1000, help="Punkteziel pro Partie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Nur bestimmte Varianten generieren (z.B. trumpf_eichel oben). "
             "Default: alle 8 Varianten.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    selected = ALL_VARIANTS
    if args.variants is not None:
        selected = [v for v in ALL_VARIANTS if v.label in set(args.variants)]
        if not selected:
            print(f"WARNUNG: keine gueltigen Varianten in {args.variants} gefunden.")
            return

    print(
        f"Balanced-Generator: {args.runs_per_variant:,} Runden je Variante, "
        f"{len(selected)} Varianten -> {output_dir}\n"
    )
    start = time.perf_counter()
    total_games = 0
    for vs in selected:
        ngames = generate_for_variant(
            output_dir=output_dir,
            variant_spec=vs,
            runs_per_variant=args.runs_per_variant,
            shard_size=args.shard_size,
            workers=args.workers,
            target_score=args.target,
            seed=args.seed + hash(vs.label) % 10000,
        )
        total_games += ngames

    elapsed = time.perf_counter() - start
    print(
        f"\nFertig: {len(selected)} Varianten, ~{total_games} Partien total. "
        f"Dauer {elapsed:.1f} s ({total_games / elapsed:.0f} Partien/s)."
    )


if __name__ == "__main__":
    main()
