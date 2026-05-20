"""Batched-GPU-Eval fuer Bodensee-Jass: viele 2-Spieler-Partien parallel,
GPU macht die NN-Inferenzen als Batch.

Architektur analog zu `evaluation/batched_solo_eval.py`, aber fuer den
2-Spieler-Bodensee mit dem Bodensee-Encoder (bodensee_1.0.0).
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluation.bodensee_eval import (
    BodenseeEvalResult,
    update_stats_from_bodensee_game,
)
from evaluation.solo_stats import PlayerStats
from jass_engine.bodensee.game import BodenseeGameResult
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card
from jass_engine.variants.bodensee_jass import play_bodensee_jass
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_player import BodenseePlayer
from training.bodensee_encoder import (
    index_to_card,
    legal_action_mask_bodensee,
)
from training.data.generate_bodensee_mcts_data import encode_state_bodensee_from_lists
from training.rl.batched_selfplay import InferenceServer


class BatchedBodenseeEvalNNPlayer(BodenseePlayer):
    """Greedy NN-Bodensee-Spieler fuer Eval, der einen InferenceServer benutzt."""

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        fallback: BodenseePlayer | None = None,
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.fallback = fallback or BodenseeHeuristicPlayer(name + "_fb")

    def choose_announcement(self, hand, visible_table, round_idx):
        return self.fallback.choose_announcement(hand, visible_table, round_idx)

    def choose_card(self, hand, visible_table, state: BodenseeGameState) -> Card:
        x = encode_state_bodensee_from_lists(
            hand=hand,
            visible_table=visible_table,
            own_hidden_count=state.own_hidden_table_count,
            game_state=state,
            i_am_announcer=False,
        ).astype(np.float32)
        mask = legal_action_mask_bodensee(hand, visible_table, state).astype(np.float32)

        policy, _value = self.inference_server.request(x, mask)
        legal_policy = policy * mask
        if legal_policy.sum() <= 0:
            return self.fallback.choose_card(hand, visible_table, state)
        action_idx = int(np.argmax(legal_policy))
        return index_to_card(action_idx)


def _build_bodensee_player(
    kind: str,
    server: InferenceServer | None,
    rng: random.Random,
    name: str,
) -> BodenseePlayer:
    if kind == "heuristic":
        return BodenseeHeuristicPlayer(name=name, rng=rng)
    if kind == "nn":
        assert server is not None, "NN-Bodensee-Player braucht einen InferenceServer"
        return BatchedBodenseeEvalNNPlayer(name=name, inference_server=server)
    raise ValueError(f"Unbekannter Bodensee-Player-Kind: {kind!r}")


def _play_one_bodensee_game(
    swap: bool,
    kind_a: str, server_a: InferenceServer | None,
    kind_b: str, server_b: InferenceServer | None,
    target_score: int,
    sub_seed: int,
) -> tuple[BodenseeGameResult, int, int]:
    """Spielt EINE Bodensee-Partie. Returnt (Result, a_seat, b_seat)."""
    sub_rng = random.Random(sub_seed)

    if swap:
        players = [
            _build_bodensee_player(kind_b, server_b, random.Random(sub_rng.randint(0, 10**9)), "B0"),
            _build_bodensee_player(kind_a, server_a, random.Random(sub_rng.randint(0, 10**9)), "A1"),
        ]
        a_seat, b_seat = 1, 0
    else:
        players = [
            _build_bodensee_player(kind_a, server_a, random.Random(sub_rng.randint(0, 10**9)), "A0"),
            _build_bodensee_player(kind_b, server_b, random.Random(sub_rng.randint(0, 10**9)), "B1"),
        ]
        a_seat, b_seat = 0, 1

    game = play_bodensee_jass(
        players,
        target_score=target_score,
        rng=random.Random(sub_rng.randint(0, 10**9)),
    )
    return game, a_seat, b_seat


def two_player_match_batched_gpu(
    label_a: str,
    kind_a: str,
    model_a: Path | None,
    label_b: str,
    kind_b: str,
    model_b: Path | None,
    num_games: int,
    target_score: int = 500,
    seed: int = 0,
    paired_eval: bool = False,
    inference_batch_size: int = 64,
    parallel_threads: int = 64,
) -> BodenseeEvalResult:
    """Bodensee-2-Spieler-Eval mit GPU-Batching.

    Args:
        label_a / label_b: Anzeigenamen
        kind_a / kind_b: "heuristic" | "nn"
        model_a / model_b: Modell-Pfade (nur wenn kind == "nn")
        num_games: Anzahl Partien (bei paired_eval gerade Zahl)
        target_score: Punkteziel
        seed: Master-Seed
        paired_eval: 2-Spiele-pro-Paar mit gespiegelten Sitzen
        inference_batch_size: max. Server-Batch
        parallel_threads: max. gleichzeitig spielende Game-Threads

    Returns:
        BodenseeEvalResult mit Stats pro Spieler.
    """
    server_a: InferenceServer | None = None
    server_b: InferenceServer | None = None

    if kind_a == "nn":
        if model_a is None:
            raise ValueError("kind_a='nn' braucht model_a")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        print(f"  Lade Bodensee-Modell A: {model_a}")
        model_a_obj = keras.models.load_model(str(model_a))
        server_a = InferenceServer(model_a_obj, max_batch_size=inference_batch_size)

    if kind_b == "nn":
        if model_b is None:
            raise ValueError("kind_b='nn' braucht model_b")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        print(f"  Lade Bodensee-Modell B: {model_b}")
        model_b_obj = keras.models.load_model(str(model_b))
        server_b = InferenceServer(model_b_obj, max_batch_size=inference_batch_size)

    try:
        rng = random.Random(seed)
        jobs: list[dict] = []
        if paired_eval:
            if num_games % 2 != 0:
                raise ValueError(
                    f"paired_eval braucht gerade num_games (uebergeben: {num_games})."
                )
            for pair_idx in range(num_games // 2):
                pair_seed = rng.randint(0, 10**9)
                jobs.append({"game_idx": pair_idx * 2, "swap": False, "sub_seed": pair_seed})
                jobs.append({"game_idx": pair_idx * 2 + 1, "swap": True, "sub_seed": pair_seed})
        else:
            for game_idx in range(num_games):
                jobs.append({
                    "game_idx": game_idx,
                    "swap": False,
                    "sub_seed": rng.randint(0, 10**9),
                })

        results: list[tuple[BodenseeGameResult, int, int] | None] = [None] * num_games
        start_time = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=parallel_threads, thread_name_prefix="BodenseeEvalGame"
        ) as pool:
            future_to_idx = {
                pool.submit(
                    _play_one_bodensee_game,
                    job["swap"],
                    kind_a, server_a,
                    kind_b, server_b,
                    target_score,
                    job["sub_seed"],
                ): job["game_idx"]
                for job in jobs
            }
            done_count = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                done_count += 1
                if done_count % max(1, num_games // 10) == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(
                        f"  Fortschritt: {done_count}/{num_games} Bodensee-Spiele "
                        f"({rate:.1f}/s)"
                    )

        stats_a = PlayerStats()
        stats_b = PlayerStats()
        for game_result, a_seat, b_seat in results:  # type: ignore[misc]
            update_stats_from_bodensee_game(
                {a_seat: stats_a, b_seat: stats_b},
                game_result,
            )

        return BodenseeEvalResult(
            label_a=label_a,
            label_b=label_b,
            stats_a=stats_a,
            stats_b=stats_b,
            games_played=num_games,
        )
    finally:
        if server_a is not None:
            server_a.shutdown()
        if server_b is not None:
            server_b.shutdown()
