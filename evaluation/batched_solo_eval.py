"""Batched-GPU-Eval fuer Solo-Jass: viele 4-Wege-Partien parallel in einem
Process, GPU macht die NN-Inferenzen als Batch.

Architektur analog zu `evaluation/batched_eval.py`, aber fuer 4-Spieler-Solo
mit dem Rollen-Setup A vs B vs H vs H (zwei NNs, zwei Heuristik-Bots):

- Modelle werden EINMAL geladen (auf GPU)
- Pro NN-Rolle ein InferenceServer (Thread)
- N parallele Game-Threads spielen die Eval-Spiele
- Solo-Heuristik-Bots brauchen keine Inferenz
- Paired-Eval mit 4-Game-Rotation wird beibehalten

Im Vergleich zum Sequential-Eval (run_solo_eval ohne --inference-mode):
- ~5-10x schneller bei 2000+ Spielen
- GPU bekommt Batches statt Einzelinferenzen
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluation.solo_eval import (
    ROLE_A,
    ROLE_B,
    SoloEvalResult,
    _random_seat_assignment,
    _seat_assignment,
)
from evaluation.solo_stats import PlayerStats, update_stats_from_solo_game
from jass_engine.game import GameResult
from jass_engine.player import GameState, Player
from jass_engine.variants.solo_jass import play_solo_jass
from players.solo_heuristic_player import SoloHeuristicPlayer
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


class BatchedSoloEvalNNPlayer(Player):
    """Greedy NN-Player fuer Solo-Eval mit InferenceServer.

    Identische Logik wie BatchedEvalNNPlayer aus evaluation/batched_eval.py,
    aber explizit fuer Solo-Setup separat instanziiert (separater Fallback,
    klarere Trennung).
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        fallback: Player | None = None,
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.fallback = fallback or SoloHeuristicPlayer(name + "_fb")

    def choose_announcement(self, hand, round_idx, can_push):
        return self.fallback.choose_announcement(hand, round_idx, can_push)

    def announce_weise(self, hand, variant, possible_weise):
        return self.fallback.announce_weise(hand, variant, possible_weise)

    def choose_card(self, hand, state: GameState):
        x = encode_state(hand, state).astype(np.float32)
        mask = legal_action_mask(hand, state).astype(np.float32)

        policy, _value = self.inference_server.request(x, mask)

        legal_policy = policy * mask
        s = legal_policy.sum()
        if s <= 0:
            return self.fallback.choose_card(hand, state)
        action_idx = int(np.argmax(legal_policy))
        return index_to_card(action_idx)


def _build_solo_player_for_role(
    role: str,
    seat: int,
    kind_a: str,
    server_a: InferenceServer | None,
    kind_b: str,
    server_b: InferenceServer | None,
    rng: random.Random,
) -> Player:
    """Erzeugt einen Player basierend auf Rolle und Modus."""
    if role == ROLE_A:
        if kind_a == "nn":
            assert server_a is not None
            return BatchedSoloEvalNNPlayer(name=f"A_seat{seat}", inference_server=server_a)
        return SoloHeuristicPlayer(name=f"A_seat{seat}", rng=rng)
    if role == ROLE_B:
        if kind_b == "nn":
            assert server_b is not None
            return BatchedSoloEvalNNPlayer(name=f"B_seat{seat}", inference_server=server_b)
        return SoloHeuristicPlayer(name=f"B_seat{seat}", rng=rng)
    # H1, H2 sind immer Solo-Heuristik
    return SoloHeuristicPlayer(name=f"H_seat{seat}", rng=rng)


def _play_one_solo_game(
    game_idx: int,
    seat_to_role: dict[int, str],
    kind_a: str,
    server_a: InferenceServer | None,
    kind_b: str,
    server_b: InferenceServer | None,
    target_score: int,
    sub_seed: int,
) -> tuple[GameResult, dict[int, str]]:
    """Spielt EINE Solo-Partie. Returnt das GameResult und das Rollen-Mapping."""
    sub_rng = random.Random(sub_seed)

    players: list[Player | None] = [None] * 4
    for seat in range(4):
        role = seat_to_role[seat]
        players[seat] = _build_solo_player_for_role(
            role=role,
            seat=seat,
            kind_a=kind_a, server_a=server_a,
            kind_b=kind_b, server_b=server_b,
            rng=random.Random(sub_rng.randint(0, 10**9)),
        )

    game = play_solo_jass(
        players,  # type: ignore[arg-type]
        target_score=target_score,
        rng=random.Random(sub_rng.randint(0, 10**9)),
    )
    return game, seat_to_role


def four_way_match_batched_gpu(
    label_a: str,
    kind_a: str,
    model_a: Path | None,
    label_b: str,
    kind_b: str,
    model_b: Path | None,
    label_h: str,
    num_games: int,
    target_score: int = 500,
    seed: int = 0,
    paired_eval: bool = False,
    inference_batch_size: int = 64,
    parallel_threads: int = 64,
) -> SoloEvalResult:
    """Solo-4-Wege-Eval mit GPU-Batching.

    Args:
        label_a / label_b / label_h: Anzeigenamen
        kind_a / kind_b: "random" | "heuristic" | "nn"
        model_a / model_b: Modell-Pfade (nur wenn kind == "nn")
        num_games: Anzahl Partien (bei paired_eval muss durch 4 teilbar sein)
        target_score: Punkteziel pro Partie
        seed: Master-Seed
        paired_eval: 4-Spiele-pro-Paar mit zyklischer Sitz-Rotation
        inference_batch_size: max. Server-Batch-Groesse
        parallel_threads: max. gleichzeitig spielende Game-Threads

    Returns:
        SoloEvalResult mit aggregierten Stats pro Rolle.
    """
    # Modelle laden + Server starten (nur fuer NN-Rollen)
    server_a: InferenceServer | None = None
    server_b: InferenceServer | None = None

    if kind_a == "nn":
        if model_a is None:
            raise ValueError("kind_a='nn' braucht model_a")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        print(f"  Lade Modell A: {model_a}")
        model_a_obj = keras.models.load_model(str(model_a))
        server_a = InferenceServer(
            model_a_obj, max_batch_size=inference_batch_size
        )

    if kind_b == "nn":
        if model_b is None:
            raise ValueError("kind_b='nn' braucht model_b")
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401
        print(f"  Lade Modell B: {model_b}")
        model_b_obj = keras.models.load_model(str(model_b))
        server_b = InferenceServer(
            model_b_obj, max_batch_size=inference_batch_size
        )

    try:
        rng = random.Random(seed)

        # Jobs vorbereiten
        jobs: list[dict] = []
        if paired_eval:
            if num_games % 4 != 0:
                raise ValueError(
                    "paired_eval=True braucht num_games als Vielfaches von 4 "
                    f"(uebergeben: {num_games})."
                )
            num_pairs = num_games // 4
            for pair_idx in range(num_pairs):
                pair_seed = rng.randint(0, 10**9)
                for pair_offset in range(4):
                    jobs.append({
                        "game_idx": pair_idx * 4 + pair_offset,
                        "seat_to_role": _seat_assignment(pair_offset),
                        "sub_seed": pair_seed,
                    })
        else:
            for game_idx in range(num_games):
                jobs.append({
                    "game_idx": game_idx,
                    "seat_to_role": _random_seat_assignment(rng),
                    "sub_seed": rng.randint(0, 10**9),
                })

        results: list[tuple[GameResult, dict[int, str]] | None] = [None] * num_games
        start_time = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=parallel_threads, thread_name_prefix="SoloEvalGame"
        ) as pool:
            future_to_idx = {
                pool.submit(
                    _play_one_solo_game,
                    job["game_idx"],
                    job["seat_to_role"],
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
                        f"  Fortschritt: {done_count}/{num_games} Solo-Spiele "
                        f"({rate:.1f}/s)"
                    )

        # Stats pro Rolle aggregieren
        stats_a = PlayerStats()
        stats_b = PlayerStats()
        stats_h = PlayerStats()

        for game_result, seat_to_role in results:  # type: ignore[misc]
            per_seat_stats: dict[int, PlayerStats] = {}
            for seat in range(4):
                role = seat_to_role[seat]
                if role == ROLE_A:
                    per_seat_stats[seat] = stats_a
                elif role == ROLE_B:
                    per_seat_stats[seat] = stats_b
                else:  # H1 oder H2 -> beide in dieselbe stats_h
                    per_seat_stats[seat] = stats_h
            update_stats_from_solo_game(per_seat_stats, game_result)

        return SoloEvalResult(
            label_a=label_a,
            label_b=label_b,
            label_h=label_h,
            stats_a=stats_a,
            stats_b=stats_b,
            stats_h=stats_h,
            games_played=num_games,
        )
    finally:
        if server_a is not None:
            server_a.shutdown()
        if server_b is not None:
            server_b.shutdown()
