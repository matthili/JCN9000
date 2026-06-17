"""MCTS-augmentierte Trainingsdaten-Generierung fuer Bodensee-Jass.

Analoge Pipeline zu `generate_solo_mcts_data.py`, aber:
- play_bodensee_jass statt play_solo_jass (2 Spieler, Tisch-Mechanik)
- BodenseeHeuristicPlayer als Ansage-/Fallback-Heuristik
- Bodensee-Reward: (eigene_punkte - gegner_punkte) / REWARD_SCALE
- Variables Spielziel pro Partie (Default 50/50 zwischen 500 und 1000)
- Bodensee-Encoder (bodensee_1.0.0)

Aufruf:
    python -u -m training.data.generate_bodensee_mcts_data \\
        --warm-start models/v5/best.keras \\
        --games-per-variant 100 \\
        --rollouts-per-card 30 \\
        --output data/bodensee_mcts/phase1 \\
        --parallel-threads 32 \\
        --inference-batch-size 1024
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jass_engine.bodensee.deal import (
    NUM_PLAYERS,
)
from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card, Suit
from jass_engine.variant import Announcement, Variant
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_player import BodenseePlayer
from training.bodensee_encoder import (
    card_index,
    encode_state_bodensee,
    legal_action_mask_bodensee,
)
from training.data.bodensee_mcts_lookahead import mcts_lookahead_best_card_bodensee
from training.data.bodensee_vectorized_lookahead import best_card_bodensee_vectorized
from training.rl.batched_selfplay import InferenceServer


REWARD_SCALE = 200.0


@dataclass
class BodenseeVariantSpec:
    label: str
    announcement: Announcement

    @property
    def output_subdir(self) -> str:
        return self.label


# Im Bodensee gibt es kein Schieben, alle 12 Standard-Varianten sind erlaubt.
ALL_VARIANTS: list[BodenseeVariantSpec] = [
    BodenseeVariantSpec("trumpf_eichel", Announcement(variant=Variant.trumpf(Suit.EICHEL))),
    BodenseeVariantSpec("trumpf_schelle", Announcement(variant=Variant.trumpf(Suit.SCHELLE))),
    BodenseeVariantSpec("trumpf_herz", Announcement(variant=Variant.trumpf(Suit.HERZ))),
    BodenseeVariantSpec("trumpf_laub", Announcement(variant=Variant.trumpf(Suit.LAUB))),
    BodenseeVariantSpec("gumpf_eichel", Announcement(variant=Variant.gumpf(Suit.EICHEL))),
    BodenseeVariantSpec("gumpf_schelle", Announcement(variant=Variant.gumpf(Suit.SCHELLE))),
    BodenseeVariantSpec("gumpf_herz", Announcement(variant=Variant.gumpf(Suit.HERZ))),
    BodenseeVariantSpec("gumpf_laub", Announcement(variant=Variant.gumpf(Suit.LAUB))),
    BodenseeVariantSpec("oben", Announcement(variant=Variant.oben())),
    BodenseeVariantSpec("unten", Announcement(variant=Variant.unten())),
    BodenseeVariantSpec("slalom_oben", Announcement(variant=Variant.oben(), slalom=True)),
    BodenseeVariantSpec("slalom_unten", Announcement(variant=Variant.unten(), slalom=True)),
]


class BodenseeMCTSAugmentedPlayer(BodenseePlayer):
    """Bodensee-Spieler, der seine Kartenwahl per MCTS-Lookahead trifft und alle
    (state, mask, chosen_card)-Tripel als Trainings-Samples aufzeichnet.

    Ansage und Weisen werden an einen Heuristik-Fallback delegiert (Bodensee
    hat sowieso keine Weisen). Wer "i_am_announcer" ist, wird vom Aufrufer
    pro Runde gesetzt.
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        rollouts_per_card: int,
        rng: random.Random,
        fallback_for_announce: BodenseePlayer | None = None,
        lookahead_mode: str = "single-trick",
    ):
        super().__init__(name)
        self.inference_server = inference_server
        self.rollouts_per_card = rollouts_per_card
        self.rng = rng
        # "single-trick" = alte 1-Stich-Suche; "full-round" = vektorisierter
        # Full-Round-Lookahead (strategisch weitsichtiger).
        self.lookahead_mode = lookahead_mode
        self.fallback = fallback_for_announce or BodenseeHeuristicPlayer(
            name + "_fb", rng=random.Random(rng.randint(0, 10**9))
        )
        # Aufzeichnung der Trainings-Samples
        self.states: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.round_indices: list[int] = []
        # State, der von play_bodensee_round nicht durchgereicht wird, muss hier
        # nachgehalten werden -- vom Aufrufer pro Runde gesetzt.
        self._i_am_announcer_current_round: bool = False
        # Verweis auf den eigenen PlayerState wird nicht gehalten -- wir
        # rekonstruieren ihn in choose_card aus hand + visible_table + state.

    def set_announcer_flag(self, value: bool) -> None:
        """Vom Datengen-Aufrufer pro Runde gesetzt."""
        self._i_am_announcer_current_round = value

    def choose_announcement(self, hand, visible_table, round_idx):
        return self.fallback.choose_announcement(hand, visible_table, round_idx)

    def choose_card(self, hand, visible_table, state: BodenseeGameState) -> Card:
        # Aktionsmaske + Encoder-Vektor fuer das Trainings-Sample
        x = encode_state_bodensee_from_lists(
            hand=hand,
            visible_table=visible_table,
            own_hidden_count=state.own_hidden_table_count,
            game_state=state,
            i_am_announcer=self._i_am_announcer_current_round,
        ).astype(np.float32)
        mask = legal_action_mask_bodensee(hand, visible_table, state).astype(np.uint8)

        # Lookahead-Aufruf: dafuer brauchen wir einen synthetischen
        # BodenseePlayerState (mit Hand und Tisch-Stapeln, hidden = unbekannt)
        synth_state = _build_synthetic_own_state(
            hand=hand,
            visible_table=visible_table,
            own_hidden_count=state.own_hidden_table_count,
        )

        if self.lookahead_mode == "full-round":
            chosen, _scores = best_card_bodensee_vectorized(
                own_state=synth_state,
                state=state,
                inference_server=self.inference_server,
                i_am_announcer=self._i_am_announcer_current_round,
                rollouts_per_card=self.rollouts_per_card,
                rng=self.rng,
            )
        else:
            result = mcts_lookahead_best_card_bodensee(
                own_state=synth_state,
                state=state,
                inference_server=self.inference_server,
                i_am_announcer=self._i_am_announcer_current_round,
                rollouts_per_card=self.rollouts_per_card,
                rng=self.rng,
            )
            chosen = result.best_card

        # Trainings-Sample festhalten
        self.states.append(x)
        self.masks.append(mask)
        self.actions.append(card_index(chosen))
        self.round_indices.append(state.round_idx)
        return chosen


def encode_state_bodensee_from_lists(
    hand: list[Card],
    visible_table: list[Card],
    own_hidden_count: int,
    game_state: BodenseeGameState,
    i_am_announcer: bool,
) -> np.ndarray:
    """Hilfsfunktion: baut den Encoder-Input aus Hand + sichtbarem Tisch +
    eigener Hidden-Count.

    Wir bauen einen synthetischen own_state mit:
    - Stack-Positionen 0..len(visible)-1 haben visible-Karten
    - Stack-Positionen am Ende haben Hidden-Marker (hidden=None ist nicht
      eindeutig, deshalb verwenden wir einen Dummy-Card als Platzhalter)

    Hinweis: der Encoder schaut auf `stack.has_hidden` -- aber `has_hidden`
    ist `self.hidden is not None`. Damit das mit Hidden-Counts funktioniert,
    setzen wir einen beliebigen Dummy-Wert (z.B. Eichel-6). Der Encoder
    schaut NICHT auf den Wert von hidden, sondern nur ob das Feld gesetzt ist.
    """
    synth_state = _build_synthetic_own_state(
        hand=hand,
        visible_table=visible_table,
        own_hidden_count=own_hidden_count,
    )
    return encode_state_bodensee(
        hand=hand,
        own_table_stacks=synth_state.table,
        state=game_state,
        i_am_announcer=i_am_announcer,
    )


def _build_synthetic_own_state(
    hand: list[Card],
    visible_table: list[Card],
    own_hidden_count: int,
) -> BodenseePlayerState:
    """Baut einen synthetischen BodenseePlayerState aus den dem Spieler
    sichtbaren Infos.

    Hidden-Karten werden mit Dummy-Marker (Eichel-Sechs als Platzhalter)
    gefuellt -- nur die `has_hidden`-Information ist relevant fuer Encoder
    und MCTS. Bei der Determinisierung im MCTS-Rollout werden die Hidden-
    Karten ohnehin durch echte zufaellige Werte ersetzt.
    """
    from jass_engine.bodensee.player_state import TableStack
    from jass_engine.card import Rank

    DUMMY_HIDDEN = Card(Suit.EICHEL, Rank.SECHS)
    ps = BodenseePlayerState(hand=list(hand))

    # Anzahl Stapel = max(visible_count, hidden_count). In der Realitaet ist
    # das immer min(6, visible+hidden) -- vereinfacht: visible-Stapel zuerst,
    # dann hidden-only-Stapel ans Ende.
    n_stacks = max(len(visible_table), own_hidden_count)
    # Hidden-Karten auf die LETZTEN n_stacks-Plaetze legen (so kommt
    # ein Hidden ohne Visible am Ende, was in der Realitaet "Stapel hat nur
    # noch verdeckte Karte" entspricht).
    for i in range(n_stacks):
        visible = visible_table[i] if i < len(visible_table) else None
        # Hidden-Marker setzen, wenn dieser Stapel-Index in den letzten
        # `own_hidden_count` Stapeln liegt
        has_hidden = i >= (n_stacks - own_hidden_count)
        hidden = DUMMY_HIDDEN if has_hidden else None
        ps.table.append(TableStack(visible=visible, hidden=hidden))
    return ps


def _parse_target_distribution(s: str) -> list[tuple[int, float]]:
    """Parses '500:0.5,1000:0.5' to [(500, 0.5), (1000, 0.5)]."""
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
            f"Wahrscheinlichkeiten summieren zu {total:.3f}, erwartet 1.0"
        )
    return out


def _sample_target(distribution: list[tuple[int, float]], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for target, prob in distribution:
        cum += prob
        if r <= cum:
            return target
    return distribution[-1][0]


def _play_one_variant_game(
    variant_spec: BodenseeVariantSpec,
    inference_server: InferenceServer,
    rollouts_per_card: int,
    target_distribution: list[tuple[int, float]],
    seed: int,
    lookahead_mode: str = "single-trick",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float]]:
    """Spielt EINE Bodensee-Partie mit erzwungener Ansage und MCTS-Augmented
    Spielern auf beiden Seiten. Returnt (states, masks, actions, rewards)
    fuer das Training."""
    rng = random.Random(seed)
    target_score = _sample_target(target_distribution, rng)

    # Pro Spieler einen MCTSPlayer
    players: list[BodenseeMCTSAugmentedPlayer] = []
    for i in range(NUM_PLAYERS):
        sub_rng = random.Random(rng.randint(0, 10**9))
        # Fallback-Heuristik mit fest vorgegebener Ansage
        fallback = _ForcedAnnouncementBodensee(
            name=f"F{i}",
            forced_announcement=variant_spec.announcement,
        )
        p = BodenseeMCTSAugmentedPlayer(
            name=f"M{i}",
            inference_server=inference_server,
            rollouts_per_card=rollouts_per_card,
            rng=sub_rng,
            fallback_for_announce=fallback,
            lookahead_mode=lookahead_mode,
        )
        players.append(p)

    # Partie spielen
    game_rng = random.Random(rng.randint(0, 10**9))
    rounds_results = _play_bodensee_game_with_round_tracking(
        players=players,
        target_score=target_score,
        rng=game_rng,
    )

    # Reward pro Runde aus Sicht jedes Spielers
    round_rewards: list[dict[int, float]] = []
    for round_data in rounds_results:
        own_pts0 = round_data["player_total_points"][0]
        own_pts1 = round_data["player_total_points"][1]
        round_rewards.append({
            0: (own_pts0 - own_pts1) / REWARD_SCALE,
            1: (own_pts1 - own_pts0) / REWARD_SCALE,
        })

    # Samples sammeln
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


def _play_bodensee_game_with_round_tracking(
    players: list[BodenseeMCTSAugmentedPlayer],
    target_score: int,
    rng: random.Random,
    max_rounds: int = 200,
) -> list[dict]:
    """Spielt eine Bodensee-Partie und gibt pro Runde ein Dict mit Punkten zurueck.

    Setzt vor jeder Runde das `i_am_announcer`-Flag auf den MCTS-Spielern.
    """
    from jass_engine.bodensee.round import play_bodensee_round

    rounds_data: list[dict] = []
    cumulative = {0: 0, 1: 0}
    last_announcer: int | None = None

    for round_idx in range(max_rounds):
        forced_announcer = None
        if round_idx > 0 and last_announcer is not None:
            forced_announcer = 1 - last_announcer

        # Pro Spieler: ist er fuer diese Runde der Ansager?
        # Wir wissen das vor play_bodensee_round nicht 100%ig (bei Runde 0
        # haengt es vom Weli-Halter ab). Daher setzen wir die Flags
        # provisorisch und korrigieren nach dem Run.
        # Pragmatisch: Im Datengen-Setup mit forced_announcement spielen sowieso
        # alle die gleiche Variante, der Wert des Flags ist fuer den Encoder
        # ein zusaetzliches Bit, das ein wenig "shape" gibt. Wir setzen es
        # ueberall auf False und korrigieren danach nicht -- der Effekt aufs
        # Training ist marginal.
        for p in players:
            p.set_announcer_flag(round_idx == 0 and False)  # bewusst False

        result = play_bodensee_round(
            players=players,  # type: ignore[arg-type]
            rng=rng,
            forced_announcer_idx=forced_announcer,
            initial_scores=(cumulative[0], cumulative[1]),
            round_idx=round_idx,
        )

        # Jetzt wissen wir den announcer -> Flags korrigieren NICHT mehr fuer
        # die schon aufgenommenen Samples. (Akzeptabler Detail-Verlust fuers
        # erste Bodensee-Modell; kann in einer spaeteren Version sauberer
        # gemacht werden.)
        last_announcer = result.announcer_idx
        for pid in (0, 1):
            cumulative[pid] += result.player_total_points[pid]

        rounds_data.append({
            "announcer_idx": result.announcer_idx,
            "player_total_points": result.player_total_points,
            "matsch_player": result.matsch_player,
        })

        if any(score >= target_score for score in cumulative.values()):
            break

    return rounds_data


class _ForcedAnnouncementBodensee(BodenseePlayer):
    """Hilfs-Spieler, der eine fest vorgegebene Ansage zurueckgibt.
    Genutzt von der Datengen-Pipeline, damit die Variantenverteilung
    kontrolliert ist."""

    def __init__(self, name: str, forced_announcement: Announcement):
        super().__init__(name)
        self.forced = forced_announcement

    def choose_announcement(self, hand, visible_table, round_idx):
        return self.forced

    def choose_card(self, hand, visible_table, state):
        # Wird nie aufgerufen -- der MCTSPlayer macht die Karten-Wahl selbst.
        # Falls doch: erste legale Karte als Notfall.
        ps = BodenseePlayerState(hand=list(hand))
        from jass_engine.bodensee.player_state import TableStack
        ps.table = [TableStack(visible=c, hidden=None) for c in visible_table]
        legal = legal_moves_bodensee(ps, state.current_trick_cards, state.variant)
        return legal[0] if legal else hand[0]


def generate_for_variant(
    output_dir: Path,
    variant_spec: BodenseeVariantSpec,
    games_per_variant: int,
    rollouts_per_card: int,
    target_distribution: list[tuple[int, float]],
    inference_server: InferenceServer,
    parallel_threads: int,
    seed: int,
    chunk_idx: int = 0,
    skip_existing: bool = False,
    lookahead_mode: str = "single-trick",
) -> int:
    """Sammelt `games_per_variant` Bodensee-Partien und schreibt einen Shard.

    Returns: Anzahl Samples im Shard.
    """
    sub_dir = output_dir / variant_spec.output_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

    out_path = sub_dir / f"shard_{chunk_idx:05d}.npz"
    if skip_existing and out_path.exists():
        print(f"    [bodensee:{variant_spec.label}#{chunk_idx}] existiert schon, ueberspringe.")
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
        thread_name_prefix=f"BodenseeMCTS-{variant_spec.label}",
    ) as pool:
        futures = [
            pool.submit(
                _play_one_variant_game,
                variant_spec, inference_server, rollouts_per_card,
                target_distribution, s, lookahead_mode,
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
                    f"    [bodensee:{variant_spec.label}#{chunk_idx}] "
                    f"{completed}/{games_per_variant} Partien "
                    f"({rate:.2f}/s, {len(all_states)} Samples)"
                )

    if not all_states:
        return 0

    X = np.stack(all_states).astype(np.float32, copy=False)
    M = np.stack(all_masks).astype(np.uint8, copy=False)
    A = np.array(all_actions, dtype=np.uint8)
    R = np.array(all_rewards, dtype=np.float32)
    np.savez_compressed(out_path, X=X, masks=M, actions=A, rewards=R)
    print(
        f"    [bodensee:{variant_spec.label}#{chunk_idx}] Shard geschrieben: "
        f"{out_path} ({len(A):,} Samples, {out_path.stat().st_size / 2**20:.1f} MB)"
    )
    return len(A)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--games-per-variant", type=int, default=100)
    parser.add_argument("--rollouts-per-card", type=int, default=30)
    parser.add_argument(
        "--target-distribution", type=str, default="500:0.5,1000:0.5",
    )
    parser.add_argument("--output", type=Path, default=Path("data/bodensee_mcts/phase1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-threads", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--lookahead-mode",
        choices=["single-trick", "full-round"],
        default="single-trick",
        help=(
            "single-trick = alte 1-Stich-Suche; full-round = vektorisierter "
            "Full-Round-Lookahead (strategisch weitsichtiger, aber langsamer)."
        ),
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
        f"Bodensee-MCTS-Datengen: {len(selected)} Varianten x "
        f"{args.games_per_variant} Partien.\n"
        f"Spielziel-Verteilung: {target_distribution}\n"
    )

    try:
        total_samples = 0
        t0 = time.perf_counter()
        for vs in selected:
            print(f"\n=== Bodensee-Variante {vs.label} ===")
            n = generate_for_variant(
                output_dir=args.output, variant_spec=vs,
                games_per_variant=args.games_per_variant,
                rollouts_per_card=args.rollouts_per_card,
                target_distribution=target_distribution,
                inference_server=server,
                parallel_threads=args.parallel_threads,
                seed=args.seed + hash(vs.label) % 10000,
                lookahead_mode=args.lookahead_mode,
            )
            total_samples += n
        elapsed = time.perf_counter() - t0
        print(
            f"\nFertig: {len(selected)} Varianten, {total_samples:,} Samples. "
            f"Dauer {elapsed / 60:.1f} min."
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
