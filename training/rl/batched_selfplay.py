"""Batched-Self-Play: GPU-Inferenz fuer N parallele Spiele in einem Process.

Architektur (Variante "D" in unserer Diskussion):

    Hauptthread                                           InferenceServer-Thread (GPU)
    ----------------------------------------              ----------------------------
    fuer 64 Spiele jeweils einen Thread starten           Loop:
        Spiel-Thread 1 (BatchedRLPlayer x4):                 sammle Anfragen aus Queue
            play_kreuz_jass(...)                              -> wenn N Anfragen oder Timeout:
                ... choose_card -> server.request(s, m) -->     model(states, masks)
                ... <- (policy, value) <-------------------     verteile Antworten
        ...                                                  ...
        Spiel-Thread 64

    Vorteile gegenueber Variante C (CPU-Worker):
    - GPU wird real ausgelastet (Batch 32-64 pro Inferenz statt Batch 1)
    - Alles in einem Process, keine IPC-Kosten
    - Workers konkurrieren auf dem GIL, aber model(...) gibt GIL frei

    Nachteile:
    - GIL-Kontention zwischen Spiel-Threads (Game-Logik laeuft Python-bound)
    - Threading bringt mehr Bug-Quellen als Multi-Processing
    - Falls Server crasht, koennen Game-Threads in Deadlock laufen -> Timeout-Schutz
"""

from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jass_engine.player import GameState, Player
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS, play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.selfplay import REWARD_SCALE, RLPlayer
from training.rl.trajectory import Trajectory, Transition


# ----- Inferenz-Server -----


@dataclass
class _Request:
    """Eine Inferenz-Anfrage aus einem Game-Thread an den Server."""
    state: np.ndarray
    mask: np.ndarray
    event: threading.Event
    response: list = field(default_factory=lambda: [None])
    # response[0] wird vom Server-Thread auf (policy_vec, value_scalar) gesetzt


class InferenceServer:
    """Threaded Inferenz-Server: sammelt Anfragen, bildet Batches, ruft das
    Modell auf der GPU auf und verteilt die Antworten an die Game-Threads.

    Lebenszyklus:
        server = InferenceServer(model)
        try:
            # Game-Threads rufen server.request(state, mask) auf
            ...
        finally:
            server.shutdown()
    """

    def __init__(
        self,
        model,
        max_batch_size: int = 64,
        request_timeout_s: float = 30.0,
    ):
        """
        Args:
            model: Keras-Modell mit Inputs {"state", "mask"} und
                Outputs {"policy", "value"}.
            max_batch_size: maximal so viele Anfragen werden zu einem
                Inferenz-Call zusammengefasst. Praktischer Wert: ~num_games / 2
                oder min(num_games, 64).
            request_timeout_s: nach so vielen Sekunden ohne Antwort wirft die
                Game-Thread-Seite einen TimeoutError. Schutz gegen Deadlocks,
                falls der Server-Thread crasht.
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.request_timeout_s = request_timeout_s
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._fatal_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._loop, name="InferenceServer", daemon=True
        )
        self._thread.start()

    def request(
        self,
        state: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Synchrone Anfrage. Blockiert bis Antwort vorliegt oder Timeout."""
        if self._fatal_error is not None:
            raise RuntimeError(
                f"InferenceServer ist gestorben: {self._fatal_error!r}"
            )
        req = _Request(state=state, mask=mask, event=threading.Event())
        self._queue.put(req)
        if not req.event.wait(timeout=self.request_timeout_s):
            raise TimeoutError(
                f"InferenceServer hat in {self.request_timeout_s}s nicht "
                f"geantwortet. Vermutlich Server-Thread tot oder hängt."
            )
        if self._fatal_error is not None:
            raise RuntimeError(
                f"InferenceServer ist während dieser Anfrage gestorben: "
                f"{self._fatal_error!r}"
            )
        if req.response[0] is None:
            raise RuntimeError("Server hat keine Antwort geschrieben.")
        return req.response[0]

    def request_many(
        self,
        states: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> list[tuple[np.ndarray, float]]:
        """Schickt N Anfragen gleichzeitig in die Queue und sammelt die
        Ergebnisse ein. Der Server-Loop wird die Anfragen in einem oder
        wenigen grossen Batches verarbeiten -- Idealfall fuer vektorisierte
        Rollouts mit ~50 Inferenzen pro Decision.

        Returns: Liste der (policy_vec, value_scalar)-Tupel, gleiche
        Reihenfolge wie die Eingaben.
        """
        if self._fatal_error is not None:
            raise RuntimeError(
                f"InferenceServer ist gestorben: {self._fatal_error!r}"
            )
        if len(states) != len(masks):
            raise ValueError(
                f"states ({len(states)}) und masks ({len(masks)}) muessen gleich lang sein"
            )
        if not states:
            return []

        # Alle Requests in einem Rutsch in die Queue. Der Server-Loop wird
        # sie ueber `get_nowait` greedy einsammeln (bis max_batch_size).
        requests: list[_Request] = []
        for s, m in zip(states, masks):
            req = _Request(state=s, mask=m, event=threading.Event())
            self._queue.put(req)
            requests.append(req)

        # Auf alle Antworten warten
        results: list[tuple[np.ndarray, float]] = []
        for req in requests:
            if not req.event.wait(timeout=self.request_timeout_s):
                raise TimeoutError(
                    f"InferenceServer hat in {self.request_timeout_s}s nicht "
                    f"geantwortet (request_many)."
                )
            if self._fatal_error is not None:
                raise RuntimeError(
                    f"InferenceServer ist während request_many gestorben: "
                    f"{self._fatal_error!r}"
                )
            if req.response[0] is None:
                raise RuntimeError("Server hat keine Antwort geschrieben.")
            results.append(req.response[0])
        return results

    def _loop(self) -> None:
        """Server-Hauptschleife (laeuft in eigenem Thread)."""
        try:
            while self._running:
                requests: list[_Request] = []
                # Auf min. eine Anfrage warten -- Timeout, damit Shutdown-Check
                # regelmaessig laeuft
                try:
                    first = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                requests.append(first)

                # Greedy weitere Anfragen sammeln, ohne zu blockieren
                while len(requests) < self.max_batch_size:
                    try:
                        requests.append(self._queue.get_nowait())
                    except queue.Empty:
                        break

                # Batch-Inferenz auf GPU
                states = np.stack([r.state for r in requests])
                masks = np.stack([r.mask for r in requests])
                out = self.model(
                    {"state": states, "mask": masks},
                    training=False,
                )
                policies = out["policy"].numpy()
                values = out["value"].numpy()

                # Antworten zurueck an die Caller
                for req, pol, val in zip(requests, policies, values):
                    req.response[0] = (pol, float(val[0]))
                    req.event.set()
        except BaseException as e:  # noqa: BLE001 -- wir wollen wirklich alles fangen
            self._fatal_error = e
            # Alle haengenden Anfragen aufwecken, sonst Deadlock
            while True:
                try:
                    req = self._queue.get_nowait()
                except queue.Empty:
                    break
                req.event.set()

    def shutdown(self) -> None:
        self._running = False
        self._thread.join(timeout=5.0)


# ----- BatchedRLPlayer: nutzt Server statt direktem Modell-Call -----


class BatchedRLPlayer(RLPlayer):
    """Wie RLPlayer, aber routet alle Inferenz-Anfragen ueber einen
    InferenceServer. Damit koennen viele Spiele parallel laufen und ihre
    Inferenzen werden vom Server zu Batches zusammengefasst.

    Beachte: `self.model` aus dem Parent ist hier auf None gesetzt; alle
    Aufrufe gehen durch self.inference_server.
    """

    def __init__(
        self,
        name: str,
        inference_server: InferenceServer,
        fallback_for_announce: Player | None = None,
    ):
        # Parent erwartet ein "model"; wir geben ihm das Server-Objekt nicht,
        # weil die choose_card-Methode komplett ueberschrieben wird.
        super().__init__(
            name=name,
            model=None,
            fallback_for_announce=fallback_for_announce,
        )
        self.inference_server = inference_server

    def choose_card(self, hand, state: GameState):
        # Trajektorien-Buchhaltung wie im Parent
        if state.round_idx != self._current_round_idx:
            self._round_start_idx.append(len(self.trajectory))
            self._current_round_idx = state.round_idx

        x = encode_state(hand, state).astype(np.float32)
        mask = legal_action_mask(hand, state).astype(np.float32)

        # *** Hier der entscheidende Unterschied zum Parent:
        # Statt self.model(...) direkt zu rufen, schicken wir die Anfrage
        # an den InferenceServer und blockieren bis zur Antwort. Waehrend
        # wir blockieren, sammelt der Server Anfragen aus anderen Threads
        # und macht GPU-Inferenz mit Batch N statt Batch 1.
        policy, value = self.inference_server.request(x, mask)

        legal_policy = policy * mask
        s = legal_policy.sum()
        if s <= 0:
            legal_indices = np.where(mask > 0.5)[0]
            action_idx = int(np.random.choice(legal_indices))
            action_prob = 1.0 / len(legal_indices)
        else:
            legal_policy = legal_policy / s
            action_idx = int(np.random.choice(len(legal_policy), p=legal_policy))
            action_prob = float(legal_policy[action_idx])
        log_prob = float(np.log(max(action_prob, 1e-9)))

        self.trajectory.append(Transition(
            state=x,
            mask=mask,
            action=action_idx,
            log_prob=log_prob,
            value=value,
            reward=0.0,
            done=False,
        ))
        return index_to_card(action_idx)


# ----- Game-Thread: eine Partie inklusive Reward-Assignment -----


def _play_one_game(
    server: InferenceServer,
    target_score: int,
    rng_seed: int,
    is_mix_game: bool,
    rl_team_id: int | None,
    result_holder: list,
) -> None:
    """Spielt EINE Partie mit BatchedRLPlayer-Inferenz. Schreibt die
    Trajektorien (nur RL-Spieler) in `result_holder`."""
    try:
        rng = random.Random(rng_seed)
        teams = list(KREUZ_JASS_TEAMS)

        if is_mix_game:
            players: list[Player] = []
            for seat in range(4):
                if teams[seat] == rl_team_id:
                    players.append(BatchedRLPlayer(
                        name=f"RL{seat}", inference_server=server,
                    ))
                else:
                    players.append(HeuristicPlayer(
                        name=f"H{seat}",
                        rng=random.Random(rng.randint(0, 10**9)),
                    ))
        else:
            players = [
                BatchedRLPlayer(name=f"RL{i}", inference_server=server)
                for i in range(4)
            ]

        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )

        # Reward-Assignment fuer RL-Spieler
        trajectories: list[Trajectory] = []
        for p_idx, player in enumerate(players):
            if not isinstance(player, BatchedRLPlayer):
                continue
            team_id = teams[p_idx]
            round_rewards = []
            for rnd in game.rounds:
                own_pts = rnd.team_total_points.get(team_id, 0)
                opp_pts = sum(
                    p for t, p in rnd.team_total_points.items() if t != team_id
                )
                round_rewards.append((own_pts - opp_pts) / REWARD_SCALE)
            player.assign_round_rewards(round_rewards)
            trajectories.append(player.trajectory)
        result_holder.append(trajectories)
    except BaseException as e:  # noqa: BLE001
        # Im Hauptthread wollen wir den Fehler sehen, nicht silent verlieren
        result_holder.append(e)


# ----- Top-level API: paralleles Sammeln von Trajektorien -----


def collect_trajectories_batched(
    model,
    num_games: int,
    target_score: int = 1000,
    seed: int = 42,
    heuristic_mix_rate: float = 0.0,
    max_batch_size: int | None = None,
) -> list[Trajectory]:
    """Drop-in-Ersatz fuer collect_trajectories aus selfplay.py, aber mit
    InferenceServer + N parallelen Game-Threads.

    Args:
        model: Keras-Modell (Inputs state+mask, Outputs policy+value).
        num_games: Wie viele Partien parallel gespielt werden.
        target_score: Punkteziel pro Partie.
        seed: Master-Seed; pro Spiel wird ein abgeleiteter Sub-Seed benutzt.
        heuristic_mix_rate: Anti-Drift-Mix wie in selfplay.collect_trajectories.
        max_batch_size: Server-Batch-Groesse. Default: min(num_games, 64).

    Returns:
        Liste aller Trajektorien aller RL-Spieler aller Spiele (in beliebiger
        Reihenfolge).
    """
    if max_batch_size is None:
        max_batch_size = min(num_games, 64)

    server = InferenceServer(model=model, max_batch_size=max_batch_size)
    threads: list[threading.Thread] = []
    holders: list[list] = []

    rng = random.Random(seed)
    try:
        for game_idx in range(num_games):
            is_mix = rng.random() < heuristic_mix_rate
            rl_team_id = rng.choice([0, 1]) if is_mix else None
            sub_seed = rng.randint(0, 10**9)
            holder: list = []
            holders.append(holder)
            t = threading.Thread(
                target=_play_one_game,
                args=(server, target_score, sub_seed, is_mix, rl_team_id, holder),
                name=f"Game-{game_idx}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Auf alle Game-Threads warten. Timeout pro Spiel grosszuegig, weil
        # mit target_score=1000 ein Spiel ~5-10s dauert.
        for t in threads:
            t.join(timeout=300.0)
            if t.is_alive():
                raise TimeoutError(
                    f"Game-Thread {t.name} ist nach 300s noch am Leben -- Deadlock?"
                )

        # Trajektorien einsammeln, Errors propagieren
        all_trajectories: list[Trajectory] = []
        for holder in holders:
            if not holder:
                continue
            item = holder[0]
            if isinstance(item, BaseException):
                raise item
            all_trajectories.extend(item)
        return all_trajectories
    finally:
        server.shutdown()
