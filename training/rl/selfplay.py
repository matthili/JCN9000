"""Self-Play-Spieler und Trajektorien-Sammler fuer RL.

Architektur:
    - Vier Instanzen desselben NN spielen eine Partie gegen sich selbst
    - Jeder Spieler zeichnet seine eigenen Karten-Entscheidungen auf
      (Trajectory: state, mask, action, log_prob, value)
    - Trumpfansage und Weise werden weiterhin vom Heuristik-Bot uebernommen
      (das ist nicht teil des RL-Trainings -- wir lernen nur die Kartenwahl)
    - Am Rundenende wird der Reward (normalisierte Punkte-Differenz) auf alle
      Transitions dieser Runde gesetzt

Wichtig: pro choose_card-Aufruf wird das NN aktuell einzeln gequeryt. Das ist
fuer einen ersten Lauf OK, aber fuer skalieren auf Mio. Partien muss spaeter
batched Inferenz rein (mehrere Spieler-Zustaende gemeinsam durch das NN).
"""

from __future__ import annotations

import random

import numpy as np

from jass_engine.card import Card
from jass_engine.player import GameState, Player
from jass_engine.variant import Announcement, Variant
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS, play_kreuz_jass
from jass_engine.weis import Weis
from players.heuristic_player import HeuristicPlayer
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.trajectory import Trajectory, Transition


REWARD_SCALE = 200.0


class RLPlayer(Player):
    """NN-gesteuerter Spieler, der pro Karten-Entscheidung eine Transition aufzeichnet.

    Aktionen werden **stochastisch** aus der Policy gesampelt (nicht greedy),
    damit die Self-Play-Trajektorien echte Vielfalt zeigen.
    """

    def __init__(self, name: str, model, fallback_for_announce: Player | None = None):
        super().__init__(name)
        self.model = model
        self.fallback = fallback_for_announce or HeuristicPlayer(name + "_fb")
        self.trajectory = Trajectory()
        self._round_start_idx: list[int] = []  # Index, an dem jede Runde in der Trajectory beginnt
        self._current_round_idx: int = -1

    def reset(self) -> None:
        self.trajectory = Trajectory()
        self._round_start_idx = []
        self._current_round_idx = -1

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        return self.fallback.choose_announcement(hand, round_idx, can_push)

    def announce_weise(
        self,
        hand: list[Card],
        variant: Variant,
        possible_weise: list[Weis],
    ) -> list[Weis]:
        return self.fallback.announce_weise(hand, variant, possible_weise)

    def choose_card(self, hand: list[Card], state: GameState) -> Card:
        # Neue Runde -> Index merken, an dem die Trajektorien-Eintraege dieser Runde beginnen
        if state.round_idx != self._current_round_idx:
            self._round_start_idx.append(len(self.trajectory))
            self._current_round_idx = state.round_idx

        x = encode_state(hand, state).astype(np.float32)
        mask = legal_action_mask(hand, state).astype(np.float32)

        # NN-Inferenz (Einzel-Sample -- spaeter batched)
        out = self.model(
            {"state": x[np.newaxis, :], "mask": mask[np.newaxis, :]},
            training=False,
        )
        policy = out["policy"].numpy()[0]
        value = float(out["value"].numpy()[0, 0])

        # Sample-Action: nutze nur legale Karten (Maske ist im Modell aber schon eingebaut,
        # ungueltige Karten haben Wahrscheinlichkeit ~0). Renormalisieren zur Sicherheit.
        legal_policy = policy * mask
        s = legal_policy.sum()
        if s <= 0:
            # Fallback: gleichverteilt unter den legalen Karten
            legal_indices = np.where(mask > 0.5)[0]
            action_idx = int(np.random.choice(legal_indices))
            action_prob = 1.0 / len(legal_indices)
        else:
            legal_policy = legal_policy / s
            action_idx = int(np.random.choice(len(legal_policy), p=legal_policy))
            action_prob = float(legal_policy[action_idx])

        log_prob = float(np.log(max(action_prob, 1e-9)))

        # Letzte Transition der letzten Runde als "done" markieren, sobald die naechste Runde startet
        # Das wird unten in finish_round() gemacht
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

    def assign_round_rewards(self, round_rewards_per_round: list[float]) -> None:
        """Setzt den Reward fuer alle Transitions der jeweiligen Runde.

        Args:
            round_rewards_per_round: Liste der Rewards pro Runde, in der Reihenfolge,
                in der dieser Spieler sie erlebt hat (Index entspricht
                self._round_start_idx).
        """
        # Pro Runde: alle Transitions dieser Runde bekommen denselben Reward;
        # die letzte Transition jeder Runde ist "done".
        starts = self._round_start_idx + [len(self.trajectory)]
        for r_idx, reward in enumerate(round_rewards_per_round):
            if r_idx >= len(self._round_start_idx):
                break
            start = starts[r_idx]
            end = starts[r_idx + 1]
            for i in range(start, end):
                self.trajectory.transitions[i].reward = reward
            # Letzte Transition der Runde: done
            if end > start:
                self.trajectory.transitions[end - 1].done = True


def collect_trajectories(
    model,
    num_games: int = 100,
    target_score: int = 200,
    seed: int = 42,
    heuristic_mix_rate: float = 0.0,
) -> list[Trajectory]:
    """Spielt num_games Partien und sammelt RL-Spieler-Trajektorien.

    Args:
        model: Das NN, mit dem die RL-Spieler entscheiden.
        num_games: Anzahl Partien insgesamt.
        target_score: Punkteziel pro Partie (1000 ueblich).
        seed: RNG-Seed.
        heuristic_mix_rate: Anteil der Partien, in denen RL nur 2 Spieler stellt
            (das andere Team ist HeuristicPlayer). Default 0 = pure Self-Play.
            Sinnvolle Werte ~0.2-0.4 als Anti-Drift-Anker.

    Setup pro Partie:
        - "pure" (Wahrscheinlichkeit 1 - heuristic_mix_rate):
            4 RL-Spieler -> 4 Trajektorien
        - "mix" (Wahrscheinlichkeit heuristic_mix_rate):
            2 RL + 2 Heuristik, RL-Team zufaellig 0 oder 1 -> 2 Trajektorien
    """
    rng = random.Random(seed)
    teams = list(KREUZ_JASS_TEAMS)
    all_trajectories: list[Trajectory] = []

    for game_idx in range(num_games):
        # Mix-Game oder Pure-Selfplay?
        is_mix = rng.random() < heuristic_mix_rate
        if is_mix:
            rl_team_id = rng.choice([0, 1])
            players: list[Player] = []
            for seat in range(4):
                if teams[seat] == rl_team_id:
                    players.append(RLPlayer(name=f"RL{seat}", model=model))
                else:
                    players.append(HeuristicPlayer(
                        name=f"H{seat}",
                        rng=random.Random(rng.randint(0, 10**9)),
                    ))
        else:
            rl_team_id = None  # alle Sitze sind RL
            players = [
                RLPlayer(name=f"RL{i}", model=model)
                for i in range(4)
            ]

        game = play_kreuz_jass(
            players,
            target_score=target_score,
            rng=random.Random(rng.randint(0, 10**9)),
        )

        # Pro Spieler die Reward-Liste basteln (nur fuer RL-Spieler)
        for p_idx, player in enumerate(players):
            if not isinstance(player, RLPlayer):
                continue
            team_id = teams[p_idx]
            round_rewards = []
            for rnd in game.rounds:
                own_pts = rnd.team_total_points.get(team_id, 0)
                opp_pts = sum(p for t, p in rnd.team_total_points.items() if t != team_id)
                round_rewards.append((own_pts - opp_pts) / REWARD_SCALE)
            player.assign_round_rewards(round_rewards)
            all_trajectories.append(player.trajectory)

    return all_trajectories
