"""Vektorisierter Full-Round-Lookahead mit GPU-Batching.

Pro Karten-Entscheidung werden ALLE Rollouts (legal_cards x rollouts_per_card)
parallel vorgerueckt: jeder Rollout-Step ist ein NN-Inferenz-Tick mit Batch
von typisch 30-100. So bekommt die GPU richtig zu tun, und Full-Round-
Lookahead (statt nur 1 Stich) liefert strategisch bessere Lehrer-Karten:

  - Single-Trick-Lookahead: "Welche Karte gewinnt mir diesen Stich?"
  - Full-Round-Lookahead:   "Welche Karte ergibt die hoechste Punkte-
                             Differenz bis Rundenende, beruecksichtigt
                             auch Matsch-Bonus und Stich-Sequenz?"

Architektur:
  1. `Rollout` ist ein Mini-State (Haende, aktueller Stich, accumulated points)
  2. `compute_card_scores_vectorized` initialisiert N Rollouts (eine pro
     Card-Rollout-Combo), tickt sie gemeinsam, sammelt am Ende die Rewards.
  3. Pro Tick: ein `server.request_many(...)` mit Batch von z.B. 50.

Vereinfachungen gegenueber der echten Engine:
- Weisen und Stoecke werden nicht modelliert (sind in der Runde schon fix).
- Stoecke-Ansage (wann der Stock gespielt wird) ist statisch (haengt nur
  vom Spieler ab, nicht von Spielzuegen).
- Reward = (own_team_card_points - opp_team_card_points) / REWARD_SCALE,
  Matsch-Bonus eingerechnet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from jass_engine.card import Card
from jass_engine.player import GameState
from jass_engine.rules import (
    LAST_TRICK_BONUS,
    MATCH_BONUS,
    legal_moves,
    trick_points,
    trick_winner,
)
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Variant
from jass_engine.void_inference import infer_forbidden_cards
from training.data.determinization import determinize_hands
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


REWARD_SCALE = 200.0


@dataclass
class Rollout:
    """Ein einzelner Lookahead-Rollout: simuliert ein Restspiel bis Round-Ende.

    Felder:
        card_idx_of_first_move: Index in der legal-Karten-Liste, fuer welche
            Karte dieser Rollout den Reward berechnet.
        hands: aktuelle Haende pro Sitz (werden modifiziert).
        current_trick_cards: Karten im laufenden Stich.
        current_trick_starter: Sitz, der den aktuellen Stich angefangen hat.
        completed_tricks: alle bereits abgeschlossenen Stiche (fuer den
            Encoder, der "played_by_<rel>"-Sektionen daraus baut).
        announcement: die Ansage (fuer Slalom-Variant-Wechsel).
        teams: teams[seat] -> team_id.
        team_points: Punktestand pro Team (akkumuliert).
        team_tricks_won: Anzahl gewonnener Stiche pro Team (fuer Matsch-Check).
        trick_idx: aktueller Stich-Index (0..8).
        root_seat: der Sitz, dessen Decision wir simulieren.
        done: True wenn Round zu Ende ist.
    """

    card_idx_of_first_move: int
    hands: list[list[Card]]
    current_trick_cards: list[Card]
    current_trick_starter: int
    completed_tricks: list[CompletedTrick]
    announcement: object  # Announcement (zirkulaerer Import vermeiden)
    teams: list[int]
    team_points: dict[int, int]
    team_tricks_won: dict[int, int]
    trick_idx: int
    root_seat: int
    done: bool = False

    def _variant_now(self) -> Variant:
        """Effektive Variante des aktuellen Stichs (beruecksichtigt Slalom)."""
        return self.announcement.variant_for_trick(self.trick_idx)

    def needs_inference(self) -> bool:
        """True, wenn der naechste Spieler entscheiden muss."""
        return (not self.done) and (len(self.current_trick_cards) < 4)

    def next_seat(self) -> int:
        return (self.current_trick_starter + len(self.current_trick_cards)) % 4

    def get_state_for_inference(self) -> tuple[list[Card], GameState]:
        seat = self.next_seat()
        cur_hand = self.hands[seat]
        state = GameState(
            player_idx=seat,
            variant=self._variant_now(),
            announcement=self.announcement,
            current_trick_cards=list(self.current_trick_cards),
            current_trick_starter=self.current_trick_starter,
            teams=list(self.teams),
            completed_tricks=list(self.completed_tricks),
            own_team_score=0,
            opp_team_score=0,
            round_idx=0,
            trick_idx=self.trick_idx,
            num_players=4,
        )
        return cur_hand, state

    def apply_action(self, action_idx: int, rng: random.Random) -> None:
        """Wendet die per NN gewaehlte Karte an. Behandelt Trick-Ende
        und ggf. Round-Ende."""
        if self.done:
            return
        seat = self.next_seat()
        cur_hand = self.hands[seat]
        variant = self._variant_now()

        # Wenn die NN-Wahl illegal ist (Race / numerische Fehler), Fallback:
        # erste legale Karte aus cur_hand
        legal = legal_moves(cur_hand, self.current_trick_cards, variant)
        chosen = index_to_card(action_idx)
        if chosen not in legal:
            chosen = legal[0] if legal else cur_hand[0]

        self.current_trick_cards.append(chosen)
        self.hands[seat] = [c for c in self.hands[seat] if c != chosen]

        if len(self.current_trick_cards) == 4:
            self._resolve_trick()

    def _resolve_trick(self) -> None:
        """Stich auswerten, Punkte zuordnen, ggf. Round-Ende."""
        variant = self._variant_now()
        win_pos = trick_winner(self.current_trick_cards, variant)
        winner_seat = (self.current_trick_starter + win_pos) % 4
        is_last = sum(len(h) for h in self.hands) == 0
        pts = trick_points(self.current_trick_cards, variant, is_last_trick=is_last)
        winner_team = self.teams[winner_seat]
        self.team_points[winner_team] = self.team_points.get(winner_team, 0) + pts
        self.team_tricks_won[winner_team] = self.team_tricks_won.get(winner_team, 0) + 1

        self.completed_tricks.append(CompletedTrick(
            starter=self.current_trick_starter,
            cards=tuple(self.current_trick_cards),
        ))
        self.current_trick_cards = []
        self.current_trick_starter = winner_seat
        self.trick_idx += 1

        if is_last:
            # Matsch-Bonus: ein Team hat alle 9 Stiche (gemessen ab Stich 0).
            # Beachte: wir starten ggf. mitten in der Runde, also "alle 9"
            # = alle Stiche dieser Runde, also team_tricks_won[t] == trick_idx.
            for tid, won in self.team_tricks_won.items():
                if won == self.trick_idx and self.trick_idx >= 1:
                    # team hat alle bisherigen Stiche -- Matsch.
                    # Aber nur, wenn ALLE Stiche der Runde gemeint sind --
                    # hier ist `is_last == True`, also Stich 8 erreicht.
                    # Wenn das Team alle 9 Stiche (inkl. Vor-Lookahead-Stiche)
                    # gewonnen hat, kriegt es +100.
                    # Bei mid-round-lookahead muessten wir wissen, ob das
                    # Team auch alle Stiche VOR dem Lookahead-Start gewonnen
                    # hat. Vereinfachung: wir bewerten nur den Effekt der
                    # Rollout-Stiche -- der Matsch-Bonus wird nur dann
                    # vergeben, wenn alle gespielten Stiche dem Team gehoeren
                    # UND mindestens 9 Stiche im Spiel waren (= komplette Runde).
                    if self.trick_idx >= 9:
                        self.team_points[tid] += MATCH_BONUS
            self.done = True

    def get_reward(self) -> float:
        own = self.team_points.get(self.teams[self.root_seat], 0)
        opp_team = next(
            t for t in self.team_points if t != self.teams[self.root_seat]
        ) if len(self.team_points) > 1 else None
        opp = self.team_points.get(opp_team, 0) if opp_team is not None else 0
        return (own - opp) / REWARD_SCALE


def _make_rollout(
    card_idx: int,
    first_move: Card,
    hand: list[Card],
    state: GameState,
    rng: random.Random,
) -> Rollout:
    """Erzeugt ein Rollout-Objekt fuer eine bestimmte first_move-Karte."""
    # Void-aware Determinisierung: Mitspieler, die eine Farbe nicht bedient
    # haben, koennen dort nichts mehr halten -> keine "halluzinierten" Truempfe
    # bei blanken Gegnern (behebt das sinnlose Trumpf-Ziehen).
    forbidden = infer_forbidden_cards(
        state.completed_tricks,
        state.announcement,
        num_players=state.num_players,
    )
    hands = determinize_hands(
        own_seat=state.player_idx,
        own_hand=hand,
        completed_tricks=state.completed_tricks,
        current_trick_cards=state.current_trick_cards,
        current_trick_starter=state.current_trick_starter,
        num_players=state.num_players,
        rng=rng,
        forbidden_by_seat=forbidden,
    )

    # first_move spielen (vor Rollout-Start)
    hands[state.player_idx] = [c for c in hands[state.player_idx] if c != first_move]
    cur_trick = list(state.current_trick_cards) + [first_move]
    cur_starter = state.current_trick_starter
    completed = list(state.completed_tricks)
    trick_idx = state.trick_idx
    team_points: dict[int, int] = {t: 0 for t in set(state.teams)}
    team_tricks_won: dict[int, int] = {t: 0 for t in set(state.teams)}

    # Wenn der first_move den Stich voll gemacht hat: sofort aufloesen
    rollout = Rollout(
        card_idx_of_first_move=card_idx,
        hands=hands,
        current_trick_cards=cur_trick,
        current_trick_starter=cur_starter,
        completed_tricks=completed,
        announcement=state.announcement,
        teams=list(state.teams),
        team_points=team_points,
        team_tricks_won=team_tricks_won,
        trick_idx=trick_idx,
        root_seat=state.player_idx,
    )
    if len(cur_trick) == 4:
        rollout._resolve_trick()
    return rollout


def compute_card_scores_vectorized(
    hand: list[Card],
    state: GameState,
    inference_server: InferenceServer,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
    max_steps_safety: int = 200,
) -> dict[Card, float]:
    """Vektorisierter Full-Round-Lookahead.

    Returns:
        Mapping `card -> mean_reward` ueber alle Rollouts dieser Karte.
    """
    if rng is None:
        rng = random.Random()

    legal = legal_moves(hand, state.current_trick_cards, state.variant)
    if len(legal) == 1:
        return {legal[0]: 0.0}

    # Initialisierung: pro Karte x Rollout je ein Rollout-Objekt
    all_rollouts: list[Rollout] = []
    for card_idx, card in enumerate(legal):
        for _ in range(rollouts_per_card):
            all_rollouts.append(_make_rollout(card_idx, card, hand, state, rng))

    # Tick all rollouts in lockstep, bis alle done sind.
    # Pro Tick: alle Rollouts, die "needs_inference" sagen, schicken EINE
    # Inferenz-Anfrage an den Server gleichzeitig -> Batch.
    for _ in range(max_steps_safety):
        waiting = [r for r in all_rollouts if r.needs_inference()]
        if not waiting:
            break

        states_batch = []
        masks_batch = []
        for r in waiting:
            cur_hand, cur_state = r.get_state_for_inference()
            x = encode_state(cur_hand, cur_state).astype(np.float32)
            m = legal_action_mask(cur_hand, cur_state).astype(np.float32)
            states_batch.append(x)
            masks_batch.append(m)

        results = inference_server.request_many(states_batch, masks_batch)

        for r, mask, (policy, _value) in zip(waiting, masks_batch, results):
            legal_policy = policy * mask
            s = legal_policy.sum()
            if s <= 0:
                legal_indices = np.where(mask > 0.5)[0]
                action_idx = int(rng.choice(legal_indices))
            else:
                action_idx = int(np.argmax(legal_policy))
            r.apply_action(action_idx, rng)

    # Aggregiere Rewards pro first_move-Karte
    rewards_per_card: dict[Card, list[float]] = {c: [] for c in legal}
    for r in all_rollouts:
        rewards_per_card[legal[r.card_idx_of_first_move]].append(r.get_reward())

    return {c: float(np.mean(rs)) if rs else 0.0 for c, rs in rewards_per_card.items()}
