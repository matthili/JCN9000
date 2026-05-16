"""MCTS-Lite: 1-Stich-Vorausschau mit NN-Rollouts via InferenceServer.

Pro Karten-Entscheidung:
  Fuer jede legale Karte c:
    Fuer rollout_i in 1..N:
      * Determinisiere die unsichtbaren Haende (zufaellige Verteilung)
      * Spiele c als die naechste Karte
      * Lasse die uebrigen Spieler im aktuellen Stich per NN entscheiden
        (Inferenz via InferenceServer)
      * Berechne den Stich-Reward fuer das eigene Team
        (Stich-Punkte wenn das eigene Team gewinnt, sonst negativ)
    avg[c] = Mittelwert ueber N Rollouts
  best = argmax(avg)

Single-Trick-Lookahead ist eine Vereinfachung gegenueber Full-Round-MCTS,
aber strategisch sinnvoll: viele Heuristik-Spielfehler entstehen schon
innerhalb eines einzelnen Stichs (z.B. schmieren vs. sparen). Das genuegt
fuer eine erste Augmentierung. Spaeter koennte man auf Full-Round erweitern.

GPU-Lastprofil:
- Pro Decision: ~5 legale Karten * N Rollouts * ~3 Mitspieler-Inferenzen
  = 15*N Inferenzen
- Mit N=10 und 64 parallelen Datengen-Spielen koennen pro "Tick" bis zu
  64 * 150 = 9600 Inferenzen anfallen, die der Server zu grossen Batches
  bundelt -> GPU bekommt richtige Arbeit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from jass_engine.card import Card
from jass_engine.player import GameState
from jass_engine.rules import legal_moves, trick_points, trick_winner
from jass_engine.variant import Variant
from training.data.determinization import determinize_hands
from training.encoder import encode_state, index_to_card, legal_action_mask
from training.rl.batched_selfplay import InferenceServer


@dataclass
class LookaheadResult:
    """Ergebnis der MCTS-Lite-Suche fuer eine Entscheidung."""
    best_card: Card
    card_scores: dict[Card, float]  # mean reward pro legale Karte
    rollouts_per_card: int


def _make_mid_trick_state(
    base_state: GameState,
    cur_trick_cards: list[Card],
    cur_player_idx: int,
) -> GameState:
    """Erzeugt einen GameState fuer einen NN-Inferenz mitten im Stich.

    Wir bauen einen lightweight-Klon des Original-States, der nur die
    Trick-Karten und den Spieler-Index ueberschreibt.
    """
    return GameState(
        player_idx=cur_player_idx,
        variant=base_state.variant,
        announcement=base_state.announcement,
        current_trick_cards=list(cur_trick_cards),
        current_trick_starter=base_state.current_trick_starter,
        teams=list(base_state.teams),
        completed_tricks=list(base_state.completed_tricks),
        own_team_score=base_state.own_team_score,
        opp_team_score=base_state.opp_team_score,
        round_idx=base_state.round_idx,
        trick_idx=base_state.trick_idx,
        num_players=base_state.num_players,
    )


def _rollout_single_trick(
    own_seat: int,
    own_hand: list[Card],
    state: GameState,
    first_move: Card,
    inference_server: InferenceServer,
    rng: random.Random,
) -> float:
    """Spielt EINEN Stich zu Ende und gibt den Reward fuers eigene Team zurueck.

    `first_move` wird als die naechste Karte des eigenen Spielers gesetzt;
    die uebrigen Karten im aktuellen Stich werden per NN-Inferenz bestimmt.
    """
    # 1) Determinisiere die Haende der Mitspieler.
    hands = determinize_hands(
        own_seat=own_seat,
        own_hand=own_hand,
        completed_tricks=state.completed_tricks,
        current_trick_cards=state.current_trick_cards,
        current_trick_starter=state.current_trick_starter,
        num_players=state.num_players,
        rng=rng,
    )

    # 2) Karten des aktuellen Stichs + eigene erste Karte.
    cur_trick = list(state.current_trick_cards) + [first_move]
    hands[own_seat] = [c for c in hands[own_seat] if c != first_move]

    # 3) Verbleibende Mitspieler im Stich spielen via NN.
    while len(cur_trick) < state.num_players:
        next_seat = (state.current_trick_starter + len(cur_trick)) % state.num_players
        cur_hand = hands[next_seat]
        cur_state = _make_mid_trick_state(state, cur_trick, next_seat)

        x = encode_state(cur_hand, cur_state).astype(np.float32)
        mask = legal_action_mask(cur_hand, cur_state).astype(np.float32)

        policy, _value = inference_server.request(x, mask)
        legal_policy = policy * mask
        if legal_policy.sum() <= 0:
            # Fallback: zufaellige legale Karte
            legal_indices = np.where(mask > 0.5)[0]
            action_idx = int(rng.choice(legal_indices))
        else:
            action_idx = int(np.argmax(legal_policy))

        chosen = index_to_card(action_idx)
        # Sicherheits-Check: chosen muss in der Hand sein
        if chosen not in cur_hand:
            # Sollte nicht passieren wenn die Mask korrekt ist
            legal_indices = np.where(mask > 0.5)[0]
            for cand_idx in sorted(legal_indices, key=lambda i: -legal_policy[i]):
                cand = index_to_card(int(cand_idx))
                if cand in cur_hand:
                    chosen = cand
                    break

        cur_trick.append(chosen)
        hands[next_seat] = [c for c in hands[next_seat] if c != chosen]

    # 4) Stich auswerten.
    win_pos = trick_winner(cur_trick, state.variant)
    winner_seat = (state.current_trick_starter + win_pos) % state.num_players
    # is_last_trick: konservativ False -- der Last-Trick-Bonus ist beim
    # Single-Trick-Lookahead nicht spielentscheidend. Aenderung waere
    # nur fuer Stich 8 in den letzten Runden relevant.
    pts = trick_points(cur_trick, state.variant, is_last_trick=False)

    own_team = state.teams[own_seat]
    if state.teams[winner_seat] == own_team:
        return float(pts)
    return -float(pts)


def mcts_lookahead_best_card(
    hand: list[Card],
    state: GameState,
    inference_server: InferenceServer,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
) -> LookaheadResult:
    """Pro legaler Karte N Rollouts, dann argmax.

    Args:
        hand: aktuelle eigene Hand.
        state: aktueller Spielzustand.
        inference_server: GPU-Inferenz-Server (fuer NN-Mitspieler in Rollouts).
        rollouts_per_card: Anzahl Determinizations pro Kandidatenkarte.
        rng: RNG fuer Reproduzierbarkeit.

    Returns:
        LookaheadResult mit beste-Karte und Score pro Karte.
    """
    if rng is None:
        rng = random.Random()

    legal = legal_moves(hand, state.current_trick_cards, state.variant)
    if len(legal) == 1:
        # Keine Wahl -- direkt zurueck (kein Rollout noetig).
        return LookaheadResult(
            best_card=legal[0],
            card_scores={legal[0]: 0.0},
            rollouts_per_card=0,
        )

    scores: dict[Card, float] = {}
    for card in legal:
        rewards: list[float] = []
        for _ in range(rollouts_per_card):
            r = _rollout_single_trick(
                own_seat=state.player_idx,
                own_hand=hand,
                state=state,
                first_move=card,
                inference_server=inference_server,
                rng=rng,
            )
            rewards.append(r)
        scores[card] = float(np.mean(rewards))

    best_card = max(scores, key=lambda c: scores[c])
    return LookaheadResult(
        best_card=best_card,
        card_scores=scores,
        rollouts_per_card=rollouts_per_card,
    )
