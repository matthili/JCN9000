"""MCTS-Lite-Lookahead fuer Bodensee-Jass.

Analog zu `training/data/mcts_lookahead.py`, aber:
- 2 Spieler statt 4
- Bodensee-Determinisierung (eigene + gegnerische Hidden-Karten + Gegnerhand)
- Pro Stich nur 2 Karten -> einfacher als bei Kreuz

Algorithmus pro Karten-Entscheidung:
  Fuer jede legale Karte c:
    Fuer rollout_i in 1..N:
      * Determinisiere die unbekannten Karten (zufaellig)
      * Spiele c als die naechste Karte
      * Lasse den anderen Spieler im aktuellen Stich per NN entscheiden
        (falls c die Lead-Karte war)
      * Berechne den Stich-Reward fuers eigene Konto
    avg[c] = Mittelwert ueber N Rollouts
  best = argmax(avg)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card
from jass_engine.rules import trick_points, trick_winner
from jass_engine.variant import Variant
from training.bodensee_encoder import (
    encode_state_bodensee,
    index_to_card,
    legal_action_mask_bodensee,
)
from training.data.bodensee_determinization import determinize_bodensee_states
from training.rl.batched_selfplay import InferenceServer


@dataclass
class BodenseeLookaheadResult:
    """Ergebnis der MCTS-Lite-Suche."""
    best_card: Card
    card_scores: dict[Card, float]
    rollouts_per_card: int


def _rollout_single_trick_bodensee(
    own_seat: int,
    own_state: BodenseePlayerState,
    state: BodenseeGameState,
    first_move: Card,
    inference_server: InferenceServer,
    i_am_announcer: bool,
    rng: random.Random,
) -> float:
    """Spielt EINEN Bodensee-Stich zu Ende und gibt den Reward fuer mein Konto zurueck.

    Wenn ich Anspieler war: `first_move` ist meine Lead-Karte, der Gegner
    spielt via NN.
    Wenn der Gegner schon angespielt hat: `first_move` ist meine Antwort,
    der Stich ist sofort komplett.
    """
    # 1) Determinisiere
    player_states = determinize_bodensee_states(
        own_state=own_state,
        opp_visible_table=state.opponent_visible_table,
        opp_hand_count=state.opponent_hand_count,
        opp_hidden_count=state.opponent_hidden_table_count,
        completed_tricks=state.completed_tricks,
        current_trick_cards=state.current_trick_cards,
        own_seat=own_seat,
        rng=rng,
    )

    # 2) Lokale Kopien fuer die Simulation
    cur_trick = list(state.current_trick_cards)

    # first_move spielen
    my_ps = player_states[own_seat]
    # Karte aus Hand oder Tisch entfernen
    if first_move in my_ps.hand:
        my_ps.hand.remove(first_move)
    else:
        for stack in my_ps.table:
            if stack.visible == first_move:
                stack.play_visible()
                break
    cur_trick.append(first_move)

    # 3) Falls Stich noch nicht voll (Bodensee = 2 Karten), Gegner spielen lassen
    if len(cur_trick) < 2:
        opp_seat = 1 - own_seat
        opp_ps = player_states[opp_seat]
        # NN-Inferenz fuer den Gegner-Zug
        opp_legal = legal_moves_bodensee(opp_ps, cur_trick, state.variant)
        if not opp_legal:
            # sollte nicht passieren -- Gegner hat noch Karten in Hand+Tisch
            raise RuntimeError("Gegner hat keine legale Karte mehr.")

        # State aus Gegner-Sicht aufbauen
        opp_state = BodenseeGameState(
            player_idx=opp_seat,
            variant=state.variant,
            announcement=state.announcement,
            current_trick_cards=list(cur_trick),
            current_trick_starter=state.current_trick_starter,
            completed_tricks=list(state.completed_tricks),
            opponent_visible_table=my_ps.visible_table_cards,
            opponent_hand_count=len(my_ps.hand),
            opponent_hidden_table_count=sum(1 for s in my_ps.table if s.has_hidden),
            own_score=state.opp_score,
            opp_score=state.own_score,
            round_idx=state.round_idx,
            trick_idx=state.trick_idx,
        )
        x = encode_state_bodensee(
            list(opp_ps.hand),
            opp_ps.table,
            opp_state,
            i_am_announcer=not i_am_announcer,
        ).astype(np.float32)
        mask = legal_action_mask_bodensee(
            list(opp_ps.hand),
            opp_ps.visible_table_cards,
            opp_state,
        ).astype(np.float32)

        policy, _value = inference_server.request(x, mask)
        legal_policy = policy * mask
        if legal_policy.sum() <= 0:
            chosen_card = rng.choice(opp_legal)
        else:
            action_idx = int(np.argmax(legal_policy))
            chosen_card = index_to_card(action_idx)
            if chosen_card not in opp_legal:
                # Fallback bei Race: hoechst-bewertete legale Karte
                chosen_card = max(
                    opp_legal,
                    key=lambda c: legal_policy[__import__('training.encoder', fromlist=['card_index']).card_index(c)],
                )
        cur_trick.append(chosen_card)

    # 4) Stich auswerten
    win_pos = trick_winner(cur_trick, state.variant)
    winner_seat = (state.current_trick_starter + win_pos) % 2
    pts = trick_points(cur_trick, state.variant, is_last_trick=False)

    if winner_seat == own_seat:
        return float(pts)
    return -float(pts)


def mcts_lookahead_best_card_bodensee(
    own_state: BodenseePlayerState,
    state: BodenseeGameState,
    inference_server: InferenceServer,
    i_am_announcer: bool,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
) -> BodenseeLookaheadResult:
    """Pro legaler Karte N Rollouts, dann argmax-Karte zurueck.

    Args:
        own_state: vollstaendiger eigener BodenseePlayerState (mit Hand und
            Tisch-Stapel). Die Hidden-Karten werden im Rollout neu determinisiert.
        state: BodenseeGameState
        inference_server: GPU-Inferenz-Server fuer den Gegner-Zug in Rollouts
        i_am_announcer: ob ich (root player) diese Runde angesagt habe
        rollouts_per_card: Anzahl Determinisierungen pro Kandidaten-Karte
        rng: optional, fuer Reproduzierbarkeit

    Returns:
        BodenseeLookaheadResult mit beste-Karte und Score pro Karte.
    """
    if rng is None:
        rng = random.Random()

    legal = legal_moves_bodensee(own_state, state.current_trick_cards, state.variant)
    if len(legal) == 1:
        return BodenseeLookaheadResult(
            best_card=legal[0],
            card_scores={legal[0]: 0.0},
            rollouts_per_card=0,
        )

    own_seat = state.player_idx
    scores: dict[Card, float] = {}
    for card in legal:
        rewards: list[float] = []
        for _ in range(rollouts_per_card):
            r = _rollout_single_trick_bodensee(
                own_seat=own_seat,
                own_state=own_state,
                state=state,
                first_move=card,
                inference_server=inference_server,
                i_am_announcer=i_am_announcer,
                rng=rng,
            )
            rewards.append(r)
        scores[card] = float(np.mean(rewards))

    best_card = max(scores, key=lambda c: scores[c])
    return BodenseeLookaheadResult(
        best_card=best_card,
        card_scores=scores,
        rollouts_per_card=rollouts_per_card,
    )
