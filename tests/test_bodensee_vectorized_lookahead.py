"""Regressionstests fuer den vektorisierten Bodensee-Full-Round-Lookahead.

Schwerpunkt: Korrektheit der Restrunden-Simulation (Stich-Aufloesung,
Letzter-Stich-Bonus, Reward, Tisch-Aufdeck-Mechanik) und das saubere
Zusammenspiel der Lockstep-Maschinerie mit einem Stub-Inferenz-Server.

Die Stich-Logik wird gegen die Engine-Funktionen `trick_winner` /
`trick_points` geprueft -- damit haengt der Lookahead-Reward direkt an der
verifizierten Regel-Engine.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from jass_engine.bodensee.deal import deal_bodensee
from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card, Rank, Suit
from jass_engine.rules import trick_points, trick_winner
from jass_engine.variant import Announcement, Variant
from training.data.bodensee_vectorized_lookahead import (
    REWARD_SCALE,
    BodenseeRollout,
    best_card_bodensee_vectorized,
    compute_card_scores_bodensee_vectorized,
)


# ---------------------------------------------------------------------------
# Stub-Inferenz-Server: gleichverteilte Policy ueber die legalen Karten.
# argmax(policy * mask) liefert damit den ersten legalen Index -> deterministisch.
# ---------------------------------------------------------------------------
class _UniformStubServer:
    def __init__(self):
        self.calls = 0

    def request_many(self, states, masks):
        out = []
        for m in masks:
            m = np.asarray(m, dtype=np.float32)
            total = m.sum()
            policy = (m / total) if total > 0 else np.ones_like(m) / len(m)
            out.append((policy, np.float32(0.0)))
        self.calls += 1
        return out

    def request(self, x, mask):
        return self.request_many([x], [mask])[0]


class _RaisingStubServer:
    """Faellt durch, wenn ueberhaupt Inferenz angefragt wird -- fuer die
    Faelle, in denen der Lookahead ohne NN auskommen muss."""

    def request_many(self, states, masks):
        raise AssertionError("Inferenz wurde unerwartet angefragt.")

    def request(self, x, mask):
        raise AssertionError("Inferenz wurde unerwartet angefragt.")


def _trumpf_eichel() -> tuple[Variant, Announcement]:
    variant = Variant.trumpf(Suit.EICHEL)
    return variant, Announcement(variant=variant)


# ---------------------------------------------------------------------------
# 1) Eine einzige legale Karte -> Score 0.0, KEINE Inferenz.
# ---------------------------------------------------------------------------
def test_single_legal_card_returns_zero_without_inference():
    variant, ann = _trumpf_eichel()
    only = Card(Suit.LAUB, Rank.ACHT)
    own_state = BodenseePlayerState(hand=[only], table=[])
    state = BodenseeGameState(
        player_idx=0,
        variant=variant,
        announcement=ann,
        current_trick_cards=[],
        current_trick_starter=0,
        completed_tricks=[],
        opponent_hand_count=1,
        trick_idx=17,
    )
    scores = compute_card_scores_bodensee_vectorized(
        own_state=own_state,
        state=state,
        inference_server=_RaisingStubServer(),
        i_am_announcer=True,
        rollouts_per_card=4,
        rng=random.Random(0),
    )
    assert scores == {only: 0.0}


# ---------------------------------------------------------------------------
# 2) Letzter-Stich-Rollout: Reward stimmt mit der Engine ueberein.
#    Beide Spieler haben genau 1 Karte; nach dem Wurzel-Zug ist der
#    Gegnerzug erzwungen (kein NN noetig).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x_card, y_card",
    [
        (Card(Suit.EICHEL, Rank.ASS), Card(Suit.SCHELLE, Rank.SIEBEN)),  # Trumpf-Ass schlaegt Abwurf
        (Card(Suit.SCHELLE, Rank.SECHS), Card(Suit.EICHEL, Rank.SECHS)),  # Gegner trumpft
        (Card(Suit.LAUB, Rank.ASS), Card(Suit.LAUB, Rank.SECHS)),         # Farb-Stich
    ],
)
def test_last_trick_rollout_reward_matches_engine(x_card, y_card):
    variant, ann = _trumpf_eichel()
    ps0 = BodenseePlayerState(hand=[x_card], table=[])
    ps1 = BodenseePlayerState(hand=[y_card], table=[])
    rollout = BodenseeRollout(
        card_idx_of_first_move=0,
        player_states=[ps0, ps1],
        current_trick_cards=[],
        current_trick_starter=0,
        completed_tricks=[],
        announcement=ann,
        root_seat=0,
        announcer_seat=0,
        card_points={0: 0, 1: 0},
        tricks_won={0: 0, 1: 0},
        trick_idx=17,
    )
    rng = random.Random(0)
    rollout._play(0, x_card)
    rollout._auto_advance_forced(rng)

    assert rollout.done, "Runde muss nach dem letzten Stich beendet sein."

    win_pos = trick_winner([x_card, y_card], variant)
    winner_seat = (0 + win_pos) % 2
    pts = trick_points([x_card, y_card], variant, is_last_trick=True)
    own = pts if winner_seat == 0 else 0
    opp = pts if winner_seat == 1 else 0
    assert rollout.get_reward() == pytest.approx((own - opp) / REWARD_SCALE)


# ---------------------------------------------------------------------------
# 3) Tisch-Aufdeck-Mechanik: spielt der Wurzel-Spieler eine sichtbare
#    Tisch-Karte, wird die verdeckte darunter neu sichtbar.
# ---------------------------------------------------------------------------
def test_table_reveal_when_playing_visible_card():
    variant, ann = _trumpf_eichel()
    visible = Card(Suit.LAUB, Rank.KOENIG)
    hidden = Card(Suit.HERZ, Rank.ASS)
    # Spieler 0: leerer Hand-Slot, ein Tisch-Stapel mit sichtbar+verdeckt
    ps0 = BodenseePlayerState(hand=[], table=[TableStack(visible=visible, hidden=hidden)])
    ps1 = BodenseePlayerState(hand=[Card(Suit.SCHELLE, Rank.SIEBEN)], table=[])
    rollout = BodenseeRollout(
        card_idx_of_first_move=0,
        player_states=[ps0, ps1],
        current_trick_cards=[],
        current_trick_starter=0,
        completed_tricks=[],
        announcement=ann,
        root_seat=0,
        announcer_seat=0,
        card_points={0: 0, 1: 0},
        tricks_won={0: 0, 1: 0},
        trick_idx=0,
    )
    before = ps0.total_cards_remaining
    rollout._play(0, visible)

    assert hidden in ps0.visible_table_cards, "Verdeckte Karte muss aufgedeckt sein."
    assert visible not in ps0.visible_table_cards, "Gespielte Karte darf nicht mehr sichtbar sein."
    assert ps0.total_cards_remaining == before - 1


# ---------------------------------------------------------------------------
# 4) End-to-End mit Stub-Server: vollstaendige 18-Stich-Rollouts aus einer
#    echten Startaufstellung. Struktur + Wertebereich der Scores.
# ---------------------------------------------------------------------------
def _trick0_state_for_seat0(rng):
    states = deal_bodensee(rng)
    variant, ann = _trumpf_eichel()
    own = states[0]
    opp = states[1]
    # Die Determinisierung liest nur Hand, sichtbare Tisch-Karten und die
    # has_hidden-Struktur des eigenen States -- die verdeckten Werte werden
    # ohnehin neu zufaellig verteilt. Daher koennen wir den echten Deal-State
    # direkt als own_state uebergeben (kein synthetischer noetig).
    own_state = own
    game_state = BodenseeGameState(
        player_idx=0,
        variant=variant,
        announcement=ann,
        current_trick_cards=[],
        current_trick_starter=0,
        completed_tricks=[],
        opponent_visible_table=opp.visible_table_cards,
        opponent_hand_count=len(opp.hand),
        opponent_hidden_table_count=opp.hidden_table_count,
        own_hidden_table_count=own.hidden_table_count,
        trick_idx=0,
    )
    return own_state, game_state, variant


def test_full_round_scores_structure_and_bounds():
    rng = random.Random(12345)
    own_state, game_state, variant = _trick0_state_for_seat0(rng)
    legal = legal_moves_bodensee(own_state, game_state.current_trick_cards, variant)
    assert len(legal) >= 2, "Test braucht eine echte Auswahl an der Wurzel."

    server = _UniformStubServer()
    scores = compute_card_scores_bodensee_vectorized(
        own_state=own_state,
        state=game_state,
        inference_server=server,
        i_am_announcer=True,
        rollouts_per_card=2,
        rng=random.Random(7),
    )

    assert set(scores.keys()) == set(legal)
    for v in scores.values():
        assert np.isfinite(v)
        # Max. Rundenpunkte ~257 (157 + 100 Matsch) -> |reward| <= 257/200
        assert -1.3 <= v <= 1.3
    assert server.calls > 0, "Full-Round-Rollouts muessen Inferenz nutzen."


def test_best_card_is_argmax_of_scores():
    rng = random.Random(999)
    own_state, game_state, variant = _trick0_state_for_seat0(rng)
    server = _UniformStubServer()
    best, scores = best_card_bodensee_vectorized(
        own_state=own_state,
        state=game_state,
        inference_server=server,
        i_am_announcer=True,
        rollouts_per_card=2,
        rng=random.Random(7),
    )
    assert best in scores
    assert scores[best] == max(scores.values())
