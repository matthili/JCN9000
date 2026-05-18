"""Tests fuer die Solo-MCTS-Datengen-Hilfsfunktionen (ohne TF).

Die TF-Importe stehen in main() und werden beim Import nicht ausgeloest.
Damit koennen wir die reinen Helfer (Target-Distribution, Solo-Reward)
direkt importieren und testen.
"""

from __future__ import annotations

import random

import pytest

from training.data.generate_solo_mcts_data import (
    _parse_target_distribution,
    _sample_target,
)
from training.data.solo_vectorized_lookahead import _solo_reward


# --- Target-Distribution-Parser ---


def test_target_distribution_parse_default():
    assert _parse_target_distribution("500:0.5,1000:0.5") == [(500, 0.5), (1000, 0.5)]


def test_target_distribution_parse_drei_eintraege():
    result = _parse_target_distribution("500:0.33,750:0.34,1000:0.33")
    assert len(result) == 3
    assert result[0] == (500, 0.33)
    assert result[2] == (1000, 0.33)


def test_target_distribution_summe_muss_eins_sein():
    with pytest.raises(ValueError, match="summieren"):
        _parse_target_distribution("500:0.5,1000:0.4")


def test_target_distribution_kaputtes_format():
    with pytest.raises(ValueError, match="Form"):
        _parse_target_distribution("500_nodoubt")


# --- Sampling aus der Distribution ---


def test_sample_target_immer_im_set():
    dist = [(500, 0.5), (1000, 0.5)]
    rng = random.Random(0)
    samples = [_sample_target(dist, rng) for _ in range(200)]
    assert all(s in (500, 1000) for s in samples)


def test_sample_target_50_50_verteilung_grob_korrekt():
    dist = [(500, 0.5), (1000, 0.5)]
    rng = random.Random(42)
    samples = [_sample_target(dist, rng) for _ in range(2000)]
    count_500 = sum(1 for s in samples if s == 500)
    # SD bei 2000 Zuegen ~22; toleriere 800-1200
    assert 800 <= count_500 <= 1200


def test_sample_target_extrem_einseitig():
    dist = [(500, 0.01), (1000, 0.99)]
    rng = random.Random(7)
    samples = [_sample_target(dist, rng) for _ in range(500)]
    count_1000 = sum(1 for s in samples if s == 1000)
    assert count_1000 >= 480


# --- Solo-Reward-Berechnung ---


class _MockRollout:
    """Minimal Rollout-aehnliches Objekt fuer Reward-Test."""

    def __init__(self, teams: list[int], root_seat: int, team_points: dict[int, int]):
        self.teams = teams
        self.root_seat = root_seat
        self.team_points = team_points


def test_solo_reward_eigene_punkte_minus_staerkster_gegner():
    r = _MockRollout(
        teams=[0, 1, 2, 3],
        root_seat=2,
        team_points={0: 30, 1: 50, 2: 80, 3: 70},
    )
    # own = 80, max(others) = 70  ->  (80 - 70) / 200 = 0.05
    assert abs(_solo_reward(r) - 0.05) < 1e-6


def test_solo_reward_negativ_wenn_eigene_punkte_niedrig():
    r = _MockRollout(
        teams=[0, 1, 2, 3],
        root_seat=0,
        team_points={0: 20, 1: 50, 2: 80, 3: 7},
    )
    # own = 20, max(others) = 80  ->  (20 - 80) / 200 = -0.30
    assert abs(_solo_reward(r) - (-0.30)) < 1e-6


def test_solo_reward_nur_ein_spieler():
    """Edge case: 1-Team-Konfiguration (sollte praktisch nie passieren)."""
    r = _MockRollout(
        teams=[0, 0, 0, 0],
        root_seat=0,
        team_points={0: 100},
    )
    # own = 100, keine anderen Teams -> opp = 0  ->  100 / 200 = 0.5
    assert abs(_solo_reward(r) - 0.5) < 1e-6


def test_solo_reward_team_2_spieler_paar():
    """Bei nur 2 Teams entspricht max(others) = einziges anderes Team --
    der Code muss in dem Fall denselben Wert liefern wie die alte Team-Version."""
    r = _MockRollout(
        teams=[0, 1, 0, 1],  # team mode
        root_seat=0,
        team_points={0: 100, 1: 57},
    )
    # own_team = 0, max(others) = 57  ->  (100 - 57) / 200 = 0.215
    assert abs(_solo_reward(r) - 0.215) < 1e-6
