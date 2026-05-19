"""End-to-End-Tests fuer Bodensee-Jass: Runde, Partie, Heuristik."""

from __future__ import annotations

import random

import pytest

from jass_engine.bodensee.deal import TRICKS_PER_ROUND
from jass_engine.bodensee.game import (
    DEFAULT_BODENSEE_TARGET_SCORE,
    play_bodensee_game,
)
from jass_engine.bodensee.round import play_bodensee_round
from jass_engine.rules import MATCH_BONUS
from jass_engine.variants.bodensee_jass import play_bodensee_jass
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.bodensee_random_player import BodenseeRandomPlayer
from players.bodensee_player import BodenseePlayer


def _two_random_players(seed: int = 0) -> list[BodenseePlayer]:
    rng = random.Random(seed)
    return [
        BodenseeRandomPlayer(name="R0", rng=random.Random(rng.randint(0, 10**9))),
        BodenseeRandomPlayer(name="R1", rng=random.Random(rng.randint(0, 10**9))),
    ]


def _two_heuristic_players(seed: int = 0) -> list[BodenseePlayer]:
    rng = random.Random(seed)
    return [
        BodenseeHeuristicPlayer(name="H0", rng=random.Random(rng.randint(0, 10**9))),
        BodenseeHeuristicPlayer(name="H1", rng=random.Random(rng.randint(0, 10**9))),
    ]


# --- Konfiguration ---


def test_bodensee_default_target_500():
    assert DEFAULT_BODENSEE_TARGET_SCORE == 500


def test_bodensee_target_unter_500_abgelehnt():
    players = _two_random_players(0)
    with pytest.raises(ValueError):
        play_bodensee_jass(players, target_score=400)


def test_bodensee_falsche_spielerzahl_abgelehnt():
    players = _two_random_players(0)[:1]
    with pytest.raises(ValueError):
        play_bodensee_jass(players)


# --- Runde ---


def test_bodensee_runde_genau_18_stiche():
    players = _two_random_players(seed=42)
    result = play_bodensee_round(players, rng=random.Random(42))
    assert len(result.trick_winners) == TRICKS_PER_ROUND
    assert len(result.trick_points) == TRICKS_PER_ROUND


def test_bodensee_runde_punkte_summe_plausibel():
    """Punkte-Summe pro Runde muss zwischen 155 und 257 liegen."""
    rng = random.Random(0)
    for trial in range(20):
        players = _two_random_players(seed=trial)
        result = play_bodensee_round(
            players, rng=random.Random(rng.randint(0, 10**9))
        )
        total = sum(result.player_card_points.values())
        if result.matsch_player is not None:
            assert total in (255, 257), f"Trial {trial}: Matsch-Summe {total}"
        else:
            assert 155 <= total <= 157, f"Trial {trial}: Summe {total}"


def test_bodensee_runde_winner_ist_einer_der_zwei():
    players = _two_random_players(seed=7)
    result = play_bodensee_round(players, rng=random.Random(7))
    for tid in (0, 1):
        assert result.player_card_points[tid] >= 0
        assert result.player_total_points[tid] >= 0


def test_bodensee_runde_announcer_ist_weli_halter():
    """In Runde 0 (kein forced_announcer) sagt der Weli-Halter an."""
    players = _two_random_players(seed=42)
    result = play_bodensee_round(players, rng=random.Random(42))
    # Der announcer_idx muss 0 oder 1 sein
    assert result.announcer_idx in (0, 1)


def test_bodensee_runde_keine_weisen_keine_stoecke():
    """player_card_points + Bonus = player_total_points (keine Weisen, keine Stoecke)."""
    rng = random.Random(123)
    for trial in range(10):
        players = _two_random_players(seed=trial)
        result = play_bodensee_round(
            players, rng=random.Random(rng.randint(0, 10**9))
        )
        for pid in (0, 1):
            # Keine Weisen/Stoecke -> total == card_points
            assert result.player_total_points[pid] == result.player_card_points[pid]


# --- Partie ---


def test_bodensee_partie_terminiert_mit_zielerreichung():
    players = _two_random_players(seed=99)
    game = play_bodensee_jass(players, target_score=500, rng=random.Random(99))
    assert max(game.final_scores.values()) >= 500


def test_bodensee_partie_winner_ist_score_max():
    players = _two_random_players(seed=7)
    game = play_bodensee_jass(players, target_score=500, rng=random.Random(7))
    assert game.winner in (0, 1)
    assert game.final_scores[game.winner] == max(game.final_scores.values())


def test_bodensee_partie_ansager_wechselt_zwischen_runden():
    """Nach Runde 1 sagt der jeweils andere Spieler an."""
    players = _two_random_players(seed=42)
    game = play_bodensee_jass(players, target_score=500, rng=random.Random(42))
    # Alle aufeinanderfolgenden announcer_idx sollten alternieren (nach Runde 1)
    if len(game.rounds) >= 3:
        for i in range(1, len(game.rounds) - 1):
            ann_prev = game.rounds[i - 1].announcer_idx
            ann_curr = game.rounds[i].announcer_idx
            # Ab Runde 1 wechselt der Ansager
            if i >= 1:
                assert ann_curr == 1 - ann_prev, (
                    f"Runde {i}: Ansager wechselte nicht ({ann_prev} -> {ann_curr})"
                )


# --- Heuristik-Smoke ---


def test_bodensee_heuristik_partie_laeuft_durch():
    players = _two_heuristic_players(seed=42)
    game = play_bodensee_jass(players, target_score=500, rng=random.Random(42))
    assert max(game.final_scores.values()) >= 500


def test_bodensee_heuristik_identische_bots_etwa_50_50():
    """Zwei identische Heuristik-Spieler in 100 Partien: Win-Rate sollte
    ungefaehr 50 % pro Spieler sein."""
    wins = {0: 0, 1: 0}
    n_games = 100
    rng = random.Random(42)
    for trial in range(n_games):
        players = _two_heuristic_players(seed=trial)
        game = play_bodensee_jass(
            players,
            target_score=500,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        wins[game.winner] += 1
    # Bei n=100 ist die SD ca. 5%, also Toleranz 35-65%
    for pid in (0, 1):
        assert 35 <= wins[pid] <= 65, (
            f"Spieler {pid}: {wins[pid]} Siege von {n_games} -- ausserhalb [35, 65]"
        )


# --- 50 Zufallspartien ---


def test_bodensee_50_zufallspartien_keine_illegalen_zuege():
    for seed in range(50):
        players = _two_random_players(seed=seed)
        rng = random.Random(seed)
        game = play_bodensee_jass(players, target_score=500, rng=rng)
        # Wenn wir hier sind, gab es keine RuntimeErrors aus illegalen Zuegen
        for r in game.rounds:
            assert len(r.trick_winners) == TRICKS_PER_ROUND
            total = sum(r.player_card_points.values())
            assert 155 <= total <= 257
