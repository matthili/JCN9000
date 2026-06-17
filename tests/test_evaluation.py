"""Tests fuer die Eval-Tools."""

from __future__ import annotations

import random


from evaluation.elo import DEFAULT_K, DEFAULT_RATING, EloRating
from evaluation.stats import TeamStats, update_stats_from_game
from evaluation.tournament import two_team_match
from jass_engine.variants.kreuz_jass import play_kreuz_jass
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer


# ---------- Elo-Tests ----------

def test_elo_default_rating():
    elo = EloRating()
    assert elo.get("neuer_spieler") == DEFAULT_RATING


def test_elo_sieg_erhoeht_rating():
    elo = EloRating()
    elo.update_team_match(["A", "A"], ["B", "B"], score_a=1.0)
    assert elo.get("A") > DEFAULT_RATING
    assert elo.get("B") < DEFAULT_RATING


def test_elo_summe_konstant():
    """Eine Elo-Partie zwischen vier verschiedenen Spielern ist Null-Summen."""
    elo = EloRating()
    elo.update_team_match(["A1", "A2"], ["B1", "B2"], score_a=1.0)
    sum_ratings = sum(elo.ratings.values())
    expected_sum = 4 * DEFAULT_RATING
    assert abs(sum_ratings - expected_sum) < 1e-6


def test_elo_unentschieden_klein_wenn_gleich_stark():
    elo = EloRating()
    elo.update_team_match(["A", "A"], ["B", "B"], score_a=0.5)
    # Bei gleichem Start-Rating und Unentschieden sollte sich kaum was aendern
    assert abs(elo.get("A") - DEFAULT_RATING) < 1.0
    assert abs(elo.get("B") - DEFAULT_RATING) < 1.0


def test_elo_seriensieger_holt_punkte_auf():
    elo = EloRating()
    for _ in range(20):
        elo.update_team_match(["A", "A"], ["B", "B"], score_a=1.0)
    assert elo.get("A") > DEFAULT_RATING + 100
    assert elo.get("B") < DEFAULT_RATING - 100


def test_elo_leaderboard_sortiert():
    elo = EloRating()
    elo.update_team_match(["A", "A"], ["B", "B"], score_a=1.0)
    elo.update_team_match(["A", "A"], ["C", "C"], score_a=1.0)
    board = elo.leaderboard()
    # A ist der staerkste
    assert board[0][0] == "A"


def test_elo_save_und_load(tmp_path):
    elo = EloRating()
    elo.update_team_match(["A", "A"], ["B", "B"], score_a=1.0)
    p = tmp_path / "elo.json"
    elo.save_json(p)
    loaded = EloRating.load_json(p)
    assert loaded.get("A") == elo.get("A")
    assert loaded.get("B") == elo.get("B")
    assert loaded.k_factor == DEFAULT_K


# ---------- Stats-Tests ----------

def test_stats_aus_einer_partie():
    """Update aus einem echten Spielergebnis."""
    players = [
        HeuristicPlayer("H0"),
        RandomPlayer("R1", rng=random.Random(0)),
        HeuristicPlayer("H2"),
        RandomPlayer("R3", rng=random.Random(1)),
    ]
    game = play_kreuz_jass(players, target_score=500, rng=random.Random(42))
    stats_a = TeamStats()
    stats_b = TeamStats()
    update_stats_from_game(stats_a, stats_b, game)
    assert stats_a.games_played == 1
    assert stats_b.games_played == 1
    # Summe der Siege/Niederlagen/Unentschieden = 1
    assert stats_a.games_won + stats_a.games_lost + stats_a.games_drawn == 1
    # Beide Teams haben gleich viele Runden gespielt
    assert stats_a.total_rounds == stats_b.total_rounds == len(game.rounds)


def test_stats_avg_score_und_winrate():
    stats = TeamStats(games_played=10, games_won=7, total_score=8000)
    assert stats.win_rate == 0.7
    assert stats.avg_score == 800.0


def test_stats_avg_score_bei_null_spielen():
    stats = TeamStats()
    assert stats.avg_score == 0.0
    assert stats.win_rate == 0.0


# ---------- Tournament-Tests ----------

def test_tournament_heuristic_gewinnt_haushoch_gegen_random():
    """Smoke-Test mit 20 Partien -- Heuristic sollte deutlich besser sein."""
    def heuristic_factory(seat, rng):
        return HeuristicPlayer(name=f"H{seat}", rng=rng)
    def random_factory(seat, rng):
        return RandomPlayer(name=f"R{seat}", rng=rng)

    result = two_team_match(
        "Heuristic", heuristic_factory,
        "Random", random_factory,
        num_games=20,
        target_score=500,
        seed=42,
    )
    assert result.games_played == 20
    assert result.stats_a.games_won > result.stats_b.games_won
    # Elo: Heuristic sollte hoeher sein
    assert result.elo.get("Heuristic") > result.elo.get("Random")


def test_tournament_sitz_tausch_balanciert_sitzeffekte():
    """Mit swap_seats_each_half wird die Sitz-Asymmetrie egalisiert -- Random vs Random
    sollte sich grob 50/50 verteilen."""
    def random_factory(seat, rng):
        return RandomPlayer(name=f"R{seat}", rng=rng)

    result = two_team_match(
        "RandomA", random_factory,
        "RandomB", random_factory,
        num_games=40,
        target_score=300,
        seed=123,
        swap_seats_each_half=True,
    )
    # Mit 40 Spielen und Sitz-Tausch sollte die Differenz nicht extrem werden
    diff = abs(result.stats_a.games_won - result.stats_b.games_won)
    assert diff <= 15, f"Erwartete Quasi-Gleichverteilung, bekam {result.stats_a.games_won} vs {result.stats_b.games_won}"
