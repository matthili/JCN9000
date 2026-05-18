"""Tests fuer Solo-Eval-Modul.

Verifiziert:
- four_way_match laeuft ohne illegale Zuege durch
- Paired-Eval rotiert die Rollen korrekt durch alle 4 Sitze
- num_games % 4 != 0 wird beim paired_eval abgelehnt
- Sanity: 4 identische Bots ergeben ungefaehr gleiche Win-Rates
"""

from __future__ import annotations

import random

import pytest

from evaluation.solo_eval import (
    ROLE_A, ROLE_B, ROLE_H1, ROLE_H2,
    _seat_assignment,
    four_way_match,
)
from jass_engine.player import Player
from players.solo_heuristic_player import SoloHeuristicPlayer


def _factory_solo_heuristic(seat: int, rng: random.Random) -> Player:
    return SoloHeuristicPlayer(name=f"H{seat}", rng=rng)


# --- Seat-Assignment ---


def test_seat_assignment_basis():
    """pair_offset 0: A=0, B=1, H1=2, H2=3"""
    m = _seat_assignment(0)
    assert m[0] == ROLE_A
    assert m[1] == ROLE_B
    assert m[2] == ROLE_H1
    assert m[3] == ROLE_H2


def test_seat_assignment_rotiert_zyklisch():
    """Bei vier verschiedenen pair_offsets besucht A jeden Sitz genau einmal."""
    a_seats = []
    for off in range(4):
        m = _seat_assignment(off)
        for seat, role in m.items():
            if role == ROLE_A:
                a_seats.append(seat)
                break
    assert sorted(a_seats) == [0, 1, 2, 3]


def test_seat_assignment_alle_rollen_immer_vorhanden():
    for off in range(4):
        m = _seat_assignment(off)
        roles = sorted(m.values())
        assert roles == sorted([ROLE_A, ROLE_B, ROLE_H1, ROLE_H2])


# --- four_way_match ---


def test_four_way_match_laeuft_durch():
    """Sanity: 8 Partien mit 4 SoloHeuristic-Rollen laufen ohne Fehler durch."""
    result = four_way_match(
        label_a="A",
        factory_a=_factory_solo_heuristic,
        label_b="B",
        factory_b=_factory_solo_heuristic,
        label_h="H",
        factory_h=_factory_solo_heuristic,
        num_games=8,
        target_score=500,
        seed=42,
        paired_eval=False,
    )
    assert result.games_played == 8
    assert result.stats_a.games_played == 8
    assert result.stats_b.games_played == 8
    # Stats_h aggregiert ueber BEIDE H-Sitze, also 16 Spieler-Spiel-Eintraege bei 8 Partien
    assert result.stats_h.games_played == 16


def test_four_way_match_paired_eval_braucht_vier_teilbar():
    with pytest.raises(ValueError, match="Vielfaches von 4"):
        four_way_match(
            label_a="A", factory_a=_factory_solo_heuristic,
            label_b="B", factory_b=_factory_solo_heuristic,
            label_h="H", factory_h=_factory_solo_heuristic,
            num_games=10,  # nicht durch 4 teilbar
            paired_eval=True,
        )


def test_four_way_match_paired_eval_laeuft_durch():
    result = four_way_match(
        label_a="A", factory_a=_factory_solo_heuristic,
        label_b="B", factory_b=_factory_solo_heuristic,
        label_h="H", factory_h=_factory_solo_heuristic,
        num_games=12,  # 3 Paare
        target_score=500,
        seed=7,
        paired_eval=True,
    )
    assert result.games_played == 12


def test_four_way_match_alle_siege_summieren_zu_games():
    """Sanity: A_wins + B_wins + H1+H2_wins = games_played"""
    result = four_way_match(
        label_a="A", factory_a=_factory_solo_heuristic,
        label_b="B", factory_b=_factory_solo_heuristic,
        label_h="H", factory_h=_factory_solo_heuristic,
        num_games=16,
        target_score=500,
        seed=1,
        paired_eval=True,
    )
    # stats_h.games_won zaehlt Siege beider H-Sitze zusammen.
    # Pro Spiel gewinnt genau einer der 4 Sitze.
    total_wins = (
        result.stats_a.games_won
        + result.stats_b.games_won
        + result.stats_h.games_won
    )
    assert total_wins == 16


def test_four_way_match_identische_bots_grob_gleich():
    """4 identische SoloHeuristic-Rollen: Win-Rate sollte um 25% streuen."""
    result = four_way_match(
        label_a="A", factory_a=_factory_solo_heuristic,
        label_b="B", factory_b=_factory_solo_heuristic,
        label_h="H", factory_h=_factory_solo_heuristic,
        num_games=80,
        target_score=500,
        seed=100,
        paired_eval=True,  # eliminiert Rauschen ueber Sitz und Karten
    )
    # Bei paired-eval und identischen Bots: alle 4 Rollen sollten praktisch gleich
    # viele Siege haben. Toleriere bis +-12%.
    a_rate = result.stats_a.win_rate  # je Spiel, also /80
    b_rate = result.stats_b.win_rate
    # stats_h.win_rate ist Siege geteilt durch 2*Partien-Anzahl, weil zwei H-Sitze
    # zusammengezaehlt werden -> ergibt natuerlich ~25% pro H-Sitz.
    h_rate = result.stats_h.games_won / result.stats_h.games_played
    assert 0.13 <= a_rate <= 0.37, f"A-Rate {a_rate:.2%} liegt zu weit weg von 25%"
    assert 0.13 <= b_rate <= 0.37
    assert 0.13 <= h_rate <= 0.37
