"""End-to-End-Tests fuer Solo-Jass: 4 Spieler, jeder gegen jeden.

Deckt ab:
- Punkte werden pro Spieler statt pro Team gefuehrt
- Schieben ist deaktiviert
- Default-Spielziel 500
- Spielende-Bedingung (erster zu 500)
- Konsistenz der Punktesumme pro Runde
"""

from __future__ import annotations

import random

import pytest

from jass_engine.rules import MATCH_BONUS, TOTAL_POINTS_PER_ROUND
from jass_engine.round import play_round
from jass_engine.variants.solo_jass import (
    DEFAULT_SOLO_TARGET_SCORE,
    SOLO_JASS_TEAMS,
    play_solo_jass,
)
from players.random_player import RandomPlayer


def _make_random_players(seed: int = 0) -> list[RandomPlayer]:
    rng = random.Random(seed)
    return [
        RandomPlayer(name=f"P{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]


# --- Konfigurations-Smoke ---


def test_solo_default_target_ist_500():
    assert DEFAULT_SOLO_TARGET_SCORE == 500


def test_solo_teams_sind_vier_separate_konten():
    """Jeder Spieler hat ein eigenes Punkte-Konto, kein Team-Sharing."""
    assert SOLO_JASS_TEAMS == (0, 1, 2, 3)
    assert len(set(SOLO_JASS_TEAMS)) == 4


def test_solo_target_unter_500_abgelehnt():
    players = _make_random_players(seed=0)
    with pytest.raises(ValueError):
        play_solo_jass(players, target_score=400)


def test_solo_falsche_spielerzahl_abgelehnt():
    players = _make_random_players(seed=0)[:3]
    with pytest.raises(ValueError):
        play_solo_jass(players)


# --- Spielablauf ---


def test_solo_vier_punkte_konten_im_ergebnis():
    """Nach einer Partie sind die final_scores Spieler-IDs 0..3."""
    rng = random.Random(42)
    game = play_solo_jass(_make_random_players(seed=42), rng=rng)
    assert len(game.final_scores) == 4
    assert set(game.final_scores.keys()) == {0, 1, 2, 3}


def test_solo_gewinner_ist_einer_der_vier_spieler():
    """winning_team enthaelt im Solo den Spieler-Index 0..3."""
    rng = random.Random(7)
    game = play_solo_jass(_make_random_players(seed=7), rng=rng)
    assert game.winning_team in (0, 1, 2, 3)
    # Und der Gewinner hat tatsaechlich den hoechsten Score
    assert game.final_scores[game.winning_team] == max(game.final_scores.values())


def test_solo_partie_terminiert_mit_zielerreichung():
    rng = random.Random(99)
    game = play_solo_jass(_make_random_players(seed=99), target_score=500, rng=rng)
    assert max(game.final_scores.values()) >= 500


def test_solo_konfigurierbares_target():
    """Auch hoehere Ziele (z.B. 1000) muessen funktionieren."""
    rng = random.Random(11)
    game = play_solo_jass(_make_random_players(seed=11), target_score=1000, rng=rng)
    assert max(game.final_scores.values()) >= 1000


# --- Punkte-Konsistenz pro Runde ---


def test_solo_summe_157_pro_runde():
    """Summe der Kartenpunkte aller 4 Spieler = 157 (oder 257 bei Matsch)."""
    rng = random.Random(0)
    for trial in range(30):
        players = _make_random_players(seed=trial)
        result = play_round(
            players=players,
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        total = sum(result.team_card_points.values())
        expected = TOTAL_POINTS_PER_ROUND + (
            MATCH_BONUS if result.matsch_team is not None else 0
        )
        assert total == expected, (
            f"Trial {trial} ({result.announcement}): {total} != {expected}"
        )


def test_solo_stichpunkte_gehen_an_stichgewinner():
    """Im Solo gibt es kein Team-Schmieren mehr: nur der Stichgewinner kassiert."""
    rng = random.Random(321)
    for trial in range(30):
        players = _make_random_players(seed=trial)
        result = play_round(
            players=players,
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        from_tricks = {0: 0, 1: 0, 2: 0, 3: 0}
        for winner, pts in zip(result.trick_winners, result.trick_points):
            from_tricks[winner] += pts
        for pid in range(4):
            expected = from_tricks[pid]
            if result.matsch_team == pid:
                expected += MATCH_BONUS
            assert result.team_card_points[pid] == expected, (
                f"Spieler {pid}: erwartet {expected}, "
                f"bekommen {result.team_card_points[pid]}"
            )


# --- Schieben deaktiviert ---


def test_solo_schieben_wird_abgelehnt():
    """Ein Spieler, der None zurueckgibt (= schieben), muss zum RuntimeError fuehren."""

    class PushyPlayer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            return None  # versuch zu schieben

    players = [PushyPlayer(name=f"P{i}", rng=random.Random(i)) for i in range(4)]
    with pytest.raises(RuntimeError):
        play_round(
            players=players,
            teams=list(SOLO_JASS_TEAMS),
            round_idx=1,  # ab Runde 2 waere Schieben sonst erlaubt
            rng=random.Random(0),
            allow_push=False,
        )


def test_kreuz_jass_default_erlaubt_weiterhin_schieben():
    """Backward-Compatibility: bei play_round ohne allow_push (Default True)
    funktioniert Schieben wie bisher (ab Runde 2)."""
    from jass_engine.variant import Announcement, Variant
    from jass_engine.card import Suit
    from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS

    class PushyAnnouncer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            if can_push:
                return None
            return Announcement(variant=Variant.trumpf(Suit.EICHEL))

    players = [
        PushyAnnouncer(name="P0", rng=random.Random(0)),
        RandomPlayer(name="P1", rng=random.Random(1)),
        PushyAnnouncer(name="P2", rng=random.Random(2)),
        RandomPlayer(name="P3", rng=random.Random(3)),
    ]
    # Kein allow_push gesetzt -> Default True, Schieben darf funktionieren
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=1,
        rng=random.Random(42),
        forced_announcer_idx=0,
    )
    assert result.pushed_to == 2  # Standard-Schieben funktioniert noch


# --- Smoke-Test ueber viele Partien ---


def test_solo_50_zufallspartien_keine_illegalen_zuege():
    for seed in range(50):
        players = _make_random_players(seed=seed)
        rng = random.Random(seed)
        game = play_solo_jass(players, target_score=500, rng=rng)
        for r in game.rounds:
            total = sum(r.team_card_points.values())
            expected = TOTAL_POINTS_PER_ROUND + (
                MATCH_BONUS if r.matsch_team is not None else 0
            )
            assert total == expected
