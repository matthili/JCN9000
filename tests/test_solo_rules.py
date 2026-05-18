"""Regel-Spezifika beim Solo-Jass: Weisen, Matsch, Stoecke."""

from __future__ import annotations

import random

from jass_engine.round import play_round
from jass_engine.variants.solo_jass import SOLO_JASS_TEAMS
from players.random_player import RandomPlayer


def _players(seed: int) -> list[RandomPlayer]:
    rng = random.Random(seed)
    return [
        RandomPlayer(name=f"P{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]


# --- Weisen ---


def test_solo_weisen_nur_hoechster_haelt_die_punkte():
    """Im Solo gibt es kein gemeinsames Team-Weisen-Konto.
    Maximal ein Spieler bekommt Weis-Punkte."""
    rng = random.Random(123)
    for trial in range(30):
        result = play_round(
            players=_players(trial),
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        weis_winners = [
            pid for pid, r in result.team_weis_results.items() if r.points > 0
        ]
        assert len(weis_winners) <= 1, (
            f"Solo erlaubt max. 1 Weis-Gewinner, aber {weis_winners}"
        )


def test_solo_weisen_alle_vier_konten_befuellt():
    """Auch Spieler ohne Weisen muessen einen 0-Punkte-Eintrag haben."""
    result = play_round(
        players=_players(0),
        teams=list(SOLO_JASS_TEAMS),
        round_idx=0,
        rng=random.Random(0),
        allow_push=False,
    )
    assert set(result.team_weis_results.keys()) == {0, 1, 2, 3}


# --- Stoecke ---


def test_solo_stoecke_an_einzelnen_spieler():
    """Stoecke (+20) gehen an den Stockhalter persoenlich, nicht ans Team."""
    rng = random.Random(456)
    for trial in range(50):
        result = play_round(
            players=_players(trial),
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        # Maximal ein Spieler hat positiv Stoecke-Punkte
        holders = [pid for pid, pts in result.team_stoecke.items() if pts > 0]
        assert len(holders) <= 1
        # Falls jemand Stoecke hat, sind es genau 20 Punkte
        for pid, pts in result.team_stoecke.items():
            assert pts in (0, 20)


# --- Matsch ---


def test_solo_matsch_falls_aufgetreten_geht_an_einzelnen_spieler():
    """Wenn Matsch passiert: +100 fuer den Spieler, der alle 9 Stiche gewinnt.
    Matsch ist sehr selten -- daher kein assert auf Auftreten, nur auf Korrektheit."""
    rng = random.Random(789)
    matsch_count = 0
    for trial in range(300):
        result = play_round(
            players=_players(trial * 7),
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        if result.matsch_team is not None:
            matsch_count += 1
            # Der Matsch-"Team"-ID ist im Solo der Spieler-Index
            assert result.matsch_team in (0, 1, 2, 3)
            # Punktesumme = 157 + 100 = 257
            assert sum(result.team_card_points.values()) == 257
            # Genau der Matsch-Spieler hat alle 9 Stiche
            counts = [result.trick_winners.count(p) for p in range(4)]
            assert counts[result.matsch_team] == 9
            # Die anderen drei haben 0 Stiche
            for p in range(4):
                if p != result.matsch_team:
                    assert counts[p] == 0


# --- total points per round (mit Weisen + Stoecke) ---


def test_solo_total_points_setzen_sich_korrekt_zusammen():
    """team_total_points = team_card_points + Weisen + Stoecke (pro Spieler)."""
    rng = random.Random(202)
    for trial in range(30):
        result = play_round(
            players=_players(trial),
            teams=list(SOLO_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
            allow_push=False,
        )
        for pid in range(4):
            expected = (
                result.team_card_points[pid]
                + result.team_weis_results[pid].points
                + result.team_stoecke[pid]
            )
            assert result.team_total_points[pid] == expected
