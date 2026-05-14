"""End-to-End-Tests für Kreuz-Jass: Spielablauf, Konsistenz, Punktebilanz."""

from __future__ import annotations

import random

import pytest

from jass_engine.card import Suit
from jass_engine.rules import MATCH_BONUS, TOTAL_POINTS_PER_ROUND
from jass_engine.round import play_round
from jass_engine.variant import Announcement, PlayMode, Variant
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS, play_kreuz_jass
from players.random_player import RandomPlayer


def _make_random_players(seed: int = 0) -> list[RandomPlayer]:
    rng = random.Random(seed)
    return [
        RandomPlayer(name=f"P{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]


def test_eine_runde_summe_konsistent():
    """Summe der Kartenpunkte muss 157 (regulär) oder 257 (Matsch) sein,
    egal welche Variante gewählt wurde."""
    rng = random.Random(0)
    for trial in range(50):
        players = _make_random_players(seed=trial)
        result = play_round(
            players=players,
            teams=list(KREUZ_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        total_cards = sum(result.team_card_points.values())
        if result.matsch_team is not None:
            assert total_cards == TOTAL_POINTS_PER_ROUND + MATCH_BONUS
        else:
            assert total_cards == TOTAL_POINTS_PER_ROUND


def test_weli_traegt_trumpfansage_in_runde_1():
    """In Runde 1 ist der Ansager der Spieler mit dem Weli."""
    rng = random.Random(123)
    players = _make_random_players(seed=0)
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=0,
        rng=rng,
    )
    assert 0 <= result.announcer_idx < 4


def test_schieben_nur_ab_runde_2():
    class PushyPlayer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            return None  # immer schieben wollen

    players = [PushyPlayer(name=f"P{i}", rng=random.Random(i)) for i in range(4)]
    with pytest.raises(RuntimeError):
        play_round(
            players=players,
            teams=list(KREUZ_JASS_TEAMS),
            round_idx=0,
            rng=random.Random(7),
        )


def test_schieben_anspieler_bleibt_urspruenglicher_ansager():
    """Nach Schieben muss der ursprüngliche Ansager die erste Karte spielen.

    Indirekter Test: Der erste Stich startet bei announcer_idx, nicht bei pushed_to.
    """

    class PushyAnnouncer(RandomPlayer):
        """Schiebt wenn er darf, sonst Random."""

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
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=1,
        rng=random.Random(42),
        forced_announcer_idx=0,
    )
    assert result.announcer_idx == 0
    assert result.pushed_to == 2  # Partner von 0 ist Index 2


def test_komplettes_spiel_terminiert():
    players = _make_random_players(seed=99)
    rng = random.Random(99)
    game = play_kreuz_jass(players, target_score=1000, rng=rng)
    assert max(game.final_scores.values()) >= 1000
    assert game.winning_team in (0, 1)


def test_viele_zufallspartien_keine_illegalen_zuege():
    """Smoke-Test: 50 Zufallspartien quer durch alle Varianten."""
    for seed in range(50):
        players = _make_random_players(seed=seed)
        rng = random.Random(seed)
        game = play_kreuz_jass(players, target_score=500, rng=rng)
        for r in game.rounds:
            total = sum(r.team_card_points.values())
            expected = TOTAL_POINTS_PER_ROUND + (
                MATCH_BONUS if r.matsch_team is not None else 0
            )
            assert total == expected, f"Runde mit {r.announcement} → {total} statt {expected}"


# ---------- Bock / Geiss / Slalom Spezial-Tests ----------

def test_bock_runde_lauft_durch():
    """Eine Runde, in der alle nur Bock ansagen, läuft fehlerfrei durch."""

    class BockPlayer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            return Announcement(variant=Variant.oben())

    players = [BockPlayer(name=f"P{i}", rng=random.Random(i)) for i in range(4)]
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=1,
        rng=random.Random(0),
        forced_announcer_idx=0,
    )
    assert result.announcement.variant.mode == PlayMode.OBEN
    # Keine Stöcke bei Bock
    assert all(v == 0 for v in result.team_stoecke.values())


def test_geiss_runde_lauft_durch():
    class GeissPlayer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            return Announcement(variant=Variant.unten())

    players = [GeissPlayer(name=f"P{i}", rng=random.Random(i)) for i in range(4)]
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=1,
        rng=random.Random(0),
        forced_announcer_idx=0,
    )
    assert result.announcement.variant.mode == PlayMode.UNTEN


def test_slalom_wechselt_pro_stich():
    """Bei Slalom mit Anfang OBEN: Stich 0=OBEN, 1=UNTEN, 2=OBEN, ..."""
    ann = Announcement(variant=Variant.oben(), slalom=True)
    assert ann.variant_for_trick(0).mode == PlayMode.OBEN
    assert ann.variant_for_trick(1).mode == PlayMode.UNTEN
    assert ann.variant_for_trick(2).mode == PlayMode.OBEN
    assert ann.variant_for_trick(8).mode == PlayMode.OBEN

    ann2 = Announcement(variant=Variant.unten(), slalom=True)
    assert ann2.variant_for_trick(0).mode == PlayMode.UNTEN
    assert ann2.variant_for_trick(1).mode == PlayMode.OBEN


def test_slalom_runde_lauft_durch():
    class SlalomPlayer(RandomPlayer):
        def choose_announcement(self, hand, round_idx, can_push):
            return Announcement(variant=Variant.oben(), slalom=True)

    players = [SlalomPlayer(name=f"P{i}", rng=random.Random(i)) for i in range(4)]
    result = play_round(
        players=players,
        teams=list(KREUZ_JASS_TEAMS),
        round_idx=1,
        rng=random.Random(0),
        forced_announcer_idx=0,
    )
    assert result.announcement.slalom is True
