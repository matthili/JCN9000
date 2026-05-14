"""Tests für Weisen-Erkennung und -Vergleich."""

from __future__ import annotations

from jass_engine.card import Card, Rank, Suit
from jass_engine.weis import (
    STOECKE_POINTS,
    Weis,
    WeisKind,
    compare_team_weise,
    find_weise,
    has_stoecke,
    stoecke_weis,
)


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


# ---------- Sequenzen ----------

def test_sequenz_3_blatt_20_punkte():
    hand = [
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        C(Suit.EICHEL, Rank.NEUN),
        C(Suit.HERZ, Rank.ASS),
    ]
    weise = find_weise(hand)
    seqs = [w for w in weise if w.kind == WeisKind.SEQUENCE]
    assert len(seqs) == 1
    assert seqs[0].points == 20
    assert len(seqs[0].cards) == 3


def test_sequenz_4_blatt_50_punkte():
    hand = [
        C(Suit.LAUB, Rank.NEUN),
        C(Suit.LAUB, Rank.ZEHN),
        C(Suit.LAUB, Rank.UNTER),
        C(Suit.LAUB, Rank.OBER),
    ]
    weise = find_weise(hand)
    seqs = [w for w in weise if w.kind == WeisKind.SEQUENCE]
    assert len(seqs) == 1
    assert seqs[0].points == 50


def test_sequenz_9_blatt_180_punkte():
    hand = [C(Suit.HERZ, r) for r in (
        Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN,
        Rank.UNTER, Rank.OBER, Rank.KOENIG, Rank.ASS,
    )]
    weise = find_weise(hand)
    seqs = [w for w in weise if w.kind == WeisKind.SEQUENCE]
    assert len(seqs) == 1
    assert seqs[0].points == 180


def test_keine_sequenz_unter_3():
    hand = [
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        C(Suit.HERZ, Rank.ASS),
    ]
    weise = find_weise(hand)
    assert all(w.kind != WeisKind.SEQUENCE for w in weise)


def test_lueckenhafte_karten_keine_sequenz():
    hand = [
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.NEUN),  # 8 fehlt
        C(Suit.EICHEL, Rank.ZEHN),
    ]
    weise = find_weise(hand)
    assert all(w.kind != WeisKind.SEQUENCE for w in weise)


def test_zwei_separate_sequenzen_in_einer_farbe():
    hand = [
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        # Lücke bei 9
        C(Suit.EICHEL, Rank.OBER),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.EICHEL, Rank.ASS),
    ]
    weise = find_weise(hand)
    seqs = [w for w in weise if w.kind == WeisKind.SEQUENCE]
    assert len(seqs) == 2


# ---------- Vierlinge ----------

def test_vier_unter_200_punkte():
    hand = [C(s, Rank.UNTER) for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)]
    weise = find_weise(hand)
    fours = [w for w in weise if w.kind == WeisKind.FOUR_OF_KIND]
    assert len(fours) == 1
    assert fours[0].points == 200


def test_vier_neuner_150_punkte():
    hand = [C(s, Rank.NEUN) for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)]
    weise = find_weise(hand)
    fours = [w for w in weise if w.kind == WeisKind.FOUR_OF_KIND]
    assert len(fours) == 1
    assert fours[0].points == 150


def test_vier_asse_100_punkte():
    hand = [C(s, Rank.ASS) for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)]
    weise = find_weise(hand)
    fours = [w for w in weise if w.kind == WeisKind.FOUR_OF_KIND]
    assert len(fours) == 1
    assert fours[0].points == 100


def test_vier_sechser_zaehlen_nicht():
    hand = [C(s, Rank.SECHS) for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)]
    weise = find_weise(hand)
    assert all(w.kind != WeisKind.FOUR_OF_KIND for w in weise)


# ---------- Stöcke ----------

def test_stoecke_erkannt():
    trumpf = Suit.EICHEL
    hand = [
        C(Suit.EICHEL, Rank.OBER),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.HERZ, Rank.ASS),
    ]
    assert has_stoecke(hand, trumpf) is True


def test_stoecke_brauchen_beide_karten():
    trumpf = Suit.EICHEL
    hand = [C(Suit.EICHEL, Rank.OBER), C(Suit.HERZ, Rank.KOENIG)]
    assert has_stoecke(hand, trumpf) is False


def test_stoecke_punktwert_20():
    weis = stoecke_weis(Suit.EICHEL)
    assert weis.points == STOECKE_POINTS == 20


def test_stoecke_nur_bei_trumpf_modus():
    from jass_engine.variant import Variant
    from jass_engine.weis import stoecke_apply
    assert stoecke_apply(Variant.trumpf(Suit.EICHEL)) is True
    assert stoecke_apply(Variant.oben()) is False
    assert stoecke_apply(Variant.unten()) is False


# ---------- Team-Vergleich ----------

def _seq(suit: Suit, ranks: list[Rank]) -> Weis:
    points_table = {3: 20, 4: 50, 5: 100, 6: 120, 7: 140, 8: 160, 9: 180}
    return Weis(
        kind=WeisKind.SEQUENCE,
        cards=tuple(Card(suit, r) for r in ranks),
        points=points_table[len(ranks)],
        top_rank=ranks[-1],
    )


def test_team_mit_hoeherem_weis_gewinnt_alle_eigenen_punkte():
    # Spieler 0 (Team 0): 4-Blatt (50)
    # Spieler 1 (Team 1): 3-Blatt (20)
    # Spieler 2 (Team 0): 3-Blatt (20)
    # Spieler 3 (Team 1): keine
    weise_per_player = [
        [_seq(Suit.EICHEL, [Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN])],
        [_seq(Suit.HERZ, [Rank.SECHS, Rank.SIEBEN, Rank.ACHT])],
        [_seq(Suit.LAUB, [Rank.SECHS, Rank.SIEBEN, Rank.ACHT])],
        [],
    ]
    teams = [0, 1, 0, 1]
    order = [0, 1, 2, 3]
    result = compare_team_weise(weise_per_player, teams, order)
    # Team 0 hat den höheren Einzel-Weis (50) → bekommt 50 + 20 = 70
    assert result[0].points == 70
    assert result[1].points == 0


def test_weis_gleichstand_zuerst_in_spielreihenfolge_gewinnt():
    # Beide haben 3-Blatt (20). Spieler 0 spielt zuerst (Spielreihenfolge [0,1,2,3]).
    weise_per_player = [
        [_seq(Suit.EICHEL, [Rank.SIEBEN, Rank.ACHT, Rank.NEUN])],
        [_seq(Suit.HERZ, [Rank.SIEBEN, Rank.ACHT, Rank.NEUN])],
        [],
        [],
    ]
    teams = [0, 1, 0, 1]
    result = compare_team_weise(weise_per_player, teams, announcement_order=[0, 1, 2, 3])
    # Beide Sequenzen haben top_rank=NEUN → strikter Gleichstand → Reihenfolge entscheidet
    assert result[0].points == 20
    assert result[1].points == 0


def test_keine_weise_kein_punkt():
    weise_per_player = [[], [], [], []]
    teams = [0, 1, 0, 1]
    result = compare_team_weise(weise_per_player, teams, [0, 1, 2, 3])
    assert result[0].points == 0
    assert result[1].points == 0
