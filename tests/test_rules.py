"""Tests für rules.py — kritische Regel-Edge-Cases für alle Varianten."""

from __future__ import annotations

from jass_engine.card import Card, Rank, Suit
from jass_engine.rules import (
    LAST_TRICK_BONUS,
    MATCH_BONUS,
    TOTAL_POINTS_PER_ROUND,
    card_strength,
    card_value,
    legal_moves,
    total_points_per_round,
    trick_points,
    trick_winner,
)
from jass_engine.variant import Variant


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


TRUMPF_EICHEL = Variant.trumpf(Suit.EICHEL)
OBEN = Variant.oben()
UNTEN = Variant.unten()


# ---------- Punktewerte: Trumpf-Modus ----------

def test_punktwerte_nicht_trumpf():
    v = TRUMPF_EICHEL
    assert card_value(C(Suit.HERZ, Rank.ASS), v) == 11
    assert card_value(C(Suit.HERZ, Rank.ZEHN), v) == 10
    assert card_value(C(Suit.HERZ, Rank.KOENIG), v) == 4
    assert card_value(C(Suit.HERZ, Rank.OBER), v) == 3
    assert card_value(C(Suit.HERZ, Rank.UNTER), v) == 2
    for r in (Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN):
        assert card_value(C(Suit.HERZ, r), v) == 0


def test_punktwerte_trumpf_buur_20_nell_14():
    v = TRUMPF_EICHEL
    assert card_value(C(Suit.EICHEL, Rank.UNTER), v) == 20
    assert card_value(C(Suit.EICHEL, Rank.NEUN), v) == 14
    assert card_value(C(Suit.EICHEL, Rank.ASS), v) == 11


def test_summe_aller_kartenpunkte_152_im_trumpf():
    v = TRUMPF_EICHEL
    total = 0
    for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB):
        for r in (
            Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN,
            Rank.UNTER, Rank.OBER, Rank.KOENIG, Rank.ASS,
        ):
            total += card_value(Card(s, r), v)
    assert total == 152


# ---------- Punktewerte: Bock/Geiss ----------

def test_punktwerte_oben_8er_zaehlt_8():
    v = OBEN
    assert card_value(C(Suit.HERZ, Rank.ACHT), v) == 8
    assert card_value(C(Suit.EICHEL, Rank.ACHT), v) == 8
    # Buur und Nell sind in Bock NICHT mehr 20/14 wert
    assert card_value(C(Suit.HERZ, Rank.UNTER), v) == 2
    assert card_value(C(Suit.HERZ, Rank.NEUN), v) == 0


def test_punktwerte_unten_8er_zaehlt_8():
    v = UNTEN
    assert card_value(C(Suit.HERZ, Rank.ACHT), v) == 8


def test_summe_kartenpunkte_oben_152():
    """Bock/Geiss: pro Farbe 11+10+4+3+2+8 = 38, mal 4 Farben = 152.

    Im Vergleich zum Trumpf: Buur (20→2) verliert 18, Nell (14→0) verliert 14, die
    4 Achter (0→8) gewinnen 32. Saldo: -18-14+32 = 0. Stichpunkte-Summe bleibt 152.
    """
    v = OBEN
    total = 0
    for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB):
        for r in (
            Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN,
            Rank.UNTER, Rank.OBER, Rank.KOENIG, Rank.ASS,
        ):
            total += card_value(Card(s, r), v)
    assert total == 152


def test_summe_kartenpunkte_unten_152():
    v = UNTEN
    total = sum(
        card_value(Card(s, r), v)
        for s in (Suit.EICHEL, Suit.SCHELLE, Suit.HERZ, Suit.LAUB)
        for r in (
            Rank.SECHS, Rank.SIEBEN, Rank.ACHT, Rank.NEUN, Rank.ZEHN,
            Rank.UNTER, Rank.OBER, Rank.KOENIG, Rank.ASS,
        )
    )
    assert total == 152


def test_total_points_konstanten():
    assert TOTAL_POINTS_PER_ROUND == 152 + LAST_TRICK_BONUS == 157
    assert total_points_per_round(TRUMPF_EICHEL) == 157
    assert total_points_per_round(OBEN) == 157
    assert total_points_per_round(UNTEN) == 157


# ---------- Kartenstärke / Stichgewinner ----------

def test_trumpf_buur_schlaegt_alles():
    v = TRUMPF_EICHEL
    trick = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.EICHEL, Rank.NEUN),
        C(Suit.EICHEL, Rank.UNTER),
    ]
    assert trick_winner(trick, v) == 3


def test_oben_ass_sticht_alles_in_lead_farbe():
    """Bei Bock sticht das Ass alles in der Lead-Farbe; andere Farben verlieren immer."""
    v = OBEN
    trick = [
        C(Suit.HERZ, Rank.KOENIG),  # Lead Herz
        C(Suit.HERZ, Rank.ASS),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.EICHEL, Rank.ASS),    # andere Farbe → kann nicht stechen
    ]
    assert trick_winner(trick, v) == 1


def test_oben_keine_andere_farbe_kann_stechen():
    v = OBEN
    s = card_strength(C(Suit.LAUB, Rank.ASS), lead_suit=Suit.HERZ, variant=v)
    assert s < 0


def test_unten_sechs_sticht_alles_in_lead_farbe():
    """Bei Geiss sticht die 6 alles in der Lead-Farbe."""
    v = UNTEN
    trick = [
        C(Suit.HERZ, Rank.ASS),       # Lead Herz
        C(Suit.HERZ, Rank.SECHS),     # niedrigster Rang = stärkste Karte bei Unten
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.SECHS),   # andere Farbe
    ]
    assert trick_winner(trick, v) == 1


def test_unten_andere_farbe_kann_nicht_stechen():
    v = UNTEN
    s = card_strength(C(Suit.LAUB, Rank.SECHS), lead_suit=Suit.HERZ, variant=v)
    assert s < 0


def test_card_strength_trumpf_andere_farbe_verliert():
    v = TRUMPF_EICHEL
    s = card_strength(C(Suit.LAUB, Rank.ASS), lead_suit=Suit.HERZ, variant=v)
    assert s < 0


# ---------- Stichpunkte ----------

def test_letzter_stich_plus_5():
    v = TRUMPF_EICHEL
    trick = [
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.HERZ, Rank.ACHT),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.HERZ, Rank.NEUN),
    ]
    assert trick_points(trick, v, is_last_trick=False) == 0
    assert trick_points(trick, v, is_last_trick=True) == LAST_TRICK_BONUS


def test_letzter_stich_plus_5_bei_oben():
    v = OBEN
    trick = [
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.HERZ, Rank.NEUN),
        C(Suit.LAUB, Rank.SIEBEN),
    ]
    assert trick_points(trick, v, is_last_trick=True) == LAST_TRICK_BONUS


def test_matsch_bonus_konstante():
    assert MATCH_BONUS == 100


# ---------- Legale Züge: Trumpf (Farbzwang, Buur, Untertrumpfen) ----------

def test_farbzwang_mit_bedienbarer_karte():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.LAUB, Rank.ASS),
    ]
    current = [C(Suit.HERZ, Rank.KOENIG)]
    legal = legal_moves(hand, current, v)
    assert C(Suit.HERZ, Rank.ASS) in legal
    assert C(Suit.HERZ, Rank.SECHS) in legal
    assert C(Suit.LAUB, Rank.ASS) not in legal


def test_buur_darf_immer_gespielt_werden_auch_wenn_bedienen_moeglich():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.UNTER),
    ]
    current = [C(Suit.HERZ, Rank.KOENIG)]
    legal = legal_moves(hand, current, v)
    assert C(Suit.HERZ, Rank.ASS) in legal
    assert C(Suit.EICHEL, Rank.UNTER) in legal


def test_trumpflead_buur_einzeln_darf_zurueckgehalten_werden():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.EICHEL, Rank.UNTER),
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.SIEBEN),
    ]
    current = [C(Suit.EICHEL, Rank.SECHS)]
    legal = legal_moves(hand, current, v)
    assert set(legal) == set(hand)


def test_trumpflead_mehrere_truempfe_muss_trumpf_spielen():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.EICHEL, Rank.UNTER),
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.HERZ, Rank.ASS),
    ]
    current = [C(Suit.EICHEL, Rank.ZEHN)]
    legal = legal_moves(hand, current, v)
    assert C(Suit.EICHEL, Rank.UNTER) in legal
    assert C(Suit.EICHEL, Rank.SECHS) in legal
    assert C(Suit.HERZ, Rank.ASS) not in legal


def test_untertrumpfen_verboten_wenn_alternative_da():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.LAUB, Rank.ASS),
        C(Suit.LAUB, Rank.SECHS),
    ]
    current = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.NEUN),
    ]
    legal = legal_moves(hand, current, v)
    assert C(Suit.EICHEL, Rank.SECHS) not in legal
    assert C(Suit.LAUB, Rank.ASS) in legal


def test_untertrumpfen_erzwungen_wenn_nur_niedrige_truempfe():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
    ]
    current = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.NEUN),
    ]
    legal = legal_moves(hand, current, v)
    assert set(legal) == set(hand)


def test_uebertrumpfen_erlaubt():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.EICHEL, Rank.UNTER),
        C(Suit.LAUB, Rank.ASS),
    ]
    current = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.NEUN),
    ]
    legal = legal_moves(hand, current, v)
    assert C(Suit.EICHEL, Rank.UNTER) in legal
    assert C(Suit.LAUB, Rank.ASS) in legal


def test_kein_stichzwang_freies_abwerfen_erlaubt():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.LAUB, Rank.ASS),
        C(Suit.SCHELLE, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.SECHS),
    ]
    current = [C(Suit.HERZ, Rank.KOENIG)]
    legal = legal_moves(hand, current, v)
    assert set(legal) == set(hand)


def test_erste_karte_alles_erlaubt():
    v = TRUMPF_EICHEL
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.EICHEL, Rank.UNTER),
    ]
    legal = legal_moves(hand, current_trick=[], variant=v)
    assert set(legal) == set(hand)


# ---------- Legale Züge: Bock/Geiss (nur Farbzwang) ----------

def test_oben_farbzwang_ohne_buur_ausnahme():
    """Bei Bock gibt es keinen Buur — Farbzwang gilt strikt für die Lead-Farbe."""
    v = OBEN
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.EICHEL, Rank.UNTER),  # bei Bock NICHT Trumpf-Buur, sondern normale Karte
    ]
    current = [C(Suit.HERZ, Rank.KOENIG)]
    legal = legal_moves(hand, current, v)
    # Nur Herz erlaubt; Eichel-Unter ist KEIN Buur in Bock-Modus
    assert legal == [C(Suit.HERZ, Rank.ASS)]


def test_oben_kein_stichzwang_aber_freier_abwurf():
    v = OBEN
    hand = [
        C(Suit.LAUB, Rank.ASS),
        C(Suit.EICHEL, Rank.SECHS),
    ]
    current = [C(Suit.HERZ, Rank.KOENIG)]
    legal = legal_moves(hand, current, v)
    assert set(legal) == set(hand)


def test_unten_farbzwang():
    v = UNTEN
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.LAUB, Rank.SECHS),
    ]
    current = [C(Suit.HERZ, Rank.SIEBEN)]
    legal = legal_moves(hand, current, v)
    # Bei Unten: Lead Herz, beide Herz-Karten müssen bedient werden
    assert C(Suit.HERZ, Rank.ASS) in legal
    assert C(Suit.HERZ, Rank.SECHS) in legal
    assert C(Suit.LAUB, Rank.SECHS) not in legal
