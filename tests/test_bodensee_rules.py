"""Tests fuer die Bodensee-Spielregeln (legal_moves) und Stich-Logik."""

from __future__ import annotations

import pytest

from jass_engine.bodensee.player_state import BodenseePlayerState, TableStack
from jass_engine.bodensee.rules import card_source, legal_moves_bodensee
from jass_engine.bodensee.trick import play_bodensee_trick, play_card_from_state
from jass_engine.card import Card, Rank, Suit
from jass_engine.variant import Variant


# --- card_source ---


def test_card_source_findet_in_hand():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS)],
        table=[TableStack(visible=Card(Suit.HERZ, Rank.SECHS), hidden=None)],
    )
    assert card_source(ps, Card(Suit.EICHEL, Rank.ASS)) == "hand"


def test_card_source_findet_auf_tisch():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS)],
        table=[TableStack(visible=Card(Suit.HERZ, Rank.SECHS), hidden=None)],
    )
    assert card_source(ps, Card(Suit.HERZ, Rank.SECHS)) == "table"


def test_card_source_wirft_bei_unbekannter_karte():
    ps = BodenseePlayerState(hand=[Card(Suit.EICHEL, Rank.ASS)])
    with pytest.raises(ValueError):
        card_source(ps, Card(Suit.LAUB, Rank.NEUN))


# --- legal_moves_bodensee: leerer Stich ---


def test_legal_moves_leerer_stich_alle_verfuegbaren_karten():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.SECHS)],
        table=[TableStack(visible=Card(Suit.LAUB, Rank.OBER), hidden=None)],
    )
    legal = legal_moves_bodensee(ps, [], Variant.oben())
    assert len(legal) == 3
    assert Card(Suit.EICHEL, Rank.ASS) in legal
    assert Card(Suit.HERZ, Rank.SECHS) in legal
    assert Card(Suit.LAUB, Rank.OBER) in legal


# --- Bedienzwang inklusive Tisch-Karten ---


def test_bedienzwang_zaehlt_tischkarten_mit():
    """Wenn Lead-Farbe nur auf dem Tisch (nicht in Hand) liegt, muss man dennoch bedienen."""
    ps = BodenseePlayerState(
        hand=[Card(Suit.HERZ, Rank.ASS), Card(Suit.LAUB, Rank.KOENIG)],
        table=[
            TableStack(visible=Card(Suit.EICHEL, Rank.NEUN), hidden=None),
        ],
    )
    # Lead = Eichel-Sieben
    legal = legal_moves_bodensee(
        ps, [Card(Suit.EICHEL, Rank.SIEBEN)], Variant.oben()
    )
    # Nur die Eichel-Neun vom Tisch ist legal (Bedienzwang)
    assert legal == [Card(Suit.EICHEL, Rank.NEUN)]


def test_bedienzwang_hand_und_tisch_kombiniert():
    """Lead = Eichel; Spieler hat Eichel in Hand UND Eichel auf Tisch -> beide legal."""
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.LAUB, Rank.KOENIG)],
        table=[TableStack(visible=Card(Suit.EICHEL, Rank.NEUN), hidden=None)],
    )
    legal = legal_moves_bodensee(
        ps, [Card(Suit.EICHEL, Rank.SIEBEN)], Variant.oben()
    )
    assert Card(Suit.EICHEL, Rank.ASS) in legal
    assert Card(Suit.EICHEL, Rank.NEUN) in legal
    assert Card(Suit.LAUB, Rank.KOENIG) not in legal


def test_keine_lead_farbe_alles_erlaubt():
    """Wenn weder in Hand noch auf Tisch die Lead-Farbe liegt, darf alles."""
    ps = BodenseePlayerState(
        hand=[Card(Suit.HERZ, Rank.ASS)],
        table=[TableStack(visible=Card(Suit.LAUB, Rank.NEUN), hidden=None)],
    )
    legal = legal_moves_bodensee(
        ps, [Card(Suit.EICHEL, Rank.SIEBEN)], Variant.oben()
    )
    assert set(legal) == {Card(Suit.HERZ, Rank.ASS), Card(Suit.LAUB, Rank.NEUN)}


# --- Buur-Ausnahme ---


def test_buur_ausnahme_bei_trumpf_lead_und_buur_einziger_trumpf():
    """Gegner spielt Trumpf-Ass, ich habe nur Trumpf-Buur als Trumpf -> ich darf
    auch andere Farbe spielen (Buur-Ausnahme)."""
    trumpf = Variant.trumpf(Suit.EICHEL)
    ps = BodenseePlayerState(
        hand=[
            Card(Suit.EICHEL, Rank.UNTER),   # Trumpf-Buur, einziger Trumpf
            Card(Suit.HERZ, Rank.ASS),
            Card(Suit.LAUB, Rank.NEUN),
        ],
    )
    legal = legal_moves_bodensee(
        ps, [Card(Suit.EICHEL, Rank.ASS)], trumpf
    )
    # Alle Karten erlaubt (Buur-Ausnahme greift)
    assert len(legal) == 3


def test_trumpf_zwang_mit_mehreren_truempfen():
    """Gegner spielt Trumpf, ich habe mehrere Truempfe -> muss Trumpf bedienen."""
    trumpf = Variant.trumpf(Suit.EICHEL)
    ps = BodenseePlayerState(
        hand=[
            Card(Suit.EICHEL, Rank.UNTER),
            Card(Suit.EICHEL, Rank.ZEHN),
            Card(Suit.HERZ, Rank.ASS),
        ],
    )
    legal = legal_moves_bodensee(
        ps, [Card(Suit.EICHEL, Rank.ASS)], trumpf
    )
    # Nur Truempfe legal
    assert set(legal) == {Card(Suit.EICHEL, Rank.UNTER), Card(Suit.EICHEL, Rank.ZEHN)}


def test_kein_untertrumpfen_irrelevant_bei_zwei_spielern():
    """Im 2-Spieler-Bodensee ist 'kein Untertrumpfen' strukturell nicht ausloesbar:
    der Trick hat hoechstens 1 vorherige Karte = Lead. Wenn der Lead Nicht-Trumpf
    ist, liegt nie schon ein Trumpf vor mir."""
    trumpf = Variant.trumpf(Suit.EICHEL)
    ps = BodenseePlayerState(
        hand=[
            Card(Suit.EICHEL, Rank.NEUN),   # niedriger Trumpf
            Card(Suit.LAUB, Rank.SIEBEN),
        ],
    )
    # Lead: Herz-Ass (Nicht-Trumpf, weil Eichel Trumpf ist)
    legal = legal_moves_bodensee(
        ps, [Card(Suit.HERZ, Rank.ASS)], trumpf
    )
    # Ich habe Herz nicht -> alles erlaubt, inkl. niedriger Trumpf
    # (keine "kein Untertrumpfen"-Restriktion, weil im current_trick nur 1 Karte ist
    # und das ein Nicht-Trumpf ist)
    assert Card(Suit.EICHEL, Rank.NEUN) in legal
    assert Card(Suit.LAUB, Rank.SIEBEN) in legal


# --- play_card_from_state ---


def test_play_card_from_state_hand():
    ps = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.SECHS)],
    )
    play_card_from_state(ps, Card(Suit.EICHEL, Rank.ASS))
    assert ps.hand == [Card(Suit.HERZ, Rank.SECHS)]


def test_play_card_from_state_tisch_deckt_auf():
    ps = BodenseePlayerState(
        table=[
            TableStack(
                visible=Card(Suit.LAUB, Rank.OBER),
                hidden=Card(Suit.SCHELLE, Rank.NEUN),
            ),
        ],
    )
    play_card_from_state(ps, Card(Suit.LAUB, Rank.OBER))
    assert ps.table[0].visible == Card(Suit.SCHELLE, Rank.NEUN)
    assert ps.table[0].hidden is None


# --- play_bodensee_trick ---


def test_play_bodensee_trick_einfacher_stich():
    """Spieler 0 spielt Eichel-Ass, Spieler 1 muss bedienen mit Eichel-Sechs."""
    ps0 = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.ASS), Card(Suit.HERZ, Rank.NEUN)],
    )
    ps1 = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.SECHS), Card(Suit.LAUB, Rank.KOENIG)],
    )

    def p0_choose(ps, legal, trick, variant):
        return Card(Suit.EICHEL, Rank.ASS)

    def p1_choose(ps, legal, trick, variant):
        # Muss Eichel bedienen
        assert Card(Suit.EICHEL, Rank.SECHS) in legal
        assert Card(Suit.LAUB, Rank.KOENIG) not in legal
        return Card(Suit.EICHEL, Rank.SECHS)

    trick, winner, points = play_bodensee_trick(
        starter_idx=0,
        player_states=[ps0, ps1],
        choose_card_fns=[p0_choose, p1_choose],
        variant=Variant.oben(),
    )
    assert winner == 0  # Eichel-Ass schlaegt Eichel-Sechs in Bock-Modus
    assert points == 11  # Ass = 11
    assert len(trick.cards) == 2
    # Karten sind aus den jeweiligen Haenden weg
    assert Card(Suit.EICHEL, Rank.ASS) not in ps0.hand
    assert Card(Suit.EICHEL, Rank.SECHS) not in ps1.hand


def test_play_bodensee_trick_tisch_karte_deckt_auf():
    """Spieler spielt eine Tisch-Karte, die verdeckte darunter wird aufgedeckt."""
    ps0 = BodenseePlayerState(
        hand=[Card(Suit.LAUB, Rank.SECHS)],
        table=[
            TableStack(
                visible=Card(Suit.EICHEL, Rank.ASS),
                hidden=Card(Suit.HERZ, Rank.NEUN),
            ),
        ],
    )
    ps1 = BodenseePlayerState(hand=[Card(Suit.EICHEL, Rank.SECHS)])

    def p0_choose(ps, legal, trick, variant):
        return Card(Suit.EICHEL, Rank.ASS)  # vom Tisch

    def p1_choose(ps, legal, trick, variant):
        return Card(Suit.EICHEL, Rank.SECHS)

    trick, winner, points = play_bodensee_trick(
        starter_idx=0,
        player_states=[ps0, ps1],
        choose_card_fns=[p0_choose, p1_choose],
        variant=Variant.oben(),
    )
    # Eichel-Ass wurde gespielt, Herz-Neun wurde aufgedeckt
    assert ps0.table[0].visible == Card(Suit.HERZ, Rank.NEUN)
    assert ps0.table[0].hidden is None
    assert winner == 0


def test_play_bodensee_trick_illegaler_zug_wirft_fehler():
    ps0 = BodenseePlayerState(hand=[Card(Suit.EICHEL, Rank.ASS)])
    ps1 = BodenseePlayerState(
        hand=[Card(Suit.EICHEL, Rank.SECHS), Card(Suit.LAUB, Rank.KOENIG)],
    )

    def p0_choose(ps, legal, trick, variant):
        return Card(Suit.EICHEL, Rank.ASS)

    def p1_choose(ps, legal, trick, variant):
        # Spielt Laub trotz Bedienzwang Eichel
        return Card(Suit.LAUB, Rank.KOENIG)

    with pytest.raises(RuntimeError, match="illegale Karte"):
        play_bodensee_trick(
            starter_idx=0,
            player_states=[ps0, ps1],
            choose_card_fns=[p0_choose, p1_choose],
            variant=Variant.oben(),
        )
