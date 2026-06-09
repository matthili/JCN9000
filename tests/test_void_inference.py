"""Tests fuer die Void-Inferenz aus der Spielhistorie.

Soundness ist hier kritisch: wir duerfen nur ausschliessen, was beweisbar
unmoeglich ist -- sonst verteilt die Determinisierung Karten falsch.
"""

from __future__ import annotations

from jass_engine.card import ALL_RANKS, Card, Rank, Suit
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant
from jass_engine.void_inference import infer_forbidden_cards, seat_is_void_in_trump


def _all_of(suit: Suit) -> set[Card]:
    return {Card(suit, r) for r in ALL_RANKS}


# ---------------------------------------------------------------------------
# Nicht-Trumpf-Lead: Abwurf einer dritten Farbe -> blank in der Lead-Farbe.
# Trumpf auf Nicht-Trumpf-Lead (Stechen) -> KEIN Schluss.
# ---------------------------------------------------------------------------
def test_nontrump_lead_discard_implies_void_in_lead_suit():
    ann = Announcement(variant=Variant.trumpf(Suit.EICHEL))  # Trumpf = Eichel
    # Sitz 0 spielt Schelle-Ass an (Lead = Schelle, Nicht-Trumpf)
    # Sitz 1 bedient Schelle -> kein Schluss
    # Sitz 2 wirft Laub ab (weder Schelle noch Trumpf) -> blank in Schelle
    # Sitz 3 sticht mit Eichel (Trumpf) -> kein Schluss (Stechen ist erlaubt)
    trick = CompletedTrick(
        starter=0,
        cards=(
            Card(Suit.SCHELLE, Rank.ASS),
            Card(Suit.SCHELLE, Rank.KOENIG),
            Card(Suit.LAUB, Rank.SIEBEN),
            Card(Suit.EICHEL, Rank.SECHS),
        ),
    )
    forbidden = infer_forbidden_cards([trick], ann, num_players=4)

    assert _all_of(Suit.SCHELLE) <= forbidden[2], "Sitz 2 muss blank in Schelle sein."
    assert forbidden[0] == set(), "Anspieler -> kein Schluss."
    assert forbidden[1] == set(), "Bedient -> kein Schluss."
    assert forbidden[3] == set(), "Trumpf auf Nicht-Trumpf-Lead -> kein Schluss."


# ---------------------------------------------------------------------------
# Trumpf-Lead: Nicht-Trumpf gespielt -> blank in Trumpf AUSSER dem Buur.
# ---------------------------------------------------------------------------
def test_trump_lead_offsuit_implies_void_in_trump_except_buur():
    ann = Announcement(variant=Variant.trumpf(Suit.EICHEL))
    # Sitz 0 spielt Trumpf an (Eichel-Ass)
    # Sitz 1 bedient Trumpf -> kein Schluss
    # Sitz 2 spielt Nicht-Trumpf (Schelle 7) -> blank in Eichel ausser Buur
    # Sitz 3 spielt Nicht-Trumpf (Laub 8) -> blank in Eichel ausser Buur
    trick = CompletedTrick(
        starter=0,
        cards=(
            Card(Suit.EICHEL, Rank.ASS),
            Card(Suit.EICHEL, Rank.KOENIG),
            Card(Suit.SCHELLE, Rank.SIEBEN),
            Card(Suit.LAUB, Rank.ACHT),
        ),
    )
    forbidden = infer_forbidden_cards([trick], ann, num_players=4)

    buur = Card(Suit.EICHEL, Rank.UNTER)
    expected = _all_of(Suit.EICHEL) - {buur}
    for seat in (2, 3):
        assert expected <= forbidden[seat], f"Sitz {seat} blank in Eichel ausser Buur."
        assert buur not in forbidden[seat], "Buur darf NICHT verboten werden (Buur-Ausnahme)."
        assert seat_is_void_in_trump(forbidden[seat], Suit.EICHEL)

    assert forbidden[1] == set(), "Hat Trumpf bedient -> nicht blank."
    assert not seat_is_void_in_trump(forbidden[1], Suit.EICHEL)


# ---------------------------------------------------------------------------
# Oben (kein Trumpf): Abwurf einer anderen Farbe -> blank in Lead-Farbe.
# ---------------------------------------------------------------------------
def test_oben_offsuit_implies_void_in_lead_suit():
    ann = Announcement(variant=Variant.oben())
    trick = CompletedTrick(
        starter=2,  # Anspieler ist Sitz 2
        cards=(
            Card(Suit.HERZ, Rank.ASS),    # Sitz 2 (Lead = Herz)
            Card(Suit.LAUB, Rank.SECHS),  # Sitz 3 wirft ab -> blank in Herz
        ),
    )
    forbidden = infer_forbidden_cards([trick], ann, num_players=4)
    assert _all_of(Suit.HERZ) <= forbidden[3]
    assert forbidden[2] == set()


# ---------------------------------------------------------------------------
# Mehrere Stiche akkumulieren; Leader nie betroffen; Bedienen nie betroffen.
# ---------------------------------------------------------------------------
def test_accumulates_over_tricks():
    ann = Announcement(variant=Variant.trumpf(Suit.LAUB))  # Trumpf = Laub
    tricks = [
        # Trick 0: Sitz 1 fuehrt Herz, Sitz 2 wirft Schelle ab -> blank Herz
        CompletedTrick(
            starter=1,
            cards=(
                Card(Suit.HERZ, Rank.ASS),
                Card(Suit.SCHELLE, Rank.NEUN),
                Card(Suit.HERZ, Rank.KOENIG),
                Card(Suit.HERZ, Rank.OBER),
            ),
        ),
    ]
    forbidden = infer_forbidden_cards(tricks, ann, num_players=4)
    # Sitz 2 (= starter 1 + 1) warf Schelle auf Herz-Lead -> blank in Herz
    assert _all_of(Suit.HERZ) <= forbidden[2]
    # Sitze 3 und 0 bedienten Herz -> kein Schluss
    assert forbidden[3] == set()
    assert forbidden[0] == set()
    assert forbidden[1] == set()  # Anspieler


# ---------------------------------------------------------------------------
# Leerer / unvollstaendiger Stich erzeugt keine Schluesse.
# ---------------------------------------------------------------------------
def test_short_trick_no_inference():
    ann = Announcement(variant=Variant.trumpf(Suit.EICHEL))
    trick = CompletedTrick(starter=0, cards=(Card(Suit.SCHELLE, Rank.ASS),))
    forbidden = infer_forbidden_cards([trick], ann, num_players=4)
    assert all(s == set() for s in forbidden.values())
