"""Tests fuer die void-tracking Trumpf-Disziplin der Heuristik (Spur B).

Wenn beide Gegner beweisbar trumpffrei sind, soll der Heuristik-Spieler beim
Anspielen KEINE Truempfe mehr ziehen (das zoege nur dem Partner die Truempfe),
sondern hohe Seitenkarten spielen.
"""

from __future__ import annotations

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant
from players.heuristic_player import HeuristicPlayer


def _state_opponents_void_in_trump(variant: Variant) -> GameState:
    """Sitz 0 ist am Anspielen; in Stich 0 wurde Trumpf gefuehrt und beide
    Gegner (Sitze 1 und 3) konnten nicht folgen -> trumpffrei."""
    ann = Announcement(variant=variant)
    trick = CompletedTrick(
        starter=0,
        cards=(
            Card(Suit.EICHEL, Rank.ASS),     # Sitz 0 fuehrt Trumpf (Eichel)
            Card(Suit.SCHELLE, Rank.SIEBEN),  # Sitz 1 wirft ab -> blank in Trumpf
            Card(Suit.EICHEL, Rank.KOENIG),   # Sitz 2 (Partner) folgt Trumpf
            Card(Suit.LAUB, Rank.ACHT),       # Sitz 3 wirft ab -> blank in Trumpf
        ),
    )
    return GameState(
        player_idx=0,
        variant=variant,
        announcement=ann,
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
        completed_tricks=[trick],
    )


def test_trumpf_leads_side_ace_when_opponents_void():
    variant = Variant.trumpf(Suit.EICHEL)
    state = _state_opponents_void_in_trump(variant)
    hand = [
        Card(Suit.EICHEL, Rank.UNTER),   # Buur (hoechster Trumpf)
        Card(Suit.SCHELLE, Rank.ASS),    # Seiten-Ass
        Card(Suit.LAUB, Rank.KOENIG),
    ]
    aware = HeuristicPlayer("aware", trump_void_awareness=True)
    chosen = aware.choose_card(hand, state)
    assert chosen == Card(Suit.SCHELLE, Rank.ASS), (
        "Bei blanken Gegnern soll das Seiten-Ass kommen, nicht der Buur."
    )


def test_trumpf_pulls_trump_when_awareness_off():
    variant = Variant.trumpf(Suit.EICHEL)
    state = _state_opponents_void_in_trump(variant)
    hand = [
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.SCHELLE, Rank.ASS),
        Card(Suit.LAUB, Rank.KOENIG),
    ]
    legacy = HeuristicPlayer("legacy", trump_void_awareness=False)
    chosen = legacy.choose_card(hand, state)
    assert chosen == Card(Suit.EICHEL, Rank.UNTER), (
        "Ohne Void-Awareness wird wie bisher der hohe Trumpf gezogen."
    )


def test_trumpf_pulls_trump_when_no_history():
    """Ohne Beweis (Stich 0, kein Verlauf) wird wie bisher Trumpf gezogen."""
    variant = Variant.trumpf(Suit.EICHEL)
    ann = Announcement(variant=variant)
    state = GameState(
        player_idx=0,
        variant=variant,
        announcement=ann,
        current_trick_cards=[],
        current_trick_starter=0,
        teams=[0, 1, 0, 1],
        completed_tricks=[],
    )
    hand = [Card(Suit.EICHEL, Rank.UNTER), Card(Suit.SCHELLE, Rank.ASS)]
    aware = HeuristicPlayer("aware", trump_void_awareness=True)
    assert aware.choose_card(hand, state) == Card(Suit.EICHEL, Rank.UNTER)


def test_gumpf_leads_side_six_when_opponents_void():
    variant = Variant.gumpf(Suit.EICHEL)
    state = _state_opponents_void_in_trump(variant)
    hand = [
        Card(Suit.EICHEL, Rank.UNTER),   # Buur
        Card(Suit.SCHELLE, Rank.SECHS),  # 6er = sicherer Sticher im Gumpf
        Card(Suit.LAUB, Rank.KOENIG),
    ]
    aware = HeuristicPlayer("aware", trump_void_awareness=True)
    chosen = aware.choose_card(hand, state)
    assert chosen == Card(Suit.SCHELLE, Rank.SECHS), (
        "Im Gumpf bei blanken Gegnern den sicheren 6er-Sticher statt Trumpf."
    )
