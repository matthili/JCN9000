"""Tests fuer die Ansage-Familien-Skalen (gumpf/oben/unten_scale, Tuning v2).

Default 1.0 = unveraendertes Verhalten (durch die bestehende Suite abgedeckt).
Hier: die Skalen greifen tatsaechlich in die Ansage-Wahl ein.
"""

from __future__ import annotations

import random

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.variant import PlayMode
from players.bodensee_heuristic_player import BodenseeHeuristicPlayer
from players.heuristic_player import HeuristicPlayer


def _lehrbuch_gumpf_hand() -> list[Card]:
    """Buur + Nell + Trumpf-Ass (Eichel) plus niedrige Nicht-Trumpf-Karten."""
    return [
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.EICHEL, Rank.NEUN),
        Card(Suit.EICHEL, Rank.ASS),
        Card(Suit.SCHELLE, Rank.SECHS),
        Card(Suit.SCHELLE, Rank.SIEBEN),
        Card(Suit.HERZ, Rank.SECHS),
        Card(Suit.HERZ, Rank.ACHT),
        Card(Suit.LAUB, Rank.SECHS),
        Card(Suit.LAUB, Rank.SIEBEN),
    ]


def test_kreuz_gumpf_hand_waehlt_gumpf_per_default():
    bot = HeuristicPlayer("Bot")
    ann = bot.choose_announcement(_lehrbuch_gumpf_hand(), round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.GUMPF, f"Lehrbuch-Gumpf-Hand, bekam {ann}"


def test_kreuz_gumpf_scale_null_unterdrueckt_gumpf():
    bot = HeuristicPlayer("Bot", gumpf_scale=0.0)
    ann = bot.choose_announcement(_lehrbuch_gumpf_hand(), round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode != PlayMode.GUMPF


def test_kreuz_gumpf_scale_null_nie_gumpf_ueber_zufallshaende():
    bot = HeuristicPlayer("Bot", gumpf_scale=0.0)
    deck = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
    rng = random.Random(42)
    for _ in range(100):
        rng.shuffle(deck)
        hand = deck[:9]
        ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
        assert ann is not None
        assert ann.variant.mode != PlayMode.GUMPF


def test_kreuz_unten_scale_hoch_bevorzugt_unten():
    """Eine neutrale Hand kippt bei stark erhoehtem unten_scale zu Unten."""
    hand = _lehrbuch_gumpf_hand()
    bot = HeuristicPlayer("Bot", unten_scale=10.0, allow_slalom=False)
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.UNTEN


def test_bodensee_gumpf_scale_null_unterdrueckt_gumpf():
    bot = BodenseeHeuristicPlayer("Bot", gumpf_scale=0.0)
    deck = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
    rng = random.Random(7)
    for _ in range(50):
        rng.shuffle(deck)
        hand = deck[:6]
        visible = deck[6:12]
        ann = bot.choose_announcement(hand, visible, round_idx=0)
        assert ann is not None
        assert ann.variant.mode != PlayMode.GUMPF
