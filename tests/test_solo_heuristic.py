"""Tests fuer den SoloHeuristicPlayer.

Verifiziert:
- Keine illegalen Zuege
- Schieben wird nie versucht (allow_push=False im Solo)
- Schmieren passiert nie (kein Partner -> Schmier-Branch tot Code)
- Vier identische Bots ergeben ueber viele Spiele eine ungefaehre 25%-Verteilung
"""

from __future__ import annotations

import random


from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.rules import legal_moves
from jass_engine.variant import Announcement, Variant
from jass_engine.variants.solo_jass import play_solo_jass
from players.solo_heuristic_player import SoloHeuristicPlayer


def _make_solo_players(seed: int) -> list[SoloHeuristicPlayer]:
    rng = random.Random(seed)
    return [
        SoloHeuristicPlayer(name=f"H{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]


# --- Konfiguration ---


def test_solo_heuristik_push_threshold_irrelevant_aber_gesetzt():
    """push_threshold ist im Solo irrelevant, sollte aber sauber auf 0 stehen."""
    p = SoloHeuristicPlayer(name="H")
    assert p.push_threshold == 0


def test_solo_heuristik_konservativerer_slalom():
    p = SoloHeuristicPlayer(name="H")
    assert p.slalom_base_factor < 0.95  # niedriger als Team-Default
    assert p.slalom_concentration_factor < 2


# --- Verhalten ---


def test_solo_heuristik_schiebt_nie():
    """SoloHeuristicPlayer.choose_announcement gibt nie None zurueck."""
    p = SoloHeuristicPlayer(name="H", rng=random.Random(0))
    rng = random.Random(0)
    for _ in range(50):
        # zufaellige Hand aus 9 verschiedenen Karten
        from jass_engine.deck import make_deck
        deck = make_deck()
        rng.shuffle(deck)
        hand = deck[:9]
        # can_push=True, um sicherzustellen dass selbst dann nicht geschoben wird
        result = p.choose_announcement(hand, round_idx=1, can_push=True)
        assert result is not None, "SoloHeuristicPlayer darf nie schieben"


def test_solo_heuristik_schmiert_nicht():
    """In einer konstruierten Situation, in der der Spieler den Stich nicht gewinnen kann,
    waehlt der Bot die niedrigste Karte (sparen), schmiert also keine Punkte auf den Stich."""
    bot = SoloHeuristicPlayer(name="H", rng=random.Random(0))

    # Konstruktion: Trumpf=Eichel; jemand anderes hat schon Eichel-Buur gespielt;
    # der Bot hat keine Trumpf, kann also nicht gewinnen.
    variant = Variant.trumpf(Suit.EICHEL)
    ann = Announcement(variant=variant)

    hand = [
        Card(Suit.HERZ, Rank.ASS),       # 11 Punkte, hoher Wert
        Card(Suit.HERZ, Rank.SECHS),     # 0 Punkte, niedrigster Wert
        Card(Suit.LAUB, Rank.NEUN),      # 0 Punkte
    ]
    state = GameState(
        player_idx=2,
        variant=variant,
        announcement=ann,
        current_trick_cards=[Card(Suit.EICHEL, Rank.UNTER)],  # Buur, kein Schlagen moeglich
        current_trick_starter=1,
        teams=[0, 1, 2, 3],
        completed_tricks=[],
        round_idx=0,
        trick_idx=0,
        num_players=4,
    )
    # Die Buur-Karte wurde von Spieler 1 gespielt, jetzt Spieler 2 dran.
    # Lead-Suit ist Eichel; Spieler 2 hat keine Eichel -> kann frei abwerfen.
    legal = legal_moves(hand, state.current_trick_cards, state.variant)
    chosen = bot.choose_card(hand, state)
    assert chosen in legal
    # Im Solo: kein Schmieren auf fremden Stich -> niedrigster Punktwert
    # erwartet: Herz-6 (0 Punkte, niedrigster Rang) oder Laub-9 (0 Punkte, hoher Rang)
    # Auf KEINEN Fall die Herz-Ass (11 Punkte!)
    assert chosen != Card(Suit.HERZ, Rank.ASS), (
        "SoloHeuristicPlayer hat Herz-Ass auf fremden Buur-Stich geschmiert!"
    )


def test_solo_heuristik_50_zufallspartien_keine_illegalen_zuege():
    """Smoke-Test: 50 Solo-Partien mit 4 SoloHeuristic-Spielern laufen sauber durch."""
    for seed in range(50):
        players = _make_solo_players(seed)
        rng = random.Random(seed)
        game = play_solo_jass(players, target_score=500, rng=rng)
        # Engine wirft RuntimeError bei illegalen Zuegen; wenn wir hier ankommen, alles OK
        assert max(game.final_scores.values()) >= 500


# --- Sieg-Verteilung bei 4 identischen Bots ---


def test_solo_heuristik_vier_identische_bots_etwa_25_prozent_pro_spieler():
    """Vier identische SoloHeuristic in 200 Partien: jeder Spieler sollte
    ungefaehr 25% der Partien gewinnen (mit Standardabweichung-Toleranz).

    Bei 200 Partien und True-Rate 25% ist die SD ca. 3% -- wir tolerieren
    bis 14-36% je Spieler (knapp 4 Sigma)."""
    wins = {0: 0, 1: 0, 2: 0, 3: 0}
    n_games = 200
    rng = random.Random(42)
    for trial in range(n_games):
        players = _make_solo_players(seed=trial)
        game = play_solo_jass(
            players,
            target_score=500,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        wins[game.winning_team] += 1

    print(f"\nWin-Verteilung in {n_games} Partien:", wins)
    for pid in range(4):
        win_rate = wins[pid] / n_games
        assert 0.14 <= win_rate <= 0.36, (
            f"Spieler {pid}: {win_rate:.1%} Win-Rate liegt ausserhalb [14%, 36%]"
        )
