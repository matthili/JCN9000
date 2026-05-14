"""Rich-basierte Konsolen-Ausgabe einer Random-vs-Random-Partie.

Aufruf:
    python -m visualization.terminal
"""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jass_engine.card import Card, Suit
from jass_engine.deck import deal, find_weli_holder
from jass_engine.player import GameState
from jass_engine.rules import legal_moves
from jass_engine.trick import Trick
from jass_engine.variant import PlayMode
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS
from jass_engine.weis import find_weise, has_stoecke, stoecke_weis
from players.random_player import RandomPlayer


SUIT_COLOR = {
    Suit.EICHEL: "yellow",
    Suit.SCHELLE: "bright_red",
    Suit.HERZ: "red",
    Suit.LAUB: "green",
}

SUIT_LETTER = {
    Suit.EICHEL: "E",
    Suit.SCHELLE: "S",
    Suit.HERZ: "H",
    Suit.LAUB: "L",
}


def card_text(card: Card) -> Text:
    return Text(
        f"{SUIT_LETTER[card.suit]}-{card.rank.german_name}",
        style=SUIT_COLOR[card.suit],
    )


def hand_text(hand: list[Card]) -> Text:
    t = Text()
    for i, c in enumerate(sorted(hand, key=lambda x: (int(x.suit), int(x.rank)))):
        if i > 0:
            t.append(" ")
        t.append_text(card_text(c))
    return t


def trick_table(trick: Trick, player_names: list[str]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Pos")
    table.add_column("Spieler")
    table.add_column("Karte")
    for i, card in enumerate(trick.cards):
        p_idx = trick.player_idx_for_card(i)
        marker = " (Anspiel)" if i == 0 else ""
        table.add_row(str(p_idx), player_names[p_idx] + marker, card_text(card))
    return table


def play_demo_game(seed: int = 42) -> None:
    console = Console()
    rng = random.Random(seed)
    players = [
        RandomPlayer(name=f"P{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]
    player_names = [p.name for p in players]
    teams = list(KREUZ_JASS_TEAMS)

    console.rule("[bold cyan]Vorarlberger Kreuz-Jass - Demo-Partie[/bold cyan]")
    console.print(f"Teams: {player_names[0]}+{player_names[2]} vs {player_names[1]}+{player_names[3]}")

    hands = deal(num_players=4, rng=rng)
    weli_holder = find_weli_holder(hands)
    console.print(
        f"\n[bold]Weli liegt bei:[/bold] {player_names[weli_holder]} "
        f"-> sagt an."
    )

    console.print()
    for idx, h in enumerate(hands):
        console.print(Panel(hand_text(h), title=f"{player_names[idx]} (Team {teams[idx]})"))

    announcement = players[weli_holder].choose_announcement(hands[weli_holder], 0, can_push=False)
    assert announcement is not None
    console.print(f"\n[bold magenta]Ansage:[/bold magenta] {announcement}")

    # Weisen (basierend auf der ersten Stich-Variante, falls Slalom)
    initial_variant = announcement.variant_for_trick(0)
    console.print("\n[bold]Weisen:[/bold]")
    for p_idx, p in enumerate(players):
        possible = find_weise(hands[p_idx])
        ann = p.announce_weise(hands[p_idx], initial_variant, possible)
        if ann:
            ws = ", ".join(repr(w) for w in ann)
            console.print(f"  {player_names[p_idx]}: {ws}")
        else:
            console.print(f"  {player_names[p_idx]}: -")

    # Stöcke nur bei Trumpf
    if initial_variant.mode == PlayMode.TRUMPF:
        assert initial_variant.trump_suit is not None
        for p_idx in range(4):
            if has_stoecke(hands[p_idx], initial_variant.trump_suit):
                sw = stoecke_weis(initial_variant.trump_suit)
                console.print(
                    f"  [italic]{player_names[p_idx]} hat {sw} (wird beim Ausspielen angesagt)[/italic]"
                )

    starter = weli_holder
    completed_tricks: list[Trick] = []
    team_card_points = {tid: 0 for tid in set(teams)}

    for trick_idx in range(9):
        variant_this = announcement.variant_for_trick(trick_idx)
        trick = Trick(starting_player_idx=starter, num_players=4)
        for _ in range(4):
            cur = trick.next_player_idx()
            state = GameState(
                player_idx=cur,
                variant=variant_this,
                announcement=announcement,
                current_trick_cards=list(trick.cards),
                current_trick_starter=trick.starting_player_idx,
                teams=list(teams),
                completed_tricks=[list(t.cards) for t in completed_tricks],
                trick_idx=trick_idx,
                num_players=4,
            )
            legal = legal_moves(hands[cur], trick.cards, variant_this)
            chosen = players[cur].choose_card(hands[cur], state)
            assert chosen in legal, f"Illegaler Zug: {chosen} nicht in {legal}"
            hands[cur].remove(chosen)
            trick.add_card(chosen)

        is_last = trick_idx == 8
        winner = trick.winner_idx(variant_this)
        pts = trick.points(variant_this, is_last=is_last)
        team_card_points[teams[winner]] += pts
        completed_tricks.append(trick)

        console.rule(
            f"Stich {trick_idx + 1} ({variant_this}) - Gewinner: {player_names[winner]} "
            f"(Team {teams[winner]}) - {pts} Pkt."
        )
        console.print(trick_table(trick, player_names))
        starter = winner

    console.rule("[bold green]Rundenende[/bold green]")
    for tid, pts in team_card_points.items():
        console.print(f"Team {tid}: {pts} Stichpunkte")


if __name__ == "__main__":
    play_demo_game()
