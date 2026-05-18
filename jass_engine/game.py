"""Eine Partie: mehrere Runden bis ein Team das Zielpunkte-Limit erreicht."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from jass_engine.deck import deal, find_weli_holder
from jass_engine.player import Player
from jass_engine.round import RoundResult, play_round


DEFAULT_TARGET_SCORE = 1000


@dataclass
class GameResult:
    rounds: list[RoundResult]
    final_scores: dict[int, int]
    winning_team: int


def play_game(
    players: list[Player],
    teams: list[int] | None = None,
    target_score: int = DEFAULT_TARGET_SCORE,
    rng: random.Random | None = None,
    max_rounds: int = 200,
    allow_push: bool = True,
) -> GameResult:
    """Spielt eine vollständige Partie bis zum Zielpunkte-Limit.

    Standardmäßig sitzen die Spieler 0+2 als Team 0, Spieler 1+3 als Team 1
    (Partner über Kreuz). Bei Solo-Jass werden vier separate "Teams" verwendet
    (jeder Spieler eigenes Team), und `allow_push=False` deaktiviert das
    Schieben (es gibt keinen Partner).

    Args:
        allow_push: erlaubt Schieben (Default True). Auf False setzen für Solo.
    """
    if rng is None:
        rng = random.Random()
    if teams is None:
        teams = [0, 1, 0, 1]
    if len(players) != 4:
        raise NotImplementedError("Aktuell nur 4-Spieler-Kreuz-Jass implementiert.")

    rounds: list[RoundResult] = []
    cumulative: dict[int, int] = {tid: 0 for tid in set(teams)}
    # Ab Runde 2 rotiert der Ansager. Wir merken uns den letzten Ansager.
    last_announcer: int | None = None

    for round_idx in range(max_rounds):
        forced_announcer = None
        if round_idx > 0 and last_announcer is not None:
            forced_announcer = (last_announcer + 1) % len(players)

        result = play_round(
            players=players,
            teams=teams,
            round_idx=round_idx,
            rng=rng,
            forced_announcer_idx=forced_announcer,
            allow_push=allow_push,
        )
        rounds.append(result)
        last_announcer = result.announcer_idx
        for tid, pts in result.team_total_points.items():
            cumulative[tid] += pts

        if any(score >= target_score for score in cumulative.values()):
            break

    winning_team = max(cumulative, key=cumulative.get)  # type: ignore[arg-type]
    return GameResult(rounds=rounds, final_scores=cumulative, winning_team=winning_team)
