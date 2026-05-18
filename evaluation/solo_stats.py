"""Solo-Jass-Statistiken: pro Spieler-Rolle aggregierte Eval-Metriken.

Anders als bei TeamStats (zwei Teams) gibt es im Solo vier Spieler-Rollen.
Im Standard-Setup vergleichen wir zwei NN-Modelle (A und B) gegen zwei
Heuristik-Bots als Anker -- daher die festen Rollen-Labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jass_engine.game import GameResult


@dataclass
class PlayerStats:
    """Aggregierte Statistik fuer eine Spieler-Rolle (z.B. "Modell A")."""

    games_played: int = 0
    games_won: int = 0
    total_score: int = 0
    total_rounds: int = 0
    matsch_for: int = 0
    # Pro-Variante: Spiele und Siege, aufgeschluesselt nach (effektiver)
    # Variante in der ERSTEN Runde. Vereinfachung wie beim Team-Eval.
    games_by_variant_id: dict[str, int] = field(default_factory=dict)
    wins_by_variant_id: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.games_won / self.games_played if self.games_played else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.games_played if self.games_played else 0.0

    @property
    def avg_rounds(self) -> float:
        return self.total_rounds / self.games_played if self.games_played else 0.0

    @property
    def matsch_rate_per_round(self) -> float:
        return self.matsch_for / self.total_rounds if self.total_rounds else 0.0

    def win_rate_for_variant(self, variant_id: str) -> float:
        n = self.games_by_variant_id.get(variant_id, 0)
        if not n:
            return 0.0
        return self.wins_by_variant_id.get(variant_id, 0) / n

    def merge(self, other: "PlayerStats") -> None:
        """In-place merge fuer parallele Worker-Aggregation."""
        self.games_played += other.games_played
        self.games_won += other.games_won
        self.total_score += other.total_score
        self.total_rounds += other.total_rounds
        self.matsch_for += other.matsch_for
        for v, c in other.games_by_variant_id.items():
            self.games_by_variant_id[v] = self.games_by_variant_id.get(v, 0) + c
        for v, c in other.wins_by_variant_id.items():
            self.wins_by_variant_id[v] = self.wins_by_variant_id.get(v, 0) + c


def _round_variant_id(round_result) -> str:
    """Eindeutiger Bezeichner der effektiven Variante der Runde 1 (zum Bucketten)."""
    ann = round_result.announcement
    variant = ann.variant
    if ann.slalom:
        return f"slalom_{variant.mode.value}"
    if variant.has_trump:
        assert variant.trump_suit is not None
        return f"{variant.mode.value}_{variant.trump_suit.name.lower()}"
    return variant.mode.value


def update_stats_from_solo_game(
    stats_by_seat: dict[int, PlayerStats],
    game: GameResult,
) -> None:
    """Schreibt die Spielergebnisse einer Solo-Partie in vier per-Sitz-Stats.

    `stats_by_seat[seat]` ist die Statistik fuer den Spieler-Index `seat`
    (0..3) in dieser Partie. Wer welcher "Rolle" entspricht (A / B / H1 / H2),
    wird im aufrufenden Code (siehe solo_eval.py) verwaltet.
    """
    if not game.rounds:
        return

    winner = game.winning_team  # im Solo = Spieler-Index 0..3

    # Per-Spiel Erhebungen
    for seat, stats in stats_by_seat.items():
        stats.games_played += 1
        stats.total_score += game.final_scores.get(seat, 0)
        stats.total_rounds += len(game.rounds)

    stats_by_seat[winner].games_won += 1

    # Pro-Variante: nehmen wir die Ansage der ERSTEN Runde
    variant_id = _round_variant_id(game.rounds[0])
    for seat, stats in stats_by_seat.items():
        stats.games_by_variant_id[variant_id] = (
            stats.games_by_variant_id.get(variant_id, 0) + 1
        )
    stats_by_seat[winner].wins_by_variant_id[variant_id] = (
        stats_by_seat[winner].wins_by_variant_id.get(variant_id, 0) + 1
    )

    # Matsch-Rate pro Spieler pro Runde
    for rnd in game.rounds:
        if rnd.matsch_team is not None and rnd.matsch_team in stats_by_seat:
            stats_by_seat[rnd.matsch_team].matsch_for += 1


def format_solo_stats_table(
    label_a: str, stats_a: PlayerStats,
    label_b: str, stats_b: PlayerStats,
    label_h: str, stats_h: PlayerStats,
) -> str:
    """Konsolen-Tabelle. `stats_h` ist die zusammengelegte Statistik der beiden
    Heuristik-Sitze (durch externes Mergen entstanden)."""
    lines = []
    headers = [label_a, label_b, label_h]
    col_w = 14
    label_w = 32

    lines.append(
        f"{'Metrik':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in headers)
    )
    lines.append("-" * (label_w + col_w * len(headers)))

    def row(name: str, vals: list[str]) -> str:
        return f"{name:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in vals)

    lines.append(row(
        "Spiele",
        [str(s.games_played) for s in (stats_a, stats_b, stats_h)],
    ))
    lines.append(row(
        "Siege",
        [str(s.games_won) for s in (stats_a, stats_b, stats_h)],
    ))
    lines.append(row(
        "Win-Rate",
        [f"{s.win_rate * 100:.1f}%" for s in (stats_a, stats_b, stats_h)],
    ))
    lines.append(row(
        "Avg-Score / Partie",
        [f"{s.avg_score:.1f}" for s in (stats_a, stats_b, stats_h)],
    ))
    lines.append(row(
        "Avg-Runden / Partie",
        [f"{s.avg_rounds:.1f}" for s in (stats_a, stats_b, stats_h)],
    ))
    lines.append(row(
        "Matsch-Rate / Runde",
        [f"{s.matsch_rate_per_round * 100:.2f}%" for s in (stats_a, stats_b, stats_h)],
    ))

    # Win-Rate pro Variante
    all_variants = sorted(
        set(stats_a.games_by_variant_id)
        | set(stats_b.games_by_variant_id)
        | set(stats_h.games_by_variant_id)
    )
    if all_variants:
        lines.append("")
        lines.append(
            f"{'Win-Rate pro Variante':<{label_w}}"
            + "".join(f"{h:>{col_w}}" for h in headers)
        )
        lines.append("-" * (label_w + col_w * len(headers)))
        for v in all_variants:
            cells = []
            for s in (stats_a, stats_b, stats_h):
                wr = s.win_rate_for_variant(v)
                n = s.games_by_variant_id.get(v, 0)
                cells.append(f"{wr * 100:.1f}% (n={n})")
            lines.append(f"  {v:<{label_w - 2}}" + "".join(f"{c:>{col_w}}" for c in cells))

    return "\n".join(lines)
