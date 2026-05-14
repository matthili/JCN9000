"""Aggregierte Statistiken pro Spieler / pro Variante.

Sammelt aus mehreren gespielten Partien Metriken, die wirklich aussagekraeftig sind:
- Win-Rate gesamt
- Win-Rate aufgeschluesselt nach Variante (Trumpf-X / Bock / Geiss / Slalom)
- Durchschnittliche Punkte je Partie
- Matsch-Quote (wie oft macht das Team einen Matsch)
- Durchschnittliche Rundenzahl pro Partie
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jass_engine.game import GameResult
from jass_engine.round import RoundResult
from jass_engine.variant import PlayMode


@dataclass
class TeamStats:
    """Aggregierte Statistik fuer eine Team-Konfiguration (z.B. NN+NN oder Heuristik+Heuristik)."""

    games_played: int = 0
    games_won: int = 0
    games_lost: int = 0
    games_drawn: int = 0
    total_score: int = 0
    total_rounds: int = 0
    matsch_for: int = 0   # Wie oft hat dieses Team einen Matsch gemacht
    matsch_against: int = 0  # Wie oft hat das Gegnerteam einen Matsch gemacht
    # Win-Rate aufgeschluesselt nach (effektive) Variante in der ersten Runde
    # (Vereinfachung: wir nehmen die Ansage der ersten Runde; bei langen Partien
    # ist das nur ein Indikator, kein exakter "Win-Rate-pro-Variante")
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
    def matsch_rate_for(self) -> float:
        return self.matsch_for / self.total_rounds if self.total_rounds else 0.0

    def win_rate_for_variant(self, variant_id: str) -> float:
        games = self.games_by_variant_id.get(variant_id, 0)
        if not games:
            return 0.0
        return self.wins_by_variant_id.get(variant_id, 0) / games


def _variant_label_for_round(rnd: RoundResult) -> str:
    """Kurz-Label der Variante einer Runde, fuer Statistik-Aggregation."""
    ann = rnd.announcement
    mode = ann.variant.mode
    if mode == PlayMode.TRUMPF:
        assert ann.variant.trump_suit is not None
        prefix = f"trumpf_{ann.variant.trump_suit.name.lower()}"
    elif mode == PlayMode.OBEN:
        prefix = "oben"
    else:
        prefix = "unten"
    if ann.slalom:
        return f"slalom_{prefix}"
    return prefix


def update_stats_from_game(
    stats_a: TeamStats,
    stats_b: TeamStats,
    game: GameResult,
    team_a_id: int = 0,
    team_b_id: int = 1,
) -> None:
    """Schreibt das Ergebnis einer Partie in beide Team-Statistiken."""
    score_a = game.final_scores.get(team_a_id, 0)
    score_b = game.final_scores.get(team_b_id, 0)

    stats_a.games_played += 1
    stats_b.games_played += 1
    stats_a.total_score += score_a
    stats_b.total_score += score_b
    stats_a.total_rounds += len(game.rounds)
    stats_b.total_rounds += len(game.rounds)

    if score_a > score_b:
        stats_a.games_won += 1
        stats_b.games_lost += 1
        winner = team_a_id
    elif score_b > score_a:
        stats_b.games_won += 1
        stats_a.games_lost += 1
        winner = team_b_id
    else:
        stats_a.games_drawn += 1
        stats_b.games_drawn += 1
        winner = None

    # Pro Runde: Matsch-Statistik + Variant-Win-Rate
    # "Wer hat den Matsch gemacht" -> direkt aus RoundResult ablesbar.
    # "Win-Rate pro Variante" -> wir betrachten je Runde, welches Team mehr
    # Stichpunkte gemacht hat (nicht die Spiel-Gesamtwertung), aggregiert
    # nach der Variante dieser Runde.
    for rnd in game.rounds:
        label = _variant_label_for_round(rnd)
        if rnd.matsch_team == team_a_id:
            stats_a.matsch_for += 1
            stats_b.matsch_against += 1
        elif rnd.matsch_team == team_b_id:
            stats_b.matsch_for += 1
            stats_a.matsch_against += 1

        # Per-Variante: das Team mit mehr Stichpunkten in dieser Runde gewinnt sie
        pts_a = rnd.team_card_points.get(team_a_id, 0)
        pts_b = rnd.team_card_points.get(team_b_id, 0)
        stats_a.games_by_variant_id[label] = stats_a.games_by_variant_id.get(label, 0) + 1
        stats_b.games_by_variant_id[label] = stats_b.games_by_variant_id.get(label, 0) + 1
        if pts_a > pts_b:
            stats_a.wins_by_variant_id[label] = stats_a.wins_by_variant_id.get(label, 0) + 1
        elif pts_b > pts_a:
            stats_b.wins_by_variant_id[label] = stats_b.wins_by_variant_id.get(label, 0) + 1


def format_stats_table(label_a: str, stats_a: TeamStats, label_b: str, stats_b: TeamStats) -> str:
    """Formatiert die Stats zweier Teams als Konsolen-Tabelle."""
    lines = []
    lines.append(f"\n{'Metrik':<32} {label_a:>16} {label_b:>16}")
    lines.append("-" * 66)
    lines.append(f"{'Spiele':<32} {stats_a.games_played:>16} {stats_b.games_played:>16}")
    lines.append(f"{'Siege':<32} {stats_a.games_won:>16} {stats_b.games_won:>16}")
    lines.append(
        f"{'Win-Rate':<32} {stats_a.win_rate * 100:>15.1f}% {stats_b.win_rate * 100:>15.1f}%"
    )
    lines.append(
        f"{'Avg-Score / Partie':<32} {stats_a.avg_score:>16.1f} {stats_b.avg_score:>16.1f}"
    )
    lines.append(
        f"{'Avg-Runden / Partie':<32} {stats_a.avg_rounds:>16.1f} {stats_b.avg_rounds:>16.1f}"
    )
    lines.append(
        f"{'Matsch-Rate (pro Runde)':<32} {stats_a.matsch_rate_for * 100:>15.2f}% {stats_b.matsch_rate_for * 100:>15.2f}%"
    )

    # Variant-Aufschluesselung
    all_variants = sorted(set(stats_a.games_by_variant_id) | set(stats_b.games_by_variant_id))
    if all_variants:
        lines.append("")
        lines.append(f"{'Win-Rate pro Variante':<32} {label_a:>16} {label_b:>16}")
        lines.append("-" * 66)
        for v in all_variants:
            wr_a = stats_a.win_rate_for_variant(v) * 100
            wr_b = stats_b.win_rate_for_variant(v) * 100
            n_a = stats_a.games_by_variant_id.get(v, 0)
            n_b = stats_b.games_by_variant_id.get(v, 0)
            lines.append(
                f"  {v:<30} {wr_a:>10.1f}% (n={n_a:<3}) {wr_b:>10.1f}% (n={n_b:<3})"
            )
    return "\n".join(lines)
