"""Elo-Rating-System fuer Jass-Spieler.

Klassisches Elo wie im Schach, mit Anpassungen fuer Team-Spiele:
- Eine "Partie" zwischen zwei Teams aktualisiert die Ratings beider Teammitglieder
- Score = 1.0 bei Sieg, 0.5 bei Unentschieden, 0.0 bei Niederlage
- K-Faktor: 32 (Standard fuer schnelle Anpassung); kann reduziert werden,
  wenn man feinere Stabilitaet braucht

Verwendung im RL-Kontext: Snapshots des NN bekommen jeweils ein eigenes
Rating, sodass man den Trainingsfortschritt zeitlich verfolgen kann
("ist die neue Version wirklich besser als die vor 100 Iterationen?").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


DEFAULT_RATING = 1500.0
DEFAULT_K = 32.0


@dataclass
class EloRating:
    """Verwaltet Elo-Ratings fuer beliebig viele benannte Spieler."""

    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)
    default_rating: float = DEFAULT_RATING
    k_factor: float = DEFAULT_K

    def get(self, name: str) -> float:
        """Aktuelles Rating; falls noch nicht vorhanden, default."""
        return self.ratings.get(name, self.default_rating)

    def _ensure(self, name: str) -> None:
        if name not in self.ratings:
            self.ratings[name] = self.default_rating
            self.games_played[name] = 0

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Erwarteter Score fuer A vs. B (Standard-Elo-Formel)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_team_match(
        self,
        team_a_players: list[str],
        team_b_players: list[str],
        score_a: float,
    ) -> None:
        """Aktualisiert die Ratings nach einer Team-Partie.

        Args:
            team_a_players: Spieler-Namen in Team A (z.B. ["NN_v1", "NN_v1"])
            team_b_players: Spieler-Namen in Team B
            score_a: 1.0 wenn Team A gewonnen, 0.5 Unentschieden, 0.0 Niederlage
        """
        for p in team_a_players + team_b_players:
            self._ensure(p)

        # Team-Rating = Durchschnitt der Mitglieder
        rating_a = sum(self.ratings[p] for p in team_a_players) / len(team_a_players)
        rating_b = sum(self.ratings[p] for p in team_b_players) / len(team_b_players)

        expected_a = self.expected_score(rating_a, rating_b)
        delta = self.k_factor * (score_a - expected_a)

        # Jeder Spieler bekommt den gleichen Delta
        for p in team_a_players:
            self.ratings[p] += delta
            self.games_played[p] += 1
        for p in team_b_players:
            self.ratings[p] -= delta
            self.games_played[p] += 1

    def leaderboard(self) -> list[tuple[str, float, int]]:
        """Liste (name, rating, games) sortiert nach Rating absteigend."""
        return sorted(
            ((n, r, self.games_played.get(n, 0)) for n, r in self.ratings.items()),
            key=lambda t: -t[1],
        )

    def to_dict(self) -> dict:
        return {
            "default_rating": self.default_rating,
            "k_factor": self.k_factor,
            "ratings": dict(self.ratings),
            "games_played": dict(self.games_played),
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "EloRating":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            ratings=dict(d.get("ratings", {})),
            games_played=dict(d.get("games_played", {})),
            default_rating=float(d.get("default_rating", DEFAULT_RATING)),
            k_factor=float(d.get("k_factor", DEFAULT_K)),
        )
