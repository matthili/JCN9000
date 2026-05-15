"""HeuristicPlayer-Variante, die eine fixe Ansage erzwingt.

Wird vom balanced Datengenerator genutzt, um pro Variante (Trumpf x4, Bock,
Geiss, Slalom) gezielt gleich viele Trainings-Runden zu erzeugen. Die normale
Heuristik-Score-Logik wuerde Trumpf-Varianten ueberproportional waehlen --
das fuehrt zu schwacher Geiss-/Bock-Performance des trainierten NN.

Schiebe-Logik: Wenn die forced_announcement ein "Schieben" (None) wuerde nicht
funktionieren -- aber wir wollen ja eine fixe Ansage. Daher gibt der Player
*immer* die gleiche Ansage zurueck, auch wenn er ueblicherweise schieben
duerfte. Der Partner ist in dem Fall trotzdem rumgekommen, daher nie ein Problem.
"""

from __future__ import annotations

import random

from jass_engine.card import Card
from jass_engine.variant import Announcement
from players.heuristic_player import HeuristicPlayer


class ForcedAnnouncementPlayer(HeuristicPlayer):
    """Wie HeuristicPlayer, aber `choose_announcement` gibt immer dieselbe
    feste Announcement zurueck."""

    def __init__(
        self,
        name: str,
        forced_announcement: Announcement,
        push_threshold: int = 55,
        slalom_base_factor: float = 0.95,
        slalom_concentration_factor: int = 2,
        slalom_spread_factor: int = 1,
        rng: random.Random | None = None,
    ):
        super().__init__(
            name=name,
            push_threshold=push_threshold,
            slalom_base_factor=slalom_base_factor,
            slalom_concentration_factor=slalom_concentration_factor,
            slalom_spread_factor=slalom_spread_factor,
            rng=rng,
        )
        self.forced = forced_announcement

    def choose_announcement(
        self,
        hand: list[Card],
        round_idx: int,
        can_push: bool,
    ) -> Announcement | None:
        # Wir schieben nie und sagen immer die forced Announcement an.
        return self.forced
