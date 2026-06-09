"""Regelbasierter Spieler fuer Solo-Jass (4 Spieler, jeder gegen jeden).

Erbt vom Team-HeuristicPlayer. Wichtige Eigenschaften:

- **Schmieren ist automatisch deaktiviert.** Der Schmier-Branch in
  `HeuristicPlayer.choose_card` wird ueber `_is_partner_winning(state)`
  geschuetzt. Im Solo (teams=[0,1,2,3]) hat jeder Spieler eine eigene
  Team-ID -- der Check liefert IMMER False, der Schmier-Zweig ist
  damit strukturell tot Code.
- **Stechen so aggressiv wie moeglich.** Ohne Partner ist jeder
  abgegebene Punkt verloren. Die geerbte Stech-Logik (uebernehmen
  wann immer moeglich, mit minimaler Karte wenn ich letzter bin,
  sonst mit hoher Karte) passt schon.
- **Konservativere Slalom-Bewertung.** Slalom braucht ohne Partner
  praktisch eine doppelseitige Hand komplett im Alleingang. Der
  `slalom_base_factor` ist hier niedriger als beim Team-Default.

Was bewusst nicht ueberschrieben wurde:
- Die Trumpf-/Gumpf-/Oben-/Unten-Scores. Im Solo sind die noetigen
  Anpassungen klein und schwer ohne empirische Daten zu kalibrieren.
  Phase 1 der MCTS-Datengen mit dieser Heuristik als Lehrer wird
  zeigen, wo der Bot zu oft / zu selten welche Ansage waehlt; danach
  ggf. Feintuning.
"""

from __future__ import annotations

import random

from players.heuristic_player import HeuristicPlayer


class SoloHeuristicPlayer(HeuristicPlayer):
    """Heuristik fuer Solo-Jass.

    Identische Trumpf-/Gumpf-/Oben-/Unten-Bewertung wie die Team-Heuristik,
    aber:
    - `push_threshold=0` -- irrelevant, weil im Solo `allow_push=False`
      gilt und damit `can_push` immer False ist; nur zur Klarheit gesetzt.
    - Die Defaults stammen aus dem Solo-Ansage-Tuning
      (scripts/tune_solo_announce.py, 301 Kandidaten, Finale mit 20.000
      paired-Partien): Win-Share 51.1 % gegen die Vorgaenger-Defaults
      (0.85 / 1 / 1, Skalen 1.0) -- knapp ueber der 2-SD-Schwelle (1.0 pp),
      also ein kleiner, aber gemessener Vorteil. Konsistentes Muster der
      Top-Kandidaten: gumpf_scale > 1 (Gumpf im Solo etwas mutiger ansagen).
    """

    def __init__(
        self,
        name: str,
        rng: random.Random | None = None,
        slalom_base_factor: float = 0.94,
        slalom_concentration_factor: int = 1,
        slalom_spread_factor: int = 1,
        gumpf_scale: float = 1.06,
        oben_scale: float = 0.91,
        unten_scale: float = 1.10,
    ):
        super().__init__(
            name=name,
            push_threshold=0,  # irrelevant, can_push ist immer False im Solo
            slalom_base_factor=slalom_base_factor,
            slalom_concentration_factor=slalom_concentration_factor,
            slalom_spread_factor=slalom_spread_factor,
            gumpf_scale=gumpf_scale,
            oben_scale=oben_scale,
            unten_scale=unten_scale,
            rng=rng,
        )
