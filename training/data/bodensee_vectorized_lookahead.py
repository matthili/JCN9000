"""Vektorisierter Full-Round-Lookahead fuer Bodensee-Jass.

Analog zu `training/data/vectorized_lookahead.py` (Kreuz) und
`solo_vectorized_lookahead.py` (Solo), aber fuer die 2-Spieler-Tisch-Mechanik:

- 2 Spieler statt 4 -> ein Stich besteht aus 2 Karten
- `BodenseePlayerState` mit Tisch-Stapeln statt reiner Hand; eine gespielte
  sichtbare Tisch-Karte deckt die darunterliegende verdeckte auf
  (`play_card_from_state`)
- eigener Encoder (`encode_state_bodensee`), eigene `legal_moves_bodensee`
- 18 Stiche pro Runde, Matsch (alle 18 Stiche) = +MATCH_BONUS

Statt nur den laufenden Stich (`bodensee_mcts_lookahead.py`, single-trick) zu
bewerten, spielt jeder Rollout die GESAMTE Restrunde aus und liefert die
Punkte-Differenz bis Rundenende. Das beseitigt die strukturelle
Kurzsichtigkeit des Single-Trick-Lookaheads (er sah nur "gewinne ich diesen
einen Stich?", nicht "wie steht die Runde am Ende?").

Performance:
- Alle Rollouts (legal_cards x rollouts_per_card) werden im Lockstep getickt;
  pro Tick eine gebatchte Inferenz (`request_many`) ueber alle Rollouts, die
  gerade eine Karten-Entscheidung brauchen -> die GPU bekommt grosse Batches.
- Erzwungene Zuege (genau 1 legale Karte) werden ohne NN-Inferenz vorgespielt.
  Im Bodensee-Endspiel ist der Verfuegbar-Pool oft erzwungen -- das spart sehr
  viele NN-Aufrufe gegenueber einem naiven Tiefe-18-Rollout.

Vereinfachungen (bewusst, analog zum Kreuz-Rollout):
- Der Matsch-Bonus wird nur vergeben, wenn der Rollout die ganze Runde abdeckt
  (Start bei Stich 0) und ein Sitz ALLE Stiche gewann. Bei einem Lookahead
  mitten in der Runde ist nicht bekannt, ob die Vor-Lookahead-Stiche demselben
  Sitz gehoerten -- dann kein Bonus.
- Der Partie-Punktestand (own_score/opp_score) wird pro Sitz aus dem
  Root-State uebernommen (er aendert sich innerhalb einer Runde nicht).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from jass_engine.bodensee.player_state import BodenseePlayerState
from jass_engine.bodensee.rules import legal_moves_bodensee
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.bodensee.trick import play_card_from_state
from jass_engine.card import Card
from jass_engine.rules import MATCH_BONUS, trick_points, trick_winner
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement
from training.bodensee_encoder import (
    encode_state_bodensee,
    index_to_card,
    legal_action_mask_bodensee,
)
from training.data.bodensee_determinization import determinize_bodensee_states

if TYPE_CHECKING:
    # Nur fuer Typ-Annotationen; dank `from __future__ import annotations` wird
    # der Typ zur Laufzeit nie ausgewertet. So zieht dieses Modul nicht
    # zwangslaeufig TensorFlow (ueber den InferenceServer) nach -- praktisch
    # fuer Tests mit Stub-Servern.
    from training.rl.batched_selfplay import InferenceServer


REWARD_SCALE = 200.0
TRICKS_PER_ROUND = 18
NUM_PLAYERS = 2


@dataclass
class BodenseeRollout:
    """Ein einzelner Lookahead-Rollout: simuliert die Restrunde bis Rundenende.

    Felder:
        card_idx_of_first_move: Index in der legal-Karten-Liste -- fuer welche
            Wurzel-Karte dieser Rollout den Reward sammelt.
        player_states: [seat0, seat1], vollstaendig determinisiert (konkrete
            Hidden-Karten). Werden waehrend der Simulation mutiert.
        current_trick_cards: Karten im laufenden Stich (0..2).
        current_trick_starter: Sitz, der den laufenden Stich angespielt hat.
        completed_tricks: abgeschlossene Stiche (fuer den Encoder).
        announcement: Ansage der Runde (fuer Slalom-Variantenwechsel).
        root_seat: der Sitz, dessen Entscheidung wir bewerten.
        announcer_seat: welcher Sitz diese Runde angesagt hat (Encoder-Bit).
        card_points: seat -> akkumulierte Stich-Punkte im Rollout.
        tricks_won: seat -> Anzahl im Rollout gewonnener Stiche (Matsch-Check).
        trick_idx: absoluter Stich-Index der Runde (0..18).
        root_own_score / root_opp_score: Partie-Punktestand aus Wurzel-Sicht.
        done: True wenn die Runde zu Ende ist.
    """

    card_idx_of_first_move: int
    player_states: list[BodenseePlayerState]
    current_trick_cards: list[Card]
    current_trick_starter: int
    completed_tricks: list[CompletedTrick]
    announcement: Announcement
    root_seat: int
    announcer_seat: int
    card_points: dict[int, int]
    tricks_won: dict[int, int]
    trick_idx: int
    root_own_score: int = 0
    root_opp_score: int = 0
    done: bool = False

    def _variant_now(self):
        """Effektive Variante des aktuellen Stichs (beruecksichtigt Slalom)."""
        return self.announcement.variant_for_trick(self.trick_idx)

    def next_seat(self) -> int:
        return (self.current_trick_starter + len(self.current_trick_cards)) % NUM_PLAYERS

    def _announcer_flag_for(self, seat: int) -> bool:
        return seat == self.announcer_seat

    def _score_for(self, seat: int) -> tuple[int, int]:
        """(own_score, opp_score) aus Sicht von `seat`."""
        if seat == self.root_seat:
            return self.root_own_score, self.root_opp_score
        return self.root_opp_score, self.root_own_score

    def needs_inference(self) -> bool:
        """True, wenn der naechste Spieler eine (nicht-erzwungene) Entscheidung
        treffen muss."""
        return (not self.done) and (len(self.current_trick_cards) < NUM_PLAYERS)

    def get_state_for_inference(self) -> tuple[np.ndarray, np.ndarray]:
        """Baut (encoder_input, mask) fuer den als naechstes ziehenden Sitz."""
        seat = self.next_seat()
        ps = self.player_states[seat]
        opp = self.player_states[1 - seat]
        own_score, opp_score = self._score_for(seat)
        state = BodenseeGameState(
            player_idx=seat,
            variant=self._variant_now(),
            announcement=self.announcement,
            current_trick_cards=list(self.current_trick_cards),
            current_trick_starter=self.current_trick_starter,
            completed_tricks=list(self.completed_tricks),
            opponent_visible_table=opp.visible_table_cards,
            opponent_hand_count=len(opp.hand),
            opponent_hidden_table_count=opp.hidden_table_count,
            own_hidden_table_count=ps.hidden_table_count,
            own_score=own_score,
            opp_score=opp_score,
            round_idx=0,
            trick_idx=self.trick_idx,
        )
        x = encode_state_bodensee(
            list(ps.hand),
            ps.table,
            state,
            i_am_announcer=self._announcer_flag_for(seat),
        ).astype(np.float32)
        m = legal_action_mask_bodensee(
            list(ps.hand),
            ps.visible_table_cards,
            state,
        ).astype(np.float32)
        return x, m

    def apply_action(self, action_idx: int, rng: random.Random) -> None:
        """Wendet die per NN gewaehlte Karte an, loest ggf. den Stich auf und
        spielt anschliessend erzwungene Folgezuege automatisch durch."""
        if self.done:
            return
        seat = self.next_seat()
        ps = self.player_states[seat]
        variant = self._variant_now()
        legal = legal_moves_bodensee(ps, self.current_trick_cards, variant)
        chosen = index_to_card(action_idx)
        if chosen not in legal:
            # Fallback bei Race / numerischem Fehler: erste legale Karte
            chosen = legal[0] if legal else ps.available_cards[0]
        self._play(seat, chosen)
        self._auto_advance_forced(rng)

    def _play(self, seat: int, card: Card) -> None:
        """Spielt `card` fuer `seat` (Hand oder Tisch, inkl. Aufdecken) und
        loest den Stich auf, sobald 2 Karten liegen."""
        play_card_from_state(self.player_states[seat], card)
        self.current_trick_cards.append(card)
        if len(self.current_trick_cards) == NUM_PLAYERS:
            self._resolve_trick()

    def _resolve_trick(self) -> None:
        variant = self._variant_now()
        is_last = all(p.total_cards_remaining == 0 for p in self.player_states)
        win_pos = trick_winner(self.current_trick_cards, variant)
        winner_seat = (self.current_trick_starter + win_pos) % NUM_PLAYERS
        pts = trick_points(self.current_trick_cards, variant, is_last_trick=is_last)
        self.card_points[winner_seat] = self.card_points.get(winner_seat, 0) + pts
        self.tricks_won[winner_seat] = self.tricks_won.get(winner_seat, 0) + 1

        self.completed_tricks.append(CompletedTrick(
            starter=self.current_trick_starter,
            cards=tuple(self.current_trick_cards),
        ))
        self.current_trick_cards = []
        self.current_trick_starter = winner_seat
        self.trick_idx += 1

        if is_last:
            # Matsch nur, wenn der Rollout die ganze Runde abgedeckt hat (Start
            # bei Stich 0) und ein Sitz ALLE Stiche gewann. tricks_won zaehlt nur
            # Rollout-Stiche; trick_idx ist der absolute Runden-Index. Beides ist
            # nur dann gleich, wenn der Rollout bei Stich 0 begann.
            for seat, won in self.tricks_won.items():
                if won == self.trick_idx and self.trick_idx >= TRICKS_PER_ROUND:
                    self.card_points[seat] += MATCH_BONUS
            self.done = True

    def _auto_advance_forced(self, rng: random.Random) -> None:
        """Spielt alle erzwungenen Zuege (genau 1 legale Karte) ohne NN-Inferenz
        durch, bis eine echte Entscheidung ansteht oder die Runde endet."""
        while self.needs_inference():
            seat = self.next_seat()
            ps = self.player_states[seat]
            variant = self._variant_now()
            legal = legal_moves_bodensee(ps, self.current_trick_cards, variant)
            if len(legal) != 1:
                return
            self._play(seat, legal[0])

    def get_reward(self) -> float:
        own = self.card_points.get(self.root_seat, 0)
        opp = self.card_points.get(1 - self.root_seat, 0)
        return (own - opp) / REWARD_SCALE


def _make_bodensee_rollout(
    card_idx: int,
    first_move: Card,
    own_state: BodenseePlayerState,
    state: BodenseeGameState,
    i_am_announcer: bool,
    rng: random.Random,
) -> BodenseeRollout:
    """Erzeugt ein determinisiertes Rollout-Objekt fuer eine Wurzel-Karte."""
    player_states = determinize_bodensee_states(
        own_state=own_state,
        opp_visible_table=state.opponent_visible_table,
        opp_hand_count=state.opponent_hand_count,
        opp_hidden_count=state.opponent_hidden_table_count,
        completed_tricks=state.completed_tricks,
        current_trick_cards=state.current_trick_cards,
        own_seat=state.player_idx,
        rng=rng,
    )
    root_seat = state.player_idx
    announcer_seat = root_seat if i_am_announcer else (1 - root_seat)

    rollout = BodenseeRollout(
        card_idx_of_first_move=card_idx,
        player_states=player_states,
        current_trick_cards=list(state.current_trick_cards),
        current_trick_starter=state.current_trick_starter,
        completed_tricks=list(state.completed_tricks),
        announcement=state.announcement,
        root_seat=root_seat,
        announcer_seat=announcer_seat,
        card_points={0: 0, 1: 0},
        tricks_won={0: 0, 1: 0},
        trick_idx=state.trick_idx,
        root_own_score=state.own_score,
        root_opp_score=state.opp_score,
    )

    # Wurzel-Karte spielen (es ist per Konstruktion der Zug des root_seat),
    # dann erzwungene Folgezuege automatisch durchspielen.
    rollout._play(root_seat, first_move)
    rollout._auto_advance_forced(rng)
    return rollout


def compute_card_scores_bodensee_vectorized(
    own_state: BodenseePlayerState,
    state: BodenseeGameState,
    inference_server: InferenceServer,
    i_am_announcer: bool,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
    max_steps_safety: int = 400,
) -> dict[Card, float]:
    """Vektorisierter Full-Round-Lookahead fuer Bodensee.

    Args:
        own_state: synthetischer eigener BodenseePlayerState (Hand + Tisch-Stapel
            mit `has_hidden`-Markern). Die Hidden-Karten werden je Rollout neu
            determinisiert.
        state: BodenseeGameState (eigene Sicht).
        inference_server: GPU-Inferenz-Server fuer die Zuege in den Rollouts.
        i_am_announcer: ob der Wurzel-Spieler diese Runde angesagt hat.
        rollouts_per_card: Determinisierungen pro Kandidaten-Karte.
        rng: optionaler RNG (Reproduzierbarkeit).
        max_steps_safety: Obergrenze fuer Lockstep-Ticks (Schutz vor Endlos).

    Returns:
        Mapping `card -> mean_reward` ueber alle Rollouts dieser Karte. Reward
        ist (eigene_Punkte - Gegner_Punkte) / REWARD_SCALE bis Rundenende.
    """
    if rng is None:
        rng = random.Random()

    legal = legal_moves_bodensee(own_state, state.current_trick_cards, state.variant)
    if len(legal) == 1:
        return {legal[0]: 0.0}

    all_rollouts: list[BodenseeRollout] = []
    for card_idx, card in enumerate(legal):
        for _ in range(rollouts_per_card):
            all_rollouts.append(
                _make_bodensee_rollout(card_idx, card, own_state, state, i_am_announcer, rng)
            )

    # Lockstep-Tick: alle Rollouts, die eine Entscheidung brauchen, gemeinsam
    # durch eine gebatchte Inferenz schicken.
    for _ in range(max_steps_safety):
        waiting = [r for r in all_rollouts if r.needs_inference()]
        if not waiting:
            break

        states_batch: list[np.ndarray] = []
        masks_batch: list[np.ndarray] = []
        for r in waiting:
            x, m = r.get_state_for_inference()
            states_batch.append(x)
            masks_batch.append(m)

        results = inference_server.request_many(states_batch, masks_batch)

        for r, mask, (policy, _value) in zip(waiting, masks_batch, results):
            legal_policy = policy * mask
            s = legal_policy.sum()
            if s <= 0:
                legal_indices = np.where(mask > 0.5)[0]
                action_idx = int(rng.choice(legal_indices))
            else:
                action_idx = int(np.argmax(legal_policy))
            r.apply_action(action_idx, rng)

    rewards_per_card: dict[Card, list[float]] = {c: [] for c in legal}
    for r in all_rollouts:
        rewards_per_card[legal[r.card_idx_of_first_move]].append(r.get_reward())

    return {c: float(np.mean(rs)) if rs else 0.0 for c, rs in rewards_per_card.items()}


def best_card_bodensee_vectorized(
    own_state: BodenseePlayerState,
    state: BodenseeGameState,
    inference_server: InferenceServer,
    i_am_announcer: bool,
    rollouts_per_card: int = 10,
    rng: random.Random | None = None,
) -> tuple[Card, dict[Card, float]]:
    """Bequemer Wrapper: liefert (beste_Karte, scores)."""
    scores = compute_card_scores_bodensee_vectorized(
        own_state=own_state,
        state=state,
        inference_server=inference_server,
        i_am_announcer=i_am_announcer,
        rollouts_per_card=rollouts_per_card,
        rng=rng,
    )
    best = max(scores, key=lambda c: scores[c])
    return best, scores


__all__ = [
    "BodenseeRollout",
    "compute_card_scores_bodensee_vectorized",
    "best_card_bodensee_vectorized",
    "REWARD_SCALE",
]
