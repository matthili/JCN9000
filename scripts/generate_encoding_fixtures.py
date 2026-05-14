"""Generator: erzeugt `spec/fixtures/encoding_fixtures.json` aus handgepflegten Szenarien.

Jede Fixture ist ein konkreter Spielzustand mit erwartetem Featurevektor + Maske.
Die Vektoren werden aus dem aktuellen Python-Encoder berechnet — eine TypeScript-
Implementierung muss exakt dieselben Werte liefern, sonst stimmt sie nicht mit
den NN-Trainingsdaten überein.

Aufruf:
    python -m scripts.generate_encoding_fixtures
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from training.encoder import encode_state, legal_action_mask


SPEC_VERSION = "1.0.0"
ENCODING_VERSION = "1.0.0"


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


def card_to_dict(c: Card) -> dict[str, str]:
    return {"suit": c.suit.name, "rank": c.rank.name}


def variant_to_dict(v: Variant) -> dict[str, Any]:
    d = {"mode": v.mode.name}
    if v.trump_suit is not None:
        d["trump_suit"] = v.trump_suit.name
    return d


def announcement_to_dict(a: Announcement) -> dict[str, Any]:
    return {
        "variant": variant_to_dict(a.variant),
        "slalom": a.slalom,
    }


def state_to_dict(hand: list[Card], state: GameState) -> dict[str, Any]:
    return {
        "hand": [card_to_dict(c) for c in hand],
        "variant_effective": variant_to_dict(state.variant),
        "announcement": announcement_to_dict(state.announcement),
        "current_trick_cards": [card_to_dict(c) for c in state.current_trick_cards],
        "current_trick_starter": state.current_trick_starter,
        "player_idx": state.player_idx,
        "teams": list(state.teams),
        "completed_tricks": [
            [card_to_dict(c) for c in t] for t in state.completed_tricks
        ],
        "own_team_score": state.own_team_score,
        "opp_team_score": state.opp_team_score,
        "round_idx": state.round_idx,
        "trick_idx": state.trick_idx,
        "num_players": state.num_players,
    }


@dataclass
class Fixture:
    id: str
    description: str
    hand: list[Card]
    state: GameState


def _make_state(
    *,
    player_idx: int = 0,
    variant: Variant,
    announcement: Announcement | None = None,
    trick: list[Card] | None = None,
    starter: int = 0,
    completed_tricks: list[list[Card]] | None = None,
    teams: list[int] | None = None,
    own_score: int = 0,
    opp_score: int = 0,
    round_idx: int = 0,
    trick_idx: int = 0,
) -> GameState:
    return GameState(
        player_idx=player_idx,
        variant=variant,
        announcement=announcement or Announcement(variant=variant),
        current_trick_cards=list(trick or []),
        current_trick_starter=starter,
        teams=list(teams or [0, 1, 0, 1]),
        completed_tricks=[list(t) for t in (completed_tricks or [])],
        own_team_score=own_score,
        opp_team_score=opp_score,
        round_idx=round_idx,
        trick_idx=trick_idx,
        num_players=4,
    )


def build_fixtures() -> list[Fixture]:
    fixtures: list[Fixture] = []

    # --- Trumpf-Szenarien ---
    fixtures.append(Fixture(
        id="fix_001_trumpf_leerer_stich_anspiel",
        description=(
            "Trumpf-Modus (Eichel), eigener Spieler am Anspielen mit Buur in der Hand. "
            "Alle Karten sind legal."
        ),
        hand=[
            C(Suit.EICHEL, Rank.UNTER),
            C(Suit.EICHEL, Rank.ASS),
            C(Suit.HERZ, Rank.ASS),
            C(Suit.LAUB, Rank.SECHS),
        ],
        state=_make_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.EICHEL),
        ),
    ))

    fixtures.append(Fixture(
        id="fix_002_trumpf_farbzwang_herz_lead",
        description=(
            "Trumpf-Eichel, Herz-Ass wurde angespielt. Eigene Hand hat 2 Herz-Karten "
            "und einen Buur. Legal: beide Herz-Karten + Buur (Buur-Ausnahme)."
        ),
        hand=[
            C(Suit.HERZ, Rank.OBER),
            C(Suit.HERZ, Rank.SECHS),
            C(Suit.EICHEL, Rank.UNTER),  # Buur
            C(Suit.LAUB, Rank.ASS),
        ],
        state=_make_state(
            player_idx=1,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.HERZ, Rank.ASS)],
            starter=0,
            trick_idx=2,
        ),
    ))

    fixtures.append(Fixture(
        id="fix_003_trumpf_untertrumpfen_verboten",
        description=(
            "Trumpf-Eichel, Herz-Lead, Trumpf-Nell liegt im Stich. Eigene Hand hat "
            "niedrige Trümpfe und Nicht-Trumpf-Karten. Untertrumpfen verboten — "
            "nur Nicht-Trumpf-Karten und höhere Trümpfe sind legal."
        ),
        hand=[
            C(Suit.EICHEL, Rank.SECHS),
            C(Suit.EICHEL, Rank.SIEBEN),
            C(Suit.LAUB, Rank.ASS),
            C(Suit.LAUB, Rank.SECHS),
        ],
        state=_make_state(
            player_idx=2,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.HERZ, Rank.ASS), C(Suit.EICHEL, Rank.NEUN)],
            starter=0,
            trick_idx=3,
            own_score=45,
            opp_score=23,
        ),
    ))

    fixtures.append(Fixture(
        id="fix_004_trumpf_buur_einzig_bei_trumpf_lead",
        description=(
            "Trumpf-Eichel, Trumpf wurde angespielt. Einziger Trumpf in der Hand ist "
            "der Buur — Buur-Ausnahme erlaubt beliebige Karte."
        ),
        hand=[
            C(Suit.EICHEL, Rank.UNTER),  # Buur
            C(Suit.HERZ, Rank.ASS),
            C(Suit.LAUB, Rank.SECHS),
        ],
        state=_make_state(
            player_idx=3,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.EICHEL, Rank.SECHS)],
            starter=0,
        ),
    ))

    fixtures.append(Fixture(
        id="fix_005_trumpf_voller_stich_letzter_spieler",
        description=(
            "Trumpf-Herz, drei Karten liegen, ich bin der vierte. Lead war Eichel, "
            "ich habe keine Eichel → frei abwerfen."
        ),
        hand=[
            C(Suit.HERZ, Rank.ZEHN),
            C(Suit.LAUB, Rank.SECHS),
            C(Suit.SCHELLE, Rank.NEUN),
        ],
        state=_make_state(
            player_idx=3,
            variant=Variant.trumpf(Suit.HERZ),
            trick=[
                C(Suit.EICHEL, Rank.KOENIG),
                C(Suit.EICHEL, Rank.SECHS),
                C(Suit.EICHEL, Rank.OBER),
            ],
            starter=0,
            trick_idx=5,
            own_score=80,
            opp_score=72,
        ),
    ))

    # --- Bock (Oben) ---
    fixtures.append(Fixture(
        id="fix_006_oben_anspielen",
        description=(
            "Bock-Modus, Anspielen. Keine Trumpf-Logik; reiner Farbzwang."
        ),
        hand=[
            C(Suit.HERZ, Rank.ASS),
            C(Suit.HERZ, Rank.SECHS),
            C(Suit.LAUB, Rank.ZEHN),
            C(Suit.EICHEL, Rank.SIEBEN),
        ],
        state=_make_state(
            player_idx=0,
            variant=Variant.oben(),
        ),
    ))

    fixtures.append(Fixture(
        id="fix_007_oben_farbzwang_streng",
        description=(
            "Bock-Modus, Herz wurde angespielt. Eigene Hand hat Herz → muss bedienen. "
            "Kein Buur-Konzept bei Bock."
        ),
        hand=[
            C(Suit.HERZ, Rank.ASS),
            C(Suit.HERZ, Rank.SECHS),
            C(Suit.EICHEL, Rank.UNTER),  # bei Bock kein Buur, normale Karte
        ],
        state=_make_state(
            player_idx=2,
            variant=Variant.oben(),
            trick=[C(Suit.HERZ, Rank.KOENIG)],
            starter=1,
            trick_idx=4,
        ),
    ))

    # --- Geiss (Unten) ---
    fixtures.append(Fixture(
        id="fix_008_unten_sechs_sticht",
        description=(
            "Geiss-Modus, Lead Herz-Ass. Eigene Hand hat Herz-6 → sticht (umgekehrte Reihenfolge)."
        ),
        hand=[
            C(Suit.HERZ, Rank.SECHS),
            C(Suit.HERZ, Rank.SIEBEN),
            C(Suit.LAUB, Rank.ASS),
        ],
        state=_make_state(
            player_idx=1,
            variant=Variant.unten(),
            trick=[C(Suit.HERZ, Rank.ASS)],
            starter=0,
        ),
    ))

    # --- Slalom ---
    fixtures.append(Fixture(
        id="fix_009_slalom_stich_0_oben",
        description=(
            "Slalom mit Start oben. Stich 0 ist effektiv Bock. `is_slalom_flag` = 1."
        ),
        hand=[
            C(Suit.EICHEL, Rank.ASS),
            C(Suit.HERZ, Rank.ASS),
            C(Suit.LAUB, Rank.SECHS),
            C(Suit.SCHELLE, Rank.SECHS),
        ],
        state=_make_state(
            player_idx=0,
            variant=Variant.oben(),  # effektive Variante in Stich 0 bei Slalom-Start-Oben
            announcement=Announcement(variant=Variant.oben(), slalom=True),
            trick_idx=0,
        ),
    ))

    fixtures.append(Fixture(
        id="fix_010_slalom_stich_1_unten",
        description=(
            "Slalom mit Start oben. Stich 1 ist effektiv Geiss (Modus-Wechsel). "
            "`is_unten=1`, `is_slalom_flag=1`."
        ),
        hand=[
            C(Suit.EICHEL, Rank.KOENIG),
            C(Suit.HERZ, Rank.SECHS),
            C(Suit.LAUB, Rank.SIEBEN),
        ],
        state=_make_state(
            player_idx=2,
            variant=Variant.unten(),  # effektive Variante in Stich 1
            announcement=Announcement(variant=Variant.oben(), slalom=True),
            trick_idx=1,
            completed_tricks=[[
                C(Suit.HERZ, Rank.ASS),
                C(Suit.HERZ, Rank.SECHS),
                C(Suit.HERZ, Rank.OBER),
                C(Suit.HERZ, Rank.SIEBEN),
            ]],
        ),
    ))

    # --- Mid-Game ---
    fixtures.append(Fixture(
        id="fix_011_trumpf_mid_game_hoher_score",
        description=(
            "Trumpf-Laub, Spieler 0 in Stich 5, eigenes Team führt 87:64. Drei "
            "abgeschlossene Stiche sind in der History."
        ),
        hand=[
            C(Suit.LAUB, Rank.OBER),
            C(Suit.LAUB, Rank.KOENIG),
            C(Suit.EICHEL, Rank.SIEBEN),
            C(Suit.SCHELLE, Rank.OBER),
        ],
        state=_make_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.LAUB),
            completed_tricks=[
                [C(Suit.LAUB, Rank.UNTER), C(Suit.LAUB, Rank.SECHS),
                 C(Suit.LAUB, Rank.SIEBEN), C(Suit.LAUB, Rank.NEUN)],
                [C(Suit.HERZ, Rank.ASS), C(Suit.HERZ, Rank.SIEBEN),
                 C(Suit.HERZ, Rank.SECHS), C(Suit.HERZ, Rank.OBER)],
                [C(Suit.SCHELLE, Rank.ASS), C(Suit.SCHELLE, Rank.SECHS),
                 C(Suit.SCHELLE, Rank.SIEBEN), C(Suit.SCHELLE, Rank.NEUN)],
            ],
            trick_idx=3,
            own_score=87,
            opp_score=64,
            round_idx=5,
        ),
    ))

    # --- Endspiel / Letzter Stich ---
    fixtures.append(Fixture(
        id="fix_012_trumpf_letzter_stich",
        description=(
            "Trumpf-Eichel, Stich 8 (letzter Stich der Runde), nur eine Karte in der Hand."
        ),
        hand=[
            C(Suit.HERZ, Rank.SECHS),
        ],
        state=_make_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[],
            trick_idx=8,
            own_score=120,
            opp_score=30,
        ),
    ))

    return fixtures


def fixture_to_dict(fixture: Fixture) -> dict[str, Any]:
    vec = encode_state(fixture.hand, fixture.state)
    mask = legal_action_mask(fixture.hand, fixture.state)
    return {
        "id": fixture.id,
        "description": fixture.description,
        "input": state_to_dict(fixture.hand, fixture.state),
        "expected": {
            "state_vector": [round(float(x), 6) for x in vec.tolist()],
            "legal_mask": [int(x) for x in mask.tolist()],
            "state_vector_shape": list(vec.shape),
            "legal_mask_shape": list(mask.shape),
        },
    }


def main():
    fixtures = build_fixtures()
    payload = {
        "spec_version": SPEC_VERSION,
        "encoding_version": ENCODING_VERSION,
        "description": (
            "Konsistenz-Fixtures für den State-Encoder. Eine TypeScript-Implementierung "
            "muss alle hier aufgeführten state_vector- und legal_mask-Werte exakt "
            "reproduzieren."
        ),
        "fixture_count": len(fixtures),
        "fixtures": [fixture_to_dict(f) for f in fixtures],
    }

    out_path = Path("spec/fixtures/encoding_fixtures.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Geschrieben: {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  {len(fixtures)} Fixtures")


if __name__ == "__main__":
    main()
