"""Generator: erzeugt `spec/fixtures/bodensee_encoding_fixtures.json`.

Jede Fixture ist ein konkreter Bodensee-Spielzustand mit erwartetem
Featurevektor (291 Dims) + Aktionsmaske (36). Eine TypeScript-Implementierung
des Bodensee-Encoders muss alle hier aufgefuehrten Werte exakt reproduzieren.

Laeuft ohne TensorFlow (nur numpy) -- kann auf jeder Maschine ausgefuehrt werden.

Aufruf:
    python -m scripts.generate_bodensee_encoding_fixtures
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jass_engine.bodensee.player_state import TableStack
from jass_engine.bodensee.state import BodenseeGameState
from jass_engine.card import Card, Rank, Suit
from jass_engine.trick import CompletedTrick
from jass_engine.variant import Announcement, Variant
from training.bodensee_encoder import (
    ENCODING_VERSION,
    encode_state_bodensee,
    legal_action_mask_bodensee,
)


SPEC_VERSION = "1.2.0"

# Platzhalter-Karte fuer verdeckte Tisch-Karten. Der Encoder schaut nur auf
# `stack.has_hidden` (= hidden is not None), nicht auf den Wert -- der konkrete
# Wert ist also fuer das Encoding irrelevant.
_DUMMY_HIDDEN = Card(Suit.EICHEL, Rank.SECHS)


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


def card_to_dict(c: Card) -> dict[str, str]:
    return {"suit": c.suit.name, "rank": c.rank.name}


def variant_to_dict(v: Variant) -> dict[str, Any]:
    d: dict[str, Any] = {"mode": v.mode.name}
    if v.trump_suit is not None:
        d["trump_suit"] = v.trump_suit.name
    return d


def announcement_to_dict(a: Announcement) -> dict[str, Any]:
    return {"variant": variant_to_dict(a.variant), "slalom": a.slalom}


def table_to_dict(stacks: list[TableStack]) -> list[dict[str, Any]]:
    """Serialisiert die eigenen Tisch-Stapel.

    Pro Stapel: sichtbare Karte (oder null) + has_hidden-Flag. Der verdeckte
    Karten-WERT wird bewusst nicht serialisiert -- der Encoder nutzt ihn nicht,
    und in der echten Spielsituation kennt der Spieler ihn auch nicht.
    """
    out = []
    for s in stacks:
        out.append({
            "visible": card_to_dict(s.visible) if s.visible is not None else None,
            "has_hidden": s.has_hidden,
        })
    return out


def bodensee_state_to_dict(
    hand: list[Card],
    own_table: list[TableStack],
    state: BodenseeGameState,
    i_am_announcer: bool,
) -> dict[str, Any]:
    return {
        "hand": [card_to_dict(c) for c in hand],
        "own_table": table_to_dict(own_table),
        "variant_effective": variant_to_dict(state.variant),
        "announcement": announcement_to_dict(state.announcement),
        "current_trick_cards": [card_to_dict(c) for c in state.current_trick_cards],
        "current_trick_starter": state.current_trick_starter,
        "player_idx": state.player_idx,
        "completed_tricks": [
            {
                "starter": t.starter,
                "cards": [card_to_dict(c) for c in t.cards],
            }
            for t in state.completed_tricks
        ],
        "opponent_visible_table": [card_to_dict(c) for c in state.opponent_visible_table],
        "opponent_hand_count": state.opponent_hand_count,
        "opponent_hidden_table_count": state.opponent_hidden_table_count,
        "own_hidden_table_count": state.own_hidden_table_count,
        "own_score": state.own_score,
        "opp_score": state.opp_score,
        "round_idx": state.round_idx,
        "trick_idx": state.trick_idx,
        "i_am_announcer": i_am_announcer,
    }


@dataclass
class Fixture:
    id: str
    description: str
    hand: list[Card]
    own_table: list[TableStack]
    state: BodenseeGameState
    i_am_announcer: bool = False


def _table(
    visibles: list[Card | None],
    hidden_flags: list[bool],
) -> list[TableStack]:
    """Baut Tisch-Stapel aus sichtbaren Karten + has-hidden-Flags."""
    assert len(visibles) == len(hidden_flags)
    return [
        TableStack(
            visible=v,
            hidden=_DUMMY_HIDDEN if h else None,
        )
        for v, h in zip(visibles, hidden_flags)
    ]


def _state(
    *,
    player_idx: int = 0,
    variant: Variant,
    announcement: Announcement | None = None,
    trick: list[Card] | None = None,
    starter: int = 0,
    completed_tricks: list[CompletedTrick] | None = None,
    opp_visible: list[Card] | None = None,
    opp_hand_count: int = 6,
    opp_hidden_count: int = 6,
    own_hidden_count: int = 6,
    own_score: int = 0,
    opp_score: int = 0,
    round_idx: int = 0,
    trick_idx: int = 0,
) -> BodenseeGameState:
    return BodenseeGameState(
        player_idx=player_idx,
        variant=variant,
        announcement=announcement or Announcement(variant=variant),
        current_trick_cards=list(trick or []),
        current_trick_starter=starter,
        completed_tricks=list(completed_tricks or []),
        opponent_visible_table=list(opp_visible or []),
        opponent_hand_count=opp_hand_count,
        opponent_hidden_table_count=opp_hidden_count,
        own_hidden_table_count=own_hidden_count,
        own_score=own_score,
        opp_score=opp_score,
        round_idx=round_idx,
        trick_idx=trick_idx,
    )


def build_fixtures() -> list[Fixture]:
    fixtures: list[Fixture] = []

    # Standard-6-Stapel-Tisch (Anfangslage) als Baustein
    def full_table(visibles: list[Card]) -> list[TableStack]:
        assert len(visibles) == 6
        return _table(visibles, [True] * 6)

    # --- 1: Trumpf, Anspielen, volle Anfangslage ---
    fixtures.append(Fixture(
        id="bfix_01_trumpf_anspiel_anfang",
        description=(
            "Trumpf-Eichel, eigener Spieler am Anspielen, volle Anfangslage "
            "(6 Hand, 6 sichtbare Tisch, 6 verdeckte). Ich bin Ansager."
        ),
        hand=[
            C(Suit.EICHEL, Rank.UNTER), C(Suit.EICHEL, Rank.ASS),
            C(Suit.HERZ, Rank.ASS), C(Suit.HERZ, Rank.KOENIG),
            C(Suit.LAUB, Rank.SECHS), C(Suit.LAUB, Rank.SIEBEN),
        ],
        own_table=full_table([
            C(Suit.EICHEL, Rank.NEUN), C(Suit.EICHEL, Rank.ZEHN),
            C(Suit.SCHELLE, Rank.OBER), C(Suit.SCHELLE, Rank.KOENIG),
            C(Suit.HERZ, Rank.SECHS), C(Suit.LAUB, Rank.OBER),
        ]),
        state=_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.EICHEL),
            starter=0,
            opp_visible=[
                C(Suit.SCHELLE, Rank.SECHS), C(Suit.SCHELLE, Rank.SIEBEN),
                C(Suit.SCHELLE, Rank.ACHT), C(Suit.SCHELLE, Rank.NEUN),
                C(Suit.HERZ, Rank.SIEBEN), C(Suit.HERZ, Rank.OBER),
            ],
        ),
        i_am_announcer=True,
    ))

    # --- 2: Trumpf, Bedienzwang aus der Hand ---
    fixtures.append(Fixture(
        id="bfix_02_trumpf_bedienzwang_hand",
        description=(
            "Trumpf-Eichel, Gegner hat Herz-Ass angespielt. Ich habe Herz in der "
            "Hand -> Bedienzwang. Ich bin nicht der Ansager."
        ),
        hand=[
            C(Suit.HERZ, Rank.SECHS), C(Suit.HERZ, Rank.NEUN),
            C(Suit.LAUB, Rank.ASS), C(Suit.EICHEL, Rank.UNTER),
        ],
        own_table=_table(
            [C(Suit.LAUB, Rank.SECHS), C(Suit.LAUB, Rank.SIEBEN), None, None, None, None],
            [True, False, False, False, False, False],
        ),
        state=_state(
            player_idx=1,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.HERZ, Rank.ASS)],
            starter=0,
            opp_visible=[C(Suit.SCHELLE, Rank.SECHS), C(Suit.SCHELLE, Rank.SIEBEN)],
            opp_hand_count=4,
            opp_hidden_count=1,
            own_hidden_count=1,
            trick_idx=7,
            own_score=42,
            opp_score=38,
        ),
    ))

    # --- 3: Trumpf, Bedienzwang nur via sichtbarer Tisch-Karte ---
    fixtures.append(Fixture(
        id="bfix_03_trumpf_bedienzwang_tisch",
        description=(
            "Trumpf-Eichel, Gegner hat Schelle-Ass angespielt. Ich habe Schelle "
            "NICHT in der Hand, aber sichtbar auf dem Tisch -> Bedienzwang gilt "
            "trotzdem, nur die Tisch-Schelle ist legal."
        ),
        hand=[
            C(Suit.HERZ, Rank.ASS), C(Suit.LAUB, Rank.KOENIG),
        ],
        own_table=_table(
            [C(Suit.SCHELLE, Rank.NEUN), C(Suit.LAUB, Rank.SIEBEN)],
            [False, False],
        ),
        state=_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.SCHELLE, Rank.ASS)],
            starter=1,
            opp_visible=[C(Suit.HERZ, Rank.SECHS)],
            opp_hand_count=2,
            opp_hidden_count=0,
            own_hidden_count=0,
            trick_idx=15,
            own_score=90,
            opp_score=60,
        ),
    ))

    # --- 4: Trumpf, Buur-Ausnahme ---
    fixtures.append(Fixture(
        id="bfix_04_trumpf_buur_ausnahme",
        description=(
            "Trumpf-Eichel, Gegner spielt Trumpf-Ass. Mein einziger Trumpf ist "
            "der Buur -> Buur-Ausnahme, alle Karten legal."
        ),
        hand=[
            C(Suit.EICHEL, Rank.UNTER), C(Suit.HERZ, Rank.ASS), C(Suit.LAUB, Rank.NEUN),
        ],
        own_table=_table(
            [C(Suit.HERZ, Rank.SECHS), C(Suit.LAUB, Rank.SECHS)],
            [False, False],
        ),
        state=_state(
            player_idx=1,
            variant=Variant.trumpf(Suit.EICHEL),
            trick=[C(Suit.EICHEL, Rank.ASS)],
            starter=0,
            opp_visible=[C(Suit.SCHELLE, Rank.OBER)],
            opp_hand_count=3,
            opp_hidden_count=0,
            own_hidden_count=0,
            trick_idx=13,
        ),
    ))

    # --- 5: Gumpf, Anspielen ---
    fixtures.append(Fixture(
        id="bfix_05_gumpf_anspiel",
        description=(
            "Gumpf-Eichel, Anspielen. Mode-Bit is_gumpf gesetzt, Trump-Suit "
            "zeigt Eichel."
        ),
        hand=[
            C(Suit.EICHEL, Rank.UNTER), C(Suit.HERZ, Rank.SECHS),
            C(Suit.LAUB, Rank.ASS), C(Suit.SCHELLE, Rank.SECHS),
            C(Suit.HERZ, Rank.SIEBEN), C(Suit.LAUB, Rank.SECHS),
        ],
        own_table=full_table([
            C(Suit.EICHEL, Rank.NEUN), C(Suit.EICHEL, Rank.ASS),
            C(Suit.HERZ, Rank.ACHT), C(Suit.LAUB, Rank.NEUN),
            C(Suit.SCHELLE, Rank.SIEBEN), C(Suit.SCHELLE, Rank.ACHT),
        ]),
        state=_state(
            player_idx=0,
            variant=Variant.gumpf(Suit.EICHEL),
            opp_visible=[
                C(Suit.EICHEL, Rank.OBER), C(Suit.EICHEL, Rank.KOENIG),
                C(Suit.HERZ, Rank.NEUN), C(Suit.HERZ, Rank.ZEHN),
                C(Suit.LAUB, Rank.OBER), C(Suit.SCHELLE, Rank.OBER),
            ],
        ),
        i_am_announcer=True,
    ))

    # --- 6: Gumpf, 6 sticht in Nicht-Trumpf ---
    fixtures.append(Fixture(
        id="bfix_06_gumpf_sechs_sticht",
        description=(
            "Gumpf-Eichel, Herz-Ass angespielt (Nicht-Trumpf). In Gumpf ist die "
            "Herz-6 die staerkste Herz-Karte. Bedienzwang -> Herz-Karten legal."
        ),
        hand=[
            C(Suit.HERZ, Rank.SECHS), C(Suit.HERZ, Rank.ZEHN), C(Suit.LAUB, Rank.ASS),
        ],
        own_table=_table(
            [C(Suit.SCHELLE, Rank.NEUN), C(Suit.EICHEL, Rank.SIEBEN)],
            [True, False],
        ),
        state=_state(
            player_idx=0,
            variant=Variant.gumpf(Suit.EICHEL),
            trick=[C(Suit.HERZ, Rank.ASS)],
            starter=1,
            opp_visible=[C(Suit.LAUB, Rank.OBER)],
            opp_hand_count=3,
            opp_hidden_count=1,
            own_hidden_count=1,
            trick_idx=12,
        ),
    ))

    # --- 7: Oben, Anspielen ---
    fixtures.append(Fixture(
        id="bfix_07_oben_anspiel",
        description="Oben (Bock), Anspielen. Kein Trumpf, mode-Bit is_oben gesetzt.",
        hand=[
            C(Suit.HERZ, Rank.ASS), C(Suit.LAUB, Rank.ZEHN),
            C(Suit.EICHEL, Rank.KOENIG), C(Suit.SCHELLE, Rank.OBER),
            C(Suit.HERZ, Rank.SECHS), C(Suit.LAUB, Rank.SIEBEN),
        ],
        own_table=full_table([
            C(Suit.EICHEL, Rank.ASS), C(Suit.EICHEL, Rank.ZEHN),
            C(Suit.SCHELLE, Rank.ASS), C(Suit.SCHELLE, Rank.KOENIG),
            C(Suit.HERZ, Rank.KOENIG), C(Suit.LAUB, Rank.ASS),
        ]),
        state=_state(
            player_idx=0,
            variant=Variant.oben(),
            opp_visible=[
                C(Suit.EICHEL, Rank.SECHS), C(Suit.EICHEL, Rank.SIEBEN),
                C(Suit.SCHELLE, Rank.SECHS), C(Suit.SCHELLE, Rank.SIEBEN),
                C(Suit.HERZ, Rank.SIEBEN), C(Suit.LAUB, Rank.SECHS),
            ],
        ),
        i_am_announcer=True,
    ))

    # --- 8: Unten ---
    fixtures.append(Fixture(
        id="bfix_08_unten_sechs_stark",
        description=(
            "Unten (Geiss), Herz-Ass angespielt. In Unten ist die 6 die "
            "staerkste Karte. Bedienzwang -> Herz-Karten legal."
        ),
        hand=[
            C(Suit.HERZ, Rank.SECHS), C(Suit.HERZ, Rank.SIEBEN), C(Suit.LAUB, Rank.ASS),
        ],
        own_table=_table(
            [C(Suit.EICHEL, Rank.SECHS), C(Suit.SCHELLE, Rank.NEUN)],
            [False, False],
        ),
        state=_state(
            player_idx=1,
            variant=Variant.unten(),
            trick=[C(Suit.HERZ, Rank.ASS)],
            starter=0,
            opp_visible=[C(Suit.LAUB, Rank.NEUN)],
            opp_hand_count=2,
            opp_hidden_count=0,
            own_hidden_count=0,
            trick_idx=15,
        ),
    ))

    # --- 9: Slalom, Stich 0 (effektiv Oben) ---
    fixtures.append(Fixture(
        id="bfix_09_slalom_stich0_oben",
        description=(
            "Slalom mit Start oben. Stich 0 ist effektiv Oben. is_oben=1 UND "
            "is_slalom=1."
        ),
        hand=[
            C(Suit.EICHEL, Rank.ASS), C(Suit.HERZ, Rank.ASS),
            C(Suit.LAUB, Rank.SECHS), C(Suit.SCHELLE, Rank.SECHS),
            C(Suit.HERZ, Rank.SECHS), C(Suit.LAUB, Rank.SIEBEN),
        ],
        own_table=full_table([
            C(Suit.EICHEL, Rank.ZEHN), C(Suit.EICHEL, Rank.KOENIG),
            C(Suit.SCHELLE, Rank.ASS), C(Suit.SCHELLE, Rank.OBER),
            C(Suit.HERZ, Rank.OBER), C(Suit.LAUB, Rank.ASS),
        ]),
        state=_state(
            player_idx=0,
            variant=Variant.oben(),
            announcement=Announcement(variant=Variant.oben(), slalom=True),
            trick_idx=0,
            opp_visible=[
                C(Suit.EICHEL, Rank.SECHS), C(Suit.EICHEL, Rank.SIEBEN),
                C(Suit.SCHELLE, Rank.SECHS), C(Suit.SCHELLE, Rank.SIEBEN),
                C(Suit.HERZ, Rank.SIEBEN), C(Suit.LAUB, Rank.SECHS),
            ],
        ),
        i_am_announcer=True,
    ))

    # --- 10: Slalom, Stich 1 (effektiv Unten) ---
    fixtures.append(Fixture(
        id="bfix_10_slalom_stich1_unten",
        description=(
            "Slalom mit Start oben. Stich 1 ist effektiv Unten (Modus-Wechsel). "
            "is_unten=1 UND is_slalom=1."
        ),
        hand=[
            C(Suit.EICHEL, Rank.KOENIG), C(Suit.HERZ, Rank.SECHS), C(Suit.LAUB, Rank.SIEBEN),
        ],
        own_table=_table(
            [C(Suit.SCHELLE, Rank.SECHS), C(Suit.LAUB, Rank.NEUN)],
            [True, True],
        ),
        state=_state(
            player_idx=1,
            variant=Variant.unten(),
            announcement=Announcement(variant=Variant.oben(), slalom=True),
            trick_idx=1,
            starter=0,
            trick=[C(Suit.HERZ, Rank.ASS)],
            completed_tricks=[
                CompletedTrick(starter=0, cards=(
                    C(Suit.EICHEL, Rank.ASS), C(Suit.EICHEL, Rank.SECHS),
                )),
            ],
            opp_visible=[C(Suit.HERZ, Rank.OBER), C(Suit.LAUB, Rank.ASS)],
            opp_hand_count=5,
            opp_hidden_count=5,
            own_hidden_count=2,
        ),
    ))

    # --- 11: Mid-Game mit abgeschlossenen Stichen ---
    fixtures.append(Fixture(
        id="bfix_11_midgame_completed_tricks",
        description=(
            "Trumpf-Laub, Stich 4, mehrere abgeschlossene Stiche in der History. "
            "Pruefen: played_cards_this_round enthaelt alle bisher gespielten Karten."
        ),
        hand=[
            C(Suit.LAUB, Rank.OBER), C(Suit.LAUB, Rank.KOENIG),
            C(Suit.EICHEL, Rank.SIEBEN), C(Suit.SCHELLE, Rank.OBER),
        ],
        own_table=_table(
            [C(Suit.HERZ, Rank.NEUN), C(Suit.HERZ, Rank.ZEHN),
             C(Suit.EICHEL, Rank.ACHT), None],
            [True, True, False, False],
        ),
        state=_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.LAUB),
            starter=0,
            completed_tricks=[
                CompletedTrick(starter=0, cards=(
                    C(Suit.LAUB, Rank.UNTER), C(Suit.LAUB, Rank.SECHS),
                )),
                CompletedTrick(starter=1, cards=(
                    C(Suit.HERZ, Rank.ASS), C(Suit.HERZ, Rank.SIEBEN),
                )),
                CompletedTrick(starter=0, cards=(
                    C(Suit.SCHELLE, Rank.ASS), C(Suit.SCHELLE, Rank.SECHS),
                )),
                CompletedTrick(starter=0, cards=(
                    C(Suit.EICHEL, Rank.ASS), C(Suit.EICHEL, Rank.SECHS),
                )),
            ],
            opp_visible=[
                C(Suit.SCHELLE, Rank.SIEBEN), C(Suit.SCHELLE, Rank.NEUN),
                C(Suit.LAUB, Rank.NEUN),
            ],
            opp_hand_count=4,
            opp_hidden_count=2,
            own_hidden_count=2,
            trick_idx=4,
            own_score=55,
            opp_score=48,
            round_idx=2,
        ),
    ))

    # --- 12: Endspiel, wenige Karten ---
    fixtures.append(Fixture(
        id="bfix_12_endspiel_letzte_karte",
        description=(
            "Trumpf-Eichel, Stich 17 (letzter Stich), nur eine Karte uebrig, "
            "kein Tisch mehr. own_hidden_count und opp_hidden_count beide 0."
        ),
        hand=[C(Suit.HERZ, Rank.SECHS)],
        own_table=_table([None] * 6, [False] * 6),
        state=_state(
            player_idx=0,
            variant=Variant.trumpf(Suit.EICHEL),
            trick_idx=17,
            opp_visible=[],
            opp_hand_count=1,
            opp_hidden_count=0,
            own_hidden_count=0,
            own_score=130,
            opp_score=22,
        ),
    ))

    return fixtures


def fixture_to_dict(fixture: Fixture) -> dict[str, Any]:
    vec = encode_state_bodensee(
        fixture.hand, fixture.own_table, fixture.state, fixture.i_am_announcer,
    )
    visible_table = [s.visible for s in fixture.own_table if s.visible is not None]
    mask = legal_action_mask_bodensee(fixture.hand, visible_table, fixture.state)
    return {
        "id": fixture.id,
        "description": fixture.description,
        "input": bodensee_state_to_dict(
            fixture.hand, fixture.own_table, fixture.state, fixture.i_am_announcer,
        ),
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
            "Konsistenz-Fixtures fuer den Bodensee-State-Encoder. Eine TypeScript-"
            "Implementierung muss alle state_vector- und legal_mask-Werte exakt "
            "reproduzieren (atol=1e-5)."
        ),
        "fixture_count": len(fixtures),
        "fixtures": [fixture_to_dict(f) for f in fixtures],
    }

    out_path = Path("spec/fixtures/bodensee_encoding_fixtures.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Geschrieben: {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  {len(fixtures)} Fixtures, Encoding-Version {ENCODING_VERSION}")


if __name__ == "__main__":
    main()
