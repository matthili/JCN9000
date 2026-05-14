"""Konsistenz-Tests fuer die Spec-Artefakte unter spec/.

Diese Tests stellen sicher, dass:
  1. jass_rules.json strikt zum mitgelieferten JSON-Schema passt
  2. jass_rules.json zu den aktuellen Python-Konstanten passt (Generator wurde
     ausgeführt, kein Drift)
  3. encoding_fixtures.json zu den aktuellen Encoder-Vektoren passt

Wenn ein Test rot ist, wurde an Engine, Encoder oder Spec etwas geändert, ohne
den Generator neu laufen zu lassen. Lösung:
    python -m scripts.generate_jass_rules_json
    python -m scripts.generate_encoding_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, Variant
from scripts.generate_jass_rules_json import build_spec
from training.encoder import encode_state, legal_action_mask


SPEC_DIR = Path("spec")
RULES_JSON = SPEC_DIR / "jass_rules.json"
RULES_SCHEMA = SPEC_DIR / "jass_rules.schema.json"
FIXTURES_JSON = SPEC_DIR / "fixtures" / "encoding_fixtures.json"


# ---------- jass_rules.json: Schema + Drift ----------

def test_rules_json_existiert():
    assert RULES_JSON.exists(), f"{RULES_JSON} fehlt — Generator laufen lassen."


def test_rules_schema_existiert():
    assert RULES_SCHEMA.exists()


def test_rules_json_valide_gegen_schema():
    rules = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    schema = json.loads(RULES_SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(rules), key=lambda e: list(e.path))
    if errors:
        msg = "\n".join(f"  - {'/'.join(map(str, e.path))}: {e.message}" for e in errors)
        pytest.fail(f"jass_rules.json verletzt das Schema:\n{msg}")


def test_rules_json_synchron_mit_python_konstanten():
    """jass_rules.json muss exakt dem entsprechen, was build_spec() generiert."""
    on_disk = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    fresh = build_spec()
    if on_disk != fresh:
        pytest.fail(
            "jass_rules.json ist veraltet. Bitte laufen lassen:\n"
            "    python -m scripts.generate_jass_rules_json"
        )


# ---------- encoding_fixtures.json: Konsistenz mit dem Encoder ----------

def test_fixtures_existieren():
    assert FIXTURES_JSON.exists(), (
        f"{FIXTURES_JSON} fehlt — laufen lassen: "
        "python -m scripts.generate_encoding_fixtures"
    )


def _card_from_dict(d: dict) -> Card:
    return Card(Suit[d["suit"]], Rank[d["rank"]])


def _variant_from_dict(d: dict) -> Variant:
    if d["mode"] == "TRUMPF":
        return Variant.trumpf(Suit[d["trump_suit"]])
    if d["mode"] == "OBEN":
        return Variant.oben()
    if d["mode"] == "UNTEN":
        return Variant.unten()
    raise ValueError(f"Unbekannter Modus: {d['mode']}")


def _announcement_from_dict(d: dict) -> Announcement:
    return Announcement(
        variant=_variant_from_dict(d["variant"]),
        slalom=d["slalom"],
    )


def _state_from_dict(d: dict) -> tuple[list[Card], GameState]:
    hand = [_card_from_dict(c) for c in d["hand"]]
    state = GameState(
        player_idx=d["player_idx"],
        variant=_variant_from_dict(d["variant_effective"]),
        announcement=_announcement_from_dict(d["announcement"]),
        current_trick_cards=[_card_from_dict(c) for c in d["current_trick_cards"]],
        current_trick_starter=d["current_trick_starter"],
        teams=list(d["teams"]),
        completed_tricks=[
            [_card_from_dict(c) for c in trick] for trick in d["completed_tricks"]
        ],
        own_team_score=d["own_team_score"],
        opp_team_score=d["opp_team_score"],
        round_idx=d["round_idx"],
        trick_idx=d["trick_idx"],
        num_players=d.get("num_players", 4),
    )
    return hand, state


@pytest.fixture(scope="module")
def fixtures_data():
    return json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))


def test_fixture_anzahl_konsistent(fixtures_data):
    assert fixtures_data["fixture_count"] == len(fixtures_data["fixtures"])
    assert fixtures_data["fixture_count"] > 0


def test_fixtures_state_vektoren_reproduzierbar(fixtures_data):
    """Jeder gespeicherte Vektor muss exakt aus dem aktuellen Encoder erzeugt werden."""
    mismatches = []
    for fix in fixtures_data["fixtures"]:
        hand, state = _state_from_dict(fix["input"])
        actual = encode_state(hand, state)
        expected = np.array(fix["expected"]["state_vector"], dtype=np.float32)
        if not np.allclose(actual, expected, atol=1e-5):
            diff_indices = np.where(np.abs(actual - expected) > 1e-5)[0]
            mismatches.append(
                f"{fix['id']}: {len(diff_indices)} Indizes weichen ab "
                f"(zuerst bei {diff_indices[:5].tolist()})"
            )
    if mismatches:
        pytest.fail(
            "Encoder weicht von gespeicherten Fixtures ab:\n  "
            + "\n  ".join(mismatches)
            + "\n\nGenerator neu laufen lassen falls der Encoder absichtlich geändert wurde:\n"
              "    python -m scripts.generate_encoding_fixtures"
        )


def test_fixtures_legal_masks_reproduzierbar(fixtures_data):
    mismatches = []
    for fix in fixtures_data["fixtures"]:
        hand, state = _state_from_dict(fix["input"])
        actual = legal_action_mask(hand, state)
        expected = np.array(fix["expected"]["legal_mask"], dtype=np.uint8)
        if not np.array_equal(actual, expected):
            mismatches.append(fix["id"])
    if mismatches:
        pytest.fail(f"Legal-Masken weichen ab in: {mismatches}")


def test_fixtures_dimensionen(fixtures_data):
    for fix in fixtures_data["fixtures"]:
        assert fix["expected"]["state_vector_shape"] == [132], fix["id"]
        assert fix["expected"]["legal_mask_shape"] == [36], fix["id"]
        assert len(fix["expected"]["state_vector"]) == 132
        assert len(fix["expected"]["legal_mask"]) == 36


def test_fixture_aktion_ist_immer_legal_falls_aktion_definiert(fixtures_data):
    """Sanity: jede Aktion in der Maske ist konsistent (nur Werte 0 und 1)."""
    for fix in fixtures_data["fixtures"]:
        mask = fix["expected"]["legal_mask"]
        assert all(v in (0, 1) for v in mask), fix["id"]
        # Mindestens eine legale Aktion muss existieren
        assert sum(mask) >= 1, fix["id"]


def test_fixture_card_indices_konsistent_mit_formel(fixtures_data):
    """Jede Hand-Karte muss in der Maske oder in der own_hand-Sektion korrekt indiziert sein."""
    from training.encoder import SECTION_OFFSETS, card_index

    for fix in fixtures_data["fixtures"]:
        hand_cards = [Card(Suit[c["suit"]], Rank[c["rank"]]) for c in fix["input"]["hand"]]
        vec = fix["expected"]["state_vector"]
        own_off = SECTION_OFFSETS["own_hand"][0]
        hand_section = vec[own_off : own_off + 36]
        # genau so viele 1en wie Karten in der Hand
        assert sum(1 for v in hand_section if v == 1.0) == len(hand_cards), fix["id"]
        # jede Karte muss am erwarteten Index 1.0 sein
        for c in hand_cards:
            assert hand_section[card_index(c)] == 1.0, (
                f"{fix['id']}: Karte {c} nicht an Index {card_index(c)}"
            )
