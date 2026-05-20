"""Tests fuer die Bodensee-Encoding-Fixtures.

Stellt sicher, dass spec/fixtures/bodensee_encoding_fixtures.json synchron mit
dem aktuellen Bodensee-Encoder ist. Wenn der Encoder geaendert wird, ohne die
Fixtures neu zu generieren, schlaegt dieser Test fehl -- analog zum
Spec-Drift-Check fuer die Kreuz/Solo-Fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_bodensee_encoding_fixtures import build_fixtures, fixture_to_dict
from training.bodensee_encoder import ENCODING_VERSION, INPUT_DIM, ACTION_DIM


FIXTURES_PATH = Path("spec/fixtures/bodensee_encoding_fixtures.json")


def _load_fixtures_file() -> dict:
    if not FIXTURES_PATH.exists():
        pytest.skip(f"{FIXTURES_PATH} existiert nicht -- erst Generator laufen lassen.")
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def test_fixtures_datei_existiert_und_hat_struktur():
    data = _load_fixtures_file()
    assert data["encoding_version"] == ENCODING_VERSION
    assert data["fixture_count"] == len(data["fixtures"])
    assert data["fixture_count"] >= 10, "Erwarte mindestens 10 Fixtures"


def test_fixtures_vektoren_haben_richtige_dimension():
    data = _load_fixtures_file()
    for fix in data["fixtures"]:
        sv = fix["expected"]["state_vector"]
        lm = fix["expected"]["legal_mask"]
        assert len(sv) == INPUT_DIM, (
            f"Fixture {fix['id']}: state_vector hat {len(sv)} statt {INPUT_DIM}"
        )
        assert len(lm) == ACTION_DIM, (
            f"Fixture {fix['id']}: legal_mask hat {len(lm)} statt {ACTION_DIM}"
        )


def test_fixtures_alle_werte_in_gueltigem_bereich():
    """state_vector-Werte muessen in [0, 1] liegen, legal_mask in {0, 1}."""
    data = _load_fixtures_file()
    for fix in data["fixtures"]:
        for v in fix["expected"]["state_vector"]:
            assert -0.001 <= v <= 1.001, f"Fixture {fix['id']}: Wert {v} ausserhalb [0,1]"
        for m in fix["expected"]["legal_mask"]:
            assert m in (0, 1), f"Fixture {fix['id']}: Maske-Wert {m} nicht 0/1"


def test_fixtures_kein_drift_gegen_aktuellen_encoder():
    """Regeneriert alle Fixtures und vergleicht mit der gespeicherten Datei.

    Schlaegt fehl, wenn der Encoder geaendert wurde, ohne die Fixtures-Datei
    neu zu generieren (python -m scripts.generate_bodensee_encoding_fixtures).
    """
    data = _load_fixtures_file()
    stored_by_id = {f["id"]: f for f in data["fixtures"]}

    regenerated = [fixture_to_dict(f) for f in build_fixtures()]

    assert len(regenerated) == len(stored_by_id), (
        "Anzahl Fixtures weicht ab -- Datei neu generieren."
    )

    for regen in regenerated:
        fid = regen["id"]
        assert fid in stored_by_id, f"Fixture {fid} fehlt in der gespeicherten Datei."
        stored = stored_by_id[fid]
        assert regen["expected"]["state_vector"] == stored["expected"]["state_vector"], (
            f"Fixture {fid}: state_vector driftet. "
            f"Bitte 'python -m scripts.generate_bodensee_encoding_fixtures' ausfuehren."
        )
        assert regen["expected"]["legal_mask"] == stored["expected"]["legal_mask"], (
            f"Fixture {fid}: legal_mask driftet. Bitte Fixtures neu generieren."
        )


def test_fixtures_legal_mask_mindestens_eine_legale_karte():
    """Jede Fixture muss mindestens eine legale Karte haben."""
    data = _load_fixtures_file()
    for fix in data["fixtures"]:
        assert sum(fix["expected"]["legal_mask"]) >= 1, (
            f"Fixture {fix['id']}: keine legale Karte -- unmoeglicher Zustand."
        )
