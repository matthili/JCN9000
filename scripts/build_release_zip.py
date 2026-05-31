"""Baut ein Release-ZIP mit allen Artefakten, die das Web-App-Projekt importiert.

Inhalt des ZIPs:
    jass-nn-<version>/
    ├── MANIFEST.json              Versionen, Hashes, Build-Info
    ├── LICENSE                    AGPL-3.0 + §7(b)-Attribution (gilt auch fuer die Gewichte)
    ├── jass_rules.json            Vollständige Regel-Spezifikation
    ├── jass_rules.schema.json     JSON-Schema zur Validierung
    ├── state_encoding.md          Encoder-Doku (132-dim Featurevektor)
    ├── encoding_fixtures.json     (state → vector)-Test-Fixtures
    ├── tfjs/
    │   ├── model.json             TensorFlow.js-Modell-Beschreibung
    │   └── *.bin                  Gewichte
    └── keras/
        └── best.keras             Original-Keras-Format (für Re-Training)

Aufruf:
    python -m scripts.build_release_zip --version v0.1.0 \
        --model models/v1/best.keras \
        --tfjs-dir models/v1/tfjs \
        --output dist/jass-nn-v0.1.0.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_spec_version(rules_json: Path) -> str:
    return json.loads(rules_json.read_text(encoding="utf-8"))["spec_version"]


def _read_encoding_version(fixtures_json: Path) -> str:
    return json.loads(fixtures_json.read_text(encoding="utf-8"))["encoding_version"]


# Spielmodus -> (team_mode-Manifest-Wert, Encoder-Doku-Dateiname, Encoding-Version)
# Kreuz und Solo teilen sich den v3.0.0-Encoder; Bodensee hat einen eigenen.
GAME_MODE_CONFIG: dict[str, dict] = {
    "kreuz": {
        "team_mode": "team",
        "encoding_doc": "state_encoding.md",
        "encoding_version": None,   # None = aus Fixtures-Datei lesen
        "include_fixtures": True,
        "fixtures_file": "encoding_fixtures.json",
    },
    "solo": {
        "team_mode": "solo",
        "encoding_doc": "state_encoding.md",
        "encoding_version": None,
        "include_fixtures": True,
        "fixtures_file": "encoding_fixtures.json",
    },
    "bodensee": {
        "team_mode": "bodensee_2p",
        "encoding_doc": "bodensee_state_encoding.md",
        "encoding_version": "bodensee_1.0.0",
        "include_fixtures": True,
        "fixtures_file": "bodensee_encoding_fixtures.json",
    },
}


def build_zip(
    version: str,
    model_path: Path | None,
    tfjs_dir: Path | None,
    spec_dir: Path,
    output: Path,
    skip_model: bool = False,
    game_mode: str = "kreuz",
) -> dict:
    """Baut das Release-ZIP und gibt das Manifest als Dict zurück.

    Args:
        game_mode: "kreuz" | "solo" | "bodensee". Steuert, welche Encoder-Doku
            ins ZIP kommt, welche encoding_version + team_mode ins MANIFEST
            geschrieben werden. Kreuz/Solo teilen sich den v3.0.0-Encoder,
            Bodensee hat einen eigenen (bodensee_1.0.0).
    """
    if game_mode not in GAME_MODE_CONFIG:
        sys.exit(
            f"FEHLER: unbekannter game_mode '{game_mode}'. "
            f"Erlaubt: {sorted(GAME_MODE_CONFIG)}"
        )
    mode_cfg = GAME_MODE_CONFIG[game_mode]

    rules_json = spec_dir / "jass_rules.json"
    schema_json = spec_dir / "jass_rules.schema.json"
    encoding_md = spec_dir / mode_cfg["encoding_doc"]
    fixtures_json = spec_dir / "fixtures" / mode_cfg["fixtures_file"]

    required_files = [rules_json, schema_json, encoding_md]
    if mode_cfg["include_fixtures"]:
        required_files.append(fixtures_json)
    for required in required_files:
        if not required.exists():
            sys.exit(f"FEHLER: Pflicht-Spec-Datei fehlt: {required}")

    spec_version = _read_spec_version(rules_json)
    if mode_cfg["encoding_version"] is not None:
        encoding_version = mode_cfg["encoding_version"]
    else:
        encoding_version = _read_encoding_version(fixtures_json)

    build_root = f"jass-nn-{version}"
    artifacts: list[tuple[Path, str]] = [
        (rules_json, f"{build_root}/jass_rules.json"),
        (schema_json, f"{build_root}/jass_rules.schema.json"),
        (encoding_md, f"{build_root}/{mode_cfg['encoding_doc']}"),
    ]
    if mode_cfg["include_fixtures"]:
        artifacts.append((fixtures_json, f"{build_root}/{mode_cfg['fixtures_file']}"))

    # LICENSE mitliefern -- die AGPL-3.0-Copyleft- und §7(b)-Attributionspflicht
    # gelten ausdruecklich auch fuer die im ZIP enthaltenen Modellgewichte.
    license_file = spec_dir.parent / "LICENSE"
    if license_file.exists():
        artifacts.append((license_file, f"{build_root}/LICENSE"))
    else:
        print(f"WARNUNG: LICENSE nicht gefunden unter {license_file}", file=sys.stderr)

    file_hashes: dict[str, str] = {}

    # Modell-Dateien
    has_model = False
    if not skip_model:
        if model_path and model_path.exists():
            artifacts.append((model_path, f"{build_root}/keras/{model_path.name}"))
            has_model = True
        else:
            print(f"WARNUNG: Keras-Modell nicht gefunden unter {model_path}", file=sys.stderr)

        if tfjs_dir and tfjs_dir.is_dir() and any(tfjs_dir.iterdir()):
            for tfjs_file in sorted(tfjs_dir.iterdir()):
                if tfjs_file.is_file():
                    artifacts.append(
                        (tfjs_file, f"{build_root}/tfjs/{tfjs_file.name}")
                    )
            has_model = True
        elif tfjs_dir is not None:
            # Nur warnen wenn der Aufrufer explizit ein Verzeichnis angegeben hat, aber es leer ist
            print(f"WARNUNG: TF.js-Verzeichnis fehlt oder leer: {tfjs_dir}", file=sys.stderr)

    manifest = {
        "release_version": version,
        "spec_version": spec_version,
        "encoding_version": encoding_version,
        "game_mode": game_mode,
        "team_mode": mode_cfg["team_mode"],
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "has_model": has_model,
        "files": [],
    }

    # ZIP schreiben
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in artifacts:
            zf.write(src, arcname=arcname)
            h = _sha256(src)
            file_hashes[arcname] = h
            manifest["files"].append({
                "path": arcname,
                "size_bytes": src.stat().st_size,
                "sha256": h,
            })

        # Manifest als letzte Datei hinzufügen (inkl. der bisherigen Hashes)
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        zf.writestr(f"{build_root}/MANIFEST.json", manifest_bytes)

    print(f"Geschrieben: {output} ({output.stat().st_size:,} bytes)")
    print(f"  Spec-Version:     {spec_version}")
    print(f"  Encoding-Version: {encoding_version}")
    print(f"  Spielmodus:       {game_mode} (team_mode={mode_cfg['team_mode']})")
    print(f"  Release-Version:  {version}")
    print(f"  Modell enthalten: {'ja' if has_model else 'NEIN (nur Spec)'}")
    print(f"  Dateien:          {len(artifacts) + 1}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="z.B. v0.1.0")
    parser.add_argument("--model", type=Path, default=Path("models/v1/best.keras"))
    parser.add_argument("--tfjs-dir", type=Path, default=Path("models/v1/tfjs"))
    parser.add_argument("--spec-dir", type=Path, default=Path("spec"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="ZIP ohne Modell bauen (nur Spec-Files) — z.B. wenn das Modell separat angehängt wird",
    )
    parser.add_argument(
        "--game-mode",
        choices=["kreuz", "solo", "bodensee"],
        default="kreuz",
        help=(
            "Spielmodus des Modells. Steuert Encoder-Doku, encoding_version "
            "und team_mode im MANIFEST. Default kreuz."
        ),
    )
    args = parser.parse_args()

    output = args.output or Path(f"dist/jass-nn-{args.version}.zip")
    build_zip(
        version=args.version,
        model_path=args.model,
        tfjs_dir=args.tfjs_dir,
        spec_dir=args.spec_dir,
        output=output,
        skip_model=args.skip_model,
        game_mode=args.game_mode,
    )


if __name__ == "__main__":
    main()
