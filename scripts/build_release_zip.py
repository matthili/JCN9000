"""Baut ein Release-ZIP mit allen Artefakten, die das Web-App-Projekt importiert.

Inhalt des ZIPs:
    jass-nn-<version>/
    ├── MANIFEST.json              Versionen, Hashes, Build-Info
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


def build_zip(
    version: str,
    model_path: Path | None,
    tfjs_dir: Path | None,
    spec_dir: Path,
    output: Path,
    skip_model: bool = False,
) -> dict:
    """Baut das Release-ZIP und gibt das Manifest als Dict zurück."""
    rules_json = spec_dir / "jass_rules.json"
    schema_json = spec_dir / "jass_rules.schema.json"
    encoding_md = spec_dir / "state_encoding.md"
    fixtures_json = spec_dir / "fixtures" / "encoding_fixtures.json"

    for required in (rules_json, schema_json, encoding_md, fixtures_json):
        if not required.exists():
            sys.exit(f"FEHLER: Pflicht-Spec-Datei fehlt: {required}")

    spec_version = _read_spec_version(rules_json)
    encoding_version = _read_encoding_version(fixtures_json)

    build_root = f"jass-nn-{version}"
    artifacts: list[tuple[Path, str]] = [
        (rules_json, f"{build_root}/jass_rules.json"),
        (schema_json, f"{build_root}/jass_rules.schema.json"),
        (encoding_md, f"{build_root}/state_encoding.md"),
        (fixtures_json, f"{build_root}/encoding_fixtures.json"),
    ]

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
    args = parser.parse_args()

    output = args.output or Path(f"dist/jass-nn-{args.version}.zip")
    build_zip(
        version=args.version,
        model_path=args.model,
        tfjs_dir=args.tfjs_dir,
        spec_dir=args.spec_dir,
        output=output,
        skip_model=args.skip_model,
    )


if __name__ == "__main__":
    main()
