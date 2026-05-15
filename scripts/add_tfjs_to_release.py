"""Laedt das Asset eines bestehenden Releases herunter, konvertiert das enthaltene
Keras-Modell nach TF.js, fuegt das tfjs/-Verzeichnis ins ZIP ein und ersetzt
das Asset im Release.

Wird vom GitHub-Actions-Workflow `add_tfjs.yml` automatisch nach jedem
veroeffentlichten Release ausgefuehrt -- damit wird die TF.js-Konvertierung
auf einem Linux-Runner gemacht, wo `tensorflowjs_converter` einfach laeuft
(unter Windows/WSL2 mit aktuellem Python ist das eine Dependency-Hoelle).

Lokal ausfuehrbar zum Testen:
    python -m scripts.add_tfjs_to_release --tag v0.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path


DEFAULT_REPO = "matthili/jass-neuronales-netz"


# Defensive Stubs fuer tensorflow_decision_forests / yggdrasil_decision_forests.
# tensorflowjs >=4.22 importiert TFDF in tf_saved_model_conversion_v2.py blind.
# TFDF hat aber einen Protobuf-Versionskonflikt mit TF 2.x (gencode 6.31 vs
# runtime 5.29) und ist fuer ein MLP-Konvertierung sowieso unnoetig. Indem
# wir leere Module in sys.modules platzieren, fragt der `import ...`-Befehl
# nicht mehr die echte (kaputte) Library, sondern unseren leeren Stub.
#
# WICHTIG: dieses Stubbing MUSS vor jedem `import tensorflowjs` stehen,
# darum auf Modul-Ebene und nicht in main().
for _stub_name in ("tensorflow_decision_forests", "yggdrasil_decision_forests"):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = types.ModuleType(_stub_name)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="z.B. v0.2.0")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    if shutil.which("gh") is None:
        sys.exit("FEHLER: gh-CLI nicht gefunden.")
    # tensorflowjs als Python-Modul pruefen (nicht als CLI-Tool). Wir benutzen
    # die Python-API, damit der MaskBias-Custom-Layer korrekt registriert wird.
    try:
        import tensorflowjs  # noqa: F401
    except ImportError:
        sys.exit("FEHLER: tensorflowjs nicht installiert. Bitte 'pip install \"tensorflowjs>=4.22\"'.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        download_dir = tmpdir_path / "download"
        download_dir.mkdir()

        # 1. Asset downloaden
        print(f"[1/6] Lade Asset fuer Release {args.tag}...")
        _run([
            "gh", "release", "download", args.tag,
            "--pattern", "jass-nn-*.zip",
            "--dir", str(download_dir),
            "--repo", args.repo,
        ])

        zips = list(download_dir.glob("jass-nn-*.zip"))
        if len(zips) != 1:
            sys.exit(f"FEHLER: erwartete genau ein ZIP, fand {len(zips)}: {zips}")
        zip_path = zips[0]
        print(f"  Asset: {zip_path.name} ({zip_path.stat().st_size:,} bytes)")

        # 2. ZIP entpacken
        print("[2/6] Entpacke ZIP...")
        extract_dir = tmpdir_path / "extract"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        sub_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(sub_dirs) != 1:
            sys.exit(f"FEHLER: erwartete genau ein Unterverzeichnis, fand {sub_dirs}")
        release_dir = sub_dirs[0]
        print(f"  Release-Verzeichnis: {release_dir.name}")

        # 3. Pruefen, ob TF.js bereits vorhanden
        tfjs_dir = release_dir / "tfjs"
        if tfjs_dir.exists() and any(tfjs_dir.iterdir()):
            print(f"[3/6] TF.js-Verzeichnis ist bereits im ZIP enthalten. Nichts zu tun.")
            return

        # 4. Keras-Modell suchen
        print("[3/6] Suche Keras-Modell im Release...")
        keras_files = list(release_dir.rglob("*.keras"))
        if not keras_files:
            print("  Kein Keras-Modell im Release-ZIP -- nichts zu konvertieren.")
            return
        keras_path = keras_files[0]
        print(f"  Keras-Modell: {keras_path.relative_to(release_dir)}")

        # 5. Konvertieren -- Python-API statt CLI-Subprocess.
        #
        # Hintergrund: unser Modell nutzt eine selbstgebaute Schicht (MaskBias),
        # registriert per @keras.saving.register_keras_serializable in
        # training/model.py. Die Registrierung wirkt nur, wenn das Modul
        # importiert wird. Wenn wir tensorflowjs_converter als externes CLI
        # aufrufen, kennt es unsere Custom-Layer nicht und das Modell-Laden
        # scheitert. Mit der Python-API geschieht das alles im gleichen
        # Prozess, in dem wir training.model importiert haben.
        print("[4/6] Konvertiere nach TF.js (Python-API mit Custom-Layer)...")
        tfjs_dir.mkdir(parents=True, exist_ok=True)
        import tensorflowjs as tfjs
        from tensorflow import keras
        from training.model import MaskBias  # noqa: F401 -- triggert Registrierung
        model = keras.models.load_model(str(keras_path))
        tfjs.converters.save_keras_model(model, str(tfjs_dir))
        tfjs_files = list(tfjs_dir.iterdir())
        print(f"  TF.js-Dateien: {[f.name for f in tfjs_files]}")

        # 6. MANIFEST aktualisieren (falls vorhanden)
        print("[5/6] Aktualisiere MANIFEST.json...")
        manifest_path = release_dir / "MANIFEST.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for tfjs_file in sorted(tfjs_files):
                if tfjs_file.is_file():
                    arcname = f"{release_dir.name}/tfjs/{tfjs_file.name}"
                    manifest.setdefault("files", []).append({
                        "path": arcname,
                        "size_bytes": tfjs_file.stat().st_size,
                        "sha256": _sha256(tfjs_file),
                    })
            manifest["has_model"] = True
            manifest["has_tfjs"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  {len(tfjs_files)} TF.js-Eintraege ins MANIFEST aufgenommen.")
        else:
            print("  Kein MANIFEST.json im Release -- ueberspringe.")

        # 7. Neues ZIP bauen (gleicher Name)
        print("[6/6] Neues ZIP bauen und im Release ersetzen...")
        new_zip_dir = tmpdir_path / "new"
        new_zip_dir.mkdir()
        new_zip = new_zip_dir / zip_path.name
        with zipfile.ZipFile(new_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in release_dir.rglob("*"):
                if item.is_file():
                    arcname = f"{release_dir.name}/{item.relative_to(release_dir)}"
                    zf.write(item, arcname=arcname)
        print(f"  Neues ZIP: {new_zip.stat().st_size:,} bytes (vorher: {zip_path.stat().st_size:,})")

        # Upload mit --clobber ersetzt das alte Asset
        _run([
            "gh", "release", "upload", args.tag,
            str(new_zip),
            "--repo", args.repo,
            "--clobber",
        ])

    print(f"\nFertig: Release {args.tag} enthaelt jetzt das TF.js-Modell.")
    print(f"URL: https://github.com/{args.repo}/releases/tag/{args.tag}")


if __name__ == "__main__":
    main()
