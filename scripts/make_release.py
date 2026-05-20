"""Vollstaendiger lokaler Release-Befehl.

Macht ein versioniertes Release auf GitHub mit allen Artefakten (Spec + Modell).
Der Workflow ist:

    1. Prueft Git-Sauberkeit und Branch
    2. Laesst alle Tests laufen
    3. Generiert die Spec-Files neu und prueft Drift
    4. Konvertiert das Keras-Modell nach TF.js (falls noch nicht da)
    5. Baut das Release-ZIP mit MANIFEST.json
    6. Fragt den Benutzer um Bestaetigung
    7. Erzeugt den Git-Tag und pusht ihn
    8. Erstellt das GitHub-Release und haengt das ZIP als Asset an

Aufruf:
    python -m scripts.make_release --version v0.1.0 --model models/v1/best.keras

Sicherheits-Flags:
    --dry-run          alles bis Schritt 5, kein Tag, kein Push, kein Release
    --skip-tests       Tests ueberspringen (NICHT empfohlen)
    --yes              alle Bestaetigungs-Prompts mit Ja beantworten
    --gh-repo OWNER/REPO  Repository angeben (Default: matthili/jass-neuronales-netz)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from scripts.build_release_zip import build_zip


DEFAULT_REPO = "matthili/jass-neuronales-netz"
VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$")


def fail(msg: str, code: int = 1):
    print(f"\n[FEHLER] {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str):
    print(f"[INFO] {msg}")


def ok(msg: str):
    print(f"[OK]   {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


def confirm(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"{prompt} [y/N]: y (auto)")
        return True
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes", "j", "ja")


def run(cmd: Sequence[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


# ---------- Vorpruefungen ----------

def check_version_format(version: str):
    if not VERSION_PATTERN.match(version):
        fail(
            f"Version '{version}' ist ungueltig. "
            f"Erwartet: vMAJOR.MINOR.PATCH (z.B. v0.1.0) oder mit Suffix (v0.1.0-beta1)."
        )


def check_git_clean():
    info("Pruefe Git-Status...")
    result = run(["git", "status", "--porcelain"], capture=True)
    if result.stdout.strip():
        warn("Working tree ist nicht clean:")
        print(result.stdout)
        fail("Bitte erst alles committen oder stashen, dann nochmal.")
    ok("Working tree clean.")


def check_branch(expected: str = "master"):
    info(f"Pruefe aktuellen Branch (erwartet: {expected})...")
    result = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True)
    current = result.stdout.strip()
    if current != expected:
        warn(f"Du bist auf Branch '{current}', nicht '{expected}'.")
        fail(f"Releases werden ueblicherweise von '{expected}' gemacht. Wechsle den Branch oder benutze --branch.")
    ok(f"Branch ist {expected}.")


def check_tag_not_taken(version: str):
    info(f"Pruefe ob Tag {version} schon existiert...")
    result = run(["git", "tag", "-l", version], capture=True)
    if result.stdout.strip() == version:
        fail(f"Tag {version} existiert bereits. Bitte hoehere Versionsnummer waehlen.")
    # Auch remote pruefen
    try:
        result = run(["git", "ls-remote", "--tags", "origin", version], capture=True)
        if version in result.stdout:
            fail(f"Tag {version} existiert bereits auf dem Remote.")
    except subprocess.CalledProcessError:
        warn("Konnte Remote-Tags nicht pruefen (kein Internet?). Fahre fort.")
    ok(f"Tag {version} ist frei.")


def check_gh_cli():
    info("Pruefe ob GitHub-CLI verfuegbar ist...")
    if shutil.which("gh") is None:
        fail(
            "GitHub-CLI ('gh') nicht gefunden. "
            "Bitte installieren: https://cli.github.com/"
        )
    # Auth pruefen
    result = run(["gh", "auth", "status"], check=False, capture=True)
    if result.returncode != 0:
        fail("GitHub-CLI ist nicht angemeldet. Bitte zuerst: gh auth login")
    ok("gh CLI ist verfuegbar und angemeldet.")


# ---------- Tests + Spec ----------

def run_tests(skip: bool):
    if skip:
        warn("Tests werden uebersprungen (--skip-tests).")
        return
    info("Lasse Tests laufen...")
    run(["python", "-m", "pytest", "-q", "--tb=short"])
    ok("Alle Tests gruen.")


def regenerate_spec_and_check_drift():
    info("Regeneriere Spec-Files...")
    run(["python", "-m", "scripts.generate_jass_rules_json"])
    run(["python", "-m", "scripts.generate_encoding_fixtures"])
    info("Pruefe auf Spec-Drift...")
    result = run(["git", "diff", "--exit-code", "spec/"], check=False, capture=True)
    if result.returncode != 0:
        warn("Spec-Drift erkannt:")
        print(result.stdout)
        fail(
            "spec/-Dateien sind nicht synchron mit dem Code. "
            "Bitte commit die neu generierten Spec-Dateien und versuche es nochmal."
        )
    ok("Spec ist synchron mit dem Code.")


# ---------- TF.js-Export ----------

def export_tfjs_if_missing(model_path: Path, tfjs_dir: Path) -> bool:
    """Versucht TF.js-Export. Gibt True bei Erfolg, False bei Fehler (Skript laeuft weiter)."""
    if tfjs_dir.is_dir() and any(tfjs_dir.iterdir()):
        info(f"TF.js-Export existiert schon unter {tfjs_dir}, wird wiederverwendet.")
        return True
    info(f"Versuche Konvertierung {model_path} -> TF.js...")
    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        warn(
            "tensorflowjs_converter nicht gefunden. ZIP wird OHNE TF.js-Modell gebaut.\n"
            "  Hinweis: Auf Windows ist der Konverter wegen Linux-only-Abhaengigkeiten\n"
            "  schwierig zu installieren. Empfohlen: Konvertierung in WSL2 oder Docker:\n"
            "    docker run --rm -v $(pwd):/work tensorflow/tensorflow:latest \\\n"
            "      bash -c 'pip install tensorflowjs && tensorflowjs_converter \\\n"
            "      --input_format=keras /work/models/v1/best.keras /work/models/v1/tfjs'"
        )
        return False
    tfjs_dir.mkdir(parents=True, exist_ok=True)
    result = run([
        converter,
        "--input_format=keras",
        str(model_path),
        str(tfjs_dir),
    ], check=False)
    if result.returncode != 0:
        warn(
            f"tensorflowjs_converter ist abgebrochen (Exit-Code {result.returncode}).\n"
            f"  ZIP wird OHNE TF.js-Modell gebaut. Bitte TF.js manuell in WSL2/Docker erzeugen."
        )
        # leeres tfjs-Verzeichnis aufraeumen
        if tfjs_dir.exists() and not any(tfjs_dir.iterdir()):
            tfjs_dir.rmdir()
        return False
    ok(f"TF.js-Export geschrieben nach {tfjs_dir}.")
    return True


# ---------- Tag + GitHub Release ----------

def create_and_push_tag(version: str, repo: str):
    info(f"Erstelle Tag {version}...")
    run(["git", "tag", "-a", version, "-m", f"Release {version}"])
    info(f"Pushe Tag nach origin...")
    run(["git", "push", "origin", version])
    ok(f"Tag {version} ist auf GitHub.")


def create_github_release(version: str, zip_path: Path, repo: str, notes_path: Path):
    info(f"Erstelle GitHub-Release {version} im Repo {repo}...")
    run([
        "gh", "release", "create", version,
        str(zip_path),
        "--repo", repo,
        "--title", version,
        "--notes-file", str(notes_path),
    ])
    ok(f"Release {version} veroeffentlicht: https://github.com/{repo}/releases/tag/{version}")


def write_release_notes(version: str, manifest: dict, model_meta_path: Path) -> Path:
    has_model = manifest.get("has_model", False)
    spec_version = manifest.get("spec_version", "?")
    encoding_version = manifest.get("encoding_version", "?")

    notes = f"""## Release {version}

Dies ist eine versionierte Veroeffentlichung der Vorarlberger Kreuz-Jass NN-Pipeline.

### Was enthalten ist (im ZIP-Anhang)

- `jass_rules.json` (Spec-Version `{spec_version}`) - vollstaendige Regel-Spezifikation
- `jass_rules.schema.json` - JSON-Schema zur Validierung
- `state_encoding.md` (Encoding-Version `{encoding_version}`) - Feature-Vektor-Layout
- `encoding_fixtures.json` - Test-Fixtures fuer den TypeScript-Port
"""
    if has_model:
        notes += """- `keras/best.keras` - Trainiertes Modell im Keras-Format
- `tfjs/` - TensorFlow.js-Modell fuer Web-Inferenz
- `MANIFEST.json` - Versionen + SHA256-Hashes aller Dateien

### So nutzt die Web-App das Release

```bash
# Asset herunterladen
gh release download """ + version + """ --repo matthili/jass-neuronales-netz --pattern "jass-nn-*.zip"
unzip jass-nn-""" + version + """.zip
```
"""
    notes += "\n### Geprueft\n\n- Alle Tests gruen\n- Spec ist synchron mit dem Code\n"

    notes_path = Path(f"dist/release-notes-{version}.md")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(notes, encoding="utf-8")
    return notes_path


# ---------- Hauptfunktion ----------

def main():
    parser = argparse.ArgumentParser(description="Lokaler Release-Befehl")
    parser.add_argument("--version", required=True, help="Release-Version, z.B. v0.1.0")
    parser.add_argument(
        "--model", type=Path, default=Path("models/v1/best.keras"),
        help="Pfad zum Keras-Modell (Default: models/v1/best.keras)",
    )
    parser.add_argument(
        "--tfjs-dir", type=Path, default=Path("models/v1/tfjs"),
        help="Pfad zum TF.js-Verzeichnis (wird bei Bedarf erzeugt)",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"GitHub-Repo (Default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--branch", default="master",
        help="Erwarteter Branch (Default: master)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="ZIP bauen, aber kein Tag erstellen / kein Push / kein Release",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-model", action="store_true",
                        help="ZIP ohne Modell bauen (nur Spec)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Erlaubt uncommitted Aenderungen (nur sinnvoll mit --dry-run)")
    parser.add_argument(
        "--game-mode",
        choices=["kreuz", "solo", "bodensee"],
        default="kreuz",
        help=(
            "Spielmodus des Modells. kreuz/solo nutzen den v3.0.0-Encoder, "
            "bodensee den bodensee_1.0.0-Encoder. Steuert, welche Encoder-Doku "
            "ins ZIP kommt und welche encoding_version + team_mode ins MANIFEST. "
            "Default kreuz."
        ),
    )
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Alle Bestaetigungs-Fragen mit ja beantworten")
    args = parser.parse_args()

    version = args.version

    print("=" * 70)
    print(f"  Release-Pipeline fuer {version}")
    print(f"  Repo: {args.repo}")
    print(f"  Dry-Run: {args.dry_run}")
    print("=" * 70)
    print()

    # --- Schritt 1: Vorpruefungen ---
    check_version_format(version)
    if args.allow_dirty:
        warn("--allow-dirty: ueberspringe Git-Clean-Check.")
    else:
        check_git_clean()
    check_branch(args.branch)
    if not args.dry_run:
        check_tag_not_taken(version)
        check_gh_cli()

    # --- Schritt 2: Tests ---
    run_tests(args.skip_tests)

    # --- Schritt 3: Spec-Drift ---
    regenerate_spec_and_check_drift()

    # --- Schritt 4: Modell + TF.js-Export ---
    tfjs_available = False
    if not args.skip_model:
        if not args.model.exists():
            fail(f"Modell nicht gefunden: {args.model}. Erst trainieren oder --skip-model.")
        tfjs_available = export_tfjs_if_missing(args.model, args.tfjs_dir)

    # --- Schritt 5: ZIP bauen ---
    zip_path = Path(f"dist/jass-nn-{version}.zip")
    manifest = build_zip(
        version=version,
        model_path=args.model if not args.skip_model else None,
        tfjs_dir=args.tfjs_dir if (not args.skip_model and tfjs_available) else None,
        spec_dir=Path("spec"),
        output=zip_path,
        skip_model=args.skip_model,
        game_mode=args.game_mode,
    )

    # --- Schritt 6: Release-Notes ---
    notes_path = write_release_notes(version, manifest, args.model)
    info(f"Release-Notes geschrieben: {notes_path}")

    if args.dry_run:
        ok("Dry-Run fertig. ZIP liegt unter " + str(zip_path))
        return

    # --- Schritt 7: Bestaetigung ---
    print()
    print("=" * 70)
    print("  Bereit zum Veroeffentlichen:")
    print(f"    - Tag {version} wird erstellt und nach origin gepusht")
    print(f"    - GitHub-Release wird auf {args.repo} angelegt")
    print(f"    - ZIP wird angehaengt: {zip_path}")
    print("=" * 70)
    if not confirm("Wirklich veroeffentlichen?", args.yes):
        warn("Abgebrochen. Lokales ZIP bleibt unter " + str(zip_path) + ".")
        return

    # --- Schritt 8: Tag + Release ---
    create_and_push_tag(version, args.repo)
    create_github_release(version, zip_path, args.repo, notes_path)

    print()
    print("=" * 70)
    print(f"  Fertig: https://github.com/{args.repo}/releases/tag/{version}")
    print("=" * 70)


if __name__ == "__main__":
    main()
