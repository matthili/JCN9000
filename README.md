# Jass Neuronales Netz

**Vorarlberger Kreuz-Jass** als regelgetreue Python-Engine plus neuronales Netz als KI-Gegner. Erzeugt versionierte Artefakte (Modell + Regel-Spezifikation), die in einer separaten Web-Anwendung als Multiplayer-Plattform eingebunden werden können.

[![Tests](https://img.shields.io/badge/tests-103%20passing-brightgreen)](#verifikation)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](#voraussetzungen)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Inhaltsverzeichnis

- [Was steckt drin?](#was-steckt-drin)
- [Voraussetzungen](#voraussetzungen)
- [Setup](#setup)
- [Schnellstart](#schnellstart)
- [Verifikation](#verifikation)
- [Architektur](#architektur)
- [Schnittstelle zur Web-Anwendung](#schnittstelle-zur-web-anwendung)
- [Roadmap](#roadmap)
- [Lizenz](#lizenz)

## Was steckt drin?

| Komponente | Bedeutung |
|---|---|
| **Spielengine** ([`jass_engine/`](jass_engine/)) | 36 Karten, alle Varianten (Trumpf / Bock / Geiss / Slalom), Weisen, Stöcke, Matsch, Schieben — 100 % regelgetreu |
| **Heuristik-Bot** ([`players/heuristic_player.py`](players/heuristic_player.py)) | Stechen, Schmieren, Sparen, Variant-Scoring; ~99 % Sieg gegen Random |
| **Trainings-Pipeline** ([`training/`](training/)) | State-Encoder, Datengenerator (350+ Partien/s mit 20 Workern), Keras-Modell, Trainings-Loop |
| **NN-Player** ([`players/nn_player.py`](players/nn_player.py)) | Lädt ein trainiertes Modell und spielt damit |
| **Visualisierung** ([`visualization/`](visualization/)) | Rich-basierte Terminal-Demo und Streamlit-App zur interaktiven Regel-Verifikation |
| **Regel-Spezifikation** ([`spec/`](spec/)) | Versionierte JSON-Spec + Encoder-Doku + Test-Fixtures als Schnittstelle für die Web-Anwendung |
| **Test-Suite** ([`tests/`](tests/)) | 103 Tests: Regeln, Weisen, Heuristik, Encoder, Spec-Konsistenz |

## Voraussetzungen

- **Python 3.11+** (getestet auf 3.13)
- Optional für Training: **NVIDIA-GPU mit CUDA** (auf Windows nur über WSL2 oder DirectML; CPU-Training funktioniert problemlos)
- ~10 GB freier Speicher für 50k generierte Trainingsdaten

## Setup

```powershell
# 1. Repository klonen
git clone https://github.com/<your-user>/jass-neuronales-netz.git
cd jass-neuronales-netz

# 2. Virtuelles Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Installation
pip install -e ".[dev]"           # nur Engine + Tools
pip install -e ".[dev,training]"  # plus TensorFlow für Training/Inferenz
```

## Schnellstart

### Terminal-Demo: eine komplette Partie zwischen Random-Bots

```powershell
python -m visualization.terminal
```

### Spielstärke-Vergleich

```powershell
python -m evaluation.compare_players --games 500
```

Erwartete Ausgabe: Heuristik ≈ 99 % vs. Random; ~50/50 Heuristik vs. Heuristik (Sitz-Symmetrie).

### Streamlit-Regelprüfer (interaktiv)

```powershell
streamlit run visualization/streamlit_app.py
```

Vier Seiten:
- **Regelwerk**: Karten, Werte, Reihenfolge je Variante
- **Regelprüfer**: Hand + Stich-Karten eingeben → erlaubte/verbotene Karten mit Begründung
- **Weis-Prüfer**: Sequenzen, Vierlinge, Stöcke aus einer Hand erkennen
- **Demo-Partie**: Random-vs-Random-Spiel Schritt für Schritt

### Trainingsdaten generieren

```powershell
# 50 000 Partien mit 20 parallelen Workern (ca. 2:20 Minuten auf 14700K)
python -m training.generate_data --games 50000 --workers 20 --output data/heuristic_50k
```

### Modell trainieren

```powershell
python -m training.train --data data/heuristic_50k --output models/v1 --epochs 20
```

Auf einer CPU dauern 20 Epochen ca. 30–40 Minuten; auf einer GPU einen Bruchteil davon.

## Verifikation

Die Engine ist über Tests abgesichert. Bei jeder Code-Änderung:

```powershell
pytest
```

Aktuell **103 Tests grün**, verteilt auf:

- `test_card.py` — Karten, Deck, Weli-Identifikation
- `test_rules.py` — Werte, Reihenfolgen, legale Züge je Variante (Trumpf, Bock, Geiss)
- `test_weis.py` — Sequenzen, Vierlinge, Stöcke, Team-Vergleich
- `test_heuristic_player.py` — Schmier-/Spar-/Stech-Verhalten, Ansage-Logik, Slalom
- `test_kreuz_jass.py` — End-to-End-Spielablauf, Konsistenz, Matsch
- `test_encoder.py` — 132-dim Featurevektor, legale Aktionsmaske
- `test_spec_consistency.py` — Spec-Drift gegen Python-Konstanten, Fixture-Reproduzierbarkeit

## Architektur

```
                ┌────────────────────────────────────┐
                │  Python-Engine (jass_engine/)      │
                │  - Karten, Deck                    │
                │  - Variant (Trumpf/Bock/...)       │
                │  - Regeln (legale Züge, Stiche)    │
                │  - Weisen, Stöcke, Matsch          │
                └─────────────┬──────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
┌──────────────┐   ┌────────────────────┐   ┌──────────────┐
│ RandomPlayer │   │ HeuristicPlayer    │   │ NNPlayer     │
│              │   │ - Score je Variant │   │ - lädt .keras│
│              │   │ - Stechen/Schmieren│   │              │
└──────┬───────┘   └─────────┬──────────┘   └──────┬───────┘
       │                     │                     │
       └──────────┬──────────┘                     │
                  │                                │
                  ▼                                │
    ┌──────────────────────────────┐               │
    │ Datengenerator               │               │
    │ training/generate_data.py    │               │
    │ → .npz-Shards (state/mask/y) │               │
    └──────────┬───────────────────┘               │
               │                                   │
               ▼                                   │
    ┌──────────────────────────────┐               │
    │ Keras-Training               │               │
    │ training/train.py            │               │
    │ → models/v*/best.keras       │───────────────┘
    └──────────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────────┐
    │ TF.js-Export                 │
    │ tensorflowjs_converter       │
    │ → GitHub Release Asset       │
    └──────────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────────┐
    │  Separates Web-App-Projekt   │
    │  (NestJS + React/Angular)    │
    │  importiert Modell + Spec    │
    └──────────────────────────────┘
```

## Schnittstelle zur Web-Anwendung

Die Web-Anwendung ist ein **separates Projekt**. Dieses Repository liefert ihr drei Artefakte:

| Artefakt | Pfad | Zweck |
|---|---|---|
| **Regel-Spezifikation** | [`spec/jass_rules.json`](spec/jass_rules.json) | Alle Spielregeln deklarativ in JSON |
| **JSON-Schema** | [`spec/jass_rules.schema.json`](spec/jass_rules.schema.json) | Validiert die Spec-Datei beim Lesen |
| **Encoder-Doku** | [`spec/state_encoding.md`](spec/state_encoding.md) | 132-dim Featurevektor-Layout für das NN |
| **Test-Fixtures** | [`spec/fixtures/encoding_fixtures.json`](spec/fixtures/encoding_fixtures.json) | (State → Vektor)-Paare zum Verifizieren des TS-Ports |
| **Trainiertes Modell** | (nicht im Repo) | TF.js-Modell, als GitHub-Release-Asset |

**Versionierung**: Jede Release-Version dieses Repos versioniert alle vier Artefakte gemeinsam. Die Web-App pinnt eine konkrete Version im Build-Prozess und prüft beim Modell-Laden, dass die Encoding-Version kompatibel ist.

**Konsistenz-Garantie**: Die Spec-Dateien werden aus den Python-Konstanten **generiert**:

```powershell
python -m scripts.generate_jass_rules_json
python -m scripts.generate_encoding_fixtures
```

Die CI-Pipeline (siehe [`.github/workflows/test.yml`](.github/workflows/test.yml)) verifiziert, dass die committeten Spec-Dateien synchron mit dem Code sind — Drift ist strukturell ausgeschlossen.

## Ein neues Release veröffentlichen

Ein "Release" ist im GitHub-Sinne eine **versionierte, eingefrorene Veröffentlichung** dieses Projekts, an die eine herunterladbare ZIP-Datei angehängt wird. Diese ZIP-Datei (im englischen Sprachgebrauch "Asset" = "Anhang") enthält alles, was die separate Web-Anwendung braucht: Regel-Spezifikation, Encoder-Doku, Test-Fixtures und das trainierte Modell. Über eine feste URL kann die Web-App genau diese ZIP-Datei beim Build herunterladen.

### Voraussetzungen einmalig

- **GitHub-CLI** installieren: <https://cli.github.com/>
- Einmal anmelden: `gh auth login`
- TensorFlow.js-Konverter installieren: `pip install tensorflowjs`

### Release in einem Befehl

Lokal ein Modell trainieren (mit voller Hardware-Power, beliebig groß), dann:

```powershell
python -m scripts.make_release --version v0.1.0
```

Das Skript läuft folgende Schritte ab und fragt vor dem Pushen ausdrücklich nach Bestätigung:

1. **Vorprüfung**: working tree sauber, richtiger Branch (`master`), Tag noch frei, `gh` angemeldet
2. **Tests laufen lassen** (gegen das Repo)
3. **Spec-Drift prüfen**: `spec/`-Dateien werden neu generiert und müssen unverändert sein
4. **TF.js-Export**: `models/v1/best.keras` → `models/v1/tfjs/` (falls noch nicht geschehen)
5. **ZIP bauen** mit allen Artefakten + MANIFEST.json (SHA256-Hashes)
6. **Bestätigung abwarten** — bis hierhin wurde nichts an GitHub gesendet
7. **Git-Tag erstellen und pushen**
8. **GitHub-Release erstellen** und ZIP als Anhang anhängen

### Trockenlauf (zum Testen)

```powershell
python -m scripts.make_release --version v0.1.0 --dry-run
```

Macht Schritte 1–5, baut das ZIP unter `dist/`, **erstellt aber keinen Tag und kein Release**. Gut zum Anschauen, ob das ZIP so aussieht, wie es soll.

### Nachträglich Assets anhängen

Wenn du nach einem Release zum Beispiel ein noch besser trainiertes Modell als zusätzlichen Anhang zur Verfügung stellen willst:

```powershell
# Lokal ein zweites ZIP bauen, z.B. mit anderem Namen
python -m scripts.build_release_zip --version v0.1.0-full `
    --model models/v2/best.keras --tfjs-dir models/v2/tfjs `
    --output dist/jass-nn-v0.1.0-full.zip

# An das bestehende Release anhängen
gh release upload v0.1.0 dist/jass-nn-v0.1.0-full.zip --repo matthili/jass-neuronales-netz
```

### Wie die Web-App das Release nutzt

Im Build-Schritt der Web-Anwendung (z.B. in einer GitHub-Actions-Pipeline):

```bash
# ZIP des gepinnten Releases herunterladen
gh release download v0.1.0 \
    --repo matthili/jass-neuronales-netz \
    --pattern "jass-nn-*.zip"

# Auspacken in einen festen Pfad
unzip jass-nn-v0.1.0.zip -d external/jass-nn/
```

Anschließend liegen `jass_rules.json`, `state_encoding.md`, `tfjs/model.json` usw. unter `external/jass-nn/jass-nn-v0.1.0/` bereit. Die Web-App referenziert diese Dateien direkt.

Damit hat die Web-App **eine einzige, versionierte Quelle** für Spielregeln und Modell. Aktualisieren bedeutet nur, im Build-Skript die Versionsnummer zu erhöhen — keine kopierten Dateien, kein Drift.

## Roadmap

- [x] Spielengine mit allen Varianten
- [x] Heuristik-Bot mit Schmieren/Sparen/Stechen
- [x] Trainings-Pipeline (Encoder, Datengenerator, MLP, Trainings-Loop)
- [x] NN-Player auf Augenhöhe mit Heuristik (Behavioral Cloning)
- [x] Schnittstellen-Spec für separate Web-App
- [x] TF.js-Export + lokaler Release-Befehl (`scripts/make_release.py`)
- [ ] Größeres Modell trainieren (mehr Daten, ggf. mit GPU via WSL2)
- [ ] Reinforcement Learning (Self-Play) für stärkeren Bot
- [ ] Steigern-Variante (Bieter-Jass)
- [ ] Bodensee-Jass (2 Spieler) und 6-Spieler-Kreuz-Jass

## Lizenz

[MIT](LICENSE) — frei nutzbar, auch kommerziell. Quellenangabe willkommen.

## Quellen für die Spielregeln

- [jassa.at/regeln](https://jassa.at/regeln/) — Grundregeln Vorarlberger Jass
- [Mohrenbrauerei FAQ](https://www.mohrenbrauerei.at/biererlebniswelt/community/haeufig-gestellte-fragen-faq/) — Weis-Tabelle
- [jasskarten.at/jassregeln](https://www.jasskarten.at/jassregeln) — Sonderregeln (Bock/Geiss/Slalom)
