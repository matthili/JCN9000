# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden hier dokumentiert.
Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

## [Unreleased]

### Hinzugefügt

- Spec-Generatoren und Konsistenz-Tests

### Geändert

- (noch nichts)

### Behoben

- (noch nichts)

## [0.1.0] - Initial Release

### Hinzugefügt

- **Spielengine** für Vorarlberger Kreuz-Jass mit allen Varianten (Trumpf / Bock / Geiss / Slalom)
- Vollständige Regelumsetzung: Farbzwang, Buur-Ausnahme, kein Untertrumpfen, kein Stichzwang
- Weisen (Sequenzen, Vierlinge), Stöcke, Matsch-Bonus
- Schiebe-Mechanik mit korrekt erhaltener Anspielreihenfolge
- **Random-Player** und **Heuristik-Bot** mit Schmieren-/Sparen-/Stechen-Strategie
- **Trainings-Pipeline**:
  - State-Encoder (132-dim Featurevektor + 36-bit Aktionsmaske)
  - Datengenerator mit Multiprocessing (~350 Partien/s auf 14700K)
  - Keras-Mask-Aware-MLP (256-256-128 + Softmax mit Mask-Bias)
  - Behavioral-Cloning-Training erreicht ~92.5 % Match-Rate mit dem Heuristik-Lehrer
  - NN-Player auf Augenhöhe mit der Heuristik in Spielstärke-Eval
- **Visualisierung**:
  - Rich-basierte Terminal-Demo
  - Streamlit-App mit vier Seiten (Regelwerk, Regelprüfer, Weis-Prüfer, Demo-Partie)
- **Schnittstellen-Spec** für die Web-Anwendung:
  - `spec/jass_rules.json` (alle Regeln deklarativ)
  - `spec/jass_rules.schema.json` (Validierung)
  - `spec/state_encoding.md` (Encoder-Dokumentation)
  - `spec/fixtures/encoding_fixtures.json` (Test-Fixtures für TypeScript-Port)
- **Tests**: 103 Tests über Regeln, Heuristik, Encoder, End-to-End und Spec-Konsistenz
- **CI**: GitHub-Actions-Workflow mit Tests auf Python 3.11/3.12/3.13 und Spec-Drift-Check
