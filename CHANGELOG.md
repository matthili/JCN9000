# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden hier dokumentiert.
Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

## [Unreleased]

### Hinzugefügt

- (noch nichts)

### Geändert

- (noch nichts)

### Behoben

- (noch nichts)

## [0.5.0] - Encoder v3, Gumpf-Variante, Geiss-Schwäche behoben

### Breaking Changes

- **Encoder bumped auf v3.0.0**: Featurevektor jetzt **421 Dims** (vorher 348).
  - Zwei neue Sections `value_per_card` (36) und `strength_per_card` (36) liefern dem NN die unter der aktuellen Variante gültigen Karten-Werte und -Kräfte vorberechnet. Damit muss das Netz die multiplikative Interaktion *Karte × Variante × Lead-Suit* nicht mehr selbst lernen.
  - `mode` wächst von 4 → 5 Bits: zusätzlich `is_gumpf`.
  - `trump_suit` ist jetzt auch im Gumpf-Modus gesetzt.
  - **Inkompatibel mit Modellen aus v0.4.0 und früher** — beim Modell-Laden muss `encoding_version == "3.0.0"` geprüft werden.
- **Spec-Version bumped auf 1.1.0** (additiv): `spec/jass_rules.json` enthält jetzt eine `gumpf`-Variante. Bestehende `trumpf`-/`oben`-/`unten`-/`slalom`-Definitionen sind unverändert.

### Hinzugefügt

- **Gumpf-Variante** (G[eiss] + Tr[umpf]):
  - Trumpf-Farbe verhält sich wie bei normalem Trumpf (Buur=20, Nell=14, Buur-Ausnahme, kein Untertrumpfen)
  - Nicht-Trumpf-Farben mit **invertierter Stärke** (6 sticht in Lead-Farbe alles)
  - Wertpunkte in Nicht-Trumpf identisch mit Trumpf-Variante (8er=0, kein Geiss-8er-Bonus)
  - Stöcke gelten im Gumpf (Trumpf-Ober + Trumpf-König = 20 Punkte)
- **Shard-Streaming-Trainings-Pipeline** (`training/dataset.py:split_shards`, `training/train.py:_make_streaming_dataset`): tf.data-`interleave` über Shard-Pfade statt In-Memory-Concat. Peak-RAM bei 21M Samples von ~80 GB → ~10 GB.
- **Parallel-Evaluation** (`evaluation/run_eval.py --workers N`): CPU-only-TF pro Worker, spawn-Context, Stats-Aggregation via `TeamStats.merge`. 2000 Spiele auf 16 Workers in ~19 min (vorher seriell ~2.5 h).
- **HeuristicPlayer.\_score_gumpf**: konservatives Gumpf-Scoring (Trumpf-Score × 0.85 + Bonus pro 6er). Heuristic-Bot wählt Gumpf nur bei klar passender Hand.

### Geändert

- **Modell-Default**: `DEFAULT_HIDDEN = (768, 768, 384)` (vorher 256/256/128). ~1.25 M Parameter, ~12 MB TF.js — passt zu den reicheren v3-Features.
- **Datengenerator**: `spawn` statt `fork` für `mp.Pool` (robust gegen Worker-RSS-Drift bei Long-Running-Jobs), neue Default `--shard-size 500` (vorher 2000), 12 Varianten (4 Trumpf + 4 Gumpf + Bock + Geiss + 2 Slalom).
- **Tests**: 126 (vorher 103) — neue Tests für Gumpf-Regeln, Encoder-v3-Sections, `TeamStats.merge`.

### Behoben

- **Geiss-Schwäche** des v4-Modells (38 % Win-Rate gegen Heuristic) → v5 erreicht 50 % auf Geiss (= BC-Plateau, so gut wie der Heuristik-Lehrer).
- `weis.py:stoecke_apply` nutzt jetzt `variant.has_trump` statt `mode == TRUMPF` → Stöcke gelten korrekt auch im Gumpf.

### Spielstärke (Eval gegen HeuristicPlayer)

| Modell | Gesamt-Win-Rate | Geiss | Schwächste Variante |
|---|---|---|---|
| v4 (v2-Encoder) | ~49 % | 38 % | Geiss |
| **v5 (v3-Encoder)** | **50.4 %** | **50.0 %** | keine |

→ BC-Plateau erreicht. Über die Heuristik hinauszukommen erfordert RL/Self-Play, was als nächste Iteration vorbereitet ist.

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
