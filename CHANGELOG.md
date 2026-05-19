# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden hier dokumentiert.
Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

## [Unreleased]

### Entwurf für v0.8.0 (Solo-Jass)

Die folgenden Punkte beschreiben den Stand kurz vor dem Release. Konkrete
Eval-Zahlen werden nach abgeschlossenem Training eingetragen.

### Hinzugefügt

- **Neue Spielart Solo-Jass** ([`jass_engine/variants/solo_jass.py`](jass_engine/variants/solo_jass.py)):
  - 4 Spieler, jeder gegen jeden
  - Punkte pro Spieler statt pro Team
  - Default-Spielziel 500 (mindestens 500, konfigurierbar)
  - Schieben deaktiviert (`allow_push=False`)
  - Matsch +100 für einen einzelnen Spieler, Stöcke +20 für den Halter persönlich
  - Weisen: nur der höchste Weis-Halter erhält die Punkte (Gleichstand → zuerst angesagt)
  - Engine-Implementierung via `teams=[0,1,2,3]` (jeder eigene Team-ID) — bestehende Punkteaggregation funktioniert dadurch automatisch korrekt
- **`SoloHeuristicPlayer`** ([`players/solo_heuristic_player.py`](players/solo_heuristic_player.py)): erbt vom Team-HeuristicPlayer; der Schmier-Branch ist im Solo strukturell tot Code, weil `_is_partner_winning` bei `teams=[0,1,2,3]` immer `False` liefert. Konservativere Slalom-Parameter.
- **Solo-MCTS-Datengen** ([`training/data/solo_vectorized_lookahead.py`](training/data/solo_vectorized_lookahead.py), [`training/data/generate_solo_mcts_data.py`](training/data/generate_solo_mcts_data.py), [`training/data/generate_solo_mcts_data_mp.py`](training/data/generate_solo_mcts_data_mp.py)):
  - Solo-Reward: `(eigene Punkte - max(andere Punkte)) / 200`
  - Variables Spielziel pro Trainings-Partie (Default 50/50 zwischen 500 und 1000)
  - Lehrer-Init für Phase 1: SoloHeuristicPlayer; für Phase 2: das Solo-Phase-1-Modell
- **Warm-Start in `training/train.py`**: neues `--warm-start MODEL_PATH`-Flag lädt Modell-Gewichte und kompiliert mit frischem Optimizer-State + konfigurierter Lernrate neu. Backwards-compatible.
- **Solo-Eval-Pipeline** ([`evaluation/solo_eval.py`](evaluation/solo_eval.py), [`evaluation/run_solo_eval.py`](evaluation/run_solo_eval.py), [`evaluation/solo_stats.py`](evaluation/solo_stats.py)):
  - `four_way_match(label_a, factory_a, label_b, factory_b, label_h, factory_h, ...)`: 1 Modell A vs 1 Modell B vs 2 SoloHeuristik
  - Paired-Eval: 4 Partien pro Kartenverteilung mit zyklischer Sitz-Rotation; eliminiert Karten-Glück und Sitz-Vorteile vollständig (Smoke-Test mit 4 identischen Bots ergibt exakt 25.0 % Win-Rate pro Rolle)
  - Win-Rate pro Variante, Avg-Score, Matsch-Rate pro Spieler-Rolle
- **Modell-Karte v0.8.0** ([`docs/model_cards/v0.8.0.md`](docs/model_cards/v0.8.0.md)) und Web-App-Briefing ([`docs/web_app_update_v0.8.0.md`](docs/web_app_update_v0.8.0.md))

### Geändert

- **Engine-Methoden `play_game()` und `play_round()`** akzeptieren jetzt einen Parameter `allow_push: bool = True`. Backwards-compatible — alle bestehenden Aufrufer nutzen den Default. Bei Solo wird `False` gesetzt, um Schieben strukturell zu unterbinden.
- **MANIFEST.json** enthält jetzt zusätzlich das Feld `team_mode` (`"team"` oder `"solo"`), damit die Web-App das richtige Modell zur gewählten Spielart laden kann. Spec-Version bleibt unverändert bei 1.2.0 — Solo ist eine Spielmodus-Frage (Teams, Schieben, Target-Score), keine Variant-Regel-Frage.

### Spielstärke (Eval mit paired-eval, 3400 Partien)

| Rolle | Win-Rate | Avg. Score / Partie | Matsch-Rate / Runde |
|---|---|---|---|
| **NN(solo_phase2) / v0.8.0** | **45.4 %** | **462.3** | **0.16 %** |
| NN(solo_phase1) | 21.1 % | 392.0 | 0.01 % |
| SoloHeuristik (zwei Sitze) | 16.8 % | 381.1 | 0.04 % |

Random-Baseline wäre 25 % pro Rolle. v0.8.0 liegt 20 Prozentpunkte darüber. Ziel ≥ 35 % wurde mit 10 Punkten Puffer erreicht. Die statistische Standardabweichung der Win-Rate bei 3400 Partien beträgt ca. 0.74 %, d.h. der Abstand zur Random-Baseline entspricht ca. 28 Standardabweichungen.

Pro Variante: alle gespielten Varianten zwischen 37.7 % (Gumpf-Schelle, schwächste) und 47.6 % (Unten, stärkste). Slalom-Varianten kamen in der Eval-Stichprobe nicht vor (Heuristik wählt sie nicht im Solo-Modus).

## [0.7.0] - 2026-05-18 - MCTS-augmentiertes Behavioral Cloning, erstes Modell stärker als v0.5.0

### Hinzugefügt

- **MCTS-augmentierte Datengen** ([`training/data/`](training/data/)):
  - `generate_mcts_data.py`: pro Spielposition Monte-Carlo-Lookahead mit Determinisierung der unsichtbaren Karten und ~30 Rollouts pro legaler Karte. Die Karte mit dem höchsten erwarteten Rundenende-Score wird als Lehrer-Aktion gespeichert.
  - `vectorized_lookahead.py`: vektorisierte Full-Round-Variante — alle Rollouts werden im Lockstep getickt und in einem einzigen Batch durch den InferenceServer geschickt. Faktor 10-20 schneller als Einzel-Inferenz.
  - `generate_mcts_data_mp.py`: Multiprocessing-Variante mit eigenem GPU-Modell pro Worker. Memory-Growth erlaubt 4-6 Worker auf einer 12-GB-GPU, umgeht den Python-GIL bei der Spiellogik.
- **Eval-Modus `batched-gpu`** ([`evaluation/batched_eval.py`](evaluation/batched_eval.py)): mehrere parallele Spiele in einem Prozess, ein InferenceServer pro NN-Team, Batch-Inferenz auf der GPU. 5-10x schneller als der CPU-Worker-Modus bei 2000 Eval-Partien.
- **Paired Evaluation** (`--paired-eval`): pro Paar zwei Spiele mit identischer Kartenverteilung — einmal Modell A auf Sitzen 0+2, einmal auf 1+3. Eliminiert Karten-Glück als Rauschquelle. Sanity-Test: zwei identische HeuristicPlayer geben Diff = 0.00 Punkte (vs. 14 Punkte Rauschen ohne paired-eval).
- **Glossar** ([`docs/glossar.md`](docs/glossar.md)): >100 Einträge in deutschem Klartext zu Jass-Begriffen, ML-Grundlagen, Trainings-Methoden, Bewertung, Hardware, Python-Parallelverarbeitung.
- **Modell-Karte** ([`docs/model_cards/v0.7.0.md`](docs/model_cards/v0.7.0.md)) und Web-App-Briefing ([`docs/web_app_update_v0.7.0.md`](docs/web_app_update_v0.7.0.md)).
- **Architektur-Diagramme** als PlantUML ([`docs/diagrams/`](docs/diagrams/)).

### Geändert

- **Spec auf 1.2.0** (additiv): neue Blöcke `scoring.score_composition`, `round_flow.play_order_anchor`, Trick-Card-Ordering. Keine breaking changes — Encoding-Version bleibt 3.0.0.

### Spielstärke (Eval gegen v0.5.0)

| Metrik | v0.7.0 | v0.5.0 |
|---|---|---|
| Win-Rate (4000 Paare, paired-eval) | **77.2 %** | 22.7 % |
| Avg. Score / Partie | 1025.2 | 829.3 |
| Matsch-Rate / Runde | 4.78 % | 1.59 % |
| Stärkste Variante | Slalom Unten (66.3 %) | — |
| Schwächste Variante | Gumpf Schelle (54.0 %) | — |

Erstes Modell, das v5 klar schlägt — die Reinforcement-Learning-Versuche (v6, v7) blieben unter dem v5-Niveau (siehe gestrichene Roadmap-Punkte).

## [0.6.0] - 2026-05-16 - TF.js-Workflow live, Heuristik-Feinschliff

### Hinzugefügt

- **GitHub-Actions-Workflow `add_tfjs.yml`**: läuft automatisch nach jedem `release:published`-Event. Konvertiert das Keras-Modell aus dem ZIP-Asset auf einem Ubuntu-Runner zu TF.js und ersetzt das Asset mit `--clobber`. Damit ist die TF.js-Konvertierung vom lokalen Setup entkoppelt — auf Windows/WSL2 mit aktuellem Python ist `tensorflowjs` eine Dependency-Hölle, auf dem Linux-Runner mit Python 3.12 läuft sie zuverlässig.
- **Defensive Stubs** in [`scripts/add_tfjs_to_release.py`](scripts/add_tfjs_to_release.py) für `tensorflow_decision_forests`, `yggdrasil_decision_forests` und `tensorflow_hub`: tensorflowjs zieht sie als Transit-Dependencies, hat aber Protobuf-Versionskonflikte mit aktuellem TF. Für die MLP-Konvertierung werden sie nicht gebraucht, daher per ModuleType-Trick stillschweigend gemockt.
- **HeuristicPlayer**: zweistufige Gumpf-Bewertung. Erster Pass schätzt Wert konservativ, zweiter Pass bestätigt nur, wenn keine bessere Trumpf-Ansage existiert. Reduziert Fehl-Ansagen.
- **`allowed_modes` und `allow_slalom`-Parameter** in HeuristicPlayer-Ansage-Logik: ermöglicht in Tests gezielt nur bestimmte Varianten zu spielen.

### Geändert

- **Zeilenenden** im Repo via `.gitattributes` auf LF normalisiert. Beseitigt CRLF-Diffs beim Wechsel zwischen WSL2 und Windows.

### Behoben

- TF.js-Konvertierung im Workflow lief vorher nicht durch (Protobuf-Konflikt, fehlende `pkg_resources` in Python 3.12, MaskBias-Custom-Layer wurde vom CLI-Konverter nicht erkannt). Jetzt komplett gefixt: Python-API statt CLI-Subprocess, Custom-Layer-Import vor dem Modell-Load, `setuptools` explizit installiert.

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
