# Architektur

<p><a href="architecture.md">English</a> · <strong>Deutsch</strong> · <a href="../README.de.md">← README</a></p>

Wie JCN9000 zusammenspielt — von der Regel-Engine in diesem Repo bis zum
TensorFlow.js-Modell, das im Browser der Web-App *„Heb ab!"* läuft.

![System-Architektur](diagrams/system_overview.png)

Diagramm-Quelle: [`diagrams/system_overview.puml`](diagrams/system_overview.puml).

## Komponenten

| Schicht | Modul | Aufgabe |
|---|---|---|
| Engine | [`jass_engine/`](../jass_engine/) | Regelgetreuer Jass: 36 Karten, alle Varianten, Stiche, Weisen, Stöcke, Matsch, Schieben. Bodensee liegt in einem eigenen Submodul (`jass_engine/bodensee/`), weil es 2 Spieler und die Tisch-Mechanik hat. |
| Spieler | [`players/`](../players/) | `RandomPlayer`, `HeuristicPlayer` (regelbasiert; zugleich der „Medium"-Gegner der App und der Ansager), `NNPlayer` (lädt ein trainiertes Modell, spielt greedy über die legale Maske). |
| Lehrer + Datengen | [`training/data/`](../training/data/) | Der MCTS-augmentierte Datengenerator — die Quelle aller Trainings-Labels. |
| Training | [`training/train.py`](../training/train.py) | Behavioral Cloning des Lehrers in ein Keras-MLP (768/768/384, ~1,25 Mio. Gewichte, `MaskBias`-Layer, Policy- + Value-Head). Shard-Streaming hält den Peak-RAM niedrig. |
| Evaluation | [`evaluation/`](../evaluation/) | paired-eval, batched-GPU-Inferenz, Win-Rate pro Variante, Elo. |
| Schnittstellen-Spec | [`spec/`](../spec/) | Versionierte Regel-JSON + Encoder-Doku + Test-Fixtures — der Vertrag, gegen den der TypeScript-Port prüft. |
| Release | [`scripts/make_release.py`](../scripts/make_release.py) + `add_tfjs.yml` | Baut das ZIP, erstellt das GitHub-Release; ein GitHub-Actions-Runner konvertiert das Modell nach TensorFlow.js und lädt das Asset erneut hoch. |

## Datenfluss

1. Die **Engine** treibt Partien zwischen Spielern.
2. Der **MCTS-Lehrer** spielt jede Stellung über determinisierte Rollouts aus und
   notiert die beste Karte als Trainings-Label → `.npz`-Shards.
3. **Training** klont diese Labels ins MLP (Warm-Start vom Vorgängermodell).
4. Das Modell wird nach **TensorFlow.js** exportiert und als **GitHub-Release**-Asset
   veröffentlicht.
5. Die **Web-App** lädt das Asset und führt die Inferenz im Browser aus.

Schritte 2–3 wiederholen sich je MCTS-Runde; die aktuellen Modelle sind das
Ergebnis dreier Runden.

## Die Lernmethode (MCTS-augmentiertes Behavioral Cloning)

Das Modell wird nicht per Reinforcement aus Spielergebnissen trainiert (das wurde
versucht und blieb unterlegen). Stattdessen ist eine **Suche** der Lehrer und das
Netz **imitiert** sie:

- **Lehrer:** Für jede Stellung werden die unsichtbaren Karten auf viele plausible
  Welten verteilt (*Determinisierung*), jede ausgespielt, das Ergebnis je
  Kandidaten-Karte gemittelt und die Stellung mit der besten Karte gelabelt. Hier
  sitzt die ganze Rechenarbeit.
- **Schüler:** Das MLP lernt, diese Labels zu reproduzieren — und generalisiert
  weit über eine Lookup-Tabelle hinaus.
- **Iterieren:** Jede Runde startet warm vom Vorgängermodell, sodass die
  Rollout-Gegner realistischer spielen und die Labels besser werden.

Zwei Verfeinerungen der letzten Runde zielen auf Fehler, die auch ein Mensch
erkennt:

- **Void-aware Determinisierung** ([`jass_engine/void_inference.py`](../jass_engine/void_inference.py)):
  Der Lehrer leitet aus der Stichhistorie ab, welche Farben ein Sitz beweisbar
  nicht halten kann (z. B. ist trumpffrei, wer auf einen Trumpf-Lead abgeworfen
  hat — außer evtl. dem Buur) und verteilt ihm diese Karten nie. Das beseitigt
  einen systematischen Bias: sinnloses Trumpf-Ziehen gegen blanke Gegner.
- **Full-Round-Lookahead für Bodensee**
  ([`training/data/bodensee_vectorized_lookahead.py`](../training/data/bodensee_vectorized_lookahead.py)):
  Der Lehrer spielt pro Kandidaten-Karte die *gesamte* Restrunde statt nur einen
  Stich — das behebt die Endspiel-Kurzsichtigkeit (sichere Stiche ordnen, den
  letzten Stich für den +5-Bonus mitnehmen).

## Inferenz-Server (batched-gpu)

Sowohl Evaluation als auch MCTS-Datengen spielen viele Partien gleichzeitig und
bündeln ihre Inferenz-Anfragen auf eine GPU. Game-Threads blockieren an einem
Event, während der Server-Thread Anfragen aus einer Queue sammelt, einen
gebündelten Forward-Pass rechnet und die Ergebnisse zurückgibt — der GIL ist also
frei, während die GPU arbeitet.

![Inferenz-Server](diagrams/inference_server.png)

Diagramm-Quelle: [`diagrams/inference_server.puml`](diagrams/inference_server.puml).

## Encoder

| Encoder | Varianten | Dimensionen | Anmerkung |
|---|---|---|---|
| `3.0.0` | Kreuz, Solo | 421 | Gemeinsam; enthält vorberechnete Karten-Werte/-Stärke |
| `bodensee_1.0.0` | Bodensee | 291 | Eigenes Layout für die Tisch-Mechanik (Hand + sichtbar + verdeckt) |

Der Aktionsraum ist immer 36 (eine Karte). Ansage und Weisen entscheidet die
Heuristik, nicht das Netz.

## Siehe auch

- [Modell-Karten](model_cards/) — pro Release: Daten, Training, Eval, Schwächen
- [Trainings-Runbook](training_runbook_mcts3.md) — die schrittweise Rezeptur
- [Diagramm-Index](diagrams/README.md) — Quellen + wie man PNGs rendert
