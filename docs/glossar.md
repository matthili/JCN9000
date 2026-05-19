# Glossar

Alle Fachbegriffe und Akronyme aus diesem Projekt — in deutschen Worten erklärt.
Sortiert nach Themenbereich. Bei Mehrfachverwendungen taucht ein Begriff in
mehreren Abschnitten auf, wenn er in unterschiedlichen Kontexten relevant ist.

---

## 1. Jass — Spielbegriffe

| Begriff | Erklärung |
|---|---|
| **Jass** | Schweizer/Vorarlberger Kartenspiel mit 4 Spielern und 36 Karten. |
| **Kreuz-Jass** | Spielvariante mit 4 Spielern, Partner sitzen über Kreuz (gegenüber). |
| **Stich** | Vier nacheinander gespielte Karten — der höchste gewinnt sie. |
| **Runde** | 9 Stiche. Am Ende werden Punkte verteilt. |
| **Partie** | Folge von Runden, bis ein Team das Punkteziel (typisch 1000) erreicht. |
| **Trumpf** | Spielart mit einer dominierenden Farbe; deren Karten stechen alles. |
| **Bock / Oben** | Spielart ohne Trumpf; Asse sind die stärksten Karten in ihrer Farbe. |
| **Geiss / Unten** | Spielart ohne Trumpf; 6er sind die stärksten Karten in ihrer Farbe (umgekehrte Reihenfolge). |
| **Slalom** | Spielart, bei der pro Stich zwischen Bock und Geiss gewechselt wird. |
| **Gumpf** | Mischung aus Trumpf und Geiss: eine Trumpf-Farbe wie üblich, aber in den Nicht-Trumpf-Farben gilt Geiss-Logik (6er stärkste). |
| **Buur** | Trumpf-Unter — die stärkste Trumpf-Karte (20 Punkte). |
| **Nell** | Trumpf-9 — zweitstärkste Trumpf-Karte (14 Punkte). |
| **Weli** | Schelle-6 — bestimmt in Runde 1, wer ansagt. |
| **Ansage** | Wahl der Spielart (Trumpf, Bock, Geiss, …) vor Stichbeginn. |
| **Schieben** | Der Ansager gibt die Ansage-Wahl an seinen Partner ab. |
| **Stöcke** | Bonus von 20 Punkten für Trumpf-König + Trumpf-Ober in derselben Hand. |
| **Weisen** | Karten-Kombinationen (Sequenzen, Vierlinge) für Bonuspunkte vor dem ersten Stich. |
| **Matsch** | Ein Team gewinnt alle 9 Stiche einer Runde → 100 Bonuspunkte. |
| **Farbzwang** | Regel: wer kann, muss die angespielte Farbe bedienen. |

---

## 2. Neuronale Netze und maschinelles Lernen — Grundbegriffe

| Begriff (Akronym) | Ausgeschrieben | Erklärung |
|---|---|---|
| **NN** | Neural Network | Neuronales Netz — eine Funktion, die durch viele kleine Rechen-Schichten lernt, von Eingaben zu Ausgaben abzubilden. |
| **ML** | Machine Learning | Maschinelles Lernen — Sammelbegriff für Verfahren, bei denen Software Muster aus Daten lernt statt fest programmiert zu werden. |
| **Modell** | – | Die konkreten gespeicherten „Gehirn-Gewichte" eines neuronalen Netzes nach dem Training. |
| **Training** | – | Phase, in der das Modell lernt — Gewichte werden iterativ angepasst. |
| **Inferenz** | – | Anwendung eines fertig trainierten Modells: Eingabe rein, Vorhersage raus. |
| **Hyperparameter** | – | Trainings-Einstellungen, die du vorgibst (Lernrate, Batch-Größe, etc.). Werden nicht gelernt. |
| **Epoche** | – | Ein vollständiger Durchlauf durch alle Trainingsdaten. |
| **Batch** | – | Eine Gruppe Eingaben, die gemeinsam durchs Modell laufen. |
| **Mini-Batch** | – | Synonym für Batch im Kontext von Stochastic Gradient Descent. |
| **Loss** | Verlust | Maß dafür, wie schlecht das Modell aktuell ist. Training minimiert den Loss. |
| **Gradient** | – | Richtung, in die die Gewichte angepasst werden müssen, um den Loss zu verringern. |
| **Lernrate** (engl. learning rate) | – | Wie stark pro Schritt die Gewichte verändert werden. |
| **Overfitting** | Überanpassung | Modell lernt Trainingsdaten auswendig, generalisiert aber nicht auf neue Fälle. |
| **Plateau** | – | Punkt, ab dem das Training nicht mehr besser wird, egal wie lange man weiterläuft. |
| **Checkpoint** | – | Gespeicherter Modellstand zu einem Trainingszeitpunkt. |
| **Snapshot** | – | Synonym für Checkpoint. |
| **Tensor** | – | Mehrdimensionales Zahlen-Array (Verallgemeinerung von Vektor und Matrix). |
| **Logits** | – | Ungenormte Modell-Ausgabewerte vor Anwendung der Softmax. |
| **Softmax** | – | Funktion, die Logits in Wahrscheinlichkeiten (alle ≥0, Summe=1) umrechnet. |
| **Policy** | Strategie | Wahrscheinlichkeitsverteilung über mögliche Aktionen. |
| **Value** | Wert | Geschätzter Punktwert für einen Spielzustand. |
| **Mask** | Maske | Filter über alle theoretischen Aktionen, der nur die in der aktuellen Situation **legalen** zulässt. |
| **Encoder** | – | Übersetzer von rohem Spielzustand in den Featurevektor, den das neuronale Netz verarbeitet. |
| **Featurevektor** | – | Zahlen-Liste, die einen Spielzustand für das Modell beschreibt (bei uns 421 Werte). |

---

## 3. Trainings-Methoden in diesem Projekt

| Begriff (Akronym) | Ausgeschrieben | Erklärung |
|---|---|---|
| **BC** | Behavioral Cloning | Verhalten-Klonen: Modell lernt durch Nachahmung eines Lehrers (typisch der Heuristik-Spieler oder ein MCTS-augmentierter Lehrer). |
| **BC-Plateau** | – | Punkt, ab dem ein BC-trainiertes Modell nicht mehr besser werden kann, weil es den Lehrer perfekt imitiert. |
| **RL** | Reinforcement Learning | Bestärkendes Lernen — Modell lernt durch Belohnungen aus Spielausgängen statt durch Vorgabe von „richtigen" Antworten. |
| **PPO** | Proximal Policy Optimization | Spezifischer RL-Algorithmus mit kontrollierter Schrittweite (verhindert wilde Policy-Sprünge). |
| **GAE** | Generalized Advantage Estimation | Methode zur Berechnung, wie gut eine einzelne Aktion über die Zeit war. |
| **Self-Play** | Selbst-Spiel | Modell spielt gegen sich selbst, um Trainingsdaten zu generieren. |
| **Reward** | Belohnung | Skalarer Wert, der die Qualität eines Spielausgangs misst (typisch: Punkte-Differenz Team A gegen Team B). |
| **Trajectory** | – | Sequenz von Spielzügen + zugehörigen Belohnungen aus einer Partie. |
| **MCTS** | Monte Carlo Tree Search | Baumsuche mit Zufalls-Simulationen: für jede Wahl-Option werden viele zufällige Spielverläufe simuliert; die Option mit dem besten durchschnittlichen Ausgang gewinnt. |
| **Lookahead** | Vorausschau | Allgemeiner Begriff für „blick in die Zukunft, bevor man eine Entscheidung trifft". |
| **Single-Trick-Lookahead** | – | Vorausschau nur bis Ende des aktuellen Stichs. |
| **Full-Round-Lookahead** | – | Vorausschau bis Ende der aktuellen Runde. |
| **Rollout** | – | Ein einzelner simulierter Spielverlauf, der für MCTS verwendet wird. |
| **Determinization** | Determinisierung | Zufällige Verteilung der unsichtbaren Karten auf die Mitspieler — nötig, weil Jass ein Spiel mit unvollständiger Information ist. |
| **AlphaZero-Stil** | – | Trainingsschema, das MCTS, Self-Play und Modell-Update iterativ kombiniert. Bekannt aus DeepMind-Projekten (Go, Schach). |
| **Datengen** | Datengenerierung | Hausinterne Kurzform für „Trainingsdaten-Erzeugung": viele simulierte Partien werden durchgespielt; pro Spielposition entstehen Tupel `(Zustand, Aktionsmaske, Lehrer-Aktion, Belohnung)`. Diese landen als `.npz`-Shard-Dateien auf der Festplatte und sind die Eingabe für das spätere Modell-Training. **Kein gängiger Begriff**, sondern Projekt-Jargon — sollte für externe Kommunikation als „Datengenerierung" oder „Trainingsdaten-Erzeugung" ausgeschrieben werden. |
| **Phase 1 / Phase 2 (Datengen)** | – | Mehrstufiges Datengen-Schema: Phase 1 erzeugt mit einem schwachen Lehrer (z.B. Heuristik) einen ersten Datensatz, das Modell wird trainiert. Phase 2 verwendet das Phase-1-Modell als Lehrer und erzeugt einen besseren Datensatz. Iterativ vergleichbar mit AlphaZero, aber ohne Suchbaum bei der Inferenz. |
| **Shard** | – | Eine `.npz`-Datei mit z.B. 50-500 Spielen, in der pro Spielposition (Zustand, Maske, Aktion, Reward) gespeichert ist. Mehrere Shards pro Variante sind möglich (siehe Chunk-Queue). |

---

## 4. Bewertung von Modellen

| Begriff | Erklärung |
|---|---|
| **Eval** | Kurz für Evaluation/Bewertung — ein Modell wird gegen einen Gegner getestet. |
| **Win-Rate** | Anteil gewonnener Partien (typisch von 0 bis 100 %). |
| **Elo** | Bewertungssystem aus dem Schach: nach jedem Spiel wird ein numerischer Stärke-Wert pro Spieler angepasst. |
| **Tournament** | Eine Eval-Serie mit mehreren Partien zwischen festen Spielertypen. |
| **Variant-Bucket** | Eval-Ergebnis aufgeschlüsselt nach Spielart (Trumpf-Eichel separat von Geiss, etc.). |
| **Paired Evaluation** | Gepaarte Bewertung — pro Paar zwei Spiele mit identischer Kartenverteilung, einmal Modell A auf Sitzen 0+2, einmal auf 1+3. Eliminiert das Karten-Glück als Rauschquelle. |
| **Statistische Signifikanz** | Aussage, ob ein beobachteter Unterschied wahrscheinlich „echt" ist oder durch Zufall entstanden. |
| **Konfidenzintervall** | Bereich, in dem mit gegebener Wahrscheinlichkeit der „wahre" Wert liegt. |
| **Sitz-Tausch** | Erste Hälfte der Spiele mit Modell A auf Sitzen 0+2, zweite Hälfte auf 1+3 — gleicht Sitz-bedingte Vorteile aus. |

---

## 5. Hardware und Tooling

| Begriff (Akronym) | Ausgeschrieben | Erklärung |
|---|---|---|
| **CPU** | Central Processing Unit | Hauptprozessor des Rechners. |
| **GPU** | Graphics Processing Unit | Grafikkarte — viel paralleler als CPU, ideal für neuronale Netze. |
| **VRAM** | Video RAM | Speicher der Grafikkarte (bei der RTX 3060: 12 GB). |
| **RAM** | Random Access Memory | Hauptspeicher des Rechners. |
| **CUDA** | Compute Unified Device Architecture | Nvidias Programmier-Schnittstelle für GPU-Berechnungen. |
| **WSL** | Windows Subsystem for Linux | Linux-Umgebung, die innerhalb von Windows läuft. |
| **TF** | TensorFlow | Maschinellen-Lernens-Bibliothek von Google. |
| **Keras** | – | Höhere Schnittstelle für neuronale Netze, läuft auf TensorFlow. |
| **TF.js** | TensorFlow.js | JavaScript-Variante von TensorFlow für Modelle im Web-Browser. |
| **XLA** | Accelerated Linear Algebra | Compiler in TensorFlow, der Berechnungen optimiert. |
| **TFDF** | TensorFlow Decision Forests | TF-Erweiterung für Entscheidungsbaum-Modelle (für uns nicht relevant). |
| **API** | Application Programming Interface | Programmier-Schnittstelle zwischen Software-Komponenten. |
| **CLI** | Command Line Interface | Kommandozeilen-Schnittstelle (Befehle im Terminal). |
| **CI/CD** | Continuous Integration / Deployment | Automatisierte Build- und Test-Pipelines (z.B. GitHub Actions). |

---

## 6. Parallelverarbeitung in Python

| Begriff (Akronym) | Ausgeschrieben | Erklärung |
|---|---|---|
| **Thread** | – | Parallel laufender „Faden" im selben Prozess. Teilt sich Speicher mit anderen Threads. |
| **Prozess** | – | Eigenständig laufendes Programm mit eigenem Speicher. |
| **Multiprocessing** | – | Nutzung mehrerer Prozesse parallel, jeder mit eigenem Python-Interpreter. |
| **Multithreading** | – | Nutzung mehrerer Threads parallel im selben Prozess. |
| **GIL** | Global Interpreter Lock | Pythons globale Sperre, die verhindert, dass mehrere Threads gleichzeitig Python-Code ausführen. Macht echtes Multi-Threading in Python schwierig. |
| **IPC** | Inter-Process Communication | Kommunikation zwischen Prozessen (typisch über Queues oder geteilten Speicher). |
| **Pickling** | – | Python-Serialisierung von Objekten zu Bytes, um sie zwischen Prozessen zu schicken. |
| **Worker** | – | Ein Hilfs-Prozess oder -Thread, der einen Teil einer großen Aufgabe übernimmt. |
| **Pool** | – | Sammlung mehrerer Worker. |
| **Queue** | – | Warteschlange — Aufgaben kommen rein, Worker holen sie raus. |
| **Spawn-Context** | – | Multiprocessing-Variante, bei der jeder Worker ein frischer Python-Interpreter ist (nötig für TensorFlow). |
| **Fork-Context** | – | Multiprocessing-Variante, bei der der Worker eine Kopie des Hauptprozesses ist (geht nicht gut mit TensorFlow). |
| **Chunk** | – | Häppchen einer großen Aufgabe. Bei uns: Teilmenge von Trainings-Partien einer Variante (z.B. 50 Spiele statt der vollen 500). |
| **Chunk-Queue** | – | Warteschlange mit Chunks als Aufgaben. Mehrere Worker-Prozesse holen sich dynamisch das nächste freie Chunk ab — bessere Auslastung als wenn jeder Worker eine fest zugeteilte Menge bekommt. |
| **Sentinel** | (auf Deutsch: „Wächter") | Spezielles Markierungs-Element in einer Queue, das den Workern „Schluss, keine Aufgaben mehr" signalisiert (typisch der Wert `None`). |
| **MP-Skript** | Multiprocessing-Skript | Hausinterne Kurzform für ein Skript, das mehrere Prozesse parallel startet (Dateinamen enden auf `_mp.py`). |
| **Smoke-Test** | – | Kurz-Test, der prüft, dass etwas grundsätzlich läuft (kein Crash, plausible Ausgaben), ohne tief in die Korrektheit zu gehen. Name kommt aus der Elektrotechnik: „rauchen die Bauteile?". |
| **Warm-Start** | – | Ein Training nicht von zufälligen Gewichten aus starten, sondern von einem bereits trainierten Modell weiterführen. Schneller und stabiler, wenn das Vorgängermodell schon was kann. |

---

## 7. Software-Entwicklung allgemein

| Begriff (Akronym) | Ausgeschrieben | Erklärung |
|---|---|---|
| **Repo** | Repository | Versioniertes Code-Verzeichnis. |
| **Commit** | – | Speicherpunkt im Repo (Schnappschuss aller Dateien zu diesem Zeitpunkt). |
| **Push** | – | Lokale Commits zum Server (z.B. GitHub) hochladen. |
| **Pull** | – | Commits vom Server herunterladen. |
| **Branch** | Zweig | Parallele Entwicklungslinie im Repo. |
| **Tag** | – | Markierte Version (typisch für Releases). |
| **Release** | Veröffentlichung | Offizielle, gekennzeichnete Version (z.B. v0.6.0). |
| **Semver** | Semantic Versioning | Versionsschema „MAJOR.MINOR.PATCH" mit klaren Regeln für jeden Teil. |
| **Patch-Release** | – | Wartungs-Release ohne neue Funktionen — nur Bugfixes. |
| **gh** | GitHub CLI | Befehlszeilen-Werkzeug von GitHub. |
| **YAML** | Yet Another Markup Language | Konfigurations-Datei-Format (z.B. GitHub-Action-Definitionen). |
| **JSON** | JavaScript Object Notation | Datenformat für strukturierte Daten (Listen, Objekte). |

---

## 8. Konzepte speziell für dieses Projekt

| Begriff | Erklärung |
|---|---|
| **v3-Encoder / Encoding v3.0.0** | Encoder-Version, die 421-dimensionale Featurevektoren erzeugt. |
| **Spec v1.x** | JSON-basierte Regel-Spezifikation, die zwischen Trainings-Repo und Web-App geteilt wird. |
| **v0.5.0, v0.6.0, …** | Release-Versionen des Modells. v0.6.0 ist aktueller Produktivstand. |
| **balanced_v3** | Datensatz, der mit dem v3-Encoder und ausgewogenen Spielvarianten erzeugt wurde. |
| **MCTS-Phase 1, Phase 2** | Aufeinanderfolgende Daten-Generierungs-Läufe mit MCTS-Augmentation. |
| **InferenceServer** | Eigene Klasse im Code, die Inferenz-Anfragen aus mehreren Threads sammelt und als Batch zur GPU schickt. |
| **HeuristicPlayer** | Regelbasierter Jass-Spieler mit Stech-/Schmier-/Spar-Strategie. |
| **ForcedAnnouncementPlayer** | Wie HeuristicPlayer, aber sagt immer eine fest vorgegebene Spielart an (für ausgewogene Datengen). |
| **NNPlayer** | Spieler, der ein neuronales Netz für die Kartenwahl nutzt. |
| **RLPlayer** | NN-Spieler im RL-Training, der zusätzlich Trajektorien für PPO-Updates aufzeichnet. |
| **BatchedRLPlayer / BatchedEvalNNPlayer** | NN-Spieler-Varianten, die ihre Inferenz über einen InferenceServer routen (für GPU-Batching). |

---

## 9. Häufige Begriffe in Konsolen-Ausgaben

| Begriff | Erklärung |
|---|---|
| **Worker N** | Prozess-ID im Multiprocessing-Lauf (z.B. „Worker 2"). |
| **Shard** | Ein einzelnes komprimiertes Daten-Stück (`.npz`-Datei) mit Trainings-Samples. |
| **OOM** (Out Of Memory) | Speicherüberlauf — der Prozess wurde vom System beendet, weil zu viel Speicher belegt wurde. |
| **Memory Growth** | TensorFlow-Modus, in dem die GPU-Speicher nur nach Bedarf allokiert wird (statt sofort komplett). |
| **Stub** | Platzhalter — eine leere Hülle einer Software-Komponente, die echte Funktionen vortäuscht, ohne sie auszuführen. |

---

## Hinweis

Wenn dir in einer Code-Stelle oder einer Diskussion ein Begriff fehlt: bitte
melden — der Glossar ist als wachsendes Dokument gedacht, nicht als
abgeschlossener Stand. Lieber einen Begriff zu viel als zu wenig.
