# Trainings-Runbook: MCTS-Runde 3 + Heuristik-Tuning (~120 h)

Geführter, schrittweiser Fahrplan zur nächsten Stärke-Iteration. Ziel: die
beobachteten **kurzsichtigen Fehler** reduzieren. Die zugrunde liegenden
Verbesserungen stecken bereits im Code (mit Tests) und greifen in den Schritten:

1. **Karten-Spiel (NN):**
   - tieferer MCTS-Lehrer (mehr Rollouts → weniger Label-Rauschen);
   - **void-aware Determinisierung** (Kreuz/Solo, automatisch aktiv): der Lehrer
     verteilt keine „halluzinierten" Trümpfe mehr an nachweislich blanke Gegner
     → behebt das sinnlose Austrumpfen (systematischer Bias, kein Rauschen);
   - **Full-Round-Lookahead für Bodensee** (vorher nur 1-Stich → strukturell
     kurzsichtig).
2. **Heuristik (eigener Schwierigkeitsgrad):** void-tracking Trumpf-Disziplin
   beim Anspielen + optionales Tuning der vier Ansage-Parameter.

GPU-Schritte laufen in einer Umgebung mit TensorFlow + CUDA (z. B. WSL2 mit
conda). Die CPU-Schritte (Heuristik) brauchen **kein** TensorFlow und laufen auf
einer beliebigen Maschine — gerne parallel zur GPU-Datengen. Nach jedem
NN-Schritt ein **Eval-Gate** — es entscheidet ehrlich, ob sich der Schritt
gelohnt hat.

> **Ehrliche Erwartung:** BC plateauiert beim Lehrer-Niveau; das wird kein
> Quantensprung. Realistisch: die kurzsichtigen Fehler werden spürbar seltener,
> am stärksten bei Bodensee (dank Full-Round). Die Eval-Gates sagen pro Variante,
> ob mcts3 die mcts2-Modelle wirklich schlägt.

---

## Zeitbudget (Schätzung, kalibriert aus Runde 2)

| Schritt | Config | Schätzung |
|---|---|---|
| 0. Bodensee-Full-Round-Smoke + Kalibrierung | 10 Spiele, 1 Variante | ~5 min |
| 1. CPU-Nebenspur: Heuristik-Void-Eval + opt. Ansage-Tuning | CPU, beliebige Maschine, parallel | Eval: Minuten · Tuning: 1–6 h |
| 2. Kreuz mcts3 (60 Rollouts) | 500 Spiele/Var | ~26 h |
| 3. Solo mcts3 (60 Rollouts) | 500 Spiele/Var | ~36 h |
| 4. Bodensee mcts3 (**full-round**) | 500 Spiele/Var (kalibriert via Schritt 0) | ~14–27 h |
| Training + Evals (alle) | — | ~3 h |
| **Summe** | | **~95–110 h** |

Der Puffer bis 120 h ist Absicht: er fängt die Bodensee-Full-Round-Unsicherheit
und evtl. Reruns ab — oder erlaubt eine **mcts4-Runde** auf der Variante, die im
Eval-Gate am meisten gewinnt.

---

## Schritt 0 — Bodensee-Full-Round-Smoke + Kalibrierung

Bestätigt, dass der neue Full-Round-Pfad mit echtem Modell läuft, **und** misst
die reale Spiel-Rate (Full-Round ist deutlich teurer als 1-Stich — wir
brauchen die Messung, um Schritt 4 budgetgerecht zu dimensionieren).

```bash
python -u -m training.data.generate_bodensee_mcts_data_mp \
    --warm-start models/bodensee_mcts2/best.keras \
    --games-per-variant 10 --games-per-chunk 10 \
    --rollouts-per-card 30 --target-distribution "500:0.5,1000:0.5" \
    --workers 4 --parallel-threads-per-worker 32 \
    --inference-batch-size 1024 \
    --variants trumpf_eichel \
    --lookahead-mode full-round \
    --output data/_smoke_bodensee_fr
```

→ **Melde die Dauer für die 10 Spiele.** Daraus rechne ich die
`--games-per-variant` für Schritt 4 so, dass Bodensee ins Budget passt.
Danach `data/_smoke_bodensee_fr` löschen.

---

## Schritt 1 — CPU-Nebenspur: Heuristik (beliebige Maschine, parallel)

Reines CPU, **kein** TensorFlow/numpy — nur Python 3.11+ und das Repo. Läuft auf
einer beliebigen freien Maschine, gerne parallel zur GPU-Datengen.

**1a) Void-Regel validieren** (misst die Trumpf-Disziplin gegen die Baseline,
paired-eval; Heuristik-Partien sind billig → wenige Minuten):

```bash
python -m scripts.eval_heuristic_void_rule --games 8000 --workers 12
```

**1b) Ansage-Tuning (optional)** — optimiert die vier Ansage-Parameter gegen die
Baseline. Übernimmt nur, was über dem 2-SD-Rauschen liegt:

```bash
python -m scripts.tune_heuristic_announce \
    --games-screen 800 --num-candidates 80 \
    --games-final 6000 --top-k 6 \
    --workers 12 \
    --output heuristic_announce_tuned.json
```

→ Signifikante Verbesserungen werden als neue `HeuristicPlayer`-Defaults
übernommen (stärkerer Heuristik-Gegner + Ansage-Fallback). Diese Spur ist vom
NN-Training **entkoppelt** — kein Re-Training der NN-Daten nötig.

---

## Schritt 2 — Kreuz mcts3

Tieferer Lehrer: **60 statt 30 Rollouts** (halbiert das Determinisierungs-
Rauschen) plus die **void-aware Determinisierung** (automatisch aktiv, kein Flag
nötig — behebt das Trumpf-Ziehen gegen blanke Gegner). Warm-Start aus dem
aktuellen Kreuz-Modell.

```bash
# Datengen (~26 h)
python -u -m training.data.generate_mcts_data_mp \
    --warm-start models/kreuz_mcts2/best.keras \
    --games-per-variant 500 --games-per-chunk 25 \
    --rollouts-per-card 60 --target 1000 \
    --workers 8 --parallel-threads-per-worker 32 \
    --inference-batch-size 1024 \
    --lookahead-mode full-round-vec \
    --skip-existing \
    --output data/mcts_fixed/phase3 \
    2>&1 | tee logs/kreuz_mcts3_datagen.log

# Training (~5 min)
python -u -m training.train \
    --data data/mcts_fixed/phase3 \
    --warm-start models/kreuz_mcts2/best.keras \
    --output models/kreuz_mcts3 \
    --epochs 20 --learning-rate 5e-4

# Eval-Gate: mcts3 vs mcts2
python -m evaluation.run_eval \
    --a nn --b nn \
    --model-a models/kreuz_mcts3/best.keras \
    --model-b models/kreuz_mcts2/best.keras \
    --games 1000 --paired-eval \
    --inference-mode batched-gpu \
    --inference-batch-size 128 --parallel-threads 128
```

**Gate-Regel:** > ~55 % paired = klarer Gewinn (mcts3 übernehmen). ~50 % =
Deepening gesättigt (mcts2 behalten).

---

## Schritt 3 — Solo mcts3

Wie Kreuz: 60 Rollouts + **void-aware Determinisierung** (automatisch aktiv).

```bash
# Datengen (~36 h)
python -u -m training.data.generate_solo_mcts_data_mp \
    --warm-start models/solo_mcts2/best.keras \
    --games-per-variant 500 --games-per-chunk 25 \
    --rollouts-per-card 60 --target-distribution "500:0.5,1000:0.5" \
    --workers 8 --parallel-threads-per-worker 32 \
    --inference-batch-size 1024 \
    --skip-existing \
    --output data/solo_mcts_fixed/phase3 \
    2>&1 | tee logs/solo_mcts3_datagen.log

# Training
python -u -m training.train \
    --data data/solo_mcts_fixed/phase3 \
    --warm-start models/solo_mcts2/best.keras \
    --output models/solo_mcts3 \
    --epochs 20 --learning-rate 5e-4

# Eval-Gate: mcts3 vs mcts2
python -m evaluation.run_solo_eval \
    --a nn --b nn \
    --model-a models/solo_mcts3/best.keras \
    --model-b models/solo_mcts2/best.keras \
    --games 1000 --paired-eval \
    --inference-mode batched-gpu
```

---

## Schritt 4 — Bodensee mcts3 (Full-Round-Lookahead)

Der Headline-Schritt: erstmals **Full-Round** statt 1-Stich. Rollouts bei 30
(die Tiefe selbst bringt schon viel Signal). 500 Spiele/Variante = dieselbe
Datenmenge wie B-mcts2 — das Eval-Gate misst dann rein die Lehrer-Qualität
(full-round vs single-trick), nicht unterschiedliche Datenmengen.

Kalibrierung aus Schritt 0: full-round erzeugt ~335 Samples/Spiel (B-mcts2:
322 — gleiches Volumen pro Spiel) und ist pro Worker max. ~2x langsamer als
single-trick (Erzwungene-Zuege-Abkuerzung wirkt).

```bash
# Datengen (~14–27 h)
python -u -m training.data.generate_bodensee_mcts_data_mp \
    --warm-start models/bodensee_mcts2/best.keras \
    --games-per-variant 500 --games-per-chunk 25 \
    --rollouts-per-card 30 --target-distribution "500:0.5,1000:0.5" \
    --workers 8 --parallel-threads-per-worker 32 \
    --inference-batch-size 1024 \
    --lookahead-mode full-round \
    --skip-existing \
    --output data/bodensee_mcts_fixed/phase3 \
    2>&1 | tee logs/bodensee_mcts3_datagen.log

# Training
python -u -m training.train \
    --data data/bodensee_mcts_fixed/phase3 \
    --warm-start models/bodensee_mcts2/best.keras \
    --output models/bodensee_mcts3 \
    --epochs 20 --learning-rate 5e-4

# Eval-Gate: mcts3 (full-round) vs mcts2 (single-trick)
python -m evaluation.run_bodensee_eval \
    --a nn --b nn \
    --model-a models/bodensee_mcts3/best.keras \
    --model-b models/bodensee_mcts2/best.keras \
    --games 1000 --paired-eval \
    --inference-mode batched-gpu
```

**Hypothese:** Hier sollte der Sprung am größten sein (Full-Round behebt die
strukturelle Kurzsichtigkeit). Wenn mcts3 die mcts2-Bodensee deutlich schlägt,
war der Code-Umbau die Arbeit wert.

---

## Abschluss

- Pro Variante, die ihr Eval-Gate besteht: neues Release **v0.7.2 / v0.8.2 /
  v0.9.2** (Model-Card + `make_release.py --game-mode …`, wie bei v0.x.1).
- Verbleibt Budget **und** war ein Gate-Gewinn besonders stark: optionale
  **mcts4**-Runde auf genau dieser Variante (Warm-Start aus mcts3).
- Reihenfolge der Datengen-Schritte ist frei; Schritt 1 (CPU) kann laufen,
  während die GPU gerade nichts tut.
