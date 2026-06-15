# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.7.2 (Kreuz-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung -- entspannt diesmal

Im NN-Repository (`matthili/JCN9000`) ist **v0.7.2** veroeffentlicht
(Kreuz-Jass). Im Gegensatz zu v0.7.1 ist das ein **reines Qualitaets-Update ohne
Pflicht-Arbeit auf der App-Seite**:

- **KEINE** Engine-Aenderung (Spielregeln unveraendert).
- **KEINE** Encoder-Aenderung (`encoding_version` weiterhin **3.0.0**).
- **KEINE** Lizenz-Aenderung (bleibt AGPL-3.0-or-later).
- **Neue, staerkere Modellgewichte** -> nur das TF.js-Modell austauschen.

Optional betroffen: der **Heuristik-Gegner** ("Medium"-Schwierigkeitsgrad) wurde
getunt -- nur relevant, falls ihr die Heuristik in TypeScript nachgebaut habt
(siehe Punkt 2).

---

## 1. Neues Modell (Pflicht, aber trivial)

```bash
gh release download v0.7.2 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.7.2.zip
```

- **Encoder + Modell-API unveraendert:** `{state:[batch,421], mask:[batch,36]}`
  -> `{policy:[batch,36], value:[batch,1]}`. Euer bestehender TS-Encoder passt
  1:1. Es ist ein **reiner Gewichts-Austausch** des TF.js-Modells.
- Was sich verbessert hat (rein intern in der Datengen, nichts fuer euch zu tun):
  void-aware Determinisierung des MCTS-Lehrers (kein sinnloses Trumpf-Ziehen mehr
  gegen blanke Gegner) + tieferer Lookahead.

---

## 2. Getunte "Medium"-Heuristik (nur falls in TypeScript portiert)

**Wenn euer "Medium"-Gegner die Python-Heuristik aus diesem Repo aufruft
(z. B. ueber den Microservice), ist nichts zu tun -- sie ist automatisch
aktualisiert.** Nur falls ihr die Ansage-Heuristik in TS *nachgebaut* habt,
hier die neuen Werte.

**2a) Neue Ansage-Parameter** (per Win-Rate-Suche getunt, +5,6 pp ueber drei
Iterationen):

| Parameter | alt (v0.7.1) | neu (v0.7.2) |
|---|---|---|
| `push_threshold` | 55 | **64** |
| `slalom_base_factor` | 0.95 | **0.90** |
| `slalom_concentration_factor` | 2 | 2 |
| `slalom_spread_factor` | 1 | 1 |
| `gumpf_scale` *(neu)* | (1.0) | **1.15** |
| `oben_scale` *(neu)* | (1.0) | **0.96** |
| `unten_scale` *(neu)* | (1.0) | **1.08** |

Die drei `*_scale`-Werte sind ein **neues Konzept**: Sie multiplizieren den
Ansage-Score der jeweiligen Familie, bevor das Maximum gewaehlt wird (Trumpf =
Anker, immer 1.0). In TS also `gumpfScore *= 1.15; obenScore *= 0.96;
untenScore *= 1.08;` vor dem argmax. Kern-Erkenntnis: Gumpf war zu zaghaft
angesagt.

**2b) Trumpf-Disziplin beim Anspielen** *(optional, geringe Prioritaet)*: Sind
beim Anspielen in Trumpf/Gumpf **beide Gegner beweisbar trumpffrei** (sie haben
auf einen Trumpf-Lead frueher Nicht-Trumpf gespielt -> blank in Trumpf, ausser
evtl. dem Buur), spielt die Heuristik **keine hohen Truempfe mehr an**, sondern
hohe Seitenkarten (Truempfe als Stich-Garanten aufsparen). Korrektes Spiel, aber
der gemessene Effekt gegen die Heuristik-Baseline war ~0 pp -- also nice-to-have,
kein Muss.

---

## Spielstaerke -- was zu erwarten ist

| Eval (paired-eval) | v0.7.2 Win-Rate |
|---|---|
| vs. eigenes Vorgaengermodell (kreuz_mcts2) | **57,9 %** (6000 Partien, ~12 SD) |
| vs. Heuristik (getunt) | **83,5 %** (4000 Partien) |

Die 83,5 % messen gegen die **staerkere** (getunte) Heuristik -- gegenueber
v0.7.1's 79,5 % ist die echte Verbesserung also groesser als die Differenz. Die
Matsch-Rate steigt auf 6,48 %: Das Modell fegt jetzt oefter alle 9 Stiche weg.

---

## Smoke-Test nach Update

1. v0.7.2-Modell laedt, `encoding_version` == "3.0.0".
2. Eine Kreuz-Partie laeuft ohne Auffaelligkeiten (Regeln/Encoder unveraendert).
3. (Falls TS-Heuristik portiert) Ansage-Verhalten plausibel; Gumpf wird etwas
   haeufiger angesagt als vorher.

---

## Fragen

Issues im NN-Repo: <https://github.com/matthili/JCN9000/issues>.
Aufwand auf eurer Seite: **ein Modell-Austausch**, sonst nichts Verpflichtendes.
