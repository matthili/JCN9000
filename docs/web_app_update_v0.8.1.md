# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.8.1 (Solo-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/jass-neuronales-netz`) ist **v0.8.1** veroeffentlicht
-- ein Punkt-Release ueber v0.8.0 fuer **Solo-Jass** (4 Spieler, jeder gegen
jeden). Drei Aenderungen, zwei davon aktiv auf der App-Seite umzusetzen:

1. **Kritischer Regelfehler in `legal_moves` behoben** -- dieselbe Engine-Funktion
   wie bei Kreuz; die TS-Engine der App muss korrigiert werden. **Wichtigster
   Punkt.**
2. **Lizenzwechsel auf AGPL-3.0-or-later** mit Attributionspflicht (§7(b)).
3. **Neue Modellgewichte** (gleiche Architektur, gleicher Encoder 3.0.0) --
   TF.js-Modell austauschen.

> Falls du parallel das v0.7.1-Briefing (Kreuz) bekommst: Punkt 1 (Regel-Fix)
> und Punkt 2 (Lizenz) sind **identisch** -- es ist derselbe Engine-Fix und
> derselbe Lizenzwechsel fuer alle drei Spielarten. Du musst den `legal_moves`-Fix
> nur **einmal** in der gemeinsamen TS-Engine machen.

---

## 1. Der Regelfehler: "bedienen ODER stechen" (PFLICHT-FIX in der TS-Engine)

Solo nutzt dieselbe `legal_moves`-Logik wie Kreuz. Der Fehler trat auf, wenn eine
**Nicht-Trumpf-Farbe angespielt** wurde und der Spieler diese Farbe **auf der
Hand hat**.

| | Verhalten |
|---|---|
| **falsch (alt)** | nur bedienen erlaubt; zusaetzlich nur der Buur (Trumpf-Unter) |
| **richtig (neu)** | **bedienen ODER stechen** -- alle Lead-Farb-Karten *und* alle spielbaren Truempfe |

```
legale_karten = lead_farb_karten ∪ spielbare_truempfe

spielbare_truempfe =
    alle Truempfe                       , wenn noch kein Trumpf im Stich liegt
    nur HOEHERE Truempfe                , wenn schon ein Trumpf im Stich liegt
                                          (kein Untertrumpfen)
```

Buur bleibt immer spielbar. Unveraendert: Trumpf angespielt -> bedienen, kein
Untertrumpfen; Lead-Farbe nicht auf der Hand -> frei abwerfen/trumpfen.

Referenz: `jass_engine/rules.py::legal_moves`, Tests in `tests/test_rules.py`.
**Solo-Besonderheit:** keine. Der Bedien-/Stech-Zwang ist regelidentisch zu
Kreuz -- nur Reward-Struktur (eigene statt Team-Punkte) und Sitz-Logik
unterscheiden sich, das betrifft `legal_moves` nicht.

---

## 2. Lizenzwechsel: AGPL-3.0-or-later + Attribution (§7(b))

Code und Modellgewichte stehen ab v0.8.1 unter **AGPL-3.0-or-later** mit
§7(b)-Attributionspflicht (Ursprung nennen:
`Based on "Jass-NN" by Matthias, https://github.com/matthili/jass-neuronales-netz`).
Kommerzielle Nutzung erlaubt; bei Netzwerkbetrieb (AGPL §13) muss die modifizierte
Version offengelegt werden. Da die App selbst AGPL-3.0 wird: kongruent, kein
Konflikt. `LICENSE` liegt im Release-ZIP.

---

## 3. Neue Gewichte herunterladen

```bash
gh release download v0.8.1 --repo matthili/jass-neuronales-netz --pattern "jass-nn-*.zip"
unzip jass-nn-v0.8.1.zip
```

- **Encoder unveraendert:** `encoding_version: "3.0.0"`, 421 Dim -- derselbe
  TS-Encoder wie Kreuz.
- **`MANIFEST.json` traegt `team_mode: "solo"`** -- damit die App das Solo-Modell
  zur Solo-Spielart laedt. Das Solo-Modell ist **nicht** mit dem Kreuz-Modell
  austauschbar (andere Reward-Struktur: jeder fuer sich, kein Schmieren).
- **Modell-API unveraendert:** `{state:[batch,421], mask:[batch,36]}` ->
  `{policy:[batch,36], value:[batch,1]}`.

---

## Spielstaerke -- was du erwarten kannst

Solo ist ein 4-Spieler-Spiel (jeder gegen jeden). Die "interessante" Baseline ist
25 % (vier gleich starke Spieler), nicht 50 %.

| Eval-Setup (paired-eval, 1000 Partien) | v0.8.1 Win-Rate |
|---|---|
| 1 NN vs. 3 Solo-Heuristiken | **77.2 %** (Heuristiken je ~7 %) |
| 1 NN(v0.8.1) vs. 1 NN(Vorlaeufer) + 2 Heuristiken | 47.2 % (Vorlaeufer 35.4 %) |

Sehr gleichmaessig ueber die Varianten (~68-84 %). Slalom kommt im Solo nicht vor
(die Solo-Heuristik sagt es nicht an).

---

## Smoke-Test nach Update

1. TS-Engine: `legal_moves`-Faelle gruen (siehe Punkt 1).
2. v0.8.1-Modell laedt im Solo-Modus, `team_mode` == "solo",
   `encoding_version` == "3.0.0".
3. Eine komplette 4-Spieler-Solo-Partie laeuft sauber durch.

---

## Fragen oder Mismatches

Issues im NN-Repo: <https://github.com/matthili/jass-neuronales-netz/issues>.
Pflicht-Aufwand: der gemeinsame `legal_moves`-Fix (einmal fuer alle Spielarten);
Encoder und Modell-API sind unveraendert.
