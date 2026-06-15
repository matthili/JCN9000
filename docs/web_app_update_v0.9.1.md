# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.9.1 (Bodensee-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/JCN9000`) ist **v0.9.1** veroeffentlicht
-- ein Punkt-Release ueber v0.9.0 fuer **Bodensee-Jass** (2 Spieler,
Tisch-Mechanik). **Vier** Aenderungen, drei davon aktiv auf der App-Seite:

1. **Kritischer Regelfehler in `legal_moves` behoben** -- dieselbe Engine-Funktion
   wie bei Kreuz/Solo (Bodensee setzt ueber `legal_moves_bodensee` darauf auf).
   Die TS-Engine der App muss korrigiert werden. **Wichtigster Punkt.**
2. **Eine Encoding-Fixture (`bfix_06`) hat sich geaendert** -- die TS-Encoder-Tests
   muessen die neue `bodensee_encoding_fixtures.json` pinnen.
3. **Lizenzwechsel auf AGPL-3.0-or-later** mit Attributionspflicht (§7(b)).
4. **Neue Modellgewichte** (gleiche Architektur, Encoder `bodensee_1.0.0`
   strukturell unveraendert) -- TF.js-Modell austauschen.

> Wenn du parallel die v0.7.1/v0.8.1-Briefings hast: Punkt 1 (Regel-Fix) und
> Punkt 3 (Lizenz) sind fuer alle drei Spielarten **identisch**. Der
> `legal_moves`-Fix muss nur **einmal** in der gemeinsamen TS-Engine erfolgen.
> Punkt 2 (Fixture) ist **Bodensee-spezifisch**.

---

## 1. Der Regelfehler: "bedienen ODER stechen" (PFLICHT-FIX in der TS-Engine)

Bodensee-Bedienzwang gilt ueber **Hand + sichtbarem Tisch gemeinsam**, aber die
zugrunde liegende Stech-Logik ist dieselbe wie bei Kreuz/Solo. Der Fehler trat
auf, wenn eine **Nicht-Trumpf-Farbe angespielt** wurde und der Spieler diese
Farbe (in Hand oder sichtbarem Tisch) hatte.

| | Verhalten |
|---|---|
| **falsch (alt)** | nur bedienen erlaubt; zusaetzlich nur der Buur (Trumpf-Unter) |
| **richtig (neu)** | **bedienen ODER stechen** -- alle Lead-Farb-Karten *und* alle spielbaren Truempfe |

```
legale_karten = lead_farb_karten ∪ spielbare_truempfe
                 (Pool = Hand + sichtbarer Tisch)

spielbare_truempfe =
    alle Truempfe                       , wenn noch kein Trumpf im Stich liegt
    nur HOEHERE Truempfe                , wenn schon ein Trumpf im Stich liegt
                                          (kein Untertrumpfen)
```

Buur bleibt immer spielbar. Referenz: `jass_engine/rules.py::legal_moves`
(Grundlogik) und `jass_engine/bodensee/rules.py::legal_moves_bodensee`
(Hand+Tisch-Pool).

---

## 2. Geaenderte Encoding-Fixture `bfix_06` (Bodensee-spezifisch)

Die Datei `bodensee_encoding_fixtures.json` enthaelt Referenz-(state -> vector)-
Paare, gegen die der TypeScript-Encoder byte-genau verifiziert wird. Eine Fixture
(`bfix_06`) basierte auf einer Stellung, in der die **alte, fehlerhafte**
`legal_moves` eine andere Aktionsmaske erzeugte. Nach dem Fix wurde sie neu
generiert.

**To-Do App-Seite:** die neue `bodensee_encoding_fixtures.json` aus dem v0.9.1-ZIP
uebernehmen und die Encoder-Tests darauf pinnen. Wenn die TS-Tests gegen die alte
`bfix_06` laufen, schlagen sie nach dem `legal_moves`-Fix fehl -- das ist
erwartet und wird durch das Fixture-Update behoben.

---

## 3. Lizenzwechsel: AGPL-3.0-or-later + Attribution (§7(b))

Code und Modellgewichte stehen ab v0.9.1 unter **AGPL-3.0-or-later** mit
§7(b)-Attributionspflicht (Ursprung nennen:
`Based on "Jass-NN" by Matthias, https://github.com/matthili/JCN9000`).
Kommerzielle Nutzung erlaubt; bei Netzwerkbetrieb (AGPL §13) muss die modifizierte
Version offengelegt werden. Da die App selbst AGPL-3.0 wird: kongruent.
`LICENSE` liegt im Release-ZIP.

---

## 4. Neue Gewichte herunterladen

```bash
gh release download v0.9.1 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.9.1.zip
```

- **Encoder-Struktur unveraendert:** `encoding_version: "bodensee_1.0.0"`,
  291 Dim. Der bestehende Bodensee-TS-Encoder bleibt gueltig -- nur die Fixtures
  (Punkt 2) muessen aktualisiert werden.
- **`MANIFEST.json` traegt `team_mode: "bodensee_2p"`.** Das Bodensee-Modell ist
  nicht mit Kreuz/Solo austauschbar (eigener Encoder, 291 statt 421 Dim).
- **Modell-API unveraendert:** `{state:[batch,291], mask:[batch,36]}` ->
  `{policy:[batch,36], value:[batch,1]}`.

---

## Spielstaerke -- was du erwarten kannst

| Eval-Setup (paired-eval, 1000 Partien) | v0.9.1 Win-Rate |
|---|---|
| vs. Bodensee-Heuristik | **77.8 %** |
| vs. eigenes Vorlaeufermodell (mcts1) | 53.0 % |

2-Spieler-Baseline ist 50 %. Gegenueber v0.9.0 (72.6 % vs. Heuristik) ein Plus von
5.2 Prozentpunkten -- Bodensee profitierte vom Re-Training am staerksten, weil der
`legal_moves`-Fehler dort am direktesten in die gelernte Policy floss. Pro
Variante 67-82 %, staerkste: Unten (81.8 %), schwaechste: Trumpf Schelle (67.1 %).

---

## Smoke-Test nach Update

1. TS-Engine: `legal_moves`-Faelle gruen (bedienen ODER stechen, Pool Hand+Tisch).
2. Neue `bodensee_encoding_fixtures.json` uebernommen; alle Fixtures
   (inkl. `bfix_06`) reproduzieren byte-genau.
3. v0.9.1-Modell laedt im Bodensee-Modus, `team_mode` == "bodensee_2p",
   `encoding_version` == "bodensee_1.0.0".
4. Eine komplette Bodensee-Partie: Tisch-Mechanik + Bedienzwang ueber Hand+Tisch
   korrekt.

---

## Fragen oder Mismatches

Issues im NN-Repo: <https://github.com/matthili/JCN9000/issues>.
Pflicht-Aufwand: der gemeinsame `legal_moves`-Fix (einmal fuer alle Spielarten)
**plus** das Bodensee-Fixture-Update (Punkt 2). Encoder-Struktur und Modell-API
sind unveraendert.
