# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.7.1 (Kreuz-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/JCN9000`) ist **v0.7.1** veroeffentlicht
-- ein Punkt-Release ueber v0.7.0 fuer **Kreuz-Jass** (4 Spieler, Teams ueber
Kreuz). Drei Dinge haben sich geaendert, zwei davon sind **aktiv auf der
App-Seite umzusetzen**:

1. **Kritischer Regelfehler in `legal_moves` behoben** -- die TS-Engine der App
   hat denselben Fehler und muss korrigiert werden. **Das ist der wichtigste
   Punkt.**
2. **Lizenzwechsel auf AGPL-3.0-or-later** mit Attributionspflicht (§7(b)).
3. **Neue Modellgewichte** (gleiche Architektur, gleicher Encoder 3.0.0) --
   einfach das TF.js-Modell austauschen.

---

## 1. Der Regelfehler: "bedienen ODER stechen" (PFLICHT-FIX in der TS-Engine)

Bisher war die Berechnung der legalen Karten falsch, wenn eine **Nicht-Trumpf-
Farbe angespielt** wurde und der Spieler diese Farbe **auf der Hand hat**.

| | Verhalten |
|---|---|
| **falsch (alt)** | nur bedienen erlaubt; zusaetzlich nur der Buur (Trumpf-Unter) |
| **richtig (neu)** | **bedienen ODER stechen** -- alle Lead-Farb-Karten *und* alle spielbaren Truempfe sind legal |

Praezise Regel fuer "Nicht-Trumpf angespielt, Spieler hat die Lead-Farbe":

```
legale_karten = lead_farb_karten ∪ spielbare_truempfe

spielbare_truempfe =
    alle Truempfe                       , wenn noch kein Trumpf im Stich liegt
    nur HOEHERE Truempfe                , wenn schon ein Trumpf im Stich liegt
                                          (kein Untertrumpfen)
```

Der Buur (Trumpf-Unter) ist ein Trumpf und damit Teil der spielbaren Truempfe;
die Sonderregel "Buur ist immer spielbar" bleibt erhalten.

**Unveraendert** (waren schon korrekt):
- Trumpf-Farbe angespielt -> Trumpf bedienen, kein Untertrumpfen.
- Lead-Farbe nicht auf der Hand -> frei abwerfen oder trumpfen (kein Stichzwang).

**Warum das auf der App-Seite zwingend ist:** Wenn die App clientseitig legale
Zuege validiert oder anzeigt, muss sie exakt dieselbe Menge berechnen wie das
Modell beim Training gesehen hat. Sonst sperrt die UI Zuege, die das Modell fuer
legal haelt (oder umgekehrt). Referenz-Implementierung: `jass_engine/rules.py`,
Funktion `legal_moves`. Testfaelle: `tests/test_rules.py`
(`test_bedienen_oder_stechen_*`).

---

## 2. Lizenzwechsel: AGPL-3.0-or-later + Attribution (§7(b))

Das NN-Repo (Code **und** Modellgewichte) steht ab v0.7.1 unter
**AGPL-3.0-or-later** mit einer Zusatzklausel nach §7(b): jede modifizierte
Version muss in ihren sichtbaren rechtlichen Hinweisen den Ursprung nennen
(`Based on "Jass-NN" by Matthias, https://github.com/matthili/JCN9000`).

Was das fuer die App bedeutet:
- Kommerzielle Nutzung ist erlaubt.
- Weil die App das Modell **als Netzwerkdienst** ausliefert, greift AGPL §13:
  Wer den Code/das Modell modifiziert und als Online-Dienst betreibt, muss den
  Nutzern den Quellcode der modifizierten Version anbieten und den Ursprung
  nennen.
- Da die App ohnehin selbst unter AGPL-3.0 veroeffentlicht wird, ist das
  kongruent -- kein Lizenzkonflikt. Die `LICENSE` liegt jetzt im Release-ZIP bei.

(Wer eine MIT-Version vor diesem Schnitt bezogen hat, behaelt fuer diesen Stand
MIT. Ab v0.7.1 gilt AGPL.)

---

## 3. Neue Gewichte herunterladen

```bash
gh release download v0.7.1 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.7.1.zip
```

- **Encoder unveraendert:** `encoding_version: "3.0.0"`, 421 Dimensionen. Der
  bestehende TypeScript-Encoder fuer Kreuz/Solo passt weiterhin.
- **Modell-API unveraendert:** `{state:[batch,421], mask:[batch,36]}` ->
  `{policy:[batch,36], value:[batch,1]}`.
- Es genuegt, das TF.js-Modell auszutauschen. **Aber:** ohne den Regel-Fix aus
  Punkt 1 spielt das neue Modell gegen eine fehlerhaft eingeschraenkte
  Zugmenge -- der Fix ist die Voraussetzung dafuer, dass v0.7.1 sein Potenzial
  zeigt.

---

## Spielstaerke -- was du erwarten kannst

| Eval-Setup (paired-eval, 1000 Partien, batched-gpu) | v0.7.1 Win-Rate |
|---|---|
| vs. Kreuz-Heuristik | **79.5 %** |
| vs. eigenes Vorlaeufermodell (mcts1) | 71.4 % |

Gegenueber v0.7.0 (77.2 % vs. Heuristik) ein Plus von 2.3 Prozentpunkten, trotz
der nun korrekt groesseren legalen Zugmenge. Staerkste Modi: Slalom Unten
(67.2 %), Unten (66.4 %). Schwaechste Familie weiterhin Gumpf (~57-58 %).

---

## Smoke-Test nach Update

1. TS-Engine: neue `legal_moves`-Faelle gruen (bedienen ODER stechen; kein
   Untertrumpfen; Buur immer spielbar).
2. v0.7.1-Modell laedt im Kreuz-Modus, `encoding_version` == "3.0.0".
3. Eine komplette Kreuz-Partie laeuft ohne von der UI faelschlich gesperrte
   legale Zuege durch.

---

## Fragen oder Mismatches

Issues im NN-Repo: <https://github.com/matthili/JCN9000/issues>.
Der einzige Pflicht-Aufwand auf der App-Seite ist der `legal_moves`-Fix (Punkt 1);
Encoder und Modell-API sind unveraendert.
