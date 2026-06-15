# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.9.0 (Bodensee-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/JCN9000`) ist Release **v0.9.0**
veroeffentlicht. Es bringt ein **eigenes Modell fuer Bodensee-Jass** -- die
2-Spieler-Variante mit Tisch-Mechanik.

**Wichtig:** v0.9.0 ist ein **drittes, eigenstaendiges Modell** neben:
- v0.7.0 (Kreuz-Jass, 4 Spieler, Teams)
- v0.8.0 (Solo-Jass, 4 Spieler, jeder gegen jeden)
- **v0.9.0 (Bodensee-Jass, 2 Spieler, Tisch-Mechanik)**

Die drei Modelle sind **nicht austauschbar**. Bodensee hat sogar einen
**eigenen Encoder** -- der TypeScript-Encoder fuer Kreuz/Solo funktioniert
hier NICHT.

---

## Das Bodensee-Spielprinzip (fuer die App-Logik)

| Aspekt | Wert |
|---|---|
| Spieler | 2 |
| Karten pro Spieler | 18: 6 Hand (privat) + 6 sichtbarer Tisch + 6 verdeckter Tisch |
| Stiche pro Runde | 18 |
| Tisch-Mechanik | Spieler waehlt pro Zug: Karte aus Hand ODER von sichtbaren Tisch-Karten. Wird eine Tisch-Karte gespielt, deckt die verdeckte darunter auf. |
| Bedienzwang | gilt ueber Hand + sichtbarem Tisch gemeinsam |
| Schieben | gibt es nicht |
| Weisen / Stoecke | gibt es nicht |
| Default-Ziel | 500 Punkte |
| Letzter Stich / Matsch | +5 / +100 wie ueblich |

Vollstaendige Engine-Referenz: `jass_engine/bodensee/` im NN-Repo.

---

## Was herunterladen

```bash
gh release download v0.9.0 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.9.0.zip
```

Inhalt:
- `MANIFEST.json` -- mit `team_mode: "bodensee_2p"` und `encoding_version: "bodensee_1.0.0"`
- `keras/best.keras`, `tfjs/`
- `bodensee_state_encoding.md` -- **Pflichtlektuere fuer den TS-Encoder**

---

## Der NEUE Encoder (das ist der Hauptaufwand)

Bodensee braucht einen **eigenen TypeScript-Encoder**. Der v3.0.0-Encoder fuer
Kreuz/Solo passt nicht -- andere Spielerzahl, andere Sektionen.

**Encoding-Version: `bodensee_1.0.0`, 291 Dimensionen.**

Vollstaendiges Layout in `bodensee_state_encoding.md` (im Release-ZIP). Kern-
Unterschiede zu v3.0.0:
- 2 Spieler statt 4 -> keine `played_by_partner`/`played_by_right`-Sektionen
- Neue Sektionen fuer die Tisch-Mechanik: `own_visible_table`,
  `own_hidden_table_mask`, `opp_visible_table`, `opp_hidden_table_count`
- `opp_hand_count` explizit als One-Hot (7 Bit)
- Eine konsolidierte `opp_lead_card`-Sektion + `i_am_leading`-Bit statt 4
  Trick-Slots

Karten-Index bleibt identisch: `card_index = suit_id * 9 + rank_id`.

**Modell-API unveraendert:** `{state: [batch, 291], mask: [batch, 36]}` ->
`{policy: [batch, 36], value: [batch, 1]}`.

---

## Modell-Lade-Logik

```typescript
// Pseudo-Code
async function loadModelForGame(gameType: "kreuz" | "solo" | "bodensee") {
    const tag = {
        kreuz: "v0.7.0",
        solo: "v0.8.0",
        bodensee: "v0.9.0",
    }[gameType];
    const manifest = await loadManifest(`/models/${tag}/MANIFEST.json`);

    const expectedEncoding = gameType === "bodensee" ? "bodensee_1.0.0" : "3.0.0";
    if (manifest.encoding_version !== expectedEncoding) {
        throw new Error(
            `Encoder ${manifest.encoding_version} passt nicht zu ${gameType}`
        );
    }
    return await tf.loadLayersModel(`/models/${tag}/tfjs/model.json`);
}
```

---

## Spielstaerke -- was du erwarten kannst

| Eval-Setup (4000 paired-Partien) | v0.9.0 Win-Rate |
|---|---|
| vs. Bodensee-Heuristik | **72.6 %** |
| vs. Bootstrap-Modell (Heuristik-Klon) | 74.0 % |

2-Spieler-Baseline ist 50 %. v0.9.0 liegt ~22 Prozentpunkte darueber. Avg-Score
524.5 vs. 435.9. Pro Variante sehr gleichmaessig (69.9 % - 74.6 %), kein
Schwachpunkt.

**Nicht-validiert:** Slalom-Varianten (kamen in der Eval nicht vor) und
Spielstaerke gegen erfahrene menschliche Bodensee-Jasser.

---

## Smoke-Test nach Update

1. v0.9.0-Modell laedt im Bodensee-Spielmodus
2. Eine komplette Bodensee-Partie: Tisch-Mechanik funktioniert (Tisch-Karte
   spielen deckt verdeckte auf), Bedienzwang ueber Hand+Tisch korrekt
3. Encoder-Verifikation: sobald `bodensee_encoding_fixtures.json` verfuegbar ist
   (folgt), alle Fixtures byte-genau reproduzieren

---

## Fragen oder Mismatches

Issues im NN-Repo: <https://github.com/matthili/JCN9000/issues>.
Der groesste Integrations-Aufwand ist der neue Bodensee-Encoder im TypeScript-
Port -- `bodensee_state_encoding.md` ist dafuer die maßgebliche Referenz.
