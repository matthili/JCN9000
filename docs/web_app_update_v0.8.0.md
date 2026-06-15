# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.8.0 (Solo-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/JCN9000`) ist Release **v0.8.0** veroeffentlicht.
Das ist ein **eigenes Modell fuer Solo-Jass** (4 Spieler, jeder gegen jeden).

**Wichtig:** v0.8.0 **ersetzt nicht** v0.7.0. v0.7.0 ist fuer Kreuz-Jass (Team-Spiel)
weiterhin das aktuelle Modell. Die beiden Modelle haben fundamental unterschiedliche
Strategien gelernt (Team-Kooperation vs. Egospiel) und sind **nicht austauschbar**.

**Was hinzukommt:**

1. **Neue Spielart "Solo-Jass"** in der App-Auswahl
2. **Eigenes Modell** v0.8.0 wird geladen, wenn der User Solo waehlt
3. **MANIFEST.json** trägt zusätzlich `team_mode: "solo"` als Indikator

**Was kompatibel bleibt:**

- Encoding-Version 3.0.0 (identisch zu v0.5.0 - v0.7.0)
- Modell-Architektur (Input/Output-Shape, MaskBias-Layer)
- **Spec-Version 1.2.0 unveraendert.** Die Solo-Spielart unterscheidet sich auf
  Spiel-Ebene (Teams, Schieben, Target-Score, Matsch-Vergabe), nicht auf
  Variant-Regel-Ebene. Karten-Werte, Trumpf-Logik, Stich-Bewertung sind
  identisch -- und die sind in der Spec beschrieben. Die Solo-spezifischen
  Regeln dokumentieren wir in der Modell-Karte und diesem Briefing, nicht in
  `jass_rules.json`.

Das v0.7.0-Modell fuer Kreuz-Jass bleibt produktiv. Die App muss anhand der
gewaehlten Spielart das passende Modell laden.

---

## Was herunterladen

```bash
gh release download v0.8.0 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.8.0.zip
```

Inhalt:
- `MANIFEST.json` -- jetzt mit Feld `team_mode: "solo"`
- `jass_rules.json` (Spec 1.3.0, mit Solo-Variant)
- `jass_rules.schema.json`
- `state_encoding.md` (3.0.0, unveraendert)
- `encoding_fixtures.json` -- 15 Fixtures (unveraendert)
- `keras/best.keras` -- Referenz-Modell
- `tfjs/` -- Browser-Modell fuer Solo

---

## Was sich beim Spielablauf unterscheidet

### Solo-Regeln (gegenueber Kreuz-Jass)

| Aspekt | Kreuz-Jass | Solo-Jass |
|---|---|---|
| Anzahl Spieler | 4 (Teams 0+2 vs 1+3) | 4 (jeder fuer sich) |
| Default-Ziel | 1000 Punkte | **500 Punkte** (anpassbar, mind. 500) |
| Schieben | erlaubt (ab Runde 2) | **nicht erlaubt** |
| Weisen | Team-Weis (hoechstes Team gewinnt) | nur hoechster Spieler gewinnt seine Weisen |
| Matsch | Team-Bonus +100 | **+100 fuer den einzelnen Spieler** mit 9/9 Stichen |
| Stoecke | Team bekommt +20 | **+20 fuer den Stockhalter persoenlich** |
| Schmieren | gelernte Strategie | **nicht erwuenscht** (kein Partner) |

### Engine-seitig

Die Solo-Spielart ist in der Engine ueber zwei Stellschrauben implementiert:

1. `teams=[0, 1, 2, 3]` -- jeder Spieler bekommt seine eigene "Team"-ID. Die existierende
   Punkteaggregation, Weisen-Vergleich und Matsch-Erkennung verhalten sich dadurch
   automatisch korrekt (jeder bekommt nur seine eigenen Punkte).
2. `allow_push=False` -- Schiebe-Mechanik komplett deaktiviert.

Wenn deine App-Engine bereits eine analoge "teams"-Parametrisierung hat, ist die
Solo-Variante ein One-Liner. Sonst muss die Score-Aufzeichnung pro Spieler statt
pro Team gefuehrt werden.

---

## Modell-Lade-Logik in der App

```typescript
// Pseudo-Code
async function loadModelForGame(gameType: "kreuz" | "solo") {
    const tag = gameType === "kreuz" ? "v0.7.0" : "v0.8.0";
    const manifest = await loadManifest(`/models/${tag}/MANIFEST.json`);

    if (manifest.encoding_version !== "3.0.0") {
        throw new Error(
            `Encoder ${manifest.encoding_version} nicht kompatibel. Erwartet: 3.0.0`
        );
    }

    // Sanity-Check: passt der team_mode zur Spielart?
    const expectedTeamMode = gameType === "kreuz" ? "team" : "solo";
    if (manifest.team_mode && manifest.team_mode !== expectedTeamMode) {
        throw new Error(
            `Modell-team_mode ${manifest.team_mode} passt nicht zu Spielart ${gameType}`
        );
    }

    return await tf.loadLayersModel(`/models/${tag}/tfjs/model.json`);
}
```

Der `team_mode`-Check ist optional, aber empfohlen -- er verhindert, dass das
Kreuz-Jass-Modell versehentlich im Solo-Modus geladen wird (oder umgekehrt) und
dann unsinnige Karten waehlt.

---

## Warum die Spec NICHT bumped

Die Spec `jass_rules.json` beschreibt **Karten, Werte, Reihenfolgen, Regeln pro
Variante** — also alles, was sich pro Spielzug entscheidet. Solo-Jass aendert
keinen einzigen dieser Punkte: Trumpf-Buur ist 20 Punkte, Geiss-6 ist die
staerkste in der Lead-Farbe, Buur darf immer gespielt werden, etc.

Was Solo aendert, ist auf einer hoeheren Ebene:
- Team-Konfiguration (4 statt 2)
- Default-Punkteziel
- Schiebe-Mechanik (deaktiviert)
- Wer bekommt den Weis/Matsch/Stoecke-Bonus (Spieler statt Team)

Diese Punkte sind **App-Konfigurationen**, keine **Spielregeln im Spec-Sinne**.
Sie sind hier in dieser Datei dokumentiert und im `MANIFEST.json` durch
`team_mode` signalisiert. Wenn der TS-Encoder die Spec liest, kann er sie 1:1
weiterverwenden -- die einzige Anpassung in der App-Logik betrifft die
Punkte-Aggregation pro Spieler statt pro Team.

---

## Spielstaerke -- was du erwarten kannst

| Eval-Setup | Win-Rate v0.8.0 |
|---|---|
| vs. Solo-Phase-1 + 2x SoloHeuristik (3400 paired-Partien) | **45.4 %** |
| Phase-1-Modell im selben Eval | 21.1 % |
| SoloHeuristik | 16.8 % |

Random-Baseline waere 25 % pro Rolle. v0.8.0 liegt 20 Prozentpunkte darueber.
Eval-Ziel war >= 35 %, erreicht mit 10 Punkten Puffer.

Avg-Score pro Partie: 462.3 (v0.8.0) vs. 392.0 (Phase 1) vs. 381.1 (Heuristik).
Matsch-Rate pro Runde: v0.8.0 macht **16x oefter Matsch** als die Heuristik.

Avg. Score pro Partie, Matsch-Rate pro Runde -- alle Detail-Zahlen kommen
nach abgeschlossenem Training. Die Modell-Karte (`docs/model_cards/v0.8.0.md`)
fuehrt sie pro Variante auf.

---

## Was sich NICHT geaendert hat

- Encoding-Version 3.0.0 -- TS-Encoder bleibt
- Featurevektor-Layout (421 Dims)
- `card_index = suit_id * 9 + rank_id`
- Aktionsraum = 36
- Maske-Semantik: `1.0 = legal`, `0.0 = illegal`
- Modell-API: `{state, mask} -> {policy, value}`

---

## Smoke-Test nach Update

1. Encoder-Fixtures alle 15 Tests reproduzieren weiterhin byte-genau (keine
   Aenderung am Encoder)
2. v0.7.0-Modell laedt weiterhin im Kreuz-Jass-Spielmodus
3. v0.8.0-Modell laedt im Solo-Spielmodus, weigert sich (idealerweise) im
   Kreuz-Modus zu laden (oder umgekehrt)
4. Eine komplette Solo-Partie zu Ende spielen, Punkte werden pro Spieler korrekt
   gezaehlt, kein Schieben angeboten
5. Optional: 10-20 reale Solo-Partien gegen v0.8.0, subjektives Feedback

---

## Fragen oder Mismatches

Issues bitte im NN-Repo melden (<https://github.com/matthili/JCN9000/issues>).
Beim Encoder-Sanity-Test (Fixtures) sollte nichts abweichen, da derselbe Encoder
wie bei v0.7.0 verwendet wird.
