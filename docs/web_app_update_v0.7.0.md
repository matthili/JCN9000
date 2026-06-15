# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.7.0

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben. Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

Im NN-Repository (`matthili/JCN9000`) ist Release **v0.7.0** veroeffentlicht. Es ersetzt v0.6.0 als Produktiv-Modell.

**Was sich geaendert hat:**
1. **Spielstaerke deutlich hoeher** (77.2 % Win-Rate gegen v0.5.0 in 4000 gepaarten Partien)
2. **Spec auf 1.2.0** (additive Erweiterung, keine breaking changes)

**Was kompatibel bleibt:**
- Encoding-Version **3.0.0** (identisch zu v0.5.0 und v0.6.0)
- Modell-Architektur (Input-Shape, Output-Heads, Custom-Layer `MaskBias`)
- ZIP-Struktur und Asset-Pfad
- Lade-Code in der App

Im besten Fall: ZIP herunterladen, austauschen, fertig.

---

## Was herunterladen

```bash
gh release download v0.7.0 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.7.0.zip
```

Inhalt:
- `MANIFEST.json` -- Versionen + SHA256-Hashes
- `jass_rules.json` (Spec 1.2.0)
- `jass_rules.schema.json`
- `state_encoding.md` (3.0.0)
- `encoding_fixtures.json` -- 15 Fixtures
- `keras/best.keras` -- Referenz
- `tfjs/` -- TF.js-Modell fuer den Browser

---

## Spec 1.2.0 -- was ist neu

Drei additive Bereiche in `jass_rules.json`. Bestehende Felder unveraendert.

### 1. `scoring.score_composition`

Neuer deklarativer Block, der erklaert, woraus sich der pro Runde gutgeschriebene Score zusammensetzt (Stichpunkte + letzter Stich + Weisen + Stoecke + optional Matsch-Bonus). Nur Dokumentation, keine Verhaltensaenderung.

### 2. `round_flow.play_order_anchor`

Praezisiert, dass beim Schieben der **urspruengliche Ansager** anspielt -- nicht der Geschobene. Das war schon immer so in der Engine, ist jetzt aber explizit in der Spec.

### 3. `trick.card_ordering`

Definiert, in welcher Reihenfolge die 4 Karten eines Stichs gespeichert werden (Anspieler an Position 0). Ist Konvention, die der Encoder annimmt.

**Was in der App zu tun ist:** wenn dein TS-Spielablauf bereits konsistent mit diesen Punkten ist (was sehr wahrscheinlich ist), nichts. Sonst gegen den Spec-Block abgleichen.

---

## Modell-Lade-Check

Versionscheck wie bisher, nur die Versionsnummer der Spec anheben:

```typescript
// Pseudo-Code
const manifest = await loadManifest("MANIFEST.json");
if (manifest.encoding_version !== "3.0.0") {
    throw new Error(`Encoder ${manifest.encoding_version} nicht kompatibel. Erwartet: 3.0.0`);
}
// Spec-Check kann jetzt 1.2.0 erlauben:
if (!manifest.spec_version.startsWith("1.")) {
    throw new Error(`Spec ${manifest.spec_version} nicht kompatibel. Erwartet: 1.x`);
}
```

---

## Spielstaerke -- was du erwarten kannst

| Eval-Setup | Win-Rate v0.7.0 |
|---|---|
| vs. v0.5.0 (alter NN-Bot) | **77.2 %** |
| vs. Heuristik | (siehe Eval-Logs) |

Avg. 1025 zu 829 Punkte pro Partie. Matsch-Rate dreimal so hoch wie bei v0.5.0 (4.78 % vs. 1.59 %). Pro Variante: zwischen 54 % (Gumpf Schelle) und 66 % (Slalom Unten).

**Bekannte Schwaeche:** Gumpf ist mit 54-57 % Win-Rate die schwaechste Variante. Andere Varianten alle ueber 56 %, Slalom besonders stark (65-66 %).

**Nicht-validiert:** Spielstaerke gegen menschliche Top-Spieler. Die 77.2 % sind gegen das eigene Vorgaengermodell.

---

## Was sich NICHT geaendert hat

- Encoding-Version 3.0.0 -- TS-Encoder unveraendert
- Featurevektor-Layout (421 Dims), `value_per_card`-/`strength_per_card`-Sections, 5-dim Mode-Feld
- `card_index = suit_id * 9 + rank_id`
- Aktionsraum = 36
- Maske-Semantik: `1.0 = legal`, `0.0 = illegal`
- Modell-API: `{state: [batch, 421], mask: [batch, 36]} -> {policy: [batch, 36], value: [batch, 1]}`
- Lade-Code im Browser

---

## Smoke-Test nach Update

1. Encoder-Fixtures: alle 15 Fixtures aus `encoding_fixtures.json` reproduzieren weiterhin byte-genau.
2. App startet, Modell laedt ohne Fehler.
3. Eine komplette Partie gegen das neue Modell spielen -- subjektive Probe, ob es "klueger" wirkt.
4. Optional: 20-30 Partien mit Auswertung der Win-Rate pro Variante, um zu verifizieren, dass die App den richtigen Verbesserungstrend zeigt.

---

## Fragen oder Mismatches

Issues bitte im NN-Repo melden (<https://github.com/matthili/JCN9000/issues>). Bei Encoder-Drift zuerst den Fixture-Test pruefen -- der zeigt sofort, wo das TS-Encoding vom Python-Encoding abweicht.
