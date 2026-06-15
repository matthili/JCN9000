# Update-Prompt für die Web-App "Heb ab!" — Integration von v0.5.0

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben. Er ist self-contained — die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Worum geht's

Im NN-Repository (`matthili/JCN9000`) gibt es ein neues Release **v0.5.0**. Das bringt drei Dinge mit, die in der Web-App eingebaut werden müssen:

1. **Encoder bumped auf v3.0.0** (Featurevektor 348 → 421 Dims, neue Sections, 5-dim Mode-Feld)
2. **Neue Spielvariante: Gumpf** (Trumpf-Farbe wie normaler Trumpf, Nicht-Trumpf-Farben mit invertierter Stärke)
3. **Spec-Version bumped auf 1.1.0** (additiv, kein Bruch — nur eine neue Variante)

**Wichtig**: das alte v0.4.0-Modell ist mit dem neuen Encoder **inkompatibel**. Beim Modell-Laden muss die App `encoding_version == "3.0.0"` prüfen und bei Mismatch eine klare Fehlermeldung werfen.

---

## Was du holen und wo es ist

```bash
# Release-Asset herunterladen (ZIP enthält alle Artefakte)
gh release download v0.5.0 \
    --repo matthili/JCN9000 \
    --pattern "jass-nn-*.zip"
unzip jass-nn-v0.5.0.zip
```

Inhalt des ZIPs:
- `MANIFEST.json` — Versionen + SHA256-Hashes
- `jass_rules.json` (Spec 1.1.0)
- `jass_rules.schema.json`
- `state_encoding.md` (3.0.0) — **Pflichtlektüre für die Encoder-Implementierung**
- `encoding_fixtures.json` — 15 konkrete Testfälle für TS-Encoder-Verifikation, inkl. 3 Gumpf-Fixtures (`fix_g01..fix_g03`)
- `keras/best.keras` — Trainiertes Modell, Referenz
- `tfjs/` — Konvertiertes TF.js-Modell für die Browser-Inferenz

---

## Konkrete Änderungen in der Web-App

### 1. Encoder anpassen

**Section-Layout neu (siehe `state_encoding.md` für Details):**

| Section | Offset | Größe | Neu in v3? |
|---|---|---|---|
| `own_hand` | 0..35 | 36 | – |
| `played_by_me/left/partner/right` | 36..179 | 4 × 36 | – |
| `current_trick_by_me/left/partner/right` | 180..323 | 4 × 36 | – |
| **`value_per_card`** | **324..359** | **36** | **JA** |
| **`strength_per_card`** | **360..395** | **36** | **JA** |
| `lead_suit` | 396..399 | 4 | – |
| `trump_suit` | 400..403 | 4 | Achtung: jetzt auch im Gumpf-Modus gesetzt |
| **`mode`** | **404..408** | **5** | **JA — 5 statt 4 Bits** (jetzt `[is_trumpf, is_gumpf, is_oben, is_unten, is_slalom]`) |
| `my_seat` | 409..412 | 4 | – |
| `starter_seat_relative` | 413..416 | 4 | – |
| `score_own_norm` / `score_opp_norm` | 417..418 | 1+1 | – |
| `trick_idx_norm` / `round_idx_norm` | 419..420 | 1+1 | – |

**Summe: 421 Dims (vorher 348).**

#### Berechnung der neuen Features

**`value_per_card[card_idx] = card_value(card, current_variant) / 20.0`**

Punktewerte exakt wie in der Engine:
- `TRUMPF` / `GUMPF`: Trumpf-Farbe → Buur=20, Nell=14, Ass=11, 10=10, K=4, O=3, U=2, 8/7/6=0
- `TRUMPF` / `GUMPF`: Nicht-Trumpf → Ass=11, 10=10, K=4, O=3, U=2, 9/8/7/6=0
- `OBEN` / `UNTEN`: Ass=11, 10=10, **8=8**, K=4, O=3, U=2, 9/7/6=0

**`strength_per_card[card_idx]`** = Kraftpunkt 1..18 unter aktueller Variante + Lead-Suit, geteilt durch 18.0:

| Variante | Karte in Trumpf-Farbe | Karte in Lead-Suit (Nicht-Trumpf) | Karte in anderer Farbe |
|---|---|---|---|
| TRUMPF | `10 + TRUMP_RANK_ORDER[rank]` (Buur=18, Nell=17, A=16, K=15, O=14, 10=13, 8=12, 7=11, 6=10) | `1 + rank_id` (6=1, …, A=9) | identisch zu mittlerer Spalte |
| GUMPF | identisch mit TRUMPF | `1 + (8 - rank_id)` (6=9, …, A=1) — **invertiert** | identisch mit mittlerer Spalte |
| OBEN | – | `10 + rank_id` (6=10, …, A=18) | `1 + rank_id` (6=1, …, A=9) |
| UNTEN | – | `10 + (8 - rank_id)` (A=10, …, 6=18) | `1 + (8 - rank_id)` (A=1, …, 6=9) |

**Wichtig für OBEN/UNTEN bei leerem Stich** (Anspielmoment, kein Lead): jede Karte bekommt den Lead-Boost, als wäre ihre eigene Suit der Lead. Spiegelt die „Anspiel-Kraft" wider.

**Mode-Feld** ist jetzt 5-dim: `[is_trumpf, is_gumpf, is_oben, is_unten, is_slalom_flag]`.
- Genau **eines** der ersten vier Bits ist gesetzt (one-hot über die effektive Stich-Variante).
- `is_slalom_flag` orthogonal — kann zusätzlich aktiv sein, wenn die Ansage Slalom war.

**Trump-Suit-One-hot** ist jetzt auch bei `is_gumpf=1` gesetzt (vorher nur bei `is_trumpf`).

#### Verifikation

Pflicht-Check: alle 15 Fixtures in `encoding_fixtures.json` müssen die TS-Implementierung exakt reproduzieren (`atol=1e-5`). Davon sind drei neu für Gumpf:
- `fix_g01_gumpf_leerer_stich_anspiel`
- `fix_g02_gumpf_6_sticht_nichttrumpf`
- `fix_g03_gumpf_trumpf_normalfall`

### 2. Modell-Lade-Check härter machen

```typescript
// Pseudo-Code
const manifest = await loadManifest("MANIFEST.json");
if (manifest.encoding_version !== "3.0.0") {
    throw new Error(
        `Modell-Encoder ${manifest.encoding_version} ist mit dieser App nicht kompatibel. ` +
        `Erwartet: 3.0.0`
    );
}
```

### 3. Gumpf in der Spielablauf-Logik

Die Engine-Seite muss `Gumpf` als zusätzliche Ansage akzeptieren. Regel-Verhalten:
- Wie Trumpf bezüglich Buur-Ausnahme, Kein-Untertrumpfen, Stöcke (Trumpf-O + Trumpf-K)
- Wie Geiss bezüglich Nicht-Trumpf-Stärke-Reihenfolge (6 sticht alles in Lead-Farbe)
- Wertpunkte wie Trumpf (8er=0)
- Slalom **darf nicht** mit Gumpf kombiniert werden (in `Announcement` validieren)

Die deklarativen Regeln sind in `jass_rules.json` unter `variants.gumpf`. Schema in `jass_rules.schema.json` hat neu `GumpfVariant` als `$def`.

### 4. UI-Anpassungen

- Ansage-Menü um „Gumpf" + Farbwahl erweitern (analog zu Trumpf)
- Spielfeld-Anzeige sollte signalisieren, wenn Gumpf aktiv ist (z. B. Trumpf-Indikator mit „G"-Suffix)
- Score-Anzeige unverändert

---

## Spielstärke-Hinweis

Das v5-Modell ist **so stark wie der Heuristik-Bot, nicht stärker** (Eval: 50.4 % vs. 49.5 % Win-Rate, 2000 Spiele). Die Geiss-Schwäche des v4-Modells (38 %) ist behoben. Über das Heuristik-Niveau hinauszukommen ist Aufgabe der RL-Iteration (v0.6.0 in Vorbereitung).

Für den Endnutzer: das v5-Modell spielt sauber und gibt ihm einen ordentlichen Gegner, aber keinen Über-Bot. Falls dir das wichtig ist, kannst du den HeuristicPlayer auch direkt aus der Spec ableiten und im Browser laufen lassen — die Spielstärke ist vergleichbar und du sparst dir die TF.js-Lade-/Inferenz-Zeit.

---

## Was sich NICHT geändert hat

- `card_index = suit_id * 9 + rank_id` (gleiche Formel)
- Aktionsraum = 36 (gleich)
- Maske-Semantik: `1.0 = legal`, `0.0 = illegal` (gleich)
- Modell-API: `{state: [batch, 421], mask: [batch, 36]} → {policy: [batch, 36], value: [batch, 1]}` (Shapes änderten sich nur für `state`)
- Relative-Position-Formel: `(other_seat - my_seat) mod 4`

---

## Fragen oder Probleme?

Fixture-Verifikation ist der definitive Lakmustest: wenn deine TS-Encoder-Implementierung alle 15 Fixtures byte-genau reproduziert, ist sie korrekt. Bei Mismatch zuerst auf die `value_per_card`- oder `strength_per_card`-Section schauen — das sind die einzigen "echten" Berechnungen im Encoder.
