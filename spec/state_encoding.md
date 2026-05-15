# State-Encoding-Spezifikation

**Version**: 3.0.0
**Breaking Change zu 2.0.0**:
- Featurevektor wächst von 348 → 421 Dimensionen.
- Zwei neue Sections `value_per_card` (36) und `strength_per_card` (36) liefern dem NN pro Karte den unter der aktuellen Variante gültigen Wert- bzw. Kraftwert vorberechnet. Damit muss das Netz die multiplikative Interaktion *Karte × Variante × Lead-Suit* nicht mehr selbst lernen.
- `mode` wächst von 4 → 5 Bits: zusätzlich `is_gumpf`. Damit ist die neue Gumpf-Variante (Trumpf-Farbe normal, Nicht-Trumpf invertiert) explizit kodiert.
- `trump_suit` ist jetzt auch im Gumpf-Modus gesetzt (nicht nur in Trumpf).

Dieses Dokument beschreibt **exakt**, wie ein Spielzustand in den Eingabevektor des
neuronalen Netzes umgewandelt wird. Eine TypeScript-Implementierung muss die hier
beschriebene Reihenfolge und Semantik 1:1 spiegeln.

Zur Verifikation existiert `spec/fixtures/encoding_fixtures.json`: konkrete
Spielzustände mit erwarteten Vektoren.

---

## Eingabe-Tensoren

| Name | Shape | dtype | Werte |
|---|---|---|---|
| `state` | `[batch, 421]` | `float32` | Featurevektor, alle Werte in `[0.0, 1.0]` |
| `mask`  | `[batch, 36]`  | `float32` | Aktionsmaske, `1.0` = legal, `0.0` = illegal |

Das Modell liefert zwei Outputs (Multi-Head):

| Output | Shape | dtype | Werte |
|---|---|---|---|
| `policy` | `[batch, 36]` | `float32` | Wahrscheinlichkeitsverteilung über Karten |
| `value`  | `[batch, 1]`  | `float32` | erwartete Punkte-Differenz, in `[-1, +1]` (tanh) |

Illegale Aktionen sind durch die Maske auf praktisch 0 gedrückt.

## Karten-Index

```
card_index = suit_id * 9 + rank_id
```

| Suit | id |
|---|---|
| EICHEL  | 0 |
| SCHELLE | 1 |
| HERZ    | 2 |
| LAUB    | 3 |

| Rank | id |
|---|---|
| SECHS  | 0 |
| SIEBEN | 1 |
| ACHT   | 2 |
| NEUN   | 3 |
| ZEHN   | 4 |
| UNTER  | 5 |
| OBER   | 6 |
| KOENIG | 7 |
| ASS    | 8 |

Beispiele: Eichel-Sechs → 0; Eichel-Ass → 8; Schelle-Sechs (Weli) → 9; Laub-Ass → 35.

## Relative Spielerposition

Vom eigenen Sitz (`player_idx`) aus zählend, im Uhrzeigersinn:

| Relative Position | Bedeutung |
|---|---|
| 0 | ich selbst |
| 1 | links neben mir (nächster Spieler im Uhrzeigersinn) |
| 2 | gegenüber (mein Partner im Kreuz-Jass) |
| 3 | rechts neben mir (vorletzter im Uhrzeigersinn) |

Formel: `relative_position = (other_seat - my_seat) mod num_players`

## Aufbau des 421-dim Featurevektors

Der Vektor besteht aus 20 zusammenhängenden Sections. Jeder Wert ist `float32`
in `[0.0, 1.0]`.

| Section | Offset | Größe | Inhalt |
|---|---|---|---|
| `own_hand`                       |   0..35  | 36 | one-hot: Karten in der eigenen Hand |
| `played_by_me`                   |  36..71  | 36 | one-hot: Karten, die ich in abgeschlossenen Stichen gespielt habe |
| `played_by_left`                 |  72..107 | 36 | Karten, die der linke Spieler (rel=1) gespielt hat |
| `played_by_partner`              | 108..143 | 36 | Karten, die mein Partner (rel=2) gespielt hat |
| `played_by_right`                | 144..179 | 36 | Karten, die der rechte Spieler (rel=3) gespielt hat |
| `current_trick_by_me`            | 180..215 | 36 | Karte, die ich im aktuellen Stich gespielt habe |
| `current_trick_by_left`          | 216..251 | 36 | Karte des linken Spielers im aktuellen Stich |
| `current_trick_by_partner`       | 252..287 | 36 | Karte des Partners im aktuellen Stich |
| `current_trick_by_right`         | 288..323 | 36 | Karte des rechten Spielers im aktuellen Stich |
| `value_per_card`                 | 324..359 | 36 | pro Karte: `card_value(card, variant) / 20.0` (NEU v3) |
| `strength_per_card`              | 360..395 | 36 | pro Karte: Kraftpunkt 1..18 / 18.0 unter aktueller Variante + Lead (NEU v3) |
| `lead_suit`                      | 396..399 |  4 | one-hot über Lead-Farbe; alle Nullen wenn Stich leer |
| `trump_suit`                     | 400..403 |  4 | one-hot über Trumpf-Farbe (bei `trumpf` UND `gumpf`) |
| `mode`                           | 404..408 |  5 | `[is_trumpf, is_gumpf, is_oben, is_unten, is_slalom_flag]` |
| `my_seat`                        | 409..412 |  4 | one-hot über eigenen Sitz 0..3 (absolut) |
| `starter_seat_relative`          | 413..416 |  4 | one-hot über Anspieler des aktuellen Stichs **relativ zu mir** |
| `score_own_norm`                 | 417..417 |  1 | eigene Team-Punkte / 1000, gekappt bei 1.0 |
| `score_opp_norm`                 | 418..418 |  1 | Gegner-Punkte / 1000, gekappt bei 1.0 |
| `trick_idx_norm`                 | 419..419 |  1 | aktueller Stich-Index 0..8, dividiert durch 9 |
| `round_idx_norm`                 | 420..420 |  1 | aktuelle Runde, dividiert durch 20 und gekappt bei 1.0 |

**Summe**: 11 × 36 + 4 + 4 + 5 + 4 + 4 + 4 × 1 = 396 + 21 + 4 = **421**.

### Detail: `value_per_card` (Offsets 324..359)

Für jede Karte `c` (Index 0..35) wird `card_value(c, variant) / 20.0` gespeichert.
Die Punktewerte stammen direkt aus den Regelkonstanten:

| Variante | Trumpf-Farbe | Nicht-Trumpf-Farbe |
|---|---|---|
| TRUMPF | Buur=20, Nell=14, sonst: A=11,10=10,K=4,O=3,U=2,9/8/7/6=0 | A=11,10=10,K=4,O=3,U=2,9/8/7/6=0 |
| GUMPF  | identisch mit TRUMPF | identisch mit TRUMPF (Wertpunkte; nur Stärke unterscheidet sich) |
| OBEN   | — | A=11,10=10,8=8,K=4,O=3,U=2,sonst=0 |
| UNTEN  | — | A=11,10=10,8=8,K=4,O=3,U=2,sonst=0 |

### Detail: `strength_per_card` (Offsets 360..395)

Für jede Karte `c` wird der unter der aktuellen Variante geltende Kraftpunkt 1..18 / 18.0 gespeichert. Höher = stärker.

**TRUMPF / GUMPF** (keine Lead-Suit-Abhängigkeit für die Trumpf-Farbe):
- Trumpf-Farbe: `10 + TRUMP_RANK_ORDER[rank]` → Buur=18, Nell=17, Ass=16, König=15, Ober=14, 10=13, 8=12, 7=11, 6=10.
- Nicht-Trumpf bei TRUMPF: `1 + int(rank)` → 6=1, 7=2, 8=3, 9=4, 10=5, U=6, O=7, K=8, A=9 (aufsteigend).
- Nicht-Trumpf bei GUMPF: `1 + (8 - int(rank))` → 6=9, 7=8, 8=7, 9=6, 10=5, U=4, O=3, K=2, A=1 (invertiert).

**OBEN / UNTEN** (Lead-Suit-abhängig):
- Karte in Lead-Suit bei OBEN: `10 + int(rank)` → 6=10, …, A=18.
- Karte in Lead-Suit bei UNTEN: `10 + (8 - int(rank))` → A=10, …, 6=18.
- Karte in Nicht-Lead-Suit: gleiche Reihenfolge wie für Lead, aber `1 + …` statt `10 + …`.
- Wenn der aktuelle Stich leer ist (Anspielmoment): die *eigene* Suit der Karte wird als hypothetischer Lead angesehen, d.h. jede Karte bekommt den 10..18-Boost. Das spiegelt die Anspiel-Kraft wider.

Diese Tabelle entspricht 1:1 der vom Domänenexperten gepflegten CSV-Tabelle (Wertpunkte/Kraftpunkte für jede Variante).

### Detail: `mode` (Offsets 404..408)

Das `mode`-Feld kodiert die **effektive Variante des aktuellen Stichs**, ergänzt um ein Flag, ob die ursprüngliche Ansage Slalom war.

| Index | Bit | Bedeutung |
|---|---|---|
| 404 | `is_trumpf` | 1 wenn aktueller Stich Trumpf-Modus |
| 405 | `is_gumpf`  | 1 wenn aktueller Stich Gumpf-Modus |
| 406 | `is_oben`   | 1 wenn aktueller Stich Bock-Modus |
| 407 | `is_unten`  | 1 wenn aktueller Stich Geiss-Modus |
| 408 | `is_slalom_flag` | 1 wenn die Ansage Slalom war (unabhängig vom aktuellen Stich-Modus) |

Bei `is_trumpf` / `is_gumpf` / `is_oben` / `is_unten` ist genau **eines** der vier Bits gesetzt (one-hot zwischen den vieren); `is_slalom_flag` ist orthogonal und kann nur zusätzlich zu `is_oben` oder `is_unten` aktiv sein.

### Detail: `trump_suit` (Offsets 400..403)

Bei `is_trumpf=1` oder `is_gumpf=1` ist genau ein Bit gesetzt. In allen anderen Modi sind alle vier Bits 0.

### Detail: Spieler-Zuordnung in Played/Current-Trick

Pro Stich (sowohl abgeschlossen als auch laufend) wird die `starter`-Position genutzt, um aus der Karten-Position im Stich die Spieler-Identität zu rekonstruieren:

```
spieler_der_karte_position_k = (starter + k) mod num_players
relative_position = (spieler_der_karte_position_k - my_seat) mod num_players
```

### Detail: leerer Stich

Wenn der aktuelle Stich noch keine Karte enthält, sind alle 4 `current_trick_by_*`-Blöcke + `lead_suit` komplett 0. `strength_per_card` nutzt in OBEN/UNTEN den hypothetischen Lead (siehe oben).

## Aktionsmaske

Die Aktionsmaske `mask` hat 36 Bits, einer pro Karte:

- `mask[i] = 1.0` → Karte `i` ist legal spielbar
- `mask[i] = 0.0` → Karte `i` ist illegal

Im Modell wird die Maske durch eine fixe Layer in den Logits-Bias übertragen:

```
masked_logits = logits + (1.0 - mask) * -1e9
policy = softmax(masked_logits)
```

## Konsistenzanforderungen für TypeScript-Port

1. **Karten-Index**: `card_index = suit_id * 9 + rank_id` — keine Abweichungen.
2. **Section-Reihenfolge** und **Section-Größen** sind fix (siehe Tabelle oben).
3. **Relative Position**: Formel `(other_seat - my_seat) mod 4`.
4. **Normalisierung**: Werte 0 bis 1, gekappt wie spezifiziert. `value_per_card` durch 20.0, `strength_per_card` durch 18.0.
5. **dtype**: `float32`, nicht `float64`.
6. **Maske**: float32 mit Werten exakt 0.0 oder 1.0.
7. **Fixture-Test**: Die TS-Implementierung muss alle (state → expected_vector)-Paare aus `spec/fixtures/encoding_fixtures.json` exakt reproduzieren.

## Versionierung

Diese Spezifikation ist Version **3.0.0**. Vorgängerversionen:

- **1.0.0**: 132-dim Featurevektor ohne Spieler-Zuordnung.
- **2.0.0**: 348-dim mit spieler-positionierter History.

**Modelle, die mit Version 2.0.0 oder älter trainiert wurden, sind mit dieser Encoder-Version nicht kompatibel** — der Featurevektor hat eine andere Form.

Beim Modell-Laden muss die Web-App prüfen:

```pseudocode
if model.metadata.encoding_version != "3.0.0":
    throw IncompatibleModelError(...)
```
