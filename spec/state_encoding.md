# State-Encoding-Spezifikation

**Version**: 2.0.0
**Breaking Change zu 1.0.0**: Played-History und Current-Trick sind jetzt **pro Spieler-Position** kodiert (relativ zu mir), nicht mehr als gemeinsames Set. Damit kann das NN lernen, wer welche Karte gespielt hat — Voraussetzung für klassisches Opponent Modeling ("Spieler X hat keinen Buur, sonst hätte er gestochen").

Dieses Dokument beschreibt **exakt**, wie ein Spielzustand in den Eingabevektor des
neuronalen Netzes umgewandelt wird. Eine TypeScript-Implementierung muss die hier
beschriebene Reihenfolge und Semantik 1:1 spiegeln.

Zur Verifikation existiert `spec/fixtures/encoding_fixtures.json`: konkrete
Spielzustände mit erwarteten Vektoren.

---

## Eingabe-Tensoren

| Name | Shape | dtype | Werte |
|---|---|---|---|
| `state` | `[batch, 348]` | `float32` | Featurevektor, alle Werte in `[0.0, 1.0]` |
| `mask` | `[batch, 36]` | `float32` | Aktionsmaske, `1.0` = legal, `0.0` = illegal |

Das Modell liefert zwei Outputs (Multi-Head):

| Output | Shape | dtype | Werte |
|---|---|---|---|
| `policy` | `[batch, 36]` | `float32` | Wahrscheinlichkeitsverteilung über Karten |
| `value` | `[batch, 1]` | `float32` | erwartete Punkte-Differenz, in `[-1, +1]` (tanh) |

Illegale Aktionen sind durch die Maske auf praktisch 0 gedrückt.

## Karten-Index

```
card_index = suit_id * 9 + rank_id
```

| Suit | id |
|---|---|
| EICHEL | 0 |
| SCHELLE | 1 |
| HERZ | 2 |
| LAUB | 3 |

| Rank | id |
|---|---|
| SECHS | 0 |
| SIEBEN | 1 |
| ACHT | 2 |
| NEUN | 3 |
| ZEHN | 4 |
| UNTER | 5 |
| OBER | 6 |
| KOENIG | 7 |
| ASS | 8 |

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

## Aufbau des 348-dim Featurevektors

Der Vektor besteht aus 18 zusammenhängenden Sections. Jeder Wert ist `float32`
in `[0.0, 1.0]`.

| Section | Offset | Größe | Inhalt |
|---|---|---|---|
| `own_hand`                       |   0..35  | 36 | one-hot: Karten in der eigenen Hand |
| `played_by_me`                   |  36..71  | 36 | one-hot: Karten, die ich in abgeschlossenen Stichen gespielt habe |
| `played_by_left`                 |  72..107 | 36 | Karten, die der linke Spieler (rel=1) gespielt hat |
| `played_by_partner`              | 108..143 | 36 | Karten, die mein Partner (rel=2) gespielt hat |
| `played_by_right`                | 144..179 | 36 | Karten, die der rechte Spieler (rel=3) gespielt hat |
| `current_trick_by_me`            | 180..215 | 36 | Karte, die ich im aktuellen Stich gespielt habe (falls schon dran) |
| `current_trick_by_left`          | 216..251 | 36 | Karte des linken Spielers im aktuellen Stich |
| `current_trick_by_partner`       | 252..287 | 36 | Karte des Partners im aktuellen Stich |
| `current_trick_by_right`         | 288..323 | 36 | Karte des rechten Spielers im aktuellen Stich |
| `lead_suit`                      | 324..327 |  4 | one-hot über Lead-Farbe; alle Nullen wenn Stich leer |
| `trump_suit`                     | 328..331 |  4 | one-hot über Trumpf-Farbe (nur bei Variante `trumpf`) |
| `mode`                           | 332..335 |  4 | `[is_trumpf, is_oben, is_unten, is_slalom]` |
| `my_seat`                        | 336..339 |  4 | one-hot über eigenen Sitz 0..3 (absolut) |
| `starter_seat_relative`          | 340..343 |  4 | one-hot über Anspieler des aktuellen Stichs **relativ zu mir** (rel=0..3) |
| `score_own_norm`                 | 344..344 |  1 | eigene Team-Punkte / 1000, gekappt bei 1.0 |
| `score_opp_norm`                 | 345..345 |  1 | Gegner-Punkte / 1000, gekappt bei 1.0 |
| `trick_idx_norm`                 | 346..346 |  1 | aktueller Stich-Index 0..8, dividiert durch 9 |
| `round_idx_norm`                 | 347..347 |  1 | aktuelle Runde, dividiert durch 20 und gekappt bei 1.0 |

**Summe**: 9 × 36 + 5 × 4 + 4 × 1 = 324 + 20 + 4 = **348**.

### Detail: `mode` (Offsets 332..335)

Das `mode`-Feld kodiert die **effektive Variante des aktuellen Stichs**, ergänzt um ein Flag, ob die ursprüngliche Ansage Slalom war.

| Index | Bit | Bedeutung |
|---|---|---|
| 332 | `is_trumpf` | 1 wenn aktueller Stich Trumpf-Modus |
| 333 | `is_oben` | 1 wenn aktueller Stich Bock-Modus |
| 334 | `is_unten` | 1 wenn aktueller Stich Geiss-Modus |
| 335 | `is_slalom_flag` | 1 wenn die Ansage Slalom war (unabhängig vom aktuellen Stich-Modus) |

Bei `is_trumpf=is_oben=is_unten` ist genau **eines** der drei Bits gesetzt (one-hot zwischen den dreien); `is_slalom` ist orthogonal.

### Detail: Spieler-Zuordnung in Played/Current-Trick

Pro Stich (sowohl abgeschlossen als auch laufend) wird die `starter`-Position genutzt, um aus der Karten-Position im Stich die Spieler-Identität zu rekonstruieren:

```
spieler_der_karte_position_k = (starter + k) mod num_players
relative_position = (spieler_der_karte_position_k - my_seat) mod num_players
```

Das Engine-Format `CompletedTrick(starter, cards)` muss diese `starter`-Information liefern.

### Detail: leerer Stich

Wenn der aktuelle Stich noch keine Karte enthält, sind alle 4 `current_trick_by_*`-Blöcke + `lead_suit` komplett 0.

## Aktionsmaske

Die Aktionsmaske `mask` hat 36 Bits, einer pro Karte:

- `mask[i] = 1.0` → Karte `i` ist legal spielbar
- `mask[i] = 0.0` → Karte `i` ist illegal

Im Modell wird die Maske durch eine fixe Layer in den Logits-Bias übertragen:

```
masked_logits = logits + (1.0 - mask) * -1e9
policy = softmax(masked_logits)
```

Dadurch sind illegale Karten effektiv auf Wahrscheinlichkeit 0 gedrückt.

## Konsistenzanforderungen für TypeScript-Port

1. **Karten-Index**: `card_index = suit_id * 9 + rank_id` — keine Abweichungen.
2. **Section-Reihenfolge** und **Section-Größen** sind fix.
3. **Relative Position**: Formel `(other_seat - my_seat) mod 4` — keine Sonderbehandlung für Index-Negativität, modulo macht das korrekt.
4. **Normalisierung**: Werte 0 bis 1, gekappt wie oben spezifiziert.
5. **dtype**: `float32`, nicht `float64`.
6. **Maske**: float32 mit Werten exakt 0.0 oder 1.0.
7. **Fixture-Test**: Die TS-Implementierung muss alle (state → expected_vector)-Paare aus `spec/fixtures/encoding_fixtures.json` exakt reproduzieren.

## Versionierung

Diese Spezifikation ist Version **2.0.0**. Die Vorgängerversion 1.0.0 hatte einen 132-dim Featurevektor ohne Spieler-Zuordnung. **Modelle, die mit Version 1.0.0 trainiert wurden, sind mit dieser Encoder-Version nicht kompatibel** — der Featurevektor hat eine andere Form.

Beim Modell-Laden muss die Web-App prüfen:

```pseudocode
if model.metadata.encoding_version != "2.0.0":
    throw IncompatibleModelError(...)
```
