# State-Encoding-Spezifikation

**Version**: 1.0.0 (zu `spec_version` in `jass_rules.json`)
**Status**: Stabil. Änderungen erhöhen die Major-Version und brechen Kompatibilität.

Dieses Dokument beschreibt **exakt**, wie ein Spielzustand in den Eingabevektor des
neuronalen Netzes umgewandelt wird. Eine TypeScript-Implementierung muss die hier
beschriebene Reihenfolge und Semantik 1:1 spiegeln, sonst stimmen die Vorhersagen
nicht mehr mit den Trainingsdaten überein.

Zur Verifikation existiert `spec/fixtures/encoding_fixtures.json`: konkrete
Spielzustände mit erwarteten Vektoren, gegen die beide Implementierungen testen.

---

## Eingabe-Tensoren

Das Modell akzeptiert **zwei** Eingaben pro Sample:

| Name | Shape | dtype | Werte |
|---|---|---|---|
| `state` | `[batch, 132]` | `float32` | Featurevektor, alle Werte in `[0.0, 1.0]` |
| `mask` | `[batch, 36]` | `float32` | Aktionsmaske, `1.0` = legal, `0.0` = illegal |

Das Modell liefert eine Wahrscheinlichkeitsverteilung über die 36 möglichen Karten;
illegale Aktionen sind durch die Maske auf praktisch 0 gedrückt.

## Karten-Index

Jeder der 36 Karten wird ein eindeutiger Index 0..35 zugewiesen:

```
card_index = suit_id * 9 + rank_id
```

Mit `suit_id ∈ {0=EICHEL, 1=SCHELLE, 2=HERZ, 3=LAUB}` und `rank_id ∈ {0=SECHS, 1=SIEBEN, 2=ACHT, 3=NEUN, 4=ZEHN, 5=UNTER, 6=OBER, 7=KOENIG, 8=ASS}`.

**Wichtig**: Die Reihenfolge muss genau diese sein. Tausche keine Werte um.

Beispiele:
- `Eichel-Sechs`  → `0 * 9 + 0 = 0`
- `Eichel-Ass`    → `0 * 9 + 8 = 8`
- `Schelle-Sechs` → `1 * 9 + 0 = 9` (= Weli)
- `Laub-Ass`      → `3 * 9 + 8 = 35`

## Aufbau des 132-dim Featurevektors

Der Vektor besteht aus 12 zusammenhängenden Sections. Jeder Wert ist `float32`
in `[0.0, 1.0]`.

| Section | Offset | Größe | Inhalt |
|---|---|---|---|
| `own_hand`         |  0..35  | 36 | one-hot: Karten in der eigenen Hand |
| `played_history`   | 36..71  | 36 | one-hot: Karten, die in vorherigen, abgeschlossenen Stichen gespielt wurden |
| `current_trick`    | 72..107 | 36 | one-hot: Karten im aktuellen, noch nicht abgeschlossenen Stich |
| `lead_suit`        | 108..111 |  4 | one-hot über Lead-Farbe des aktuellen Stichs; alle Nullen wenn der Stich noch leer ist |
| `trump_suit`       | 112..115 |  4 | one-hot über Trumpf-Farbe (nur bei Variante `trumpf`); alle Nullen bei Bock/Geiss/Slalom |
| `mode`             | 116..119 |  4 | `[is_trumpf, is_oben, is_unten, is_slalom_flag]` (siehe unten) |
| `my_seat`          | 120..123 |  4 | one-hot über die eigene Sitz-Position 0..3 (absolut, nicht relativ) |
| `starter_seat`     | 124..127 |  4 | one-hot über den Spieler, der den aktuellen Stich begonnen hat |
| `score_own_norm`   | 128..128 |  1 | eigene Team-Punkte / 1000, gekappt bei 1.0 |
| `score_opp_norm`   | 129..129 |  1 | Gegner-Punkte / 1000, gekappt bei 1.0 |
| `trick_idx_norm`   | 130..130 |  1 | aktueller Stich-Index 0..8, dividiert durch 9 |
| `round_idx_norm`   | 131..131 |  1 | aktuelle Runde, dividiert durch 20 und gekappt bei 1.0 |

**Summe**: 36 + 36 + 36 + 4 + 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1 = **132**.

### Detail: `mode` (Offsets 116..119)

Das `mode`-Feld kodiert die **effektive Variante des aktuellen Stichs**, ergänzt um
ein Flag, ob die ursprüngliche Ansage Slalom war.

| Index | Bit | Bedeutung |
|---|---|---|
| 116 | `is_trumpf` | 1 wenn aktueller Stich Trumpf-Modus |
| 117 | `is_oben` | 1 wenn aktueller Stich Bock-Modus |
| 118 | `is_unten` | 1 wenn aktueller Stich Geiss-Modus |
| 119 | `is_slalom_flag` | 1 wenn die **Ansage** Slalom war (unabhängig vom aktuellen Stich-Modus) |

Bei einer Slalom-Runde, die mit Bock beginnt, sind im ersten Stich `is_oben=1`
**und** `is_slalom_flag=1`. Im zweiten Stich `is_unten=1` und `is_slalom_flag=1`.
Bei reinem Bock ohne Slalom: `is_oben=1`, `is_slalom_flag=0`.

Bei den Modi `is_trumpf`, `is_oben` und `is_unten` ist genau **eines** dieser drei Bits
gesetzt (one-hot zwischen den dreien); `is_slalom_flag` ist orthogonal.

### Detail: `lead_suit` (Offsets 108..111) und leerer Stich

Wenn der aktuelle Stich noch keine Karte enthält (der Spieler ist am Anspielen),
sind **alle 4 Lead-Suit-Bits gleich 0**. Ein "leerer Stich"-Marker wird nicht
zusätzlich kodiert; das Netz erkennt es daran, dass auch der `current_trick`-Block
komplett 0 ist.

### Detail: Sitz-Indizes

Sitze sind 0..3, gegen den Uhrzeigersinn. Spieler 0 und 2 sind Team 0, Spieler 1
und 3 sind Team 1 (Partner über Kreuz). Die Sitz-Information ist **absolut** kodiert,
nicht relativ zum Spieler — das Netz lernt selbst, was "Partner" und "Gegner" bedeutet.

## Aktionsmaske

Die Aktionsmaske `mask` hat 36 Bits, einer pro Karte (gleiches Indexschema wie oben).

- `mask[i] = 1.0` → Karte `i` ist legal spielbar
- `mask[i] = 0.0` → Karte `i` ist illegal (in der Hand nicht vorhanden, oder verletzt
  Farbzwang / Untertrumpfen-Verbot)

Die Maske wird im Modell durch eine fixe Layer in den Logits-Bias übertragen:

```
masked_logits = logits + (1.0 - mask) * -1e9
probs = softmax(masked_logits)
```

Dadurch sind die Wahrscheinlichkeiten für illegale Karten effektiv 0 — der Spieler
wählt mit `argmax(probs)` zwingend einen legalen Zug.

## Inferenz-Algorithmus (clientseitig)

```pseudocode
def choose_card(hand, game_state):
    state_vector = encode_state(hand, game_state)  # (132,)
    legal_mask   = compute_legal_mask(hand, game_state)  # (36,)

    probs = model.predict({
        "state": state_vector.reshape(1, 132),
        "mask":  legal_mask.reshape(1, 36),
    })[0]  # (36,)

    chosen_idx = argmax(probs)
    return index_to_card(chosen_idx)
```

Für eine stochastische Variante (für Datenaugmentation oder Anti-Vorhersagbarkeit
bei Spielern, die das NN "auswendig lernen"):

```pseudocode
chosen_idx = sample(probs)   # zufällig nach Wahrscheinlichkeit
```

## Konsistenzanforderungen für TypeScript-Port

1. **Karten-Index**: `card_index = suit_id * 9 + rank_id` — keine Abweichungen.
2. **Section-Reihenfolge** und **Section-Größen** sind fix; weder ergänzen noch umsortieren.
3. **Normalisierung**: Werte 0 bis 1, gekappt wie oben spezifiziert.
4. **dtype**: `float32` (nicht `float64`), sonst weicht das Modell numerisch ab.
5. **Maske**: float32 mit Werten exakt 0.0 oder 1.0 (kein int).
6. **Fixture-Test**: Die TS-Implementierung muss alle (state → expected_vector)-Paare
   aus `spec/fixtures/encoding_fixtures.json` exakt reproduzieren.

## Versionierung

Bei jeder breaking change wird die **Major-Version** dieser Spezifikation erhöht
und im Modell-Metadata vermerkt. Die Web-App muss beim Laden des Modells prüfen:

```pseudocode
if model.metadata.encoding_version != EXPECTED_ENCODING_VERSION:
    throw IncompatibleModelError(...)
```

Eine **Minor-Version**-Erhöhung darf nur additiv sein (z.B. neue optionale Features
am Ende des Vektors). Alle bestehenden Offsets bleiben in jedem Minor-Update gleich.
