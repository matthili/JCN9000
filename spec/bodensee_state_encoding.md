# Bodensee-State-Encoding-Spezifikation

**Encoding-Version**: `bodensee_1.0.0`

**Status**: separates Encoding parallel zu `3.0.0` (Kreuz-/Solo-Jass). Beide
existieren nebeneinander, weil Bodensee strukturell eine andere Spielsituation
ist (2 Spieler, Tisch-Mechanik). Die Web-App lädt anhand der gewählten Spielart
das passende Modell.

Dieses Dokument beschreibt **exakt**, wie ein Bodensee-Spielzustand in den
Eingabevektor des neuronalen Netzes umgewandelt wird. Eine TypeScript-
Implementierung muss die hier beschriebene Reihenfolge und Semantik 1:1
spiegeln. Verifikations-Fixtures folgen in einem späteren Release-Schritt.

---

## Eingabe-Tensoren

| Name | Shape | dtype | Werte |
|---|---|---|---|
| `state` | `[batch, 291]` | `float32` | Featurevektor, alle Werte in `[0.0, 1.0]` |
| `mask`  | `[batch, 36]`  | `float32` | Aktionsmaske, `1.0` = legal, `0.0` = illegal |

Modell-Outputs (Multi-Head, identisch zu v3-Format):

| Output | Shape | dtype | Werte |
|---|---|---|---|
| `policy` | `[batch, 36]` | `float32` | Wahrscheinlichkeitsverteilung über Karten |
| `value`  | `[batch, 1]`  | `float32` | Erwarteter Endausgang in `[-1, +1]` (tanh) |

## Karten-Index

Identisch zu `state_encoding.md` (Version 3.0.0):
```
card_index = suit_id * 9 + rank_id
```

EICHEL=0, SCHELLE=1, HERZ=2, LAUB=3. SECHS=0, ..., ASS=8.

## Sections (Reihenfolge im Vektor)

| Section | Offset | Size | Erklärung |
|---|---|---|---|
| `own_hand` | 0 | 36 | One-hot über die privaten Handkarten |
| `own_visible_table` | 36 | 36 | One-hot über die eigenen sichtbaren Tisch-Karten |
| `own_hidden_table_mask` | 72 | 6 | Pro Stapel-Position (0..5): hat noch eine verdeckte Karte? |
| `opp_visible_table` | 78 | 36 | One-hot über die sichtbaren Tisch-Karten des Gegners |
| `opp_hand_count` | 114 | 7 | One-hot für die Anzahl Hand-Karten des Gegners (0..6) |
| `opp_hidden_table_count` | 121 | 7 | One-hot für die Anzahl verdeckter Tisch-Karten beim Gegner (0..6) |
| `played_cards_this_round` | 128 | 36 | One-hot: alle Karten, die in dieser Runde schon gespielt wurden (inkl. laufender Stich) |
| `opp_lead_card` | 164 | 36 | Lead-Karte des Gegners, falls ich nicht selbst leite (sonst alle 0) |
| `i_am_leading` | 200 | 1 | `1.0` wenn ich Anspieler des aktuellen Stichs bin |
| `value_per_card` | 201 | 36 | Normalisierter Wertpunkt jeder der 36 Karten unter der aktuellen Variante (`Punkte / 20.0`) |
| `strength_per_card` | 237 | 36 | Normalisierter Kraftwert jeder Karte unter Variante + Lead-Suit (`Stärke / 18.0`) |
| `lead_suit` | 273 | 4 | One-hot Lead-Farbe des laufenden Stichs (alle 0 wenn ich leite) |
| `trump_suit` | 277 | 4 | One-hot Trumpf-Farbe (gesetzt bei `TRUMPF` und `GUMPF`, sonst alle 0) |
| `mode` | 281 | 5 | `[is_trumpf, is_gumpf, is_oben, is_unten, is_slalom]` |
| `i_am_announcer` | 286 | 1 | `1.0` wenn ich diese Runde angesagt habe (also Weli-Halter war) |
| `score_own_norm` | 287 | 1 | Eigener Punktestand `/ 1000`, gekappt bei 1 |
| `score_opp_norm` | 288 | 1 | Punktestand Gegner `/ 1000`, gekappt bei 1 |
| `trick_idx_norm` | 289 | 1 | Aktueller Stich-Index `/ 18` |
| `round_idx_norm` | 290 | 1 | Aktuelle Runde `/ 20`, gekappt bei 1 |

**Summe: 291 Dimensionen.**

## Wichtige Unterschiede zu v3.0.0 (Kreuz/Solo)

| Aspekt | v3.0.0 (Kreuz/Solo) | bodensee_1.0.0 |
|---|---|---|
| Spieler-Slots in `played_by_*` | 4 (me/left/partner/right) | nicht relevant — Bodensee hat eigene `played_cards_this_round`-Section |
| `current_trick_by_*` | 4 Slots à 36 Bits | konsolidiert zu **einer** `opp_lead_card`-Section + Bit `i_am_leading` |
| Tisch-Mechanik | nicht existent | neue Sections `own_visible_table`, `own_hidden_table_mask`, `opp_visible_table`, `opp_hidden_table_count` |
| Gegner-Hand-Größe | implizit aus Gespielten ableitbar | explizit als `opp_hand_count` (7 Bit one-hot) |
| `my_seat` | 4 Bit (Sitz 0-3) | entfällt — bei 2 Spielern reicht `i_am_leading` + `i_am_announcer` |
| `starter_seat_relative` | 4 Bit | entfällt — ist immer "ich" oder "der Gegner" |

## Aktions-Raum (Maske)

36 Karten. Eine Karte ist zu jedem Zeitpunkt eindeutig entweder in der Hand
**oder** sichtbar auf dem Tisch (sie kann nicht an zwei Stellen sein).
Daher reicht ein 36-bit Aktions-Raum. Der Spieler-Code findet die Quelle
("Hand" oder "Tisch") automatisch.

Maske-Semantik wie bisher: `1.0 = legal`, `0.0 = illegal`. Bedienzwang gilt
**gemeinsam für Hand und sichtbaren Tisch** (siehe `legal_moves_bodensee` in
`jass_engine/bodensee/rules.py`).

## value_per_card und strength_per_card

Identische Logik wie in v3.0.0 — siehe `state_encoding.md`. Der einzige
Unterschied ist die Variante-Verteilung: in Bodensee kommen alle 12 üblichen
Varianten vor (4× Trumpf, 4× Gumpf, Oben, Unten, 2× Slalom).

## Verifikation

Pflicht-Check beim TypeScript-Port: alle Test-Fixtures aus
`spec/fixtures/bodensee_encoding_fixtures.json` müssen byte-genau reproduziert
werden (`atol=1e-5`). Die Datei liegt jedem v0.9.0+-Release-ZIP bei und
enthält 12 konkrete Spielzustände mit erwartetem `state_vector` (291) und
`legal_mask` (36).

Jede Fixture hat ein `input`-Objekt (vollständige Zustandsbeschreibung) und
ein `expected`-Objekt (erwartete Encoder-Ausgabe). Der TS-Encoder liest das
`input`, encodiert es und vergleicht mit `expected.state_vector` /
`expected.legal_mask`.

Erzeugt wird die Datei mit `python -m scripts.generate_bodensee_encoding_fixtures`.
Der Test `tests/test_bodensee_fixtures.py` stellt sicher, dass sie nicht
gegenüber dem Encoder driftet.
