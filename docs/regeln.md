# Vorarlberger Kreuz-Jass — Regelwerk

Diese Regeln sind die **Single Source of Truth** für die Engine. Bei Diskrepanz zwischen
Code und diesem Dokument: Code ist falsch.

**Quellen:**
- [jassa.at/regeln](https://jassa.at/regeln/)
- [Mohrenbrauerei FAQ (Weisen)](https://www.mohrenbrauerei.at/biererlebniswelt/community/haeufig-gestellte-fragen-faq/)
- [jasskarten.at/jassregeln](https://www.jasskarten.at/jassregeln)

---

## Karten

**36 Karten** mit einfachdeutschem Blatt: vier Farben (Eichel, Schelle, Herz, Laub) zu je
neun Rängen (6, 7, 8, 9, 10, Unter, Ober, König, Ass).

**Weli**: Die Schelle-6 heißt "Weli" und hat in Runde 1 die Sonderrolle, dass ihr Halter
den Trumpf ansagt. **Sonst spielt sie wie jede andere 6** (0 Punkte, keine Joker-Funktion).

## Kartenwerte

### Nicht-Trumpf

| Karte | Punkte |
|---|---|
| Ass | 11 |
| Zehner | 10 |
| König | 4 |
| Ober | 3 |
| Unter | 2 |
| 9, 8, 7, 6 | 0 |

### Trumpf

| Karte | Punkte |
|---|---|
| Unter (Buur) | **20** |
| 9 (Nell) | **14** |
| Ass | 11 |
| Zehner | 10 |
| König | 4 |
| Ober | 3 |
| 8, 7, 6 | 0 |

## Kartenstärke im Stich

**Nicht-Trumpf** (hoch → niedrig): Ass, König, Ober, Unter, 10, 9, 8, 7, 6
**Trumpf** (hoch → niedrig): Buur (Unter), Nell (9), Ass, König, Ober, 10, 8, 7, 6

Trumpf sticht jede Nicht-Trumpf-Karte.

## Spielablauf

1. **4 Spieler** in **2 Teams** (Partner sitzen gegenüber — über Kreuz)
2. Jeder erhält 9 Karten
3. **Trumpfansage Runde 1**: Spieler mit dem Weli (Schelle-6) sagt Trumpf an
4. **Trumpfansage ab Runde 2**: Anspieler darf zum Mitspieler **schieben**
   - Kein Reden während der Trumpfansage
   - Der Geschobene wählt Trumpf nur anhand seiner eigenen Hand
   - **Der ursprüngliche Ansager spielt trotzdem die erste Karte aus** (nicht der Geschobene)
5. **Weisen** werden vor dem ersten Stich angesagt (Stöcke separat, siehe unten)
6. Es werden **9 Stiche** gespielt
7. Wer einen Stich gewinnt, spielt den nächsten an

## Regelzwänge beim Kartenausspielen

- **Farbzwang**: Wer kann, muss die angespielte Farbe bedienen
- **Ausnahme Buur**: Der Trumpf-Unter darf **immer** gespielt werden, auch wenn man bedienen könnte
- **Kein Untertrumpfen**: Liegt schon ein Trumpf im Stich, darf man nur **höher** trumpfen (außer man hat keine andere Wahl)
- **Kein Stichzwang**: Wer die angespielte Farbe nicht bedienen kann, darf frei abwerfen *oder* trumpfen

## Stichgewinn

- Wenn Trumpf im Stich liegt: höchster Trumpf gewinnt
- Sonst: höchste Karte der Lead-Farbe gewinnt
- Karten anderer Farben (ohne Trumpf) können nicht gewinnen

## Punktezählung pro Runde

| Posten | Punkte |
|---|---|
| Stichpunkte (Summe aller Karten) | 152 |
| Letzter Stich (Bonus) | 5 |
| **Summe regulär** | **157** |
| Matsch (alle 9 Stiche ein Team) | +100 |
| **Summe bei Matsch** | **257** |

**Zählweise:** Üblich wird nur das Team mit weniger Stichen seine Punkte zählen, das andere
bekommt `157 - x`. In der Engine zählen wir beide und verifizieren die Summe als
Konsistenzprüfung.

## Weisen

Werden vor dem ersten Stich der Runde angesagt. **Nur der höchste Weis je Team zählt** —
das Team mit dem höheren Weis bekommt **alle** seine Weis-Punkte gutgeschrieben, das
andere Team bekommt **nichts**.

Gleichstand-Regel: Wer in Spielreihenfolge zuerst gewiesen hat, gewinnt.

### Sequenzen (gleiche Farbe, aufeinanderfolgende Ränge)

| Länge | Punkte |
|---|---|
| 3 Blatt | 20 |
| 4 Blatt | 50 |
| 5 Blatt | 100 |
| 6 Blatt | 120 |
| 7 Blatt | 140 |
| 8 Blatt | 160 |
| 9 Blatt | 180 |

### Vierlinge

| Kombination | Punkte |
|---|---|
| 4× Zehner / Ober / König / Ass | 100 |
| 4× Neuner | 150 |
| 4× Unter | 200 |

### Stöcke

**Trumpf-Ober + Trumpf-König** = 20 Punkte. Werden **nicht** mit den anderen Weisen
verglichen, sondern gelten **immer**. Sie werden angesagt, sobald die zweite Stockkarte
ausgespielt wird.

## Spielende

Die Partie endet, wenn ein Team den **Zielpunktestand** (Default 1000, konfigurierbar)
erreicht oder überschreitet.

---

## Sondervarianten: Bock, Geiss, Slalom

Diese Varianten kann der Ansager **immer** statt einer Trumpf-Farbe wählen — auch in Runde 1.
Schieben bleibt ab Runde 2 möglich.

### Bock (auch "oben")

- **Kein Trumpf**. Es zählt nur die jeweils angespielte (Lead-)Farbe.
- Reihenfolge in der Lead-Farbe (hoch → niedrig): Ass > König > Ober > Unter > 10 > 9 > 8 > 7 > 6
- Andere Farben können nie stechen.
- Strategie: Wer Ass anspielt, kann nicht gestochen werden — Karten ziehen ist mächtig.

### Geiss (auch "unten")

- **Kein Trumpf**. Wie Bock, aber **umgekehrte Reihenfolge** in der Lead-Farbe:
- Reihenfolge (hoch → niedrig): 6 > 7 > 8 > 9 > 10 > Unter > Ober > König > Ass
- Die 6 sticht alles (in der Lead-Farbe), zählt aber weiterhin 0 Punkte.

### Slalom

- Ansager wählt, ob mit *oben* (Bock) oder *unten* (Geiss) begonnen wird.
- **Pro Stich wird gewechselt**: Stich 1=oben → Stich 2=unten → Stich 3=oben → …
- Kartenwerte und Regeln wie bei Bock/Geiss.

### Kartenwerte bei Bock/Geiss/Slalom

| Karte | Punkte |
|---|---|
| Ass | 11 |
| Zehner | 10 |
| **Acht** | **8** |
| König | 4 |
| Ober | 3 |
| Unter | 2 |
| 9, 7, 6 | 0 |

- **Kein Buur-Bonus (20)**, **kein Nell-Bonus (14)** — diese gelten nur im Trumpf-Modus.
- Stattdessen werden die **8er auf 8 Punkte** aufgewertet.
- Summe pro Farbe: 11+10+4+3+2+8 = 38; mal 4 Farben = **152 Stichpunkte** (gleich wie im Trumpf-Modus!).
- Mit letztem Stich: **157**, mit Matsch: **257**.

### Regeln bei Bock/Geiss/Slalom

- **Farbzwang** gilt strikt — Lead-Farbe bedienen, wenn möglich.
- **Kein Buur, kein Untertrumpfen** — diese Konzepte existieren ohne Trumpf nicht.
- **Kein Stichzwang** — wer nicht bedienen kann, darf frei abwerfen.
- **Stöcke gibt es nicht** (mangels Trumpf).
- **Weisen** (Sequenzen, Vierlinge) sind weiterhin möglich.

---

## Spätere Erweiterungen (nicht Teil der aktuellen Engine)

- Steigern (Bieter-Variante)
- Bodensee-Jass (2 Spieler)
- 6-Spieler-Kreuz-Jass
