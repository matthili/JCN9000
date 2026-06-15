# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.9.2 (Bodensee-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung -- das ist der grosse Sprung

**v0.9.2** (Bodensee-Jass) ist veroeffentlicht und bringt den **mit Abstand
groessten Staerke-Zuwachs der ganzen Serie**. Trotzdem auf eurer Seite **keine
Pflicht-Arbeit ausser dem Modell-Austausch**:

- **KEINE** Encoder-Aenderung (`encoding_version` weiterhin **bodensee_1.0.0**).
- **KEINE** Fixture-Aenderung (`bodensee_encoding_fixtures.json` unveraendert --
  anders als bei v0.9.1, wo `bfix_06` neu war; diesmal nichts zu pinnen).
- **KEINE** Engine-/Lizenz-Aenderung.
- **Neues, drastisch staerkeres Modell** -> TF.js-Modell austauschen.

---

## 1. Neues Modell (Pflicht, trivial)

```bash
gh release download v0.9.2 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
unzip jass-nn-v0.9.2.zip
```

- **Encoder-Struktur + API unveraendert:** `{state:[batch,291], mask:[batch,36]}`
  -> `{policy:[batch,36], value:[batch,1]}`. `team_mode: "bodensee_2p"`. Reiner
  Gewichts-Austausch -- der bestehende Bodensee-TS-Encoder und die Fixtures
  bleiben gueltig.
- Was sich verbessert hat (intern in der Datengen): Der MCTS-Lehrer plant jetzt
  die **gesamte Restrunde** voraus statt nur einen Stich. Das behebt die
  strukturelle Kurzsichtigkeit der bisherigen Bodensee-Modelle (z. B. das
  Endspiel-Verhalten "erst Schrott werfen, dann selbst stechen, +5 mitnehmen").

---

## 2. Getunte Bodensee-Heuristik (nur falls in TypeScript portiert)

Laeuft die Heuristik aus diesem Repo, ist nichts zu tun. Falls in TS nachgebaut
-- die neuen Ansage-Werte:

| Parameter | alt | neu (v0.9.2) |
|---|---|---|
| `slalom_base_factor` | 0.85 | **0.88** |
| `gumpf_scale` *(neu)* | (1.0) | **1.02** |
| `oben_scale` *(neu)* | (1.0) | **0.92** |
| `unten_scale` *(neu)* | (1.0) | **0.89** |

`*_scale` = Multiplikator auf den Ansage-Score der Familie vor dem argmax
(Trumpf = Anker 1.0). **Wichtig:** Die Bodensee-Heuristik sagt auch in den
**NN-Partien** an (das NN trifft nur Karten-Entscheidungen) -- diese Werte wirken
also in *beiden* Schwierigkeitsgraden. (Die Trumpf-Disziplin-Regel aus den
Kreuz/Solo-Briefings gibt es im Bodensee-Heuristik-Player **nicht**; nur die
Ansage-Parameter oben.)

---

## Spielstaerke -- aussergewoehnlich

| Eval (paired-eval) | v0.9.2 Win-Rate |
|---|---|
| vs. Vorgaengermodell (bodensee_mcts2, 1-Stich-Lookahead) | **92,4 %** (8000 Partien, ~140 SD) |
| vs. Heuristik (getunt) | **96,8 %** (4000 Partien) |

Ueber alle Varianten 91-99 %, extrem gleichmaessig. Wichtige Konsequenz fuer
euer Test-Setup: **Die Heuristik ist als Bodensee-Gegner gesaettigt** (sie
verliert quasi immer) -- sie taugt nicht mehr als Mess-Latte. Der einzige noch
aussagekraeftige Test ist **menschliches Spiel**. Wenn ihr das Modell selbst
testet, achtet besonders aufs Endspiel: Die Reihenfolge-Entscheidungen (sichere
Stiche zuerst kassieren, +5 im letzten Stich mitnehmen) sollten jetzt sauber sein.

---

## Smoke-Test

1. v0.9.2-Modell laedt, `team_mode` == "bodensee_2p", `encoding_version` ==
   "bodensee_1.0.0".
2. Bestehende `bodensee_encoding_fixtures.json` reproduzieren weiterhin
   byte-genau (keine Fixture-Aenderung in diesem Release).
3. Eine Bodensee-Partie: Tisch-Mechanik + Endspiel plausibel.

---

## Fragen

Issues: <https://github.com/matthili/JCN9000/issues>. Aufwand: ein
Modell-Austausch -- und dann am besten ein paar Partien selbst spielen, denn das
ist bei dieser Spielstaerke die letzte echte Pruefinstanz.
