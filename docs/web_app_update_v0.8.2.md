# Update-Briefing fuer die Web-App "Heb ab!" -- Integration von v0.8.2 (Solo-Jass)

> Diesen Text dem Web-App-Kollegen (oder dessen Claude-Session) als Briefing geben.
> Self-contained -- die andere Session braucht den NN-Repo-Verlauf nicht zu kennen.

---

## Kurzfassung

**v0.8.2** (Solo-Jass) ist veroeffentlicht. Wie bei Kreuz v0.7.2 ein reines
Qualitaets-Update **ohne Pflicht-Arbeit**:

- **KEINE** Engine-/Encoder-/Lizenz-Aenderung (`encoding_version` weiterhin **3.0.0**).
- **Neue, staerkere Modellgewichte** -> TF.js-Modell austauschen.
- Heuristik ("Medium"-Gegner) getunt -- nur relevant bei TS-Port (Punkt 2).

> Falls du parallel das v0.7.2-Briefing (Kreuz) hast: Punkt 1 und 2 sind
> strukturell identisch. Solo-Spezifika sind nur `team_mode` und die etwas
> anderen Heuristik-Werte unten.

---

## 1. Neues Modell (Pflicht, trivial)

```bash
gh release download v0.8.2 --repo matthili/jass-neuronales-netz --pattern "jass-nn-*.zip"
unzip jass-nn-v0.8.2.zip
```

- **Encoder + API unveraendert:** `{state:[batch,421], mask:[batch,36]}` ->
  `{policy:[batch,36], value:[batch,1]}`. Reiner Gewichts-Austausch.
- `MANIFEST.json` traegt weiterhin `team_mode: "solo"`. Nicht mit dem
  Kreuz-Modell verwechseln (andere Reward-Struktur).

---

## 2. Getunte Solo-Heuristik (nur falls in TypeScript portiert)

Laeuft euer "Medium"-Gegner als Python-Heuristik aus diesem Repo, ist nichts zu
tun. Falls in TS nachgebaut -- die neuen Werte:

| Parameter | alt | neu (v0.8.2) |
|---|---|---|
| `slalom_base_factor` | 0.85 | **0.94** |
| `slalom_concentration_factor` | 1 | 1 |
| `slalom_spread_factor` | 1 | 1 |
| `gumpf_scale` *(neu)* | (1.0) | **1.06** |
| `oben_scale` *(neu)* | (1.0) | **0.91** |
| `unten_scale` *(neu)* | (1.0) | **1.10** |

`*_scale` wie bei Kreuz: Multiplikator auf den Ansage-Score der Familie vor dem
argmax (Trumpf = Anker 1.0). Schieben gibt es im Solo nicht (`push_threshold`
irrelevant). Die optionale Trumpf-Disziplin beim Anspielen (siehe Kreuz-Briefing
Punkt 2b) gilt auch hier -- gleiche geringe Prioritaet.

---

## Spielstaerke

Solo ist 4-Spieler (jeder fuer sich); die "interessante" Baseline ist 25 %.

| Eval (paired-eval, Sitz-Rotation) | v0.8.2 Win-Rate |
|---|---|
| 1 NN vs. 1 Vorgaenger-NN (+ 2 Heuristiken) | **46,8 %** (4000; bedingt 53,7 %, ~4,4 SD) |
| 1 NN vs. 3 Heuristiken | **78,8 %** (4000) |

Solider Schritt ueber v0.8.1 (gegen die jetzt staerkere Heuristik gemessen).

---

## Smoke-Test

1. v0.8.2-Modell laedt im Solo-Modus, `team_mode` == "solo", `encoding_version`
   == "3.0.0".
2. Eine 4-Spieler-Solo-Partie laeuft sauber durch.

---

## Fragen

Issues: <https://github.com/matthili/jass-neuronales-netz/issues>. Aufwand: ein
Modell-Austausch.
