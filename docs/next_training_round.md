# Nächste Trainingsrunde — Verbesserungs-Sammlung

**Lebendes Dokument.** Sammelstelle für gezielte Verbesserungen der nächsten
Trainingsrunde (geplant v0.7.3 / v0.8.3 / v0.9.3). Pro Posten: Befund, Ursache,
Fix-Hebel, Verifikation. Neue Beobachtungen unten unter **„Weitere
Beobachtungen"** anhängen — dann gehen sie bis zum nächsten Training nicht verloren.

_Stand: 2026-06-17. Kein Training aktiv (bewusst gesammelt, nicht sofort ausgeführt)._

## Verifikations-Harness (für alle Punkte)
- `scripts/probe_unten_lead.py` — Einzelfall-Policy in reinem Unten (n=1, zum Anschauen)
- `scripts/probe_lead_sweep.py` — Statistik Unten vs Oben (blunder / strongmass / mean_top1 / agree_Heur)

Vor **und** nach dem Training laufen lassen und die Zahlen vergleichen — so sehen
wir schwarz auf weiß, ob ein Fix gewirkt hat.

---

## Posten 1 — Anspiel-Policy unterfittet ⚑ Priorität
**Betrifft:** Kreuz v0.7.2 (`models/kreuz_mcts3`); vermutlich Solo/Bodensee analog.

### Befund (`probe_lead_sweep`, N=1000)
| Metrik | Modell | Heuristik (Referenz) |
|---|---|---|
| Unten-blunder (Ass angespielt) | **26 %** | **0 %** |
| strongmass / strong-Anspiel | ~27 % Masse ≈ uniform | 94 % |
| agree_Heur (argMax = Regel-Karte) | **16 %** (Zufall ≈ 11 %) | — |
| mean_top1 | 0,19 | (deterministisch) |

Die Eröffnungs-Policy ist **flach** und mit der regel-korrekten Wahl praktisch
unkorreliert. Sichtbar/teuer in Unten (schwächste Karte = Ass = 11 Punkte
verschenkt), in Oben harmlos (schwächste = wertlose 6). Lokaler Leak, kein
Totalausfall (Modell gewinnt Eval 83,5 % vs Heuristik).

### Ursache
Anspiel = info-ärmste Entscheidung (leere Stich-Historie) und nur ~1/9 der
Lead-Situationen. Labels sind hart (`sparse_categorical_crossentropy` auf die
Lehrer-Aktion, `training/train.py`). Über viele Anspiele mittelt sich die
Schüler-Policy flach.

### Fix-Hebel (billig → gründlich)
**B) Lead-Stellungen im Loss höher gewichten — empfohlen, KEIN Daten-Re-Gen.**
- Anspiel ist aus dem State-Vektor erkennbar: alle `current_trick_by_*`-Sektionen = 0.
- Helper ist fertig + getestet: `training/sample_weights.py` → `lead_sample_weights(X, w)`.
- Integration in `training/train.py` (Rezept unten), neuer Flag `--lead-loss-weight` (Default 1.0 = aus).
- Retrain aus den vorhandenen mcts3-Shards (oder Warm-Start von v0.7.2). **Erst 1 Epoche Smoke-Test**, dann voller Lauf.
- Startwert ausprobieren: `--lead-loss-weight 3.0` (Bereich ~2–5 testen).

**A) Heuristik-Anspiel als Label — gründlicher, braucht Daten-Re-Gen.**
In der Datengen (`training/data/generate_mcts_data.py`, ForcedAnnouncementPlayer /
Teacher) für Lead-Stellungen das entschiedene (0-%-blunder) Heuristik-Anspiel als
Label verwenden statt der weichen MCTS-Wahl. Injiziert scharfe, korrekte Leads direkt.

**C) MCTS-Budget an Lead-Stellungen erhöhen** (mehr Determinisierungen/Rollouts),
damit der Lehrer sich beim Anspiel entschiedener festlegt.

→ Reihenfolge: erst **B** (billig, aus vorhandenen Daten). Nur wenn B nicht
reicht, A/C beim nächsten Daten-Re-Gen mitnehmen.

### Integrations-Rezept Hebel B (in `training/train.py`)
1. `--lead-loss-weight` (float, Default 1.0) in `main()` ergänzen + an `train()` durchreichen.
2. In `_make_shard_to_sample_dataset` / `_shard_to_sample_dataset`: wenn `lead_weight != 1.0`,
   pro Sample ein Policy-Gewicht als 3. Tupel-Element anhängen:
   ```python
   lo, hi = current_trick_span(SECTION_OFFSETS)   # aus training.sample_weights
   is_lead = tf.reduce_sum(X[:, lo:hi], axis=1) == 0
   w = tf.where(is_lead, tf.constant(lead_weight, tf.float32), tf.constant(1.0))
   return tf.data.Dataset.from_tensor_slices((
       {"state": X, "mask": masks},
       {"policy": actions, "value": rewards},
       {"policy": w, "value": tf.ones_like(w)},   # nur die Policy gewichten
   ))
   ```
3. Bei `lead_weight == 1.0` den bestehenden **2-Tupel-Pfad unverändert** lassen (kein Risiko für den Default).
4. **Vor dem echten Lauf `--epochs 1` smoke-testen** (3-Tupel + `fit` zusammen).

### Verifikation
`probe_lead_sweep` nach dem Training: Ziel **Unten-blunder deutlich < 26 %**
(Richtung Heuristik 0 %), **agree_Heur deutlich > 16 %**, **strongmass hoch**.
Gegenprobe: die Gesamt-Eval (paired vs mcts2 / Heuristik) darf **nicht fallen** —
sonst hat das Lead-Upweighting den Rest verschlechtert.

---

## Posten 2 — Modell sieht keine gewiesenen Karten 🧩 größer (encoder-breaking)
**Betrifft:** alle Modelle.

### Befund (am Code bestätigt)
`GameState` (`jass_engine/player.py`) hat **kein Weis-Feld**, und die 421
Encoder-`SECTIONS` (`training/encoder.py`) enthalten **keine** Weis-Info. Das NN
kann gewiesene (offen gelegte) Karten also nicht berücksichtigen.

### Warum es zählt
Beim Weisen werden Karten allen Mitspielern gezeigt. Ein starker Spieler merkt
sich „Gegner hat Sequenz gewiesen → hält diese Karten" und spielt entsprechend.
Aktuell ist das Modell dafür blind.

### Scope (deutlich größer als Posten 1 — eigene Mini-Roadmap)
1. **Engine:** gewiesene Karten je Sitz in `GameState` surfacen.
2. **Encoder:** neue Sektion, z.B. `weis_revealed_by_<relpos>` (4×36) → **Encoder-Version-Bump** (3.0.0 → 3.1.0 / 4.0.0).
3. **Datengen:** Weis-Info in die Rollout-States schreiben.
4. **Teacher:** muss die Info **nutzen** — sonst lernt der Schüler nur Rauschen.
5. **App + Spec:** `@jass/engine`-Encoder nachziehen, `encoding_fixtures.json` erweitern, `ENCODING_VERSION` hochziehen.

→ Kein Quick-Fix. **Zuerst entscheiden**, ob der erwartete Spielstärke-Gewinn den
Encoder-Bruch (App-Koordination!) wert ist. Wenn ja: eigene Runde dafür.

---

## Weitere Beobachtungen (hier anhängen)
Format: _(Datum) — Beobachtung — Modus/Modell — grobe Fix-Idee — Status_

- _(2026-06-17) — Anspiel flach / Ass in Unten — Kreuz v0.7.2 — siehe Posten 1 — analysiert, Fix vorbereitet_
- _(2026-06-17) — keine Weis-Wahrnehmung — alle — siehe Posten 2 — offen (Entscheidung nötig)_
- …
