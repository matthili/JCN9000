"""Parameter-Tuning fuer die Ansage-Heuristik (Kreuz-Jass).

Die `HeuristicPlayer`-Ansage haengt an vier Parametern:
    push_threshold              (int)   -- ab welchem Score nicht geschoben wird
    slalom_base_factor          (float) -- Gewicht max(oben,unten) im Slalom-Score
    slalom_concentration_factor (int)   -- Bonus fuer Dominanz am selben Ende/Farbe
    slalom_spread_factor        (int)   -- kleiner Bonus pro Balance-Karte

Das Kartenspiel der Heuristik haengt NICHT an diesen Parametern. Wenn also
beide Seiten denselben Heuristik-Kartenspieler nutzen und sich nur in den
Ansage-Parametern unterscheiden, misst die Win-Rate ausschliesslich die
Qualitaet der Ansage -- mit paired-eval (identische Kartenverteilung,
gespiegelte Sitze) faellt zusaetzlich das Karten-Glueck als Rauschen weg.

Suche:
1. Screening: viele zufaellige Kandidaten rund um die Baseline, je mit einer
   kleinen Partienzahl gegen die Baseline bewertet (parallel ueber Kandidaten).
2. Finale: die besten K Kandidaten mit grosser Partienzahl re-evaluiert, damit
   das Ergebnis nicht im statistischen Rauschen untergeht.

Reines CPU-Tooling (keine NN-Inferenz, kein TensorFlow) -- skaliert linear mit
den Worker-Prozessen.

Aufruf (WSL2 oder PowerShell):
    python -m scripts.tune_heuristic_announce \\
        --games-screen 600 --num-candidates 60 \\
        --games-final 4000 --top-k 5 \\
        --workers 12 --output models/heuristic_announce_tuned.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.tournament import two_team_match
from players.heuristic_player import HeuristicPlayer


# ---------------------------------------------------------------------------
# Parameter-Modell
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnnounceParams:
    # Muss die aktuellen HeuristicPlayer-Defaults spiegeln -- BASELINE ist der
    # Vergleichsanker. Stand: Tuning-Lauf vom 2026-06 (52.9 % gegen die
    # Vorgaenger-Defaults 55 / 0.95 / 2 / 1).
    push_threshold: int = 59
    slalom_base_factor: float = 0.86
    slalom_concentration_factor: int = 0
    slalom_spread_factor: int = 1

    def as_tuple(self) -> tuple:
        return (
            self.push_threshold,
            self.slalom_base_factor,
            self.slalom_concentration_factor,
            self.slalom_spread_factor,
        )


BASELINE = AnnounceParams()


def _sample_candidate(rng: random.Random) -> AnnounceParams:
    """Zufaelliger Kandidat in plausiblen Bereichen rund um die Baseline."""
    return AnnounceParams(
        push_threshold=rng.randint(40, 75),
        slalom_base_factor=round(rng.uniform(0.80, 1.05), 2),
        slalom_concentration_factor=rng.randint(0, 4),
        slalom_spread_factor=rng.randint(0, 3),
    )


def _make_factory(params: AnnounceParams):
    """Baut eine PlayerFactory (seat, rng) -> HeuristicPlayer mit diesen
    Ansage-Parametern. Das Kartenspiel ist davon unabhaengig."""
    def factory(seat: int, rng: random.Random) -> HeuristicPlayer:
        return HeuristicPlayer(
            name=f"H{seat}",
            push_threshold=params.push_threshold,
            slalom_base_factor=params.slalom_base_factor,
            slalom_concentration_factor=params.slalom_concentration_factor,
            slalom_spread_factor=params.slalom_spread_factor,
            rng=rng,
        )
    return factory


# ---------------------------------------------------------------------------
# Bewertung eines Kandidaten gegen die Baseline (Modul-Ebene fuer mp.Pool)
# ---------------------------------------------------------------------------
def _eval_candidate_worker(task: tuple) -> tuple[tuple, float]:
    """task = (params_tuple, baseline_tuple, num_games, target_score, seed).

    Returns (params_tuple, win_rate_des_Kandidaten_gegen_Baseline).
    """
    params_tuple, baseline_tuple, num_games, target_score, seed = task
    cand = AnnounceParams(*params_tuple)
    base = AnnounceParams(*baseline_tuple)
    result = two_team_match(
        label_a="cand",
        factory_a=_make_factory(cand),
        label_b="base",
        factory_b=_make_factory(base),
        num_games=num_games,
        target_score=target_score,
        seed=seed,
        paired_eval=True,
    )
    return params_tuple, result.stats_a.win_rate


def _winrate_sd(num_games: int) -> float:
    """Grobe Standardabweichung der Win-Rate bei num_games (p=0.5)."""
    return math.sqrt(0.25 / num_games) if num_games > 0 else 0.0


def _evaluate_many(
    candidates: list[AnnounceParams],
    baseline: AnnounceParams,
    num_games: int,
    target_score: int,
    workers: int,
    base_seed: int,
) -> list[tuple[AnnounceParams, float]]:
    """Bewertet alle Kandidaten parallel (ein Prozess pro Kandidat-Match)."""
    tasks = [
        (c.as_tuple(), baseline.as_tuple(), num_games, target_score, base_seed + i * 101)
        for i, c in enumerate(candidates)
    ]
    ctx = mp.get_context("spawn")
    results: list[tuple[AnnounceParams, float]] = []
    with ctx.Pool(processes=workers) as pool:
        for params_tuple, wr in pool.imap_unordered(_eval_candidate_worker, tasks):
            results.append((AnnounceParams(*params_tuple), wr))
    # In Eingabe-Reihenfolge sortieren ist egal -- wir sortieren nach Win-Rate.
    results.sort(key=lambda t: t[1], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-screen", type=int, default=600,
                        help="Partien pro Kandidat im Screening (durch 4 teilbar empfohlen).")
    parser.add_argument("--num-candidates", type=int, default=60,
                        help="Anzahl zufaelliger Kandidaten im Screening.")
    parser.add_argument("--games-final", type=int, default=4000,
                        help="Partien pro Kandidat in der finalen Re-Evaluation.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Wieviele Top-Screening-Kandidaten final re-evaluiert werden.")
    parser.add_argument("--target", type=int, default=1000, help="Punkteziel pro Partie.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/heuristic_announce_tuned.json"))
    args = parser.parse_args()

    # num_games muss fuer paired_eval gerade sein.
    if args.games_screen % 2 != 0 or args.games_final % 2 != 0:
        parser.error("--games-screen und --games-final muessen gerade sein (paired-eval).")

    rng = random.Random(args.seed)
    # Baseline ist immer Kandidat 0 (Sanity: sollte ~50 % gegen sich selbst geben).
    candidates = [BASELINE] + [_sample_candidate(rng) for _ in range(args.num_candidates)]

    sd_screen = _winrate_sd(args.games_screen) * 100
    sd_final = _winrate_sd(args.games_final) * 100
    print("=" * 70)
    print("  Ansage-Heuristik-Tuning (Kreuz-Jass, paired-eval gegen Baseline)")
    print(f"  Baseline: {BASELINE}")
    print(f"  Screening: {len(candidates)} Kandidaten x {args.games_screen} Partien "
          f"(Win-Rate-SD ~{sd_screen:.1f} %)")
    print(f"  Finale:    Top-{args.top_k} x {args.games_final} Partien "
          f"(Win-Rate-SD ~{sd_final:.1f} %)")
    print(f"  Worker: {args.workers}")
    print("=" * 70)

    t0 = time.perf_counter()
    print("\n[1/2] Screening laeuft ...")
    screened = _evaluate_many(
        candidates, BASELINE, args.games_screen, args.target, args.workers, args.seed,
    )
    print(f"  Screening fertig in {(time.perf_counter() - t0) / 60:.1f} min.")
    print("  Top-Kandidaten (Win-Rate gegen Baseline):")
    for params, wr in screened[:min(args.top_k, len(screened))]:
        print(f"    {wr * 100:5.1f} %  {params}")

    # Finale: die besten K (ohne exakte Baseline-Duplikate) gross re-evaluieren.
    top = [p for p, _ in screened[:args.top_k]]
    print(f"\n[2/2] Finale Re-Evaluation der Top-{len(top)} ...")
    t1 = time.perf_counter()
    final = _evaluate_many(top, BASELINE, args.games_final, args.target, args.workers, args.seed + 7)
    print(f"  Finale fertig in {(time.perf_counter() - t1) / 60:.1f} min.")

    best_params, best_wr = final[0]
    # Verbesserung nur ernst nehmen, wenn klar ueber dem Rausch-Floor.
    margin = (best_wr - 0.5) * 100
    significant = margin > 2 * sd_final

    print("\n" + "=" * 70)
    print("  ERGEBNIS")
    for params, wr in final:
        flag = "  <-- beste" if (params, wr) == (best_params, best_wr) else ""
        print(f"    {wr * 100:5.1f} %  {params}{flag}")
    print("-" * 70)
    if significant:
        print(f"  Beste Konfiguration schlaegt die Baseline mit {best_wr * 100:.1f} % "
              f"(+{margin:.1f} pp, > 2 SD = {2 * sd_final:.1f} pp). UEBERNEHMEN sinnvoll.")
    else:
        print(f"  Beste Konfiguration: {best_wr * 100:.1f} % (+{margin:.1f} pp). "
              f"NICHT signifikant ueber dem Rauschen (2 SD = {2 * sd_final:.1f} pp) -- "
              f"Baseline beibehalten.")
    print("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": asdict(BASELINE),
        "best": asdict(best_params),
        "best_winrate_vs_baseline": best_wr,
        "improvement_pp": margin,
        "significant": significant,
        "winrate_sd_final_pp": sd_final,
        "games_screen": args.games_screen,
        "games_final": args.games_final,
        "screening": [{"params": asdict(p), "winrate": wr} for p, wr in screened],
        "final": [{"params": asdict(p), "winrate": wr} for p, wr in final],
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nErgebnis geschrieben: {args.output}")
    if significant:
        print("Naechster Schritt: diese Parameter als neue HeuristicPlayer-Defaults "
              "uebernehmen (players/heuristic_player.py).")


if __name__ == "__main__":
    main()
