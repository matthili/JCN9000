"""Parameter-Tuning fuer die Ansage-Heuristik des Solo-Jass.

Analog zu `scripts/tune_heuristic_announce.py` (Kreuz), aber fuer den
`SoloHeuristicPlayer`. Getunte Parameter:

    slalom_base_factor / slalom_concentration_factor / slalom_spread_factor
    gumpf_scale / oben_scale / unten_scale   (Trumpf = Anker, immer 1.0)

Schieben gibt es im Solo nicht (push_threshold irrelevant).

Messung: `four_way_match` -- Kandidat (A) + Baseline (B) + 2x Baseline (H)
am selben Tisch, paired-eval mit zyklischer Sitz-Rotation (4 Partien pro
Kartenverteilung -> Sitz-Vorteile und Karten-Glueck eliminiert). Da das
Kartenspiel beider Seiten identisch ist (gleiche geerbte Logik), misst der
Unterschied ausschliesslich die Ansage-Qualitaet.

Metrik: **bedingter Sieganteil** wins_A / (wins_A + wins_B). Baseline = 0.5.
Die zwei H-Sitze fangen die restlichen Siege ab, verzerren den A-vs-B-
Vergleich aber nicht (paired-Rotation).

Reines CPU-Tooling, kein TensorFlow.

Aufruf:
    python -m scripts.tune_solo_announce \\
        --games-screen 1000 --num-candidates 300 \\
        --games-final 20000 --top-k 6 \\
        --workers 12 --output solo_announce_tuned.json
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

from evaluation.solo_eval import four_way_match
from players.solo_heuristic_player import SoloHeuristicPlayer


@dataclass(frozen=True)
class SoloAnnounceParams:
    # Spiegelt die aktuellen SoloHeuristicPlayer-Defaults (= BASELINE-Anker).
    slalom_base_factor: float = 0.85
    slalom_concentration_factor: int = 1
    slalom_spread_factor: int = 1
    gumpf_scale: float = 1.0
    oben_scale: float = 1.0
    unten_scale: float = 1.0

    def as_tuple(self) -> tuple:
        return (
            self.slalom_base_factor,
            self.slalom_concentration_factor,
            self.slalom_spread_factor,
            self.gumpf_scale,
            self.oben_scale,
            self.unten_scale,
        )


BASELINE = SoloAnnounceParams()


def _sample_candidate(rng: random.Random) -> SoloAnnounceParams:
    return SoloAnnounceParams(
        slalom_base_factor=round(rng.uniform(0.70, 1.00), 2),
        slalom_concentration_factor=rng.randint(0, 3),
        slalom_spread_factor=rng.randint(0, 3),
        gumpf_scale=round(rng.uniform(0.75, 1.15), 2),
        oben_scale=round(rng.uniform(0.85, 1.15), 2),
        unten_scale=round(rng.uniform(0.85, 1.15), 2),
    )


def _make_factory(params: SoloAnnounceParams):
    def factory(seat: int, rng: random.Random) -> SoloHeuristicPlayer:
        return SoloHeuristicPlayer(
            name=f"S{seat}",
            slalom_base_factor=params.slalom_base_factor,
            slalom_concentration_factor=params.slalom_concentration_factor,
            slalom_spread_factor=params.slalom_spread_factor,
            gumpf_scale=params.gumpf_scale,
            oben_scale=params.oben_scale,
            unten_scale=params.unten_scale,
            rng=rng,
        )
    return factory


def _eval_candidate_worker(task: tuple) -> tuple[tuple, float, int]:
    """task = (params_tuple, baseline_tuple, num_games, target_score, seed).

    Returns (params_tuple, win_share_A_vs_B, n_decisive) mit
    win_share = wins_A / (wins_A + wins_B).
    """
    params_tuple, baseline_tuple, num_games, target_score, seed = task
    cand = SoloAnnounceParams(*params_tuple)
    base = SoloAnnounceParams(*baseline_tuple)
    res = four_way_match(
        label_a="cand",
        factory_a=_make_factory(cand),
        label_b="base",
        factory_b=_make_factory(base),
        label_h="base_h",
        factory_h=_make_factory(base),
        num_games=num_games,
        target_score=target_score,
        seed=seed,
        paired_eval=True,
    )
    wins_a = res.stats_a.games_won
    wins_b = res.stats_b.games_won
    decisive = wins_a + wins_b
    share = wins_a / decisive if decisive else 0.5
    return params_tuple, share, decisive


def _evaluate_many(
    candidates: list[SoloAnnounceParams],
    baseline: SoloAnnounceParams,
    num_games: int,
    target_score: int,
    workers: int,
    base_seed: int,
) -> list[tuple[SoloAnnounceParams, float, int]]:
    tasks = [
        (c.as_tuple(), baseline.as_tuple(), num_games, target_score, base_seed + i * 101)
        for i, c in enumerate(candidates)
    ]
    ctx = mp.get_context("spawn")
    results: list[tuple[SoloAnnounceParams, float, int]] = []
    with ctx.Pool(processes=workers) as pool:
        for params_tuple, share, decisive in pool.imap_unordered(_eval_candidate_worker, tasks):
            results.append((SoloAnnounceParams(*params_tuple), share, decisive))
    results.sort(key=lambda t: t[1], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-screen", type=int, default=1000,
                        help="Partien pro Kandidat im Screening (Vielfaches von 4).")
    parser.add_argument("--num-candidates", type=int, default=300)
    parser.add_argument("--games-final", type=int, default=20000,
                        help="Partien pro Finalist (Vielfaches von 4).")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("solo_announce_tuned.json"))
    args = parser.parse_args()

    if args.games_screen % 4 != 0 or args.games_final % 4 != 0:
        parser.error("--games-screen und --games-final muessen Vielfache von 4 sein "
                     "(paired-eval rotiert 4 Sitze).")

    rng = random.Random(args.seed)
    candidates = [BASELINE] + [_sample_candidate(rng) for _ in range(args.num_candidates)]

    # Entscheidende Partien (A- oder B-Sieg) sind ~die Haelfte aller Partien
    # (die zwei H-Sitze gewinnen die andere Haelfte) -> SD entsprechend.
    sd_screen = math.sqrt(0.25 / max(1, args.games_screen // 2)) * 100
    sd_final = math.sqrt(0.25 / max(1, args.games_final // 2)) * 100
    print("=" * 70)
    print("  Solo-Ansage-Tuning (4-Spieler, paired-eval gegen Baseline)")
    print(f"  Baseline: {BASELINE}")
    print(f"  Screening: {len(candidates)} Kandidaten x {args.games_screen} Partien "
          f"(Win-Share-SD ~{sd_screen:.1f} %)")
    print(f"  Finale:    Top-{args.top_k} x {args.games_final} Partien "
          f"(Win-Share-SD ~{sd_final:.1f} %)")
    print(f"  Worker: {args.workers}")
    print("=" * 70)

    t0 = time.perf_counter()
    print("\n[1/2] Screening laeuft ...")
    screened = _evaluate_many(
        candidates, BASELINE, args.games_screen, args.target, args.workers, args.seed,
    )
    print(f"  Screening fertig in {(time.perf_counter() - t0) / 60:.1f} min.")
    print("  Top-Kandidaten (Win-Share Kandidat vs Baseline):")
    for params, share, _n in screened[:min(args.top_k, len(screened))]:
        print(f"    {share * 100:5.1f} %  {params}")

    top = [p for p, _, _ in screened[:args.top_k]]
    print(f"\n[2/2] Finale Re-Evaluation der Top-{len(top)} ...")
    t1 = time.perf_counter()
    final = _evaluate_many(top, BASELINE, args.games_final, args.target, args.workers, args.seed + 7)
    print(f"  Finale fertig in {(time.perf_counter() - t1) / 60:.1f} min.")

    best_params, best_share, best_n = final[0]
    sd_emp = math.sqrt(0.25 / best_n) * 100 if best_n else 99.0
    margin = (best_share - 0.5) * 100
    significant = margin > 2 * sd_emp

    print("\n" + "=" * 70)
    print("  ERGEBNIS")
    for params, share, n in final:
        flag = "  <-- beste" if params == best_params else ""
        print(f"    {share * 100:5.1f} % (n={n})  {params}{flag}")
    print("-" * 70)
    if significant:
        print(f"  Beste Konfiguration: Win-Share {best_share * 100:.1f} % "
              f"(+{margin:.1f} pp, > 2 SD = {2 * sd_emp:.1f} pp). UEBERNEHMEN sinnvoll.")
    else:
        print(f"  Beste Konfiguration: {best_share * 100:.1f} % (+{margin:.1f} pp). "
              f"NICHT signifikant (2 SD = {2 * sd_emp:.1f} pp) -- Baseline behalten.")
    print("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": asdict(BASELINE),
        "best": asdict(best_params),
        "best_win_share_vs_baseline": best_share,
        "improvement_pp": margin,
        "significant": significant,
        "decisive_games_final": best_n,
        "screening": [{"params": asdict(p), "win_share": s, "decisive": n}
                      for p, s, n in screened],
        "final": [{"params": asdict(p), "win_share": s, "decisive": n}
                  for p, s, n in final],
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nErgebnis geschrieben: {args.output}")


if __name__ == "__main__":
    main()
