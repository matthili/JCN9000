"""Misst den Spielstaerke-Effekt der Trumpf-Disziplin-Regel (Spur B).

Vergleicht zwei Heuristik-Konfigurationen, die sich NUR in der
void-tracking-Trumpf-Disziplin unterscheiden:

    A = HeuristicPlayer(trump_void_awareness=True)   # neu
    B = HeuristicPlayer(trump_void_awareness=False)  # alt (Baseline)

Beide sagen identisch an (gleiche Parameter), nur das Anspiel-Verhalten bei
blanken Gegnern differiert. Mit paired-eval (identische Kartenverteilung,
gespiegelte Sitze) faellt das Karten-Glueck als Rauschen weg, sodass die
Win-Rate-Differenz die Regel isoliert misst.

Die Ansage wird auf TRUMPF/GUMPF beschraenkt (`allowed_modes`), damit die Regel
ueberhaupt greifen kann (in Oben/Unten/Slalom gibt es keinen Trumpf).

Reines CPU-Tooling, kein TensorFlow -- ideal fuer die zweite Maschine.

Aufruf:
    python -m scripts.eval_heuristic_void_rule --games 8000 --workers 12
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time

from evaluation.tournament import two_team_match
from jass_engine.variant import PlayMode
from players.heuristic_player import HeuristicPlayer


TRUMP_MODES = {PlayMode.TRUMPF, PlayMode.GUMPF}


def _make_factory(void_aware: bool):
    def factory(seat: int, rng):
        return HeuristicPlayer(
            name=f"{'aware' if void_aware else 'base'}{seat}",
            trump_void_awareness=void_aware,
            allowed_modes=TRUMP_MODES,
            allow_slalom=False,
            rng=rng,
        )
    return factory


def _batch_worker(task: tuple) -> tuple[int, int, int]:
    """task = (num_games, target_score, seed). A = void-aware, B = Baseline.
    Returns (wins_a, wins_b, games_played)."""
    num_games, target_score, seed = task
    res = two_team_match(
        label_a="aware",
        factory_a=_make_factory(True),
        label_b="base",
        factory_b=_make_factory(False),
        num_games=num_games,
        target_score=target_score,
        seed=seed,
        paired_eval=True,
    )
    return res.stats_a.games_won, res.stats_b.games_won, res.games_played


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8000,
                        help="Gesamtanzahl Partien (durch 2*workers teilbar empfohlen).")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # In gerade Batches aufteilen (paired-eval braucht gerade Partienzahlen).
    per_batch = max(2, (args.games // args.workers) // 2 * 2)
    n_batches = max(1, args.games // per_batch)
    tasks = [(per_batch, args.target, args.seed + i * 101) for i in range(n_batches)]
    total_games = per_batch * n_batches

    print("=" * 70)
    print("  Trumpf-Disziplin-Eval: void-aware (A) vs. Baseline (B)")
    print(f"  {total_games} Partien (paired), Ansage beschraenkt auf Trumpf/Gumpf")
    print(f"  {n_batches} Batches x {per_batch} Partien, {args.workers} Worker")
    print("=" * 70)

    t0 = time.perf_counter()
    wins_a = wins_b = played = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for wa, wb, g in pool.imap_unordered(_batch_worker, tasks):
            wins_a += wa
            wins_b += wb
            played += g

    win_rate = wins_a / played if played else 0.0
    sd = math.sqrt(0.25 / played) if played else 0.0
    margin_pp = (win_rate - 0.5) * 100
    significant = margin_pp > 2 * sd * 100

    print(f"\nFertig in {(time.perf_counter() - t0) / 60:.1f} min.")
    print(f"  void-aware (A): {wins_a} Siege  ({win_rate * 100:.1f} %)")
    print(f"  Baseline   (B): {wins_b} Siege  ({(1 - win_rate) * 100:.1f} %)")
    print(f"  Win-Rate-SD: ~{sd * 100:.2f} pp,  2-SD-Schwelle: {2 * sd * 100:.1f} pp")
    print("-" * 70)
    if significant:
        print(f"  Die Trumpf-Disziplin verbessert die Heuristik signifikant "
              f"(+{margin_pp:.1f} pp). Default trump_void_awareness=True ist berechtigt.")
    elif margin_pp >= 0:
        print(f"  Leichter Vorteil (+{margin_pp:.1f} pp), aber NICHT signifikant "
              f"(< 2 SD). Schadet nicht -- die Regel ist beweisbar sound.")
    else:
        print(f"  Kein Vorteil ({margin_pp:.1f} pp). Unerwartet -- bitte melden, "
              f"dann schauen wir uns die Regel nochmal an.")
    print("=" * 70)


if __name__ == "__main__":
    main()
