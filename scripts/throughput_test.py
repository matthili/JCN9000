"""Throughput-Vergleichstest fuer die MCTS-Datengen.

Fuehrt mehrere Konfigurationen von --workers / --parallel-threads-per-worker
gegeneinander aus, misst jeweils die Wallzeit und gibt am Ende eine
Vergleichstabelle aus. Damit kann man empirisch finden, welche Verteilung von
Worker-Prozessen und Game-Threads auf der eigenen Hardware am schnellsten ist.

Aufruf:
    python -m scripts.throughput_test \
        --warm-start models/v5/best.keras \
        --games-per-variant 10 \
        --variants trumpf_eichel gumpf_eichel oben slalom_unten \
        --configs 4x8,4x32,2x64,1x64

Die Konfigs sind als "WxT" notiert: W Worker-Prozesse, T Threads pro Worker.

Default-Konfigs decken eine sinnvolle Bandbreite ab:
    4x8   = wenig Threads, viele Worker (typisch fuer GIL-Vorsicht)
    4x32  = vierfache Threads, weiterhin 4 Worker
    2x64  = weniger Worker, mehr Threads pro Worker
    1x64  = ein Prozess, viele Threads (maximal moegliche Batch-Buendelung)

Was die Tabelle zeigt:
    Spiele       Anzahl effektiv gespielter Partien (games_per_variant * Varianten)
    Dauer        Wallzeit fuer diese Konfig (inkl. TF-Start pro Worker)
    Spiele/min   Durchsatz -- die Hauptmetrik

Tipps:
    - games_per_variant zu klein -> Startup-Overhead dominiert -> Zahlen unzuverlaessig
    - games_per_variant zu gross -> sehr lange Testdauer
    - Faustregel: pro Konfig mindestens 3-5 Minuten reine Spielzeit
    - 1x128 oder mehr Threads kann auf manchen WSL2-Setups haengen -- erst klein anfangen
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_config(s: str) -> tuple[int, int]:
    """'4x8' -> (4, 8)"""
    parts = s.split("x")
    if len(parts) != 2:
        raise ValueError(f"Konfig '{s}' muss Form 'WxT' haben, z.B. '4x8'")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Konfig '{s}' enthaelt keine Ganzzahlen")


def run_one_config(workers: int, threads: int, args: argparse.Namespace) -> dict:
    """Fuehrt EINE Konfig aus, liefert dict mit Metriken."""
    out_dir = Path(args.tmp_root) / f"run_{workers}x{threads}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable, "-u", "-m", "training.data.generate_mcts_data_mp",
        "--warm-start", args.warm_start,
        "--games-per-variant", str(args.games_per_variant),
        "--rollouts-per-card", str(args.rollouts_per_card),
        "--target", str(args.target),
        "--workers", str(workers),
        "--parallel-threads-per-worker", str(threads),
        "--inference-batch-size", str(args.inference_batch_size),
        "--lookahead-mode", args.lookahead_mode,
        "--output", str(out_dir),
    ]
    if args.variants:
        cmd.extend(["--variants", *args.variants])

    print()
    print("=" * 70)
    print(f"  Konfig: {workers} Worker x {threads} Threads pro Worker")
    print("=" * 70)
    print("  $ " + " ".join(cmd))
    print()

    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=False)
        success = result.returncode == 0
    except KeyboardInterrupt:
        print("\n[abgebrochen]")
        raise
    elapsed = time.perf_counter() - start

    # Anzahl Varianten ermitteln (entweder explizit oder Default 12)
    if args.variants:
        n_variants = len(args.variants)
    else:
        n_variants = 12  # ALL_VARIANTS in generate_mcts_data.py

    total_games = args.games_per_variant * n_variants

    if not args.keep_data:
        shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "workers": workers,
        "threads": threads,
        "total_games": total_games,
        "elapsed_s": elapsed,
        "games_per_sec": total_games / elapsed if elapsed > 0 else 0.0,
        "games_per_min": (total_games / elapsed * 60) if elapsed > 0 else 0.0,
        "success": success,
    }


def format_table(results: list[dict]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Throughput-Vergleich")
    lines.append("=" * 70)
    lines.append("")
    header = f"{'Konfig':<14} {'Spiele':>8} {'Dauer':>10} {'Spiele/s':>10} {'Spiele/Min':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    successful = [r for r in results if r["success"]]
    baseline = successful[0] if successful else None

    for r in results:
        config_str = f"{r['workers']}w x {r['threads']}t"
        if not r["success"]:
            lines.append(f"{config_str:<14} {'FEHLER':>8}")
            continue
        elapsed_str = f"{r['elapsed_s'] / 60:>5.1f} min"
        rate_s = r["games_per_sec"]
        rate_m = r["games_per_min"]
        speedup_marker = ""
        if baseline and r is not baseline:
            speedup = rate_s / baseline["games_per_sec"] if baseline["games_per_sec"] > 0 else 0
            speedup_marker = f"  ({speedup:.2f}x)"
        lines.append(
            f"{config_str:<14} {r['total_games']:>8} {elapsed_str:>10} "
            f"{rate_s:>10.2f} {rate_m:>12.1f}{speedup_marker}"
        )

    lines.append("")
    if successful:
        winner = max(successful, key=lambda r: r["games_per_sec"])
        lines.append(
            f"Schnellste Konfig: {winner['workers']}w x {winner['threads']}t "
            f"({winner['games_per_min']:.1f} Spiele/Min)"
        )
        if baseline and winner is not baseline and baseline["games_per_sec"] > 0:
            total_speedup = winner["games_per_sec"] / baseline["games_per_sec"]
            lines.append(f"Speedup gegenueber erster Konfig: {total_speedup:.2f}x")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Vergleicht Durchsatz verschiedener Worker/Thread-Konfigs der MCTS-Datengen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--warm-start", required=True,
        help="Pfad zum NN-Modell, das als Lehrer-Init dient (z.B. models/v5/best.keras).",
    )
    parser.add_argument(
        "--games-per-variant", type=int, default=10,
        help="Wieviele Partien pro Variante in jedem Testlauf (Default 10). "
             "Klein machen fuer schnelle Tests, gross genug fuer stabile Zahlen.",
    )
    parser.add_argument(
        "--rollouts-per-card", type=int, default=30,
        help="Wie im echten Lauf -- aendert den Inferenz-Druck pro Zug (Default 30).",
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument(
        "--inference-batch-size", type=int, default=1024,
        help="Max. Batch-Groesse pro InferenceServer (Default 1024).",
    )
    parser.add_argument(
        "--lookahead-mode",
        choices=["single-trick", "full-round-vec"],
        default="full-round-vec",
    )
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Bestimmte Varianten zum Testen, z.B. 'trumpf_eichel oben slalom_unten'. "
             "Weniger Varianten = schnellerer Test, aber sollte mind. so gross sein "
             "wie die hoechste Worker-Zahl, damit jeder Worker etwas zu tun bekommt.",
    )
    parser.add_argument(
        "--configs", default="4x8,4x32,2x64,1x64",
        help="Komma-separierte WxT-Liste. Default: '4x8,4x32,2x64,1x64'.",
    )
    parser.add_argument(
        "--tmp-root", default="data/mcts/throughput_test",
        help="Wo die temporaeren Test-Outputs landen (werden danach geloescht, ausser --keep-data).",
    )
    parser.add_argument(
        "--keep-data", action="store_true",
        help="Generierte Test-Daten nach jedem Lauf behalten (sonst geloescht).",
    )
    args = parser.parse_args()

    # Konfigs parsen
    try:
        configs = [parse_config(s.strip()) for s in args.configs.split(",")]
    except ValueError as e:
        parser.error(str(e))

    # Vorab-Sanity-Check: variants mind. so viele wie max(workers)
    max_workers = max(w for w, _ in configs)
    n_variants_test = len(args.variants) if args.variants else 12
    if n_variants_test < max_workers:
        print(
            f"WARNUNG: {n_variants_test} Varianten < {max_workers} max. Worker -- "
            f"einige Worker werden in 'WxT'-Konfigs mit grossem W leerlaufen.\n"
            f"Empfehlung: mind. {max_workers} Varianten waehlen oder Worker reduzieren.",
            file=sys.stderr,
        )

    print(f"Geplante Konfigs: {[f'{w}x{t}' for w, t in configs]}")
    print(f"Pro Konfig: {args.games_per_variant} Spiele x {n_variants_test} Varianten "
          f"= {args.games_per_variant * n_variants_test} Spiele")
    print()

    results = []
    total_start = time.perf_counter()
    try:
        for workers, threads in configs:
            r = run_one_config(workers, threads, args)
            results.append(r)
    except KeyboardInterrupt:
        print("\nDurchlauf abgebrochen. Bisherige Ergebnisse:")

    total_elapsed = time.perf_counter() - total_start

    print(format_table(results))
    print()
    print(f"Gesamtdauer: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
