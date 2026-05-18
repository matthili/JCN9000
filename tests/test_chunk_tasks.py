"""Tests fuer den Chunk-Task-Builder in den MP-Datengen-Skripten.

Diese Tests sind TF-frei -- sie pruefen nur die reine Task-Berechnung.
"""

from __future__ import annotations

from jass_engine.card import Suit
from jass_engine.variant import Announcement, Variant
from training.data.generate_mcts_data_mp import _build_chunk_tasks as build_team_chunks
from training.data.generate_solo_mcts_data_mp import _build_chunk_tasks as build_solo_chunks
from training.data.generate_mcts_data import VariantSpec as TeamVariantSpec
from training.data.generate_solo_mcts_data import VariantSpec as SoloVariantSpec


def _three_variants_team():
    return [
        TeamVariantSpec("trumpf_eichel", Announcement(variant=Variant.trumpf(Suit.EICHEL))),
        TeamVariantSpec("oben", Announcement(variant=Variant.oben())),
        TeamVariantSpec("slalom_unten", Announcement(variant=Variant.unten(), slalom=True)),
    ]


def _three_variants_solo():
    return [
        SoloVariantSpec("trumpf_eichel", Announcement(variant=Variant.trumpf(Suit.EICHEL))),
        SoloVariantSpec("oben", Announcement(variant=Variant.oben())),
        SoloVariantSpec("slalom_unten", Announcement(variant=Variant.unten(), slalom=True)),
    ]


# --- Team-Chunks ---


def test_team_chunks_glatte_teilung():
    """100 Spiele, Chunks à 50 -> 2 Chunks pro Variante, 3 Varianten = 6 Tasks."""
    variants = _three_variants_team()
    tasks = build_team_chunks(
        selected=variants, games_per_variant=100, games_per_chunk=50, base_seed=42
    )
    assert len(tasks) == 6
    # Jede Variante hat genau 2 Chunks
    chunks_per_variant: dict[str, list[int]] = {}
    for vs, chunk_idx, games_in_chunk, _seed in tasks:
        chunks_per_variant.setdefault(vs.label, []).append(chunk_idx)
        assert games_in_chunk == 50
    for v in variants:
        assert sorted(chunks_per_variant[v.label]) == [0, 1]


def test_team_chunks_rest_im_letzten_chunk():
    """123 Spiele, Chunks à 50 -> Chunks von 50, 50, 23."""
    variants = _three_variants_team()[:1]  # nur eine Variante
    tasks = build_team_chunks(
        selected=variants, games_per_variant=123, games_per_chunk=50, base_seed=42
    )
    assert len(tasks) == 3
    games = sorted([t[2] for t in tasks])
    assert games == [23, 50, 50]
    assert sum(games) == 123


def test_team_chunks_chunk_groesser_als_total():
    """Wenn chunk_size > games_per_variant, gibt es genau einen Chunk."""
    variants = _three_variants_team()[:1]
    tasks = build_team_chunks(
        selected=variants, games_per_variant=30, games_per_chunk=50, base_seed=42
    )
    assert len(tasks) == 1
    assert tasks[0][2] == 30  # games_in_chunk


def test_team_chunks_eindeutige_seeds_pro_chunk():
    """Verschiedene chunk_idx muessen verschiedene Seeds liefern.

    Sonst wuerde jeder Chunk derselben Variante dieselben Spiele simulieren.
    """
    variants = _three_variants_team()[:1]
    tasks = build_team_chunks(
        selected=variants, games_per_variant=200, games_per_chunk=50, base_seed=42
    )
    seeds = [t[3] for t in tasks]
    assert len(set(seeds)) == len(seeds), f"Seed-Kollision in {seeds}"


def test_team_chunks_seeds_unterscheiden_sich_pro_variante():
    """Chunks verschiedener Varianten haben verschiedene Seeds."""
    variants = _three_variants_team()
    tasks = build_team_chunks(
        selected=variants, games_per_variant=50, games_per_chunk=50, base_seed=42
    )
    # Pro Variante 1 Chunk -> 3 Tasks, 3 verschiedene Seeds
    seeds = [t[3] for t in tasks]
    assert len(set(seeds)) == len(seeds)


# --- Solo-Chunks ---


def test_solo_chunks_genau_wie_team():
    """Die Solo-Variante des Builders verhaelt sich identisch zur Team-Variante."""
    variants = _three_variants_solo()
    tasks = build_solo_chunks(
        selected=variants, games_per_variant=100, games_per_chunk=25, base_seed=42
    )
    assert len(tasks) == 12  # 3 Varianten x 4 Chunks
    for _vs, _chunk_idx, games_in_chunk, _seed in tasks:
        assert games_in_chunk == 25


def test_solo_chunks_aufstellung_korrekt():
    """Prueft die exakte Aufstellung der Chunks."""
    variants = _three_variants_solo()[:1]  # nur trumpf_eichel
    tasks = build_solo_chunks(
        selected=variants, games_per_variant=50, games_per_chunk=20, base_seed=42
    )
    # 50 / 20 = 2.5 -> 3 Chunks: 20, 20, 10
    assert len(tasks) == 3
    games_in_chunks = sorted([t[2] for t in tasks])
    assert games_in_chunks == [10, 20, 20]
    # Chunk-Indices 0, 1, 2
    chunk_indices = sorted([t[1] for t in tasks])
    assert chunk_indices == [0, 1, 2]


# --- Allgemein ---


def test_chunks_reihenfolge_nach_variante_dann_chunk_idx():
    """Tasks sollten in der Reihenfolge (Variant 0 Chunks, Variant 1 Chunks, ...) sein.

    Das ist nicht zwingend erforderlich (die Queue mischt eh dynamisch), aber gibt
    eine deterministische Eingabereihenfolge fuer Reproduzierbarkeit.
    """
    variants = _three_variants_team()
    tasks = build_team_chunks(
        selected=variants, games_per_variant=100, games_per_chunk=50, base_seed=42
    )
    labels_in_order = [t[0].label for t in tasks]
    chunks_in_order = [t[1] for t in tasks]
    # Erwartete Reihenfolge: trumpf_eichel#0, trumpf_eichel#1, oben#0, oben#1, slalom_unten#0, slalom_unten#1
    assert labels_in_order == [
        "trumpf_eichel", "trumpf_eichel",
        "oben", "oben",
        "slalom_unten", "slalom_unten",
    ]
    assert chunks_in_order == [0, 1, 0, 1, 0, 1]
