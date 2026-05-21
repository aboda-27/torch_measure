"""Unit tests for pipeline benchmark/row sampling (no HF network)."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "start_kit" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from sampling import (  # noqa: E402
    resolve_benchmark_ids,
    select_benchmarks_stratified,
    subsample_rows,
)


def test_stratified_selects_from_each_domain():
    domains = {
        "a1": "math",
        "a2": "math",
        "b1": "code",
        "c1": "misc",
    }
    selected = select_benchmarks_stratified(domains, per_domain_cap=1, seed=42)
    assert len(selected) == 3
    assert len({domains[b] for b in selected}) == 3


def test_subsample_rows_respects_frac():
    rows = [(f"s{i}", f"i{i}", 1.0) for i in range(1000)]
    out = subsample_rows(rows, frac=0.25, seed=42)
    assert 150 < len(out) < 350


def test_resolve_legacy_max_mode():
    domains = {f"b{i}": "d" for i in range(10)}
    config = {
        "mode": "legacy_max",
        "row_sample_frac": 1.0,
        "per_domain_cap": None,
        "legacy_max_datasets": 3,
        "seed": 42,
        "bulk_load": False,
    }
    ids, summary = resolve_benchmark_ids(domains, config)
    assert len(ids) == 3
    assert summary["mode"] == "legacy_max"
