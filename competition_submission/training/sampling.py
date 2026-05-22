"""Benchmark and row sampling for representative training sets."""

from __future__ import annotations

import os
import random
from typing import Any

from utils import SPLIT_SEED

RawTriple = tuple[str, str, float]

DEFAULT_SAMPLE_MODE = "stratified"
DEFAULT_ROW_SAMPLE_FRAC = 0.25


def primary_domain(domain: list[str] | None) -> str:
    """Stratum key: first domain tag, or ``misc``."""
    if domain:
        return domain[0]
    return "misc"


def select_benchmarks_stratified(
    benchmark_domains: dict[str, str],
    *,
    per_domain_cap: int | None = None,
    seed: int = SPLIT_SEED,
) -> list[str]:
    """Pick benchmark ids with at least one per domain stratum (up to cap per domain)."""
    by_domain: dict[str, list[str]] = {}
    for bid, dom in benchmark_domains.items():
        by_domain.setdefault(dom, []).append(bid)

    rng = random.Random(seed)
    selected: list[str] = []

    for dom in sorted(by_domain.keys()):
        ids = sorted(by_domain[dom])
        rng.shuffle(ids)
        if per_domain_cap is not None:
            ids = ids[:per_domain_cap]
        selected.extend(ids)

    return sorted(selected)


def select_benchmarks_all(benchmark_domains: dict[str, str]) -> list[str]:
    return sorted(benchmark_domains.keys())


def select_benchmarks_legacy_max(
    benchmark_domains: dict[str, str],
    *,
    max_count: int,
) -> list[str]:
    return sorted(benchmark_domains.keys())[:max_count]


def subsample_rows(
    rows: list[RawTriple],
    *,
    frac: float,
    seed: int = SPLIT_SEED,
) -> list[RawTriple]:
    """Bernoulli row subsample: keep each row independently with probability ``frac``."""
    if frac >= 1.0:
        return list(rows)
    if frac <= 0.0:
        return []

    rng = random.Random(seed)
    return [row for row in rows if rng.random() < frac]


def get_sampling_config() -> dict[str, Any]:
    """Read sampling knobs from environment."""
    mode = os.environ.get("PIPELINE_SAMPLE_MODE", DEFAULT_SAMPLE_MODE).strip().lower()
    row_frac = float(os.environ.get("PIPELINE_ROW_SAMPLE_FRAC", str(DEFAULT_ROW_SAMPLE_FRAC)))

    per_domain = os.environ.get("PIPELINE_BENCHMARKS_PER_DOMAIN")
    per_domain_cap = int(per_domain) if per_domain else None

    max_ds = os.environ.get("PIPELINE_MAX_DATASETS")
    legacy_max = int(max_ds) if max_ds else None

    bulk = os.environ.get("PIPELINE_HF_BULK_LOAD", "").strip().lower() in ("1", "true", "yes")

    return {
        "mode": mode,
        "row_sample_frac": row_frac,
        "per_domain_cap": per_domain_cap,
        "legacy_max_datasets": legacy_max,
        "seed": SPLIT_SEED,
        "bulk_load": bulk,
    }


def resolve_benchmark_ids(
    benchmark_domains: dict[str, str],
    config: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return benchmark ids to load and a summary dict for artifacts."""
    config = config or get_sampling_config()
    mode = config["mode"]
    seed = config["seed"]

    if mode == "all":
        ids = select_benchmarks_all(benchmark_domains)
    elif mode == "legacy_max":
        n = config["legacy_max_datasets"]
        if n is None:
            raise ValueError(
                "PIPELINE_SAMPLE_MODE=legacy_max requires PIPELINE_MAX_DATASETS=N"
            )
        ids = select_benchmarks_legacy_max(benchmark_domains, max_count=n)
    elif mode == "stratified":
        ids = select_benchmarks_stratified(
            benchmark_domains,
            per_domain_cap=config["per_domain_cap"],
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown PIPELINE_SAMPLE_MODE={mode!r}; use stratified, all, or legacy_max"
        )

    by_domain: dict[str, list[str]] = {}
    for bid in ids:
        dom = benchmark_domains.get(bid, "misc")
        by_domain.setdefault(dom, []).append(bid)

    summary = {
        "mode": mode,
        "seed": seed,
        "per_domain_cap": config["per_domain_cap"],
        "legacy_max_datasets": config["legacy_max_datasets"],
        "row_sample_frac": config["row_sample_frac"],
        "n_benchmarks_available": len(benchmark_domains),
        "n_benchmarks_selected": len(ids),
        "benchmark_ids": ids,
        "by_domain": {d: sorted(v) for d, v in sorted(by_domain.items())},
    }
    return ids, summary
