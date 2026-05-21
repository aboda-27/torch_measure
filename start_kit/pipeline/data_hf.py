"""Load measurement-db with domain-stratified benchmark selection (start_kit/README-aligned)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sampling import (
    RawTriple,
    get_sampling_config,
    primary_domain,
    resolve_benchmark_ids,
    subsample_rows,
)
from utils import ARTIFACTS_DIR, SPLIT_SEED, build_index, render_subject_content

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def load_benchmark_manifest() -> dict[str, str]:
    """Return ``{benchmark_id: primary_domain}`` from ``benchmarks.parquet`` on HF."""
    from huggingface_hub import hf_hub_download

    from torch_measure.datasets._manifest import _coerce_list

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="benchmarks.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    if "benchmark_id" not in df.columns:
        raise ValueError("benchmarks.parquet missing benchmark_id column")

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        bid = str(row["benchmark_id"])
        domain = _coerce_list(row.get("domain"))
        out[bid] = primary_domain(domain)
    return out


def triples_from_long_form(data) -> list[RawTriple]:
    """Join responses to item/subject registries (same rules as utils.load_training_data)."""
    items_df = data.items.set_index("item_id", drop=False)
    subjects_df = data.subjects.set_index("subject_id", drop=False)
    rows: list[RawTriple] = []

    for _, row in data.responses.iterrows():
        item_id = row["item_id"]
        subject_id = row["subject_id"]

        if item_id not in items_df.index:
            continue
        item_row = items_df.loc[[item_id]].iloc[0]

        content = item_row.get("content")
        if content is None or (isinstance(content, float) and np.isnan(content)):
            continue
        item_content = str(content).strip()
        if not item_content:
            continue

        if subject_id not in subjects_df.index:
            continue
        subject_row = subjects_df.loc[[subject_id]].iloc[0]
        subject_content = render_subject_content(subject_row.to_dict(), str(subject_id))

        label = 1.0 if float(row["response"]) >= 0.5 else 0.0
        rows.append((subject_content, item_content, label))

    return rows


def load_triples_per_benchmark(benchmark_ids: list[str]) -> list[RawTriple]:
    """Load each benchmark via torch_measure (one parquet at a time)."""
    from torch_measure.datasets import load

    raw: list[RawTriple] = []
    for name in benchmark_ids:
        print(f"Loading {name}...")
        data = load(name)
        raw.extend(triples_from_long_form(data))
    return raw


def load_triples_bulk_hf(benchmark_ids: list[str]) -> list[RawTriple]:
    """README-style bulk load: all response parquets + registries (high memory)."""
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    selected = set(benchmark_ids)
    response_files = sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
        and name.removesuffix(".parquet") in selected
    )

    response_features = Features(
        {
            "subject_id": Value("string"),
            "item_id": Value("string"),
            "benchmark_id": Value("string"),
            "trial": Value("int64"),
            "test_condition": Value("string"),
            "response": Value("float64"),
            "correct_answer": Value("string"),
            "trace": Value("string"),
        }
    )

    print(f"Bulk loading {len(response_files)} response parquets from HF...")
    responses = load_dataset(
        REPO_ID,
        data_files=response_files,
        features=response_features,
        split="train",
    )
    items_ds = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects_ds = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")

    items_by_id = {row["item_id"]: row for row in items_ds}
    subjects_by_id = {row["subject_id"]: row for row in subjects_ds}

    rows: list[RawTriple] = []
    for row in responses:
        if row["benchmark_id"] not in selected:
            continue
        item = items_by_id.get(row["item_id"], {})
        subject = subjects_by_id.get(row["subject_id"], {})
        content = item.get("content")
        if content is None or (isinstance(content, float) and np.isnan(content)):
            continue
        item_content = str(content).strip()
        if not item_content:
            continue
        subject_content = render_subject_content(subject, row["subject_id"])
        label = 1.0 if float(row["response"]) >= 0.5 else 0.0
        rows.append((subject_content, item_content, label))

    return rows


def save_sampling_artifacts(
    benchmark_summary: dict[str, Any],
    sampling_meta: dict[str, Any],
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "selected_benchmarks.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    with open(root / "sampling_meta.json", "w") as f:
        json.dump(sampling_meta, f, indent=2)


def load_representative_training_data(
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> tuple[list[str], list[str], list[tuple[int, int, float]], dict[str, int], dict[str, int]]:
    """Domain-stratified benchmarks + row subsample → indexed triples."""
    config = get_sampling_config()
    benchmark_domains = load_benchmark_manifest()
    benchmark_ids, benchmark_summary = resolve_benchmark_ids(benchmark_domains, config)

    print(
        f"Sampling mode={config['mode']}: {len(benchmark_ids)} benchmarks "
        f"(of {len(benchmark_domains)} in manifest)"
    )

    if config["bulk_load"]:
        raw_triples = load_triples_bulk_hf(benchmark_ids)
    else:
        raw_triples = load_triples_per_benchmark(benchmark_ids)

    n_raw = len(raw_triples)
    raw_triples = subsample_rows(
        raw_triples,
        frac=config["row_sample_frac"],
        seed=config["seed"],
    )
    n_sampled = len(raw_triples)
    print(
        f"Row subsample: {n_raw} raw observations → {n_sampled} "
        f"({config['row_sample_frac']:.0%}, seed={config['seed']})"
    )

    sampling_meta = {
        **benchmark_summary,
        "n_rows_raw": n_raw,
        "n_rows_after_subsample": n_sampled,
        "bulk_load": config["bulk_load"],
    }
    save_sampling_artifacts(benchmark_summary, sampling_meta, artifacts_dir)

    subject_strings = [t[0] for t in raw_triples]
    item_strings = [t[1] for t in raw_triples]
    subject2idx = build_index(subject_strings)
    item2idx = build_index(item_strings)

    unique_subjects = sorted(subject2idx.keys())
    unique_items = sorted(item2idx.keys())

    triples = [
        (subject2idx[s], item2idx[i], y)
        for s, i, y in raw_triples
    ]

    print(
        f"Observations: {len(triples)}, subjects: {len(unique_subjects)}, "
        f"items: {len(unique_items)}"
    )
    return unique_items, unique_subjects, triples, item2idx, subject2idx
