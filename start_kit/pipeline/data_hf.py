"""Load measurement-db with domain-stratified benchmark selection (start_kit/README-aligned)."""

from __future__ import annotations

import json
import os
import random
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
from utils import (
    ARTIFACTS_DIR,
    SPLIT_SEED,
    build_index,
    format_item_content,
    render_subject_content,
)

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

_RESPONSE_COLS = (
    "subject_id",
    "item_id",
    "benchmark_id",
    "response",
    "test_condition",
)
_ITEM_COLS = ("item_id", "content", "benchmark_id")
_SUBJECT_COLS = ("subject_id", "display_name", "provider", "params", "release_date", "family")
_DEFAULT_PARQUET_BATCH_SIZE = 65_536


def _parquet_batch_size() -> int:
    raw = os.environ.get("PIPELINE_PARQUET_BATCH_SIZE", str(_DEFAULT_PARQUET_BATCH_SIZE))
    return max(1024, int(raw))


def _use_legacy_load() -> bool:
    return os.environ.get("PIPELINE_LEGACY_LOAD", "").strip().lower() in ("1", "true", "yes")


def _use_full_parquet_read() -> bool:
    return os.environ.get("PIPELINE_FULL_PARQUET_READ", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
    )


def _read_parquet(path: Path, columns: tuple[str, ...] | None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    except (KeyError, ValueError):
        return pd.read_parquet(path)


def _pick_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    return [c for c in wanted if c in names]


def load_benchmark_manifest() -> dict[str, str]:
    """Return ``{benchmark_id: primary_domain}`` from ``benchmarks.parquet`` on HF."""
    from torch_measure.datasets._manifest import _coerce_list

    df = _read_parquet(_hf_download("benchmarks.parquet"), ("benchmark_id", "domain"))
    if "benchmark_id" not in df.columns:
        raise ValueError("benchmarks.parquet missing benchmark_id column")

    domains = df["domain"] if "domain" in df.columns else pd.Series([None] * len(df))
    out: dict[str, str] = {}
    for bid, domain in zip(df["benchmark_id"], domains, strict=False):
        out[str(bid)] = primary_domain(_coerce_list(domain))
    return out


def _scan_parquet_filtered(
    path: Path,
    columns: tuple[str, ...],
    id_column: str,
    needed_ids: set[str],
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    if not needed_ids:
        return pd.DataFrame(columns=list(columns))

    cols = _pick_columns(path, columns)
    if id_column not in cols:
        return _read_parquet(path, tuple(cols) if cols else None)

    parts: list[pd.DataFrame] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_parquet_batch_size(), columns=cols):
        df = batch.to_pandas()
        hit = df[id_column].astype(str).isin(needed_ids)
        if hit.any():
            parts.append(df.loc[hit])

    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def _subsample_response_rows(
    responses: pd.DataFrame,
    *,
    frac: float,
    rng: random.Random,
) -> pd.DataFrame:
    if frac >= 1.0 or len(responses) == 0:
        return responses
    if frac <= 0.0:
        return responses.iloc[0:0]

    keep = [rng.random() < frac for _ in range(len(responses))]
    return responses.loc[keep].reset_index(drop=True)


def _load_responses_sampled(
    benchmark_id: str,
    *,
    row_sample_frac: float,
    rng: random.Random,
) -> tuple[pd.DataFrame, int]:
    path = _hf_download(f"{benchmark_id}.parquet")
    cols = _pick_columns(path, _RESPONSE_COLS)
    if not cols:
        return pd.DataFrame(), 0

    if _use_full_parquet_read() or row_sample_frac >= 1.0:
        df = _read_parquet(path, tuple(cols))
        n_raw = len(df)
        if row_sample_frac < 1.0:
            df = _subsample_response_rows(df, frac=row_sample_frac, rng=rng)
        return df, n_raw

    import pyarrow.parquet as pq

    parts: list[pd.DataFrame] = []
    n_raw = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_parquet_batch_size(), columns=cols):
        df = batch.to_pandas()
        n_raw += len(df)
        keep = [rng.random() < row_sample_frac for _ in range(len(df))]
        if any(keep):
            parts.append(df.loc[keep])

    if not parts:
        return pd.DataFrame(columns=cols), n_raw
    return pd.concat(parts, ignore_index=True), n_raw


def _bench_items_subjects(
    items: pd.DataFrame,
    subjects: pd.DataFrame,
    benchmark_id: str,
    response_subject_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "benchmark_id" in items.columns:
        items_b = items.loc[items["benchmark_id"] == benchmark_id]
    else:
        items_b = items

    present = set(response_subject_ids.dropna().unique())
    if present and "subject_id" in subjects.columns:
        subjects_b = subjects.loc[subjects["subject_id"].isin(present)]
    else:
        subjects_b = subjects

    return items_b, subjects_b


def triples_from_frames(
    responses: pd.DataFrame,
    items: pd.DataFrame,
    subjects: pd.DataFrame,
    benchmark_id: str,
) -> list[RawTriple]:
    if responses.empty:
        return []

    items_b, subjects_b = _bench_items_subjects(
        items, subjects, benchmark_id, responses["subject_id"]
    )
    if items_b.empty or subjects_b.empty:
        return []

    items_idx = items_b.drop_duplicates(subset=["item_id"], keep="first").set_index(
        "item_id", drop=False
    )
    subjects_idx = subjects_b.drop_duplicates(subset=["subject_id"], keep="first").set_index(
        "subject_id", drop=False
    )

    merged = responses.merge(
        items_idx[["content"]],
        left_on="item_id",
        right_index=True,
        how="inner",
    )
    content = merged["content"]
    str_content = content.astype(str).str.strip()
    valid = content.notna() & str_content.ne("") & str_content.ne("nan")
    merged = merged.loc[valid]
    if merged.empty:
        return []

    merged = merged.loc[merged["subject_id"].isin(subjects_idx.index)]
    if merged.empty:
        return []

    unique_subject_ids = merged["subject_id"].unique()
    subject_content_by_id = {
        str(sid): render_subject_content(subjects_idx.loc[sid].to_dict(), str(sid))
        for sid in unique_subject_ids
    }

    if "benchmark_id" in merged.columns:
        bench_series = merged["benchmark_id"].fillna(benchmark_id).astype(str)
    else:
        bench_series = pd.Series([benchmark_id] * len(merged), index=merged.index)

    if "test_condition" in merged.columns:
        cond_series = merged["test_condition"].fillna("none").astype(str)
    else:
        cond_series = pd.Series(["none"] * len(merged), index=merged.index)

    content_series = merged["content"].astype(str).str.strip()
    item_contents = [
        format_item_content(b, c, t)
        for b, c, t in zip(bench_series, cond_series, content_series, strict=True)
    ]
    subject_contents = merged["subject_id"].map(subject_content_by_id).tolist()
    labels = (merged["response"].astype(float) >= 0.5).astype(float).tolist()
    return list(zip(subject_contents, item_contents, labels, strict=True))


def triples_from_long_form(data) -> list[RawTriple]:
    return triples_from_frames(
        data.responses, data.items, data.subjects, data.name
    )


def load_triples_fast(
    benchmark_ids: list[str],
    *,
    row_sample_frac: float,
    seed: int = SPLIT_SEED,
) -> tuple[list[RawTriple], int, str]:
    rng = random.Random(seed)
    n_raw_rows = 0
    sampled: list[tuple[str, pd.DataFrame]] = []
    needed_item_ids: set[str] = set()
    needed_subject_ids: set[str] = set()
    stream = row_sample_frac < 1.0 and not _use_full_parquet_read()

    print(
        f"Pass 1/2: {'streaming' if stream else 'full read'} response parquets "
        f"(keep≈{row_sample_frac:.0%}, seed={seed})..."
    )
    for name in benchmark_ids:
        print(f"  {name}...")
        responses, n_raw = _load_responses_sampled(
            name, row_sample_frac=row_sample_frac, rng=rng
        )
        n_raw_rows += n_raw
        if responses.empty:
            print(f"    {n_raw} rows scanned → 0 kept")
            continue
        needed_item_ids.update(responses["item_id"].dropna().astype(str).tolist())
        needed_subject_ids.update(responses["subject_id"].dropna().astype(str).tolist())
        sampled.append((name, responses))
        print(f"    {n_raw} rows scanned → {len(responses)} kept")

    if not sampled:
        return [], n_raw_rows, "streaming_sample_empty"

    if stream and row_sample_frac < 1.0:
        print(
            f"Pass 2/2: registry scan for {len(needed_item_ids)} items, "
            f"{len(needed_subject_ids)} subjects..."
        )
        items = _scan_parquet_filtered(
            _hf_download("items.parquet"), _ITEM_COLS, "item_id", needed_item_ids
        )
        subjects = _scan_parquet_filtered(
            _hf_download("subjects.parquet"),
            _SUBJECT_COLS,
            "subject_id",
            needed_subject_ids,
        )
        mode = "modal_streaming_sample_filtered_registry"
    else:
        print("Pass 2/2: loading full registries...")
        items = _read_parquet(_hf_download("items.parquet"), _ITEM_COLS)
        subjects = _read_parquet(_hf_download("subjects.parquet"), _SUBJECT_COLS)
        mode = "modal_full_read_registry"

    print("Pass 3/3: joining...")
    raw: list[RawTriple] = []
    for name, responses in sampled:
        n_before = len(raw)
        raw.extend(triples_from_frames(responses, items, subjects, name))
        print(f"  {name}: {len(responses)} rows → {len(raw) - n_before} triples")

    return raw, n_raw_rows, mode


def load_triples_per_benchmark(benchmark_ids: list[str]) -> list[RawTriple]:
    from torch_measure.datasets import load

    raw: list[RawTriple] = []
    for name in benchmark_ids:
        print(f"Loading {name}...")
        data = load(name, skip_traces=True)
        raw.extend(triples_from_long_form(data))
    return raw


def load_triples_bulk_hf(benchmark_ids: list[str]) -> list[RawTriple]:
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

    items_df = pd.DataFrame([row for row in items_ds])
    subjects_df = pd.DataFrame([row for row in subjects_ds])
    responses_df = pd.DataFrame(responses)

    rows: list[RawTriple] = []
    for bench_id in sorted(selected):
        bench_resp = responses_df.loc[responses_df["benchmark_id"] == bench_id]
        if bench_resp.empty:
            continue
        rows.extend(triples_from_frames(bench_resp, items_df, subjects_df, bench_id))
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

    row_frac = config["row_sample_frac"]
    early_subsample = False
    n_raw_rows = 0

    if config["bulk_load"]:
        raw_triples = load_triples_bulk_hf(benchmark_ids)
        n_raw = len(raw_triples)
        raw_triples = subsample_rows(raw_triples, frac=row_frac, seed=config["seed"])
        n_sampled = len(raw_triples)
        load_mode = "bulk_hf"
    elif _use_legacy_load():
        raw_triples = load_triples_per_benchmark(benchmark_ids)
        n_raw = len(raw_triples)
        raw_triples = subsample_rows(raw_triples, frac=row_frac, seed=config["seed"])
        n_sampled = len(raw_triples)
        load_mode = "legacy_per_benchmark"
    else:
        raw_triples, n_raw_rows, load_mode = load_triples_fast(
            benchmark_ids,
            row_sample_frac=row_frac,
            seed=config["seed"],
        )
        n_raw = n_raw_rows
        n_sampled = len(raw_triples)
        early_subsample = row_frac < 1.0

    print(
        f"Row subsample: {n_raw} raw observations → {n_sampled} "
        f"({row_frac:.0%}, seed={config['seed']}"
        f"{', before join' if early_subsample else ''})"
    )

    sampling_meta = {
        **benchmark_summary,
        "n_rows_raw": n_raw,
        "n_rows_after_subsample": n_sampled,
        "bulk_load": config["bulk_load"],
        "load_mode": load_mode,
        "early_row_subsample": early_subsample,
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
