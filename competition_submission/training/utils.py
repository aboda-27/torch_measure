"""Shared helpers for the amortized IRT pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ENCODER_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 256
VAL_ITEM_FRAC = 0.2  # hold out 20% of items (all their responses go to val)
SPLIT_SEED = 42


def format_item_content(
    benchmark: str,
    condition: str | None,
    item_content: str,
) -> str:
    """Text fed to MPNet for items; matches ``predict()`` (competition_submission ``input`` fields)."""
    bench = (benchmark or "").strip() or "unknown"
    cond = (condition or "").strip() or "none"
    body = (item_content or "").strip()
    return f"Benchmark: {bench}\nCondition: {cond}\n\n{body}"


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    """Match hosted runtime subject_content format (see competition_submission/README.md)."""
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_index(strings: list[str]) -> dict[str, int]:
    """Map unique strings to contiguous integer indices (sorted for stability)."""
    unique = sorted(set(strings))
    return {s: i for i, s in enumerate(unique)}


def load_mappings(artifacts_dir: Path | str = ARTIFACTS_DIR) -> tuple[dict[str, int], dict[str, int]]:
    root = Path(artifacts_dir)
    with open(root / "item2idx.json") as f:
        item2idx = json.load(f)
    with open(root / "subject2idx.json") as f:
        subject2idx = json.load(f)
    return item2idx, subject2idx


def save_mappings(
    item2idx: dict[str, int],
    subject2idx: dict[str, int],
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "item2idx.json", "w") as f:
        json.dump(item2idx, f)
    with open(root / "subject2idx.json", "w") as f:
        json.dump(subject2idx, f)


def _triples_to_array(triples: list[tuple[int, int, float]]) -> np.ndarray:
    return np.array(triples, dtype=np.float64)


def _array_to_triples(arr: np.ndarray) -> list[tuple[int, int, float]]:
    return [(int(r[0]), int(r[1]), float(r[2])) for r in arr]


def split_triples_by_item(
    triples: list[tuple[int, int, float]],
    val_frac: float = VAL_ITEM_FRAC,
    seed: int = SPLIT_SEED,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]], set[int]]:
    """Hold out a fraction of *items* (not random rows).

    All (subject, item) rows for a held-out item go to validation. This matches
    cold-item prediction at competition time better than a random row split.
    """
    import random

    item_indices = sorted({t[1] for t in triples})
    rng = random.Random(seed)
    rng.shuffle(item_indices)
    n_val = max(1, int(len(item_indices) * val_frac))
    val_items = set(item_indices[:n_val])

    train: list[tuple[int, int, float]] = []
    val: list[tuple[int, int, float]] = []
    for t in triples:
        if t[1] in val_items:
            val.append(t)
        else:
            train.append(t)
    return train, val, val_items


def save_triples(triples: list[tuple[int, int, float]], artifacts_dir: Path | str = ARTIFACTS_DIR) -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "triples.npy", _triples_to_array(triples))


def save_train_val_triples(
    train_triples: list[tuple[int, int, float]],
    val_triples: list[tuple[int, int, float]],
    val_item_indices: set[int],
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "train_triples.npy", _triples_to_array(train_triples))
    np.save(root / "val_triples.npy", _triples_to_array(val_triples))
    with open(root / "val_item_indices.json", "w") as f:
        json.dump(sorted(val_item_indices), f)


def load_triples(artifacts_dir: Path | str = ARTIFACTS_DIR) -> list[tuple[int, int, float]]:
    arr = np.load(Path(artifacts_dir) / "triples.npy")
    return _array_to_triples(arr)


def load_train_val_triples(
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Load train/val splits from embed. Falls back to splitting triples.npy in-place."""
    root = Path(artifacts_dir)
    train_path = root / "train_triples.npy"
    val_path = root / "val_triples.npy"

    if train_path.exists() and val_path.exists():
        return _array_to_triples(np.load(train_path)), _array_to_triples(np.load(val_path))

    triples = load_triples(root)
    train, val, _ = split_triples_by_item(triples)
    print(
        f"No train_triples.npy / val_triples.npy found; split triples.npy "
        f"({VAL_ITEM_FRAC:.0%} of items held out, seed={SPLIT_SEED})."
    )
    return train, val


def load_training_data() -> tuple[list[str], list[str], list[tuple[int, int, float]], dict[str, int], dict[str, int]]:
    """Load measurement-db with representative sampling and build indexed triples.

    Sampling is configured via environment variables (see pipeline README):
    ``PIPELINE_SAMPLE_MODE`` (default ``stratified``); row subsample is fixed at 25% in ``sampling.py``
    (default ``0.25``), ``PIPELINE_BENCHMARKS_PER_DOMAIN``, ``PIPELINE_MAX_DATASETS``
    (``legacy_max`` mode only).

    Labels are binarized per benchmark ``response_type`` (see ``labeling.py``):
    binary as-is, likert_10 (mtbench) >= 7, likert_5 (ultrafeedback) >= 4,
    fraction/mixed with scale-appropriate rules, rewardbench ties -> 0.5.
    Writes ``selected_benchmarks.json`` and ``sampling_meta.json`` under artifacts/.
    """
    from data_hf import load_representative_training_data

    return load_representative_training_data(ARTIFACTS_DIR)
