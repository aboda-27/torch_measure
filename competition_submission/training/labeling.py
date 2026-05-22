"""Labeling for the Predictive AI Evaluation Challenge.

This file serves two roles:

1. **Gradescope / reproducibility** — Documents how raw Hugging Face ``response``
   values were converted to training targets in ``{0, 1}`` (and soft ties for
   rewardbench) before fitting AmortizedIRT. Used offline in ``embed.py`` /
   ``data_hf.py``.

2. **Codabench (optional)** — Defines ``acquisition_function()`` for adaptive
   labeling (Section 2.2 of the competition handbook). **This submission does not
   use adaptive labeling:** ``model.py`` ignores the ``labeled`` argument. The
   stub below is included so the file matches the starter-kit interface if present
   in a zip; do **not** add this file to the Codabench zip unless you implement
   active acquisition.

Handbook references:
- ``predict(input, labeled=None) -> float`` — in ``model.py``
- ``acquisition_function(input) -> float`` — optional; higher = more desired label
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Codabench adaptive labeling (optional — not used by this submission)
# ---------------------------------------------------------------------------


def acquisition_function(input: dict) -> float:
    """Score how much this candidate should receive a revealed ground-truth label.

    Called once per hidden (subject, item) pair **before** ``predict()``. The
    platform reveals the top **K** inputs per data category (default K = 5) and
    passes them as ``labeled`` to ``predict()``.

    Parameters
    ----------
    input : dict
        Same four string keys as ``predict()``:
        ``benchmark``, ``condition``, ``subject_content``, ``item_content``.

    Returns
    -------
    float
        Finite score; **higher** = more desirable for labeling. Only ranking
        matters, not the absolute value. Wrap numpy/torch scalars with
        ``float(...)``.

    Notes
    -----
    This baseline does not use adaptive labeling. Returning ``0.0`` for every
    input is equivalent to not shipping ``labeling.py`` (random K per category).
    """
    del input
    return 0.0


# ---------------------------------------------------------------------------
# Offline training labels (measurement-db → BCE targets)
# ---------------------------------------------------------------------------

DEFAULT_REPO_ID = "aims-foundations/measurement-db"
BENCHMARKS_FILE = "benchmarks.parquet"

DEFAULT_LIKERT_10_MIN = 7.0   # mtbench (likert_10)
DEFAULT_LIKERT_5_MIN = 4.0    # ultrafeedback (likert_5)
DEFAULT_FRACTION_MIN = 0.5    # cybench (fraction in [0, 1])
REWARD_TIE_VALUE = 0.5
REWARD_EPS = 1e-6

_specs_cache: dict[str, dict[str, Any]] | None = None


def _repo_id() -> str:
    return os.environ.get("PIPELINE_REPO_ID", DEFAULT_REPO_ID).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def load_benchmark_label_specs(*, force_download: bool = False) -> dict[str, dict[str, Any]]:
    """Load per-benchmark ``response_type`` metadata from ``benchmarks.parquet``.

    Returns
    -------
    dict
        ``{benchmark_id: {response_type, response_scale, categorical}}``
    """
    global _specs_cache
    if _specs_cache is not None and not force_download:
        return _specs_cache

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_repo_id(),
        filename=BENCHMARKS_FILE,
        repo_type="dataset",
        force_download=force_download,
    )
    df = pd.read_parquet(path)
    if "benchmark_id" not in df.columns:
        raise ValueError(f"{BENCHMARKS_FILE} missing benchmark_id column")

    specs: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        bid = str(row["benchmark_id"])
        specs[bid] = {
            "response_type": str(row.get("response_type") or "binary").strip().lower(),
            "response_scale": str(row.get("response_scale") or ""),
            "categorical": bool(row.get("categorical", True)),
        }
    _specs_cache = specs
    return specs


def label_rule_summary(specs: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
    """Human-readable binarization rule per benchmark (stored in ``sampling_meta.json``)."""
    specs = specs or load_benchmark_label_specs()
    likert_10_min = _env_float("PIPELINE_LIKERT_10_MIN", DEFAULT_LIKERT_10_MIN)
    likert_5_min = _env_float("PIPELINE_LIKERT_5_MIN", DEFAULT_LIKERT_5_MIN)
    fraction_min = _env_float("PIPELINE_FRACTION_MIN", DEFAULT_FRACTION_MIN)

    out: dict[str, str] = {}
    for bid, meta in sorted(specs.items()):
        rt = meta.get("response_type") or "binary"
        if bid == "rewardbench" or (rt == "binary" and "0.5" in meta.get("response_scale", "")):
            out[bid] = f"rewardbench_map: 1→1, 0→0, tie→{REWARD_TIE_VALUE}"
        elif rt == "likert_10":
            out[bid] = f"response >= {likert_10_min}"
        elif rt == "likert_5":
            out[bid] = f"response >= {likert_5_min}"
        elif rt == "fraction":
            out[bid] = f"clip_[0,1] then response >= {fraction_min}"
        elif rt == "mixed":
            out[bid] = "near_0/1 → round else response >= 0.5"
        else:
            out[bid] = "binary in {0,1} else response >= 0.5"
    return out


def responses_to_binary_labels(
    benchmark_id: str,
    responses: pd.Series,
    specs: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Map raw parquet ``response`` values to training targets in [0, 1].

    Competition test labels are binary pass/fail. Public training parquets mix
    binary, Likert, fraction, and tie-aware scales. A global ``response >= 0.5``
    rule mis-labels most mtbench (1–10) and ultrafeedback (1–5) rows.

    Parameters
    ----------
    benchmark_id : str
        Benchmark identifier (e.g. ``mmlupro``, ``mtbench``).
    responses : pd.Series
        Raw ``response`` column for one benchmark parquet.
    specs : dict, optional
        Output of ``load_benchmark_label_specs()``; loaded from HF if omitted.

    Returns
    -------
    np.ndarray
        Float targets for BCE: ``0.0`` / ``1.0``, or ``0.5`` for rewardbench ties.
    """
    specs = specs or load_benchmark_label_specs()
    meta = specs.get(benchmark_id, {"response_type": "binary", "response_scale": ""})
    rt = (meta.get("response_type") or "binary").lower()
    scale = meta.get("response_scale") or ""

    r = responses.astype(float).to_numpy()
    likert_10_min = _env_float("PIPELINE_LIKERT_10_MIN", DEFAULT_LIKERT_10_MIN)
    likert_5_min = _env_float("PIPELINE_LIKERT_5_MIN", DEFAULT_LIKERT_5_MIN)
    fraction_min = _env_float("PIPELINE_FRACTION_MIN", DEFAULT_FRACTION_MIN)

    if benchmark_id == "rewardbench" or (
        rt == "binary" and "0.5" in scale and np.any(np.abs(r - 0.5) < REWARD_EPS)
    ):
        labels = np.empty_like(r)
        labels[r >= 1.0 - REWARD_EPS] = 1.0
        labels[r <= REWARD_EPS] = 0.0
        tie = (r > REWARD_EPS) & (r < 1.0 - REWARD_EPS)
        labels[tie] = REWARD_TIE_VALUE
        return np.clip(labels, 0.0, 1.0)

    if rt == "likert_10":
        return (r >= likert_10_min).astype(np.float64)

    if rt == "likert_5":
        return (r >= likert_5_min).astype(np.float64)

    if rt == "fraction":
        clipped = np.clip(r, 0.0, 1.0)
        return (clipped >= fraction_min).astype(np.float64)

    if rt == "mixed":
        is_near_binary = (r <= REWARD_EPS) | (r >= 1.0 - REWARD_EPS)
        out = np.where(is_near_binary, np.round(np.clip(r, 0.0, 1.0)), (r >= 0.5).astype(np.float64))
        return out.astype(np.float64)

    if rt == "binary":
        uniq = set(np.unique(np.round(r, 3)))
        if uniq.issubset({0.0, 1.0}):
            return np.clip(np.round(r), 0.0, 1.0).astype(np.float64)
        return (r >= 0.5).astype(np.float64)

    if r.min() >= 0.0 and r.max() <= 1.0:
        return (r >= 0.5).astype(np.float64)
    return (r >= np.median(r)).astype(np.float64)
