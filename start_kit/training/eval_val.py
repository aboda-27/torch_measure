"""Evaluate competition predict() on held-out val triples."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = REPO_ROOT / "start_kit" / "submission"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"

sys.path.insert(0, str(SUBMISSION_DIR))
import model as submission_model  # noqa: E402


def _invert_index(mapping: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in mapping.items()}


def _require_models() -> None:
    if submission_model.ENCODER is None or submission_model.MODEL is None:
        raise RuntimeError(
            "model.py did not load (missing submission artifacts?). "
            "Copy weights: cp start_kit/training/artifacts/{amortized_irt.pt,model_meta.json,subject2idx.json} "
            "start_kit/submission/artifacts/"
        )


def _tensor_row(emb) -> torch.Tensor:
    if isinstance(emb, torch.Tensor):
        return emb.detach().clone()
    return torch.tensor(emb, dtype=torch.float32)


def main(max_rows: int | None = None) -> None:
    for name in ("amortized_irt.pt", "model_meta.json", "val_triples.npy", "item2idx.json", "subject2idx.json"):
        if not (ARTIFACTS_DIR / name).exists():
            raise FileNotFoundError(
                f"Missing {ARTIFACTS_DIR / name}\n"
                "Pull: bash start_kit/training/scripts/pull_artifacts.sh"
            )

    print("Loading model.py (IRT + MPNet at import)...", flush=True)
    t0 = time.perf_counter()
    _require_models()
    encoder = submission_model.ENCODER
    print(f"  ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Loading index maps...", flush=True)
    with open(ARTIFACTS_DIR / "item2idx.json") as f:
        item2idx = json.load(f)
    with open(ARTIFACTS_DIR / "subject2idx.json") as f:
        subject2idx = json.load(f)

    idx2item = _invert_index(item2idx)
    idx2subject = _invert_index(subject2idx)

    val = np.load(ARTIFACTS_DIR / "val_triples.npy")
    rows = val[: len(val) if max_rows is None else min(len(val), max_rows)]
    n = len(rows)
    print(f"Val rows to score: {n} / {len(val)}", flush=True)

    unique_item_texts = sorted({idx2item[int(r[1])] for r in rows})
    print(f"Batch-encoding {len(unique_item_texts)} unique val items...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        embs = encoder.encode(
            unique_item_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
    item_emb_by_text = {
        text: _tensor_row(embs[i]) for i, text in enumerate(unique_item_texts)
    }
    print(f"  embeddings done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Scoring rows...", flush=True)
    probs: list[float] = []
    labels: list[float] = []
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        s_idx, i_idx, y = int(row[0]), int(row[1]), float(row[2])
        item_text = idx2item[i_idx]
        subj_text = idx2subject[s_idx]
        p = submission_model._probability(subj_text, item_emb_by_text[item_text])
        probs.append(p)
        labels.append(y)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"  {i + 1}/{n} rows ({rate:.0f} rows/s)", flush=True)

    y_t = torch.tensor(labels, dtype=torch.float32)
    p_t = torch.tensor(probs, dtype=torch.float32)
    bce = torch.nn.functional.binary_cross_entropy(p_t, y_t).item()
    acc = ((p_t >= 0.5) == (y_t >= 0.5)).float().mean().item()

    print(f"val BCE (_probability path): {bce:.4f}")
    print(f"val accuracy @ 0.5:         {acc:.4f}")


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_rows=cap)
