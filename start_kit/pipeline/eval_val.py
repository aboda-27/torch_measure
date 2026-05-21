"""Evaluate competition predict() on held-out val triples."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PIPELINE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"

sys.path.insert(0, str(PIPELINE_DIR))
from model import _ensure_loaded, _get_encoder, _probability  # noqa: E402


def _invert_index(mapping: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in mapping.items()}


def main(max_rows: int | None = None) -> None:
    for name in ("amortized_irt.pt", "model_meta.json", "val_triples.npy", "item2idx.json", "subject2idx.json"):
        if not (ARTIFACTS_DIR / name).exists():
            raise FileNotFoundError(
                f"Missing {ARTIFACTS_DIR / name}\n"
                "Pull weights: modal volume get --force irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/\n"
                "              modal volume get --force irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/"
            )

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

    print("Loading IRT checkpoint (amortized_irt.pt)...", flush=True)
    t0 = time.perf_counter()
    _ensure_loaded()
    print(f"  checkpoint ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Loading MPNet encoder (first run may download ~400MB — not hung)...", flush=True)
    t0 = time.perf_counter()
    encoder = _get_encoder()
    print(f"  encoder ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    unique_item_texts = sorted({idx2item[int(r[1])] for r in rows})
    print(f"Batch-encoding {len(unique_item_texts)} unique val items...", flush=True)
    t0 = time.perf_counter()
    embs = encoder.encode(
        unique_item_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    item_emb_by_text = {text: embs[i] for i, text in enumerate(unique_item_texts)}
    print(f"  embeddings done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Scoring rows...", flush=True)
    probs: list[float] = []
    labels: list[float] = []
    t0 = time.perf_counter()

    for i, row in enumerate(rows):
        s_idx, i_idx, y = int(row[0]), int(row[1]), float(row[2])
        item_text = idx2item[i_idx]
        subj_text = idx2subject[s_idx]
        p = _probability(subj_text, item_emb_by_text[item_text])
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

    print(f"val BCE (predict API): {bce:.4f}")
    print(f"val accuracy @ 0.5:    {acc:.4f}")


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_rows=cap)
