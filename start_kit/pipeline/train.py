"""Modal app: train AmortizedIRT on precomputed embeddings."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn

app = modal.App("irt-train")
vol = modal.Volume.from_name("irt-pipeline-artifacts", create_if_missing=True)

_base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.3",
    "pyarrow>=0.12",
    "huggingface_hub>=0.16",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "sentence-transformers>=5.4",
    "pyro-ppl>=1.8",
)

image = _base_image
if modal.is_local():
    _pipeline_dir = Path(__file__).resolve().parent
    _repo_src = (_pipeline_dir / ".." / ".." / "src").resolve()
    sys.path.insert(0, str(_pipeline_dir))
    image = image.add_local_dir(str(_repo_src), remote_path="/root/src")
    image = image.add_local_dir(str(_pipeline_dir), remote_path="/root/pipeline")

if modal.is_local():
    from utils import EMBEDDING_DIM, load_mappings, load_train_val_triples  # noqa: E402
else:
    sys.path.insert(0, "/root/pipeline")
    from utils import EMBEDDING_DIM, load_mappings, load_train_val_triples  # noqa: E402

ARTIFACTS_PATH = "/artifacts"
LOCAL_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REQUIRED_ARTIFACTS = (
    "item_embs.npy",
    "item2idx.json",
    "subject2idx.json",
    "train_triples.npy",
    "val_triples.npy",
)

N_EPOCHS = 30
LR = 1e-3
TRAIN_BATCH_SIZE = 512


def _upload_local_artifacts_to_volume() -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (LOCAL_ARTIFACTS_DIR / name).exists()]
    if missing and (LOCAL_ARTIFACTS_DIR / "triples.npy").exists():
        missing = [n for n in missing if n not in ("train_triples.npy", "val_triples.npy")]
    if missing:
        raise FileNotFoundError(
            f"Missing embed artifacts in {LOCAL_ARTIFACTS_DIR}: {missing}\n"
            "Re-run: modal run start_kit/pipeline/embed.py"
        )

    print(f"Uploading artifacts from {LOCAL_ARTIFACTS_DIR} to volume ...")
    with vol.batch_upload(force=True) as batch:
        for path in LOCAL_ARTIFACTS_DIR.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                batch.put_file(str(path), path.name)
    print("Volume ready for training.")


def _require_volume_artifacts() -> None:
    root = Path(ARTIFACTS_PATH)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    if missing and (root / "triples.npy").exists():
        missing = [n for n in missing if n not in ("train_triples.npy", "val_triples.npy")]
    if missing:
        raise FileNotFoundError(f"Missing on Modal volume at {ARTIFACTS_PATH}: {missing}")


def _run_epoch(
    model,
    triples: list[tuple[int, int, float]],
    criterion: nn.Module,
    device: str,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    if train:
        model.train()
        random.shuffle(triples)
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for start in range(0, len(triples), TRAIN_BATCH_SIZE):
            batch = triples[start : start + TRAIN_BATCH_SIZE]
            s_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            i_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            y = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device)

            probs = model.predict({"subject_idx": s_idx, "item_idx": i_idx})
            loss = criterion(probs, y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@app.function(
    image=image,
    gpu="A10G",
    volumes={ARTIFACTS_PATH: vol},
    timeout=7200,
)
def train():
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")

    from torch_measure.models.amortized import AmortizedIRT

    vol.reload()
    _require_volume_artifacts()

    item_embs = np.load(f"{ARTIFACTS_PATH}/item_embs.npy")
    item2idx, subject2idx = load_mappings(ARTIFACTS_PATH)
    train_triples, val_triples = load_train_val_triples(ARTIFACTS_PATH)

    n_subjects = len(subject2idx)
    n_items = len(item2idx)
    device = "cuda"

    print(f"train observations: {len(train_triples)}, val observations: {len(val_triples)}")

    model = AmortizedIRT(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=256,
        n_layers=3,
        pl=2,
        dropout=0.1,
        device=device,
    )
    model.set_embeddings(torch.from_numpy(item_embs).float().to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = _run_epoch(
            model, train_triples, criterion, device, train=True, optimizer=optimizer
        )
        val_loss = _run_epoch(model, val_triples, criterion, device, train=False)

        print(f"epoch {epoch}/{N_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_path = f"{ARTIFACTS_PATH}/amortized_irt.pt"
    state = best_state if best_state is not None else model.state_dict()
    torch.save(state, out_path)

    meta_path = f"{ARTIFACTS_PATH}/model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_subjects": n_subjects,
                "n_items": n_items,
                "best_val_loss": best_val,
            },
            f,
        )
    vol.commit()
    print(f"Saved best checkpoint (val_loss={best_val:.4f}) to {out_path}")


def _print_artifact_status() -> None:
    print(f"Artifacts directory: {LOCAL_ARTIFACTS_DIR}")
    for name in REQUIRED_ARTIFACTS:
        path = LOCAL_ARTIFACTS_DIR / name
        print(f"  {name}: {'OK' if path.exists() else 'MISSING'}")
    if (LOCAL_ARTIFACTS_DIR / "triples.npy").exists():
        print("  triples.npy: OK (legacy; train will split if train/val npy missing)")


@app.local_entrypoint()
def main(upload: bool = True, no_upload: bool = False):
    """Train on Modal. Embeds must be on volume (from embed.py) or local artifacts/."""
    if no_upload:
        upload = False
    if modal.is_local():
        _print_artifact_status()
        if upload:
            if not (LOCAL_ARTIFACTS_DIR / "item_embs.npy").exists():
                raise FileNotFoundError(
                    "Run embed first: modal run start_kit/pipeline/embed.py"
                )
            _upload_local_artifacts_to_volume()
        else:
            print("Skipping volume upload (using existing files on irt-pipeline-artifacts).")
    train.remote()
