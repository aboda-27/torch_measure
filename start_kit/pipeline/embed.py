"""Modal app: parallel sentence embeddings for items and subjects."""

from __future__ import annotations

import os

import modal
import numpy as np

# Constants used by Modal workers (embed_batch) — do not import utils at module level.
ENCODER_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 256


def _hf_secrets() -> list[modal.Secret]:
    """HF auth for MPNet download. Set HF_TOKEN in shell or create a Modal secret."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return [modal.Secret.from_dict({"HF_TOKEN": token})]
    # modal secret create huggingface HF_TOKEN=hf_...
    return [modal.Secret.from_name("huggingface")]


app = modal.App("irt-embed")
vol = modal.Volume.from_name("irt-pipeline-artifacts", create_if_missing=True)

_hf = _hf_secrets()

# Bake MPNet into the image once (avoids 32 workers each hitting HF Hub unauthenticated).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=5.4",
        "numpy>=1.24",
        "torch>=2.0",
    )
    .run_commands(
        f'python -c "from sentence_transformers import SentenceTransformer; '
        f"SentenceTransformer('{ENCODER_NAME}')\"",
        secrets=_hf,
    )
)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder


@app.function(image=image, secrets=_hf, timeout=3600)
def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed one batch of strings (up to BATCH_SIZE)."""
    encoder = _get_encoder()
    embs = encoder.encode(texts, batch_size=min(len(texts), BATCH_SIZE), show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)


def _chunk(strings: list[str], size: int) -> list[list[str]]:
    return [strings[i : i + size] for i in range(0, len(strings), size)]


def _upload_artifacts_to_volume(artifacts_dir) -> None:
    """Upload local artifacts/ to Modal volume."""
    from pathlib import Path

    artifacts_dir = Path(artifacts_dir)
    print(f"Uploading {artifacts_dir} to volume irt-pipeline-artifacts ...")
    with vol.batch_upload(force=True) as batch:
        for path in artifacts_dir.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                batch.put_file(str(path), path.name)
    print("Synced artifacts to Modal volume.")


@app.local_entrypoint()
def main():
    import sys
    from pathlib import Path

    if not os.environ.get("HF_TOKEN"):
        print(
            "Tip: export HF_TOKEN=hf_... for faster HF downloads "
            "(or: modal secret create huggingface HF_TOKEN=hf_...)"
        )

    pipeline_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(pipeline_dir))

    from utils import (  # noqa: E402
        ARTIFACTS_DIR,
        VAL_ITEM_FRAC,
        load_training_data,
        save_mappings,
        save_train_val_triples,
        split_triples_by_item,
    )

    unique_items, unique_subjects, triples, item2idx, subject2idx = load_training_data()

    item_batches = _chunk(unique_items, BATCH_SIZE)
    subject_batches = _chunk(unique_subjects, BATCH_SIZE)

    print(f"Embedding {len(unique_items)} items in {len(item_batches)} batches...")
    print("Use: modal run --detach start_kit/pipeline/embed.py  (so Ctrl+C does not cancel workers)")
    item_parts = list(embed_batch.map(item_batches))
    item_embs = np.vstack(item_parts)

    print(f"Embedding {len(unique_subjects)} subjects in {len(subject_batches)} batches...")
    subject_parts = list(embed_batch.map(subject_batches))
    subject_embs = np.vstack(subject_parts)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / "item_embs.npy", item_embs)
    np.save(ARTIFACTS_DIR / "subject_embs.npy", subject_embs)
    save_mappings(item2idx, subject2idx, ARTIFACTS_DIR)

    train_triples, val_triples, val_items = split_triples_by_item(triples)
    save_train_val_triples(train_triples, val_triples, val_items, ARTIFACTS_DIR)

    print(f"Saved artifacts to {ARTIFACTS_DIR}")
    print(f"  item_embs: {item_embs.shape}, subject_embs: {subject_embs.shape}")
    print(f"  train triples: {len(train_triples)}, val triples: {len(val_triples)} ({VAL_ITEM_FRAC:.0%} items held out)")
    for meta_name in ("selected_benchmarks.json", "sampling_meta.json"):
        if (ARTIFACTS_DIR / meta_name).exists():
            print(f"  {meta_name}: OK")

    _upload_artifacts_to_volume(ARTIFACTS_DIR)
