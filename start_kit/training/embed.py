"""Modal app: load on Modal, parallel GPU embed shards, write artifacts to volume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal
import numpy as np

ENCODER_NAME = "all-mpnet-base-v2"
BATCH_SIZE = int(os.environ.get("PIPELINE_EMBED_BATCH_SIZE", "128"))
SHARD_SIZE = int(os.environ.get("PIPELINE_EMBED_SHARD_SIZE", "512"))
MAX_CONTAINERS = int(os.environ.get("PIPELINE_EMBED_MAX_CONTAINERS", "16"))
ARTIFACTS_PATH = "/artifacts"
VOLUME_NAME = "irt-pipeline-artifacts"
HF_SECRET_NAME = "huggingface"
SHARDS_DIR = "embed_shards"

# Shard kind (internal) → final embedding filename on the volume.
KIND_TO_EMB_FILE = {
    "items": "item_embs.npy",
    "subjects": "subject_embs.npy",
}
LEGACY_EMB_FILE = {
    "items": "items_embs.npy",  # bug in earlier merge_shards
}
UNIQUE_STRINGS_FILE = {
    "items": "unique_items_strings.npy",
    "subjects": "unique_subjects_strings.npy",
}

HF_SECRETS = [modal.Secret.from_name(HF_SECRET_NAME)]

ARTIFACT_FILENAMES = (
    "item_embs.npy",
    "subject_embs.npy",
    "item2idx.json",
    "subject2idx.json",
    "train_triples.npy",
    "val_triples.npy",
    "val_item_indices.json",
    "selected_benchmarks.json",
    "sampling_meta.json",
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _hf_secrets_for_run() -> list[modal.Secret]:
    secrets: list[modal.Secret] = list(HF_SECRETS)
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token:
        secrets.append(
            modal.Secret.from_dict(
                {"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}
            )
        )
    return secrets


def _ensure_hf_hub_auth() -> None:
    token = (
        (os.environ.get("HF_TOKEN") or "").strip()
        or (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    )
    if not token:
        print(
            f"WARNING: No HF_TOKEN in container. "
            f"Set Modal secret '{HF_SECRET_NAME}' (key HF_TOKEN) or export HF_TOKEN locally."
        )
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"huggingface_hub.login note: {exc}")
    print(f"HF Hub authenticated (token …{token[-4:]})")


def _setup_training_path() -> None:
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/training")


app = modal.App("irt-embed")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=5.4",
        "numpy>=1.24",
        "torch>=2.0",
        "pandas>=2.3",
        "pyarrow>=0.12",
        "huggingface_hub>=0.16",
        "datasets>=2.14",
    )
)

image = _base_image.run_commands(
    f'python -c "from sentence_transformers import SentenceTransformer; '
    f"SentenceTransformer('{ENCODER_NAME}')\"",
    secrets=HF_SECRETS,
)

if modal.is_local():
    _training_dir = Path(__file__).resolve().parent
    _repo_src = (_training_dir / ".." / ".." / "src").resolve()
    image = image.add_local_dir(str(_repo_src), remote_path="/root/src")
    image = image.add_local_dir(str(_training_dir), remote_path="/root/training")

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _encoder = SentenceTransformer(ENCODER_NAME, device=device)
    return _encoder


def _chunk(strings: list[str], size: int) -> list[list[str]]:
    return [strings[i : i + size] for i in range(0, len(strings), size)]


def _emb_path(root: Path, kind: str) -> Path:
    return root / KIND_TO_EMB_FILE[kind]


def _normalize_legacy_emb_files(root: Path) -> None:
    """Rename mistaken ``items_embs.npy`` → ``item_embs.npy`` if needed."""
    for kind, legacy_name in LEGACY_EMB_FILE.items():
        target = root / KIND_TO_EMB_FILE[kind]
        legacy = root / legacy_name
        if legacy.exists() and not target.exists():
            legacy.rename(target)
            print(f"Renamed legacy {legacy_name} → {target.name}")


def _emb_is_ready(root: Path, kind: str) -> bool:
    path = _emb_path(root, kind)
    return path.exists() and path.stat().st_size > 0


def _save_unique_strings(strings: list[str], path: Path) -> None:
    np.save(path, np.asarray(strings, dtype=object))


def _load_unique_strings(path: Path) -> list[str]:
    return np.load(path, allow_pickle=True).tolist()


def _clear_shards(root: Path, kind: str | None = None) -> None:
    shard_dir = root / SHARDS_DIR
    if not shard_dir.exists():
        return
    for path in shard_dir.glob("*.npy"):
        if kind is None or path.name.startswith(f"{kind}_"):
            path.unlink()


@app.function(
    image=image,
    gpu="T4",
    secrets=HF_SECRETS,
    volumes={ARTIFACTS_PATH: vol},
    timeout=3600,
    max_containers=MAX_CONTAINERS,
)
def embed_shard(shard_id: int, texts: list[str], kind: str) -> int:
    """Embed one shard of strings on GPU; write to volume (no large gRPC return)."""
    if not texts:
        return 0

    vol.reload()
    root = Path(ARTIFACTS_PATH)
    shard_dir = root / SHARDS_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)

    encoder = _get_encoder()
    embs = encoder.encode(
        texts,
        batch_size=min(len(texts), BATCH_SIZE),
        show_progress_bar=False,
    )
    out = np.asarray(embs, dtype=np.float32)
    path = shard_dir / f"{kind}_{shard_id:05d}.npy"
    np.save(path, out)
    print(f"  shard {kind} {shard_id:05d}: {len(texts)} texts → {path.name} {out.shape}")
    vol.commit()
    return len(texts)


@app.function(
    image=image,
    secrets=HF_SECRETS,
    volumes={ARTIFACTS_PATH: vol},
    timeout=3600,
)
def merge_shards(kind: str, n_shards: int) -> tuple[int, int]:
    """Stack shard files in order → ``item_embs.npy`` / ``subject_embs.npy``."""
    vol.reload()
    root = Path(ARTIFACTS_PATH)
    out_path = _emb_path(root, kind)
    shard_dir = root / SHARDS_DIR
    parts: list[np.ndarray] = []
    for i in range(n_shards):
        path = shard_dir / f"{kind}_{i:05d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard {path}")
        parts.append(np.load(path))
    merged = np.vstack(parts)
    np.save(out_path, merged)
    print(f"Merged {n_shards} {kind} shards → {out_path.name} {merged.shape}")
    vol.commit()
    return merged.shape[0], merged.shape[1]


@app.function(
    image=image,
    secrets=HF_SECRETS,
    volumes={ARTIFACTS_PATH: vol},
    timeout=7200,
)
def embed_orchestrator() -> dict[str, int]:
    """Load HF, save triples/mappings, parallel ``embed_shard.starmap``, merge."""
    _ensure_hf_hub_auth()
    _setup_training_path()

    from data_hf import load_representative_training_data
    from utils import VAL_ITEM_FRAC, save_mappings, save_train_val_triples, split_triples_by_item

    vol.reload()
    root = Path(ARTIFACTS_PATH)
    root.mkdir(parents=True, exist_ok=True)
    _normalize_legacy_emb_files(root)

    subjects_only = _env_truthy("PIPELINE_EMBED_SUBJECTS_ONLY")
    skip_items = _env_truthy("PIPELINE_EMBED_SKIP_ITEMS") or _emb_is_ready(root, "items")

    n_train = n_val = 0
    if subjects_only and (root / "train_triples.npy").exists():
        print("PIPELINE_EMBED_SUBJECTS_ONLY: skipping HF load (artifacts on volume).")
        subj_path = root / UNIQUE_STRINGS_FILE["subjects"]
        if subj_path.exists():
            unique_subjects = _load_unique_strings(subj_path)
        elif (root / "subject2idx.json").exists():
            with open(root / "subject2idx.json") as f:
                unique_subjects = sorted(json.load(f).keys())
            print(f"Loaded {len(unique_subjects)} subject strings from subject2idx.json")
        else:
            raise FileNotFoundError(
                f"Missing {subj_path.name} and subject2idx.json; run full embed or unset "
                "PIPELINE_EMBED_SUBJECTS_ONLY."
            )
        unique_items = []
        item_shards: list[list[str]] = []
    else:
        unique_items, unique_subjects, triples, item2idx, subject2idx = (
            load_representative_training_data(root)
        )
        save_mappings(item2idx, subject2idx, root)
        train_triples, val_triples, val_items = split_triples_by_item(triples)
        save_train_val_triples(train_triples, val_triples, val_items, root)
        _save_unique_strings(unique_items, root / UNIQUE_STRINGS_FILE["items"])
        _save_unique_strings(unique_subjects, root / UNIQUE_STRINGS_FILE["subjects"])
        n_train, n_val = len(train_triples), len(val_triples)
        vol.commit()
        item_shards = _chunk(unique_items, SHARD_SIZE)

    subject_shards = _chunk(unique_subjects, SHARD_SIZE)

    print(
        f"Parallel embed: {len(unique_items)} items in {len(item_shards)} shards, "
        f"{len(unique_subjects)} subjects in {len(subject_shards)} shards "
        f"(shard_size={SHARD_SIZE}, max_containers={MAX_CONTAINERS}, "
        f"skip_items={skip_items})..."
    )

    if item_shards and not skip_items:
        _clear_shards(root, "items")
        item_args = [(i, batch, "items") for i, batch in enumerate(item_shards)]
        item_counts = list(embed_shard.starmap(item_args))
        print(f"Item embed shards finished: {len(item_counts)} shards, {sum(item_counts)} texts")
        merge_shards.local("items", len(item_shards))
    elif skip_items:
        print(f"Skipping item embed; using existing {_emb_path(root, 'items').name}")

    if subject_shards and not _emb_is_ready(root, "subjects"):
        _clear_shards(root, "subjects")
        subj_args = [(i, batch, "subjects") for i, batch in enumerate(subject_shards)]
        subj_counts = list(embed_shard.starmap(subj_args))
        print(
            f"Subject embed shards finished: {len(subj_counts)} shards, {sum(subj_counts)} texts"
        )
        merge_shards.local("subjects", len(subject_shards))
    elif _emb_is_ready(root, "subjects"):
        print(f"Skipping subject embed; using existing {_emb_path(root, 'subjects').name}")

    vol.commit()
    summary = {
        "n_items": len(unique_items) if unique_items else int(np.load(_emb_path(root, "items")).shape[0]),
        "n_subjects": len(unique_subjects),
        "n_item_shards": len(item_shards),
        "n_subject_shards": len(subject_shards),
        "n_train_triples": n_train,
        "n_val_triples": n_val,
        "skipped_item_embed": skip_items,
    }
    with open(root / "embed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    print(f"Done. {summary}")
    return summary


def _sync_volume_to_local(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILENAMES:
        dest = local_dir / name
        print(f"Downloading {name} ...")
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                VOLUME_NAME,
                name,
                str(dest),
            ],
            check=True,
        )
    print(f"Synced volume → {local_dir}")


@app.local_entrypoint()
def main(sync_local: bool = True):
    """Run load + parallel embed on Modal. ``--no-sync-local`` = volume only."""
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print(
            f"Tip: export HF_TOKEN=hf_... or Modal secret '{HF_SECRET_NAME}' with key HF_TOKEN."
        )
    else:
        print(f"Using HF_TOKEN from shell (…{token[-4:]})")

    run_fn = embed_orchestrator.with_options(secrets=_hf_secrets_for_run())

    print("Starting embed_orchestrator on Modal (load + parallel GPU shards)...")
    call = run_fn.spawn()
    print(f"Spawned job: {call.object_id}")
    call.get()
    print("Embed pipeline finished.")

    if sync_local:
        local_dir = Path(__file__).resolve().parent / "artifacts"
        _sync_volume_to_local(local_dir)
