"""Competition submission: predict P(correct) from (subject_content, item_content)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EMBEDDING_DIM = 768
LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
DEFAULT_ENCODER_REPO = "sentence-transformers/all-mpnet-base-v2"


def _format_item_content(benchmark: str, condition: str | None, item_content: str) -> str:
    bench = (benchmark or "").strip() or "unknown"
    cond = (condition or "").strip() or "none"
    body = (item_content or "").strip()
    return f"Benchmark: {bench}\nCondition: {cond}\n\n{body}"


def _declared_encoder_repo() -> str:
    models_path = Path(__file__).with_name("models.txt")
    if not models_path.exists():
        return DEFAULT_ENCODER_REPO
    declared = [
        line.strip()
        for line in models_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return declared[0] if declared else DEFAULT_ENCODER_REPO


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(Path(__file__).with_name(".hf_cache")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.R_OK):
            return str(path)
    return None


def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_irt(device: torch.device) -> tuple[object, dict[str, int]]:
    from torch_measure.models.amortized import AmortizedIRT

    with open(ARTIFACTS_DIR / "subject2idx.json") as f:
        subject2idx = json.load(f)
    with open(ARTIFACTS_DIR / "model_meta.json") as f:
        meta = json.load(f)

    state = _load_state_dict(ARTIFACTS_DIR / "amortized_irt.pt", device)
    model = AmortizedIRT(
        n_subjects=meta["n_subjects"],
        n_items=meta["n_items"],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=256,
        n_layers=3,
        pl=2,
        dropout=0.1,
        device=str(device),
    )
    model.load_state_dict(state)
    model.eval()
    return model, subject2idx


def _hosted_eval_environment() -> bool:
    """True on Codabench workers (pre-downloaded HF cache, no outbound network)."""
    if _local_smoke_test_enabled():
        return False
    return Path("/app/hf_cache").is_dir()


def _load_encoder(repo_id: str, device: torch.device) -> object:
    from sentence_transformers import SentenceTransformer

    if not _hosted_eval_environment():
        return SentenceTransformer(repo_id, device=str(device))

    cache_dir = _resolve_cache_dir() or "/app/hf_cache"
    candidates = [repo_id]
    if repo_id != "all-mpnet-base-v2":
        candidates.append("all-mpnet-base-v2")

    last_exc: Exception | None = None
    for name in candidates:
        try:
            return SentenceTransformer(
                name,
                device=str(device),
                cache_folder=cache_dir,
                local_files_only=True,
            )
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Could not load encoder from the local HuggingFace cache (tried {candidates}). "
        "Declare sentence-transformers/all-mpnet-base-v2 in models.txt."
    ) from last_exc


def _init_at_import() -> None:
    global ENCODER, MODEL, SUBJECT2IDX
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL, SUBJECT2IDX = _load_irt(device)
    repo_id = _declared_encoder_repo()
    ENCODER = _load_encoder(repo_id, device)


# Module-level init (competition: load once before any predict() call).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER: object | None = None
MODEL: object | None = None
SUBJECT2IDX: dict[str, int] = {}

try:
    _init_at_import()
except FileNotFoundError:
    # Local `python model.py` without artifacts; predict() will fail clearly.
    pass


def _probability(subject_content: str, item_emb: torch.Tensor) -> float:
    if MODEL is None or SUBJECT2IDX is None:
        raise RuntimeError("IRT weights not loaded; missing artifacts/amortized_irt.pt?")

    subj_key = subject_content or ""
    idx = SUBJECT2IDX.get(subj_key)
    if idx is not None:
        theta = MODEL.ability[idx]
    else:
        theta = MODEL.ability.mean()

    emb = item_emb.to(_DEVICE)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    params = MODEL.item_net(emb)
    b = params[:, 0]
    a = torch.exp(params[:, 1]) if MODEL.pl >= 2 else None
    p = MODEL._irt_probability(theta.unsqueeze(0), b, a)[0].item()
    return float(np.clip(p, 0.0, 1.0))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return predicted probability that the subject answers the item correctly."""
    del labeled  # optional adaptive labels; not used in this baseline
    if ENCODER is None:
        raise RuntimeError(
            "MPNet encoder not loaded. Check models.txt and HuggingFace cache "
            "(sentence-transformers/all-mpnet-base-v2)."
        )

    item_text = _format_item_content(
        str(input.get("benchmark") or ""),
        str(input.get("condition") or ""),
        str(input.get("item_content") or ""),
    )
    with torch.no_grad():
        item_emb = ENCODER.encode([item_text], convert_to_tensor=True)[0]
        if isinstance(item_emb, torch.Tensor):
            item_emb = item_emb.detach().clone()
        return _probability(input.get("subject_content") or "", item_emb)


if __name__ == "__main__":
    fake = {
        "benchmark": "test",
        "condition": "none",
        "subject_content": "Name: fake\nOrganization: test",
        "item_content": "What is 2+2?",
    }
    try:
        prob = predict(fake)
        print(f"predict() -> {prob:.4f}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Smoke test skipped: {e}")
