"""Competition submission: predict P(correct) from (subject_content, item_content)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ENCODER_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# Module-level init (runs once when the container starts)
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_encoder = None
_model = None
_subject2idx: dict[str, int] | None = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder


def _load_model():
    from torch_measure.models import AmortizedIRT

    with open(ARTIFACTS_DIR / "subject2idx.json") as f:
        subject2idx = json.load(f)

    with open(ARTIFACTS_DIR / "model_meta.json") as f:
        meta = json.load(f)

    state = torch.load(ARTIFACTS_DIR / "amortized_irt.pt", map_location=_DEVICE, weights_only=True)

    model = AmortizedIRT(
        n_subjects=meta["n_subjects"],
        n_items=meta["n_items"],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=256,
        n_layers=3,
        pl=2,
        dropout=0.1,
        device=str(_DEVICE),
    )
    model.load_state_dict(state)
    model.eval()
    return model, subject2idx


def _ensure_loaded():
    global _model, _subject2idx
    if _model is None:
        _model, _subject2idx = _load_model()


def _probability(subject_content: str, item_emb: torch.Tensor) -> float:
    """P(correct) from subject text key and a precomputed item embedding."""
    _ensure_loaded()
    subj_key = subject_content or ""
    idx = _subject2idx.get(subj_key) if _subject2idx else None
    if idx is not None:
        theta = _model.ability[idx]
    else:
        theta = _model.ability.mean()

    emb = item_emb.to(_DEVICE)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    params = _model.item_net(emb)
    b = params[:, 0]
    a = torch.exp(params[:, 1]) if _model.pl >= 2 else None

    p = _model._irt_probability(theta.unsqueeze(0), b, a)[0].item()
    return float(np.clip(p, 0.0, 1.0))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return predicted probability that the subject answers the item correctly."""
    _ensure_loaded()
    encoder = _get_encoder()

    item_text = input.get("item_content") or ""
    item_emb = encoder.encode([item_text], convert_to_tensor=True)[0]
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
    except FileNotFoundError as e:
        print(f"Smoke test skipped (artifacts missing): {e}")
