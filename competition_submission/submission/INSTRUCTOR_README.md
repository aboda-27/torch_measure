# Instructor guide (Gradescope + Codabench)

**For graders and course staff.** Gradescope links this **GitHub repo**. Start here for inference; training code is in [`../training/`](../training/).

## Layout in this repo

```text
competition_submission/submission/          ← you are here (inference)
  model.py
  models.txt
  artifacts/
    amortized_irt.pt
    model_meta.json
    subject2idx.json

competition_submission/training/            ← embed, train, labeling rules
  labeling.py
  embed.py, train.py, ...
```

**Codabench:** build `my_submission.zip` with `bash competition_submission/training/scripts/sync_codabench.sh` (zip has no `labeling.py`).

---

## 1. Install dependencies

From repo root (`torch_measure/`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "torch>=2.0" "sentence-transformers>=2.0" "numpy>=1.24"
```

---

## 2. Smoke test

```bash
cd competition_submission/submission
python model.py
```

Expected: `predict() -> 0.xxxx`

---

## 3. Call `predict()`

```python
from model import predict

prob = predict({
    "benchmark": "mmlupro",
    "condition": "none",
    "subject_content": "Name: Example\nOrganization: ...",
    "item_content": "Your question text here.",
})
print(prob)   # P(correct) in [0, 1]
```

---

## 4. Training labels

See [`../training/labeling.py`](../training/labeling.py): benchmark-aware binarization (mtbench ≥ 7, ultrafeedback ≥ 4, etc.). Not used at inference.

Adaptive labeling (`acquisition_function`) is stubbed — this submission ignores Codabench `labeled` hints.

---

## 5. Reproduce training

See [`../training/README.md`](../training/README.md). Quick sync from Modal:

```bash
bash competition_submission/training/scripts/pull_artifacts.sh
```

**This checkpoint:** 16 benchmarks, `row_sample_frac=0.25`, live-encode val checkpoint — see `model_meta.json`.

---

## 6. Full re-embed and train (your own run)

From repo root, with Modal + HF token configured:

```bash
source .venv/bin/activate
export HF_TOKEN=hf_...   # or: modal secret create huggingface HF_TOKEN=...

# optional: full rows instead of 25% subsample
export PIPELINE_ROW_SAMPLE_FRAC=1.0

bash competition_submission/training/scripts/run_full_pipeline.sh
```

Or step by step (same paths Modal uses):

```bash
modal run competition_submission/training/embed.py
modal run competition_submission/training/train.py
bash competition_submission/training/scripts/pull_artifacts.sh
bash competition_submission/training/scripts/sync_codabench.sh
```

Use **`competition_submission/training/`** only — not `competition_submission/pipeline/` (stale duplicate).
