# Course submission (Gradescope + Codabench)

Gradescope links this **GitHub repo**. Start here for inference; training code is in [`../training/`](../training/).

## Layout in this repo

```text
start_kit/submission/          ← you are here (inference)
  model.py
  models.txt
  artifacts/
    amortized_irt.pt
    model_meta.json
    subject2idx.json

start_kit/training/            ← embed, train, labeling rules
  labeling.py
  embed.py, train.py, ...
```

**Codabench:** build `my_submission.zip` with `bash start_kit/training/scripts/sync_codabench.sh` (zip has no `labeling.py`).

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
cd start_kit/submission
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
bash start_kit/training/scripts/pull_artifacts.sh
```

**This checkpoint:** 16 benchmarks, `row_sample_frac=0.25`, live-encode val checkpoint — see `model_meta.json`.
