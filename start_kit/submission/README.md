# How to run this submission

Unzip this folder, `cd` into it, and follow the steps below. Everything needed for **`predict()`** is in this directory.

## What’s in this folder

```text
.
├── README.md                 # this file
├── model.py                  # entry point: predict() loads weights + MPNet here
├── models.txt                # Hugging Face id for the text encoder (one line)
├── labeling.py               # how training labels were built (not used at inference)
└── artifacts/
    ├── amortized_irt.pt      # trained AmortizedIRT weights
    ├── model_meta.json       # n_subjects, n_items, training metadata
    └── subject2idx.json      # maps subject_content strings → ability indices
```

**Codabench** runs `model.py` only (no `labeling.py`). **Gradescope** should include the same `model.py` + `artifacts/` + `models.txt` as your best Codabench zip, plus `labeling.py` and this README.

---

## 1. Install dependencies

Python **3.10+** recommended.

```bash
cd /path/to/unzipped/submission   # folder that contains model.py

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
# Inference: PyTorch, MPNet encoder, and torch_measure (AmortizedIRT class used in model.py)
pip install "torch>=2.0" "sentence-transformers>=2.0" "numpy>=1.24" torch_measure
```

First run downloads the encoder in `models.txt` (`sentence-transformers/all-mpnet-base-v2`) from Hugging Face unless a cache already exists.

---

## 2. Smoke test (one prediction)

```bash
# Still inside the submission folder (same directory as model.py)
python model.py
```

Expected: a line like `predict() -> 0.xxxx`.  
If you see `Smoke test skipped: ...`, check that `artifacts/amortized_irt.pt` exists and dependencies installed correctly.

---

## 3. Call `predict()` from Python

```python
from model import predict

prob = predict({
    "benchmark": "mmlupro",
    "condition": "none",
    "subject_content": "Name: Example\nOrganization: ...",
    "item_content": "Your question text here.",
})
print(prob)   # float in [0, 1] — P(correct)
```

**Required keys:** `benchmark`, `condition`, `subject_content`, `item_content`  
**Returns:** probability the subject answers correctly.

The model encodes **item text** with MPNet at runtime (same format as training). If `subject_content` is not in `artifacts/subject2idx.json`, it uses the mean subject ability.

---

## 4. `labeling.py` (training only)

Not imported by `model.py`. It documents how raw benchmark `response` values were turned into binary training labels (Likert thresholds, fractions, etc.) when the checkpoint in `artifacts/` was trained.

Open `labeling.py` in an editor to read the binarization rules.

---

## 5. Retrain from scratch (optional)

This zip is the **inference bundle**. Full re-embed + train on measurement-db uses the parent repo (`torch_measure`), Modal, and Hugging Face — not runnable from this folder alone. See the course repo `start_kit/pipeline/` for embed/train scripts if you need to reproduce training.

**Training settings for this checkpoint:** 16 public benchmarks, `PIPELINE_ROW_SAMPLE_FRAC=1.0`, labels via `labeling.py`, 10-epoch AmortizedIRT, best checkpoint by validation BCE with live MPNet on held-out items. Details in `artifacts/model_meta.json`.

---


*Authors: rebuild with `bash start_kit/pipeline/sync_submission.sh` from the `torch_measure` repo (Modal steps in `start_kit/pipeline/OUTLINE.md`).*
