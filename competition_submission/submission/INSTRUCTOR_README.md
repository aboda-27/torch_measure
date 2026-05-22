# Instructor guide 

Run all commands from **repo root** (the folder that contains `pyproject.toml`).

---

## Verify the submission

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

cd competition_submission/submission
python model.py
```

Expected: `predict() -> 0.xxxx`

Weights live in `competition_submission/submission/artifacts/` (`amortized_irt.pt`, `model_meta.json`, `subject2idx.json`). First run downloads MPNet from Hugging Face (~400MB); needs network.

Training code and label rules: [`../training/`](../training/) · Pipeline details: [`../training/README.md`](../training/README.md)

---

## Pull weights from Modal

Only if artifacts are missing and a prior run exists on volume `irt-pipeline-artifacts`.

```bash
source .venv/bin/activate
pip install "modal>=0.64"
modal setup

bash competition_submission/training/scripts/pull_artifacts.sh

cd competition_submission/submission
python model.py
```

---

## Re-embed and train (full pipeline)

Embed/train run on Modal GPUs. From **repo root**:

```bash
source .venv/bin/activate
pip install "modal>=0.64"
modal setup
modal secret create huggingface HF_TOKEN=hf_...   # once
export HF_TOKEN=hf_...

export PIPELINE_ROW_SAMPLE_FRAC=1.0   # optional; script default is 0.1 (smoke)

bash competition_submission/training/scripts/run_full_pipeline.sh
```

That embeds, trains, pulls artifacts, smoke-tests `predict()`, and builds `my_submission.zip`.

---

## Codabench zip only

```bash
bash competition_submission/training/scripts/sync_codabench.sh
```

Upload `my_submission.zip` from repo root.
