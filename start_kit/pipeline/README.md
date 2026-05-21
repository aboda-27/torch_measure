# Amortized IRT Pipeline

Barebones pipeline for the Predictive AI Evaluation Challenge: embed item and subject text with MPNet, train `AmortizedIRT` from [torch_measure](https://github.com/aims-foundations/torch_measure), and expose a `predict()` function that returns P(correct) for a `(subject_content, item_content)` pair.

## What it does

1. **embed.py** — Loads [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db), embeds unique item/subject strings with `all-mpnet-base-v2` on Modal (parallel `.map()`).
2. **train.py** — Trains `AmortizedIRT` on an Modal A10G GPU (AdamW + BCE).
3. **model.py** — Loads weights at import time; `predict(input)` encodes the item at runtime and looks up subject ability (mean fallback if unknown).

```
HF measurement-db  →  embed.py  →  artifacts/  →  train.py  →  amortized_irt.pt
                                                              →  model.py predict()
```

## Prerequisites

- Python 3.10+
- **Virtual environment (recommended)** — see [AGENT_SETUP.md](./AGENT_SETUP.md)
- [Modal](https://modal.com/) account (`modal setup` / `modal token new`)
- **Hugging Face token (recommended)** — for measurement-db + MPNet downloads
- Enough disk for HF cache and `artifacts/` (full run: several GB)
- Repo cloned with submodules not required

### Hugging Face token

Create a token at https://huggingface.co/settings/tokens (read access is enough).

**Option A — export in your shell (easiest for local + Modal embed):**

```bash
export HF_TOKEN=hf_your_token_here
```

`embed.py` passes this to Modal workers and bakes MPNet into the image at build time.

**Option B — Modal secret (persists across terminals):**

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

If `HF_TOKEN` is not in your shell, embed falls back to the `huggingface` Modal secret.

Local `load_training_data()` also respects `HF_TOKEN` / `huggingface-cli login` for downloading measurement-db.

## Install

Use a venv so `torch`, `sentence-transformers`, `modal`, and editable `torch_measure` do not conflict with system Python. Modal GPU jobs use their own images; the venv is only for **local** CLI and smoke tests.

From the **repository root** (`torch_measure/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e .
pip install -r start_kit/pipeline/requirements.txt
modal setup
```

**Cursor agent?** Follow the step-by-step checklist in [AGENT_SETUP.md](./AGENT_SETUP.md) (includes `.venv/bin/python` / `.venv/bin/modal` one-liners that work without `source`).

`pip install -e .` installs `torch_measure` (needed for training). `model.py` only needs `torch`, `sentence-transformers`, and the artifact files for local inference.

## Representative training data (default)

Embed loads measurement-db using the [starting-kit parquet model](../README.md#loading-the-public-training-data):

1. Read **`benchmarks.parquet`** from HF for domain metadata.
2. **Domain-stratified benchmark selection** — include benchmarks from every primary `domain` stratum (see `sampling.py`).
3. Load each selected benchmark’s responses via **`torch_measure.datasets.load`** (one parquet at a time; avoids OOM).
4. **Random 25% row subsample** (seed `42`) so huge suites do not dominate.
5. **20% item holdout** for val (unchanged).

Artifacts: `selected_benchmarks.json`, `sampling_meta.json`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIPELINE_SAMPLE_MODE` | `stratified` | `stratified` \| `all` \| `legacy_max` |
| `PIPELINE_ROW_SAMPLE_FRAC` | `0.25` | Bernoulli fraction of rows to keep |
| `PIPELINE_BENCHMARKS_PER_DOMAIN` | unset | Max benchmarks per domain (unset = all in each domain) |
| `PIPELINE_MAX_DATASETS` | — | Only with `PIPELINE_SAMPLE_MODE=legacy_max` (old “first N alphabetically” smoke) |
| `PIPELINE_HF_BULK_LOAD` | off | `1` = README bulk `load_dataset` for all response parquets (high RAM) |

**Production representative run:**

```bash
unset PIPELINE_MAX_DATASETS
export PIPELINE_SAMPLE_MODE=stratified
export PIPELINE_ROW_SAMPLE_FRAC=0.25
modal run start_kit/pipeline/embed.py
```

**Small dev run:**

```bash
export PIPELINE_SAMPLE_MODE=stratified
export PIPELINE_BENCHMARKS_PER_DOMAIN=1
export PIPELINE_ROW_SAMPLE_FRAC=0.1
modal run start_kit/pipeline/embed.py
```

## Train/val split workflow (explicit splits)

Split rule: **20% of items** held out → all their rows in `val_triples.npy`; rest in `train_triples.npy` (`utils.py`: `VAL_ITEM_FRAC=0.2`, `SPLIT_SEED=42`). Training saves the **best val-loss** checkpoint.

**One command per line** from repo root:

```bash
cd /path/to/torch_measure
source .venv/bin/activate

# 1) Embed + create train_triples.npy / val_triples.npy
modal run start_kit/pipeline/embed.py

# 2) Confirm split files exist locally
ls -lh start_kit/pipeline/artifacts/train_triples.npy start_kit/pipeline/artifacts/val_triples.npy start_kit/pipeline/artifacts/val_item_indices.json

# 3) Train (uploads artifacts to volume, then GPU train with train_loss + val_loss)
modal run start_kit/pipeline/train.py

# 4) Pull best checkpoint + smoke test
modal volume get irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/
modal volume get irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/
python3 start_kit/pipeline/model.py
```

If embed artifacts are **already on the volume** and you only need to re-train with splits from legacy `triples.npy`:

```bash
modal run start_kit/pipeline/train.py --upload false
```

(Train splits `triples.npy` on the GPU if `train_triples.npy` / `val_triples.npy` are missing.)

Long embed run (use detach so Ctrl+C does not kill workers):

```bash
modal run --detach start_kit/pipeline/embed.py
# wait for completion in Modal dashboard, then:
modal run start_kit/pipeline/train.py
```

## Quick smoke run (recommended first)

Uses stratified sampling with **1 benchmark per domain** and **10%** of rows (see `run_smoke.sh`).

**Run one command per line** (do not paste the block as a single line — a `# comment` glued to `train.py` breaks the shell).

```bash
cd /path/to/torch_measure
source .venv/bin/activate
export HF_TOKEN=hf_your_token_here
export PIPELINE_SAMPLE_MODE=stratified
export PIPELINE_BENCHMARKS_PER_DOMAIN=1
export PIPELINE_ROW_SAMPLE_FRAC=0.1

modal run --detach start_kit/pipeline/embed.py
modal run start_kit/pipeline/train.py

modal volume get irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/
modal volume get irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/

python3 start_kit/pipeline/model.py
```

Or use the helper script (same steps):

```bash
chmod +x start_kit/pipeline/run_smoke.sh
./start_kit/pipeline/run_smoke.sh
```

You should see a line like `predict() -> 0.xxxx` (value depends on weights).

## Full production run

Uses **all benchmarks** (stratified mode includes every domain) and **25%** of rows by default.

```bash
unset PIPELINE_MAX_DATASETS
export PIPELINE_SAMPLE_MODE=stratified
export PIPELINE_ROW_SAMPLE_FRAC=0.25

modal run start_kit/pipeline/embed.py
modal run start_kit/pipeline/train.py

modal volume get irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/
modal volume get irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/

python3 start_kit/pipeline/model.py
```

## Step-by-step

### Step 1 — Embed

```bash
export HF_TOKEN=hf_your_token_here   # recommended
modal run --detach start_kit/pipeline/embed.py
```

Use `--detach` so closing the terminal or pressing Ctrl+C does not cancel the 32 embedding workers.

**What happens**

- Loads a **representative benchmark set** via `data_hf.py` / `sampling.py` (manifest + per-benchmark `torch_measure.datasets.load` on your machine).
- Sends text batches to Modal workers (`embed_batch.map`, batch size 256).
- Writes under `start_kit/pipeline/artifacts/`:

| File | Description |
|------|-------------|
| `item_embs.npy` | `(n_items, 768)` MPNet embeddings |
| `subject_embs.npy` | `(n_subjects, 768)` (saved for future use; not used in `predict`) |
| `item2idx.json` | item text → index |
| `subject2idx.json` | subject text → index |
| `train_triples.npy` | Training rows (items in train split) |
| `val_triples.npy` | Validation rows (held-out items) |
| `val_item_indices.json` | Which item indices are in val |
| `selected_benchmarks.json` | Domain-stratified benchmark ids used for this run |
| `sampling_meta.json` | Mode, row fraction, seeds, row counts |
| `triples.npy` | Legacy: all rows (train splits this if train/val npy missing) |

- Syncs the same files to Modal volume `irt-pipeline-artifacts` for training.

**Sampling env** — see [Representative training data](#representative-training-data-default) above.

### Step 2 — Train

**Requires step 1** — `start_kit/pipeline/artifacts/` must contain `item_embs.npy`, `triples.npy`, and the JSON maps. `train.py` uploads those files to the Modal volume automatically before starting the GPU job.

```bash
modal run start_kit/pipeline/train.py
```

**What happens**

- Uploads local `artifacts/` to volume `irt-pipeline-artifacts` (from your laptop).
- Reloads artifacts from volume on an **A10G** worker.
- Trains `AmortizedIRT` (768-d embeddings, hidden 256, 3 layers, 2PL, 30 epochs, batch 512).
- Saves on the volume:
  - `amortized_irt.pt` — `state_dict`
  - `model_meta.json` — `n_subjects`, `n_items`

Training does not re-download HF data; it uses `triples.npy` from step 1.

### Step 3 — Local predict smoke test

Ensure these exist under `start_kit/pipeline/artifacts/`:

- `amortized_irt.pt`
- `model_meta.json`
- `subject2idx.json`

(from embed + volume get after train)

```bash
python3 start_kit/pipeline/model.py
```

### Step 3b — Run `predict()` on the val split

Uses `val_triples.npy` + index maps to call the same `predict()` API as the competition (encodes item text at runtime).

```bash
modal volume get irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/
modal volume get irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/

python3 start_kit/pipeline/eval_val.py
# faster smoke: first 1000 val rows only
python3 start_kit/pipeline/eval_val.py 1000
```

Prints val BCE and accuracy. **First run can look hung for 1–2 minutes** while MPNet downloads (~400MB); you will see `Loading MPNet encoder...` then a progress bar for batch encoding. Eval batch-encodes unique val items once (not one encode per row).

### Step 4 — Competition submission

Layout for a Codabench zip (see [start_kit/README.md](../README.md)):

```text
my_submission/
  model.py
  artifacts/
    amortized_irt.pt
    subject2idx.json
    model_meta.json
```

Copy `model.py` from this directory. It does **not** require `utils.py` in the zip.

Validate:

```bash
python3 start_kit/tools/check_submission_zip.py my_submission.zip
python3 start_kit/tools/run_smoke_test.py my_submission/
```

## `predict()` API

```python
predict(input: dict, labeled: list[dict] | None = None) -> float
```

**`input` keys:** `benchmark`, `condition`, `subject_content`, `item_content`

**Returns:** float in `[0, 1]`

Example:

```python
from model import predict

p = predict({
    "benchmark": "mmlupro",
    "condition": "zero-shot",
    "subject_content": "Name: Example Model\nOrganization: Example Org",
    "item_content": "What is the capital of France?",
})
```

## Pipeline files

| File | Run with |
|------|----------|
| `utils.py` | Imported by embed/train (not shipped in submission zip) |
| `embed.py` | `modal run start_kit/pipeline/embed.py` |
| `train.py` | `modal run start_kit/pipeline/train.py` |
| `model.py` | `python3 start_kit/pipeline/model.py` or competition ingestion |
| `requirements.txt` | `pip install -r ...` |

## Troubleshooting

**`modal: command not found`** — `pip install modal` and ensure your venv is active.

**HF download errors** — Check network and Hugging Face access. Try `PIPELINE_BENCHMARKS_PER_DOMAIN=1` and `PIPELINE_ROW_SAMPLE_FRAC=0.05` for a tiny run.

**`FileNotFoundError` in `model.py`** — Run embed + train and `modal volume get` for `amortized_irt.pt` and `model_meta.json`.

**`sync_to_volume` / Mount errors** — Run embed through to completion so `artifacts/` is populated before the sync function runs.

**Subject always gets mean ability** — `subject_content` at runtime must match the strings in `subject2idx.json` (same format as `render_subject_content` in `utils.py`). Mismatched formatting falls back to mean ability.

**Train OOM on A10G** — Reduce `TRAIN_BATCH_SIZE` in `train.py` or lower `PIPELINE_ROW_SAMPLE_FRAC` / `PIPELINE_BENCHMARKS_PER_DOMAIN`.

**`IsADirectoryError: .../torch_measure`** — Fixed in `train.py` (use `add_local_dir(src)` instead of `pip_install_from_pyproject` on the repo root). Re-run `modal run start_kit/pipeline/train.py`.

**`add_local_*` after build step** — Modal requires `add_local_dir` last in the image chain (no `.env()` after it). Re-run `modal run start_kit/pipeline/train.py` after pulling latest `train.py`.

**`IndexError` on `PIPELINE_DIR.parents[1]` in embed.py** — Modal workers mount the file at `/root/embed.py`; `utils` must only load in `main()`, not at import. Pull latest `embed.py` and re-run embed.

**`IndexError` on `PIPELINE_DIR.parents[1]` in train.py** — Same fix pattern in `train.py`. Pull latest and re-run.

**`ModuleNotFoundError: sentence_transformers`** — The Modal train image installs `sentence-transformers` (required by `torch_measure.models.NCF` during package import). Re-run `modal run start_kit/pipeline/train.py` to rebuild the image.

**`zsh: command not found: #`** — Commands were pasted on one line; run each command separately or use `run_smoke.sh`.

**`modal volume get` — No such file** — Training did not finish; complete `train.py` successfully before pulling `amortized_irt.pt`.

**`FileNotFoundError: /artifacts/item_embs.npy`** — Embed was not run, or volume upload failed. Run `modal run start_kit/pipeline/embed.py` first and confirm `start_kit/pipeline/artifacts/item_embs.npy` exists locally, then re-run `train.py`.

**`modal has no attribute 'Mount'`** — Your Modal version removed `modal.Mount`; use the latest `embed.py` / `train.py` (they use `Volume.batch_upload()` instead).

**HF Hub unauthenticated warnings** — `export HF_TOKEN=hf_...` before `modal run embed.py`, or `modal secret create huggingface HF_TOKEN=hf_...`.

**Embed cancelled mid-run** — Re-run with `modal run --detach start_kit/pipeline/embed.py`; check progress in the Modal dashboard.

## Notes

- Responses are binarized with `label = 1.0 if response >= 0.5 else 0.0` for BCE (some benchmarks are continuous).
- Items are **cold-encoded** at inference; only subject indices and IRT weights are baked in.
- No W&B or `labeled`-example adaptation in this starter.

**Train/val split:** embed holds out **20% of items** (all their responses → `val_triples.npy`). Training optimizes train rows only; `train.py` prints `train_loss` and `val_loss` each epoch and saves the **best val** checkpoint to `amortized_irt.pt`. If you only have legacy `triples.npy`, train splits on the fly with the same rule (`VAL_ITEM_FRAC=0.2`, `SPLIT_SEED=42` in `utils.py`).
