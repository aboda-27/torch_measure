# Amortized IRT Pipeline

End-to-end flow for the Predictive AI Evaluation Challenge: load **measurement-db** on Modal, embed text with MPNet, train **AmortizedIRT**, expose **`predict()`** for P(correct).

Everything heavy runs on **Modal** (`irt-pipeline-artifacts` volume). Your laptop only needs a venv for `modal run`, `modal volume get`, and local `eval_val.py` / `model.py`.

```
measurement-db (HF)
       │
       ▼
  embed.py  ──►  volume: embeddings, maps, train/val triples
       │
       ▼
  train.py  ──►  volume: amortized_irt.pt, model_meta.json
       │
       ▼
  model.py / eval_val.py  ──►  predict() at runtime (encode item text live)
```

---

## System overview

| Stage | Script | Where it runs | Output |
|-------|--------|---------------|--------|
| Load + split | `data_hf.py` (called by embed) | Modal | `item2idx.json`, `subject2idx.json`, `train_triples.npy`, `val_triples.npy` |
| Embed | `embed.py` | Modal (up to 16× T4) | `item_embs.npy`, `subject_embs.npy` |
| Train | `train.py` | Modal (A10G) | `amortized_irt.pt` (best **val** epoch), `model_meta.json` |
| Inference | `model.py` | Competition container / local | `predict(input) → float` |
| Holdout eval | `eval_val.py` | Local | Val BCE + accuracy on `val_triples.npy` |

**Item text** (embed + predict) is always:

```text
Benchmark: {benchmark}
Condition: {condition}

{content}
```

**Val split:** 20% of **items** held out (all their rows → `val_triples.npy`). Train rows use the other 80% of items. Same rule as competition-style cold items: val items never appear in `train_triples.npy`, but their text is encoded at eval time like new items.

**Training** uses precomputed `item_embs.npy` + indices. **`predict()`** encodes the item with MPNet on each call and runs the amortized head (subject ability from `subject2idx.json`, or mean ability if unknown).

---

## Prerequisites

- Python 3.10+, repo cloned
- [Modal](https://modal.com/) (`modal setup`)
- Hugging Face read token for measurement-db + MPNet

```bash
# repo root
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e . && pip install -r start_kit/pipeline/requirements.txt
modal setup
export HF_TOKEN=hf_...   # or: modal secret create huggingface HF_TOKEN=hf_...
```

Cursor agents: see [AGENT_SETUP.md](./AGENT_SETUP.md).

---

## Default data policy

Controlled by env vars (read in `sampling.py` / `data_hf.py`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIPELINE_SAMPLE_MODE` | `stratified` | Pick benchmarks per domain (`all` = every benchmark; `legacy_max` = old smoke) |
| `PIPELINE_ROW_SAMPLE_FRAC` | `0.25` | Keep ~25% of rows per benchmark parquet (before join, seed 42) |
| `PIPELINE_BENCHMARKS_PER_DOMAIN` | unset | Cap benchmarks per domain (unset = all in each domain) |
| `PIPELINE_EMBED_SHARD_SIZE` | `512` | Texts per embed worker |
| `PIPELINE_EMBED_MAX_CONTAINERS` | `16` | Max parallel T4 embed jobs |
| `PIPELINE_EMBED_SKIP_ITEMS` | off | `1` = skip item embed if `item_embs.npy` exists |
| `PIPELINE_EMBED_SUBJECTS_ONLY` | off | `1` = resume subjects only (no HF reload) |

HF still downloads full parquets; sampling reduces **join work and RAM**, not download size.

---

## Run (production)

From **repo root**, one command per line. Use **`--detach`** on long Modal jobs so Ctrl+C on your machine does not kill them.

```bash
source .venv/bin/activate
export HF_TOKEN=hf_...
export PIPELINE_SAMPLE_MODE=stratified
export PIPELINE_ROW_SAMPLE_FRAC=0.25
unset PIPELINE_MAX_DATASETS

# 1) Load + embed + splits → volume (no local copy)
modal run --detach start_kit/pipeline/embed.py --no-sync-local

# 2) Train on volume (local artifacts/ can be empty)
modal run --detach start_kit/pipeline/train.py --no-upload

# 3) Pull weights for local predict / eval
modal volume get --force irt-pipeline-artifacts amortized_irt.pt start_kit/pipeline/artifacts/
modal volume get --force irt-pipeline-artifacts model_meta.json start_kit/pipeline/artifacts/

python3 start_kit/pipeline/model.py
```

**Sync embed artifacts locally** (optional): omit `--no-sync-local` on embed, or `modal volume get` individual files.

**Upload local artifacts before train** (if you synced embed locally): `modal run start_kit/pipeline/train.py` (default uploads `artifacts/` then trains).

---

## Small smoke run

```bash
export PIPELINE_BENCHMARKS_PER_DOMAIN=1
export PIPELINE_ROW_SAMPLE_FRAC=0.1
modal run start_kit/pipeline/embed.py
modal run start_kit/pipeline/train.py
```

Or: `./start_kit/pipeline/run_smoke.sh`

---

## Resume / fix volume

**Subjects only** (items already on volume):

```bash
export PIPELINE_EMBED_SKIP_ITEMS=1
export PIPELINE_EMBED_SUBJECTS_ONLY=1
modal run --detach start_kit/pipeline/embed.py --no-sync-local
```

**Wrong item embeddings:** train expects **`item_embs.npy`** with shape `(n_items, 768)` matching `item2idx.json`. If you have a large **`items_embs.npy`** (~642 MiB) and a small **`item_embs.npy`** (~296 MiB), replace before re-training:

```bash
modal volume rm irt-pipeline-artifacts item_embs.npy
modal volume cp irt-pipeline-artifacts items_embs.npy item_embs.npy
modal volume rm irt-pipeline-artifacts items_embs.npy
```

Then re-run `train.py --no-upload`. Delete stale `amortized_irt.pt` if it was trained on old/smaller data.

---

## Evaluate holdout (val)

Uses the same logic as `predict()` on rows in `val_triples.npy`:

```bash
modal volume get --force irt-pipeline-artifacts val_triples.npy start_kit/pipeline/artifacts/
modal volume get --force irt-pipeline-artifacts item2idx.json start_kit/pipeline/artifacts/
modal volume get --force irt-pipeline-artifacts subject2idx.json start_kit/pipeline/artifacts/
# plus amortized_irt.pt + model_meta.json (see above)

python3 start_kit/pipeline/eval_val.py 1000   # quick
python3 start_kit/pipeline/eval_val.py         # full val set
```

First run downloads MPNet (~400 MB); then batch-encodes unique val items once.

Training already prints **train_loss** and **val_loss** each epoch and saves the **best val** checkpoint. `eval_val.py` is the local check that matches the competition API.

---

## Artifacts

All live on Modal volume **`irt-pipeline-artifacts`** (and optionally `start_kit/pipeline/artifacts/`).

| File | Used by |
|------|---------|
| `item_embs.npy` | **train** (fixed embeddings per item index) |
| `subject_embs.npy` | Saved only; not used in `predict()` |
| `item2idx.json`, `subject2idx.json` | Train indices; eval maps index → text |
| `train_triples.npy`, `val_triples.npy` | Train / val loss |
| `val_item_indices.json` | Which item indices are held out |
| `selected_benchmarks.json`, `sampling_meta.json`, `embed_summary.json` | Run metadata |
| `amortized_irt.pt`, `model_meta.json` | **predict** / submission |
| `triples.npy` | Legacy; train can split on the fly if train/val npy missing |

---

## `predict()` API

```python
predict(input: dict, labeled: list[dict] | None = None) -> float
```

**`input` keys:** `benchmark`, `condition`, `subject_content`, `item_content`  
**Returns:** P(correct) in `[0, 1]`

```python
from model import predict

p = predict({
    "benchmark": "mmlupro",
    "condition": "zero-shot",
    "subject_content": "Name: Example\nOrganization: Example",
    "item_content": "What is the capital of France?",
})
```

---

## Competition zip

Source of truth: [`model.py`](model.py) + [`models.txt`](models.txt) in this directory. After training, sync to `my_submission/`:

```bash
bash start_kit/pipeline/sync_submission.sh
```

```text
my_submission/
  model.py
  models.txt
  artifacts/
    amortized_irt.pt
    subject2idx.json
    model_meta.json
```

```bash
python3 start_kit/tools/check_submission_zip.py my_submission.zip
python3 start_kit/tools/run_smoke_test.py my_submission/
```

**Local val eval** (uses same `model.py` as submission):

```bash
python3 start_kit/pipeline/eval_val.py 1000
```

See [start_kit/README.md](../README.md) and repo-root [`SUBMIT.md`](../../SUBMIT.md).

---

## Source files

| File | Role |
|------|------|
| `data_hf.py` | HF load, row sampling, joins |
| `sampling.py` | Env-based benchmark selection |
| `utils.py` | Splits, `format_item_content`, mappings |
| `embed.py` | `modal run` — orchestrator + parallel `embed_shard` |
| `train.py` | `modal run` — AmortizedIRT, BCE, best val checkpoint |
| `model.py` | Submission `predict()` |
| `eval_val.py` | Local val metrics |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Local `artifacts/` all MISSING with `--no-upload` | Normal; train reads the **volume** |
| `Expected N embeddings, got M` | Fix `item_embs.npy` vs `items_embs.npy` (see Resume) |
| Embed/train killed by Ctrl+C | `modal run --detach ...` |
| `FileNotFoundError` in `model.py` | Pull `amortized_irt.pt` + `model_meta.json` from volume |
| Subject always mean ability | `subject_content` must match keys in `subject2idx.json` |
| HF auth warnings | `export HF_TOKEN` or Modal `huggingface` secret |
| Train OOM | Lower `TRAIN_BATCH_SIZE` in `train.py` or `PIPELINE_ROW_SAMPLE_FRAC` |

---

## Notes

- Labels: `1.0 if response >= 0.5 else 0.0` (BCE).
- Val = **item** holdout, not random rows or subject holdout.
- No W&B or `labeled`-example fine-tuning in this starter.
