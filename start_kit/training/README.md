# Pipeline outline

How the amortized IRT pipeline works end-to-end for the [Predictive AI Evaluation Challenge](https://www.codabench.org/). For Gradescope/Codabench replication steps, see [../submission/INSTRUCTOR_README.md](../submission/INSTRUCTOR_README.md).

Related docs: [start_kit/README.md](../README.md) (HF data + Codabench contract), [SUBMIT.md](../../SUBMIT.md) (upload).

---

## What this pipeline does

1. **Load** public training data from Hugging Face (`aims-foundations/measurement-db`).
2. **Label** each response row as a binary target (or soft tie for rewardbench).
3. **Embed** item text with MPNet; save maps and train/val triples on a Modal volume.
4. **Train** AmortizedIRT: a small network maps item embeddings → IRT difficulty/discrimination; learn per-subject ability.
5. **Submit** `predict()` that encodes **new** item text at runtime and outputs P(correct) ∈ [0, 1].

Heavy steps run on Modal; your laptop runs `modal`, optional `eval_val.py`, and zip checks.

**Pipeline flow** (embed → train → submit):

```
measurement-db (HF, 16 benchmarks)
         │
         ├──────────────────────────────┐
         ▼                              ▼
   labeling.py                    MPNet (embed.py)
   response → y                   item text → vectors
         │                              │
         ▼                              ▼
 train_triples.npy              item_embs.npy
 val_triples.npy                subject_embs.npy
         │                              │
         └──────────────┬───────────────┘
                        ▼
                 train.py (Modal)
           BCE on (subject_idx, item_idx, y)
           + frozen item_embs for train rows
           + val_loss = live MPNet on held-out items
                        │
                        ▼
              amortized_irt.pt + model_meta.json
                        │
                        ▼
              predict() on Codabench
           live MPNet on item text → item_net → P(correct)
```

---

## Core concepts (read this first)

### Training data: 16 benchmarks, not 92

The competition points at `aims-foundations/measurement-db`, which currently ships **16** benchmarks as long-form response parquets plus `benchmarks.parquet`. The wider Measurement Data Bank catalogs **92** “ready” benchmarks; the rest need parquets on HF or a local build before this pipeline can train on them.

`PIPELINE_SAMPLE_MODE=all` loads every benchmark in the manifest (16 today). `stratified` loads all of them grouped by domain (same set when no per-domain cap).

### Generalization at test time

**Amortized IRT** is built for **cold items**: questions you did not train on. At submit time, `predict()`:

- Formats item text like training (`Benchmark` / `Condition` / content).
- Encodes with MPNet.
- Runs `item_net(embedding)` → IRT params → P(correct).

It does **not** look up a training `item_idx`. Training on 16 benchmarks teaches a mapping from **embedding space** to difficulty, not memorization of benchmark names.

### Embeddings vs labels (different objects)

| | Embeddings | Labels |
|---|------------|--------|
| **What** | MPNet vector of item **text** | Binary target **y** per (subject, item) row |
| **Where** | `item_embs.npy` | `train_triples.npy`, `val_triples.npy` |
| **Used in train** | Frozen rows via `item_idx` | BCE loss vs model output |
| **Used in predict** | Encoded **live** from text | N/A (no labels at test) |

Changing label rules does **not** change embeddings. **Re-run embed** only if you need new triples/labels (or new data). Re-train after label or data changes.

### Training vs validation vs submission paths

| Phase | Item side | Subject side |
|-------|-----------|--------------|
| **Train (fast)** | Precomputed `item_embs[i]` | `ability[subject_idx]` |
| **Val in train (`val_loss`)** | Live MPNet on held-out item **text** | Same indices |
| **Val in train (`val_loss_idx`)** | Precomputed `item_embs[i]` | Same (diagnostic only) |
| **`predict()` / Codabench** | Live MPNet on input text | `subject2idx` string match, else mean ability |

The **best checkpoint** is saved using **`val_loss`** (live encode), so it aligns with submission better than `val_loss_idx`.

---

## Repository layout

```text
torch_measure/                      # modal run from here
  start_kit/training/
    README.md                         # quick run (commands)
    OUTLINE.md                        # this file
    embed.py  train.py  data_hf.py  labeling.py  ...
    artifacts/                        # optional copy from Modal volume
  start_kit/tools/                    # zip / smoke checks
  submission/                         # inference (Gradescope entry)
  my_submission.zip                   # built by scripts/sync_codabench.sh
```

Modal volume **`irt-pipeline-artifacts`** is the canonical store between embed and train.

Local mirror: `artifacts/` (see `artifacts/MANIFEST.md`). Re-sync: `bash start_kit/training/scripts/pull_artifacts.sh`.

---

## Stage-by-stage

### 1. Load and sample (`data_hf.py`, `sampling.py`)

- Reads `benchmarks.parquet` for the benchmark list and metadata.
- Loads each `{benchmark_id}.parquet` (responses), joins `items.parquet` and `subjects.parquet`.
- Optional row subsample: `PIPELINE_ROW_SAMPLE_FRAC` (default 0.25 in code; use `1.0` for full replication).
- Builds unique item/subject strings and integer indices.

### 2. Label (`labeling.py`)

Raw `response` values are not all 0/1. Rules follow `response_type`:

| Type | Examples | Rule |
|------|----------|------|
| `binary` | mmlupro, swebench, … | Keep 0/1; else ≥ 0.5 |
| `likert_10` | mtbench | score ≥ 7 → 1 |
| `likert_5` | ultrafeedback | score ≥ 4 → 1 |
| `fraction` | cybench | clip [0,1], ≥ 0.5 → 1 |
| `mixed` | matharena | round near 0/1; else ≥ 0.5 |
| ties | rewardbench | 0 / 0.5 / 1 explicit |

A global `response >= 0.5` rule wrongly marks almost all mtbench (1–10) and ultrafeedback (1–5) rows as correct.

Rules are logged in `sampling_meta.json` → `label_rules` after embed.

### 3. Split (`utils.py`)

**20% of items** (all their rows) → `val_triples.npy`. Remaining items → `train_triples.npy`. This matches evaluating on **new questions**, not random rows from seen items.

### 4. Embed (`embed.py`)

- Parallel GPU workers encode unique item strings → shards → `item_embs.npy`.
- Also saves `subject_embs.npy` (not used in current `predict()`).
- Writes `item2idx.json`, `subject2idx.json`, metadata JSONs.

### 5. Train (`train.py`)

- **AmortizedIRT** from `torch_measure`: `item_net` (MLP on 768-d embeddings) + per-subject `ability`.
- **10 epochs**, AdamW with `weight_decay=1e-2`.
- Saves `amortized_irt.pt` at epoch with lowest **live-encode** `val_loss`.
- `model_meta.json`: `best_val_loss`, `val_metric: "live_encode"`, counts.

### 6. Submit (`model.py`)

Loads at import: MPNet + IRT weights + `subject2idx.json`. Each `predict()` call encodes one item and returns a calibrated probability.

Zip layout (via `sync_submission.sh`):

```text
model.py
models.txt          # sentence-transformers/all-mpnet-base-v2
artifacts/
  amortized_irt.pt
  model_meta.json
  subject2idx.json
```

---

## Item text format

Same at embed, train-val, and predict:

```text
Benchmark: {benchmark}
Condition: {condition}

{content}
```

---

## Configuration reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIPELINE_REPO_ID` | `aims-foundations/measurement-db` | HF dataset |
| `PIPELINE_SAMPLE_MODE` | `stratified` | Benchmark selection (`all` = all in manifest) |
| `PIPELINE_ROW_SAMPLE_FRAC` | `0.25` | Row subsample (use `1.0` for full data) |
| `PIPELINE_LIKERT_10_MIN` | `7` | mtbench threshold |
| `PIPELINE_LIKERT_5_MIN` | `4` | ultrafeedback threshold |
| `PIPELINE_FRACTION_MIN` | `0.5` | fractional benchmarks |
| `PIPELINE_EMBED_MAX_CONTAINERS` | `16` | Parallel T4 workers (not “16 benchmarks”) |
| `PIPELINE_EMBED_SKIP_ITEMS` | off | Skip item re-embed if `item_embs.npy` exists |
| `PIPELINE_EMBED_SUBJECTS_ONLY` | off | Resume subjects without HF reload |

---

## Artifacts on the volume

| File | Role |
|------|------|
| `train_triples.npy`, `val_triples.npy` | `(subject_idx, item_idx, label)` |
| `item_embs.npy` | Training-time item vectors |
| `item2idx.json`, `subject2idx.json` | Text ↔ index maps |
| `sampling_meta.json` | Includes `label_rules`, row counts |
| `amortized_irt.pt`, `model_meta.json` | Trained model for submission |

---

## `predict()` API

```python
predict(input: dict, labeled: list[dict] | None = None) -> float
```

**Input keys:** `benchmark`, `condition`, `subject_content`, `item_content`  
**Output:** P(correct) in `[0, 1]`

---

## Source files

| File | Role |
|------|------|
| `data_hf.py` | HF load, joins |
| `labeling.py` | Response → label |
| `sampling.py` | Benchmark selection |
| `utils.py` | Splits, formatting |
| `embed.py` | Modal embed orchestrator |
| `train.py` | Modal training |
| `model.py` | Submission |
| `eval_val.py` | Local val BCE |
| `scripts/sync_codabench.sh` | Build `my_submission.zip` from `submission/` |

Optional tests: `tests/test_pipeline_sampling.py` at repo root.

---

## Resume / fix volume

**Subjects only** (items already on volume):

```bash
export PIPELINE_EMBED_SKIP_ITEMS=1
export PIPELINE_EMBED_SUBJECTS_ONLY=1
modal run --detach start_kit/training/embed.py --no-sync-local
```

**Legacy typo `items_embs.npy`:**

```bash
modal volume rm irt-pipeline-artifacts item_embs.npy
modal volume cp irt-pipeline-artifacts items_embs.npy item_embs.npy
modal volume rm irt-pipeline-artifacts items_embs.npy
```

Re-run `train.py` after fixing embeddings; delete stale `amortized_irt.pt` if trained on old labels.

**Upload local artifacts before train:** run `train.py` without `--no-upload`.

---

## Training on more than 16 benchmarks

Point `PIPELINE_REPO_ID` at a bucket with long-form parquets for more benchmarks, set `PIPELINE_SAMPLE_MODE=all`, re-embed and re-train. `labeling.py` applies by `response_type` automatically.

---

## Notes

- Val = **item** holdout, not random row or subject holdout.
- No fine-tuning on Codabench `labeled` examples in this starter.
- Leaderboard metric is log-loss; calibrated probabilities matter.
