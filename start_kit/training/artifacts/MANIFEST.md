# Latest run artifacts (local mirror)

Synced from Modal volume **`irt-pipeline-artifacts`**. Keep this whole folder for the current pipeline run.

**Run config** (see `sampling_meta.json`): 16 benchmarks, `row_sample_frac=0.25`, benchmark-aware `label_rules`.

## Submission (Codabench zip)

| File | Role |
|------|------|
| `../model.py` | `predict()` |
| `../models.txt` | MPNet repo id |
| `amortized_irt.pt` | Trained weights |
| `model_meta.json` | Shapes + `best_val_loss` |
| `subject2idx.json` | Subject → ability index |

## Embed outputs

| File | Role |
|------|------|
| `item_embs.npy` | Frozen MPNet vectors (train forward pass) |
| `subject_embs.npy` | Subject embeddings (saved; not used in `predict()`) |
| `item2idx.json` | Item text → index |
| `train_triples.npy` | Train (subject_idx, item_idx, label) |
| `val_triples.npy` | Val holdout triples |
| `val_item_indices.json` | Held-out item ids |
| `unique_items_strings.npy` | Resume / debug item list |
| `unique_subjects_strings.npy` | Resume / debug subject list |
| `sampling_meta.json` | Benchmarks, row counts, `label_rules` |
| `selected_benchmarks.json` | Benchmark selection summary |
| `embed_summary.json` | Shard counts, triple counts |

## Not kept locally (still on Modal volume)

| Path | Why skip |
|------|----------|
| `embed_shards/` | Merged into `item_embs.npy` |
| `triples.npy` | Legacy; use `train_triples` / `val_triples` |
| `ncf_head.pt`, `blend_meta.json` | Old ensemble experiment |

## Re-sync from volume

From repo root:

```bash
bash start_kit/training/scripts/pull_artifacts.sh
```
