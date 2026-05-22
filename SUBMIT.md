# Codabench upload (ready)

Pre-built submission at repo root:

- **Directory:** `my_submission/`
- **ZIP:** `my_submission.zip` (~972 KB)

Validated:

```bash
python3 competition_submission/tools/check_submission_zip.py my_submission.zip
python3 competition_submission/tools/run_smoke_test.py my_submission/
```

## Upload

1. Open [Predictive Evaluation Competition](https://www.codabench.org/) → **Amortized Prediction** phase.
2. **Submit** → upload `my_submission.zip` (not a CSV).
3. Wait for scoring (no stdout in hosted logs).

Limits: 50 submissions/team/day (UTC).

## Rebuild after re-training

```bash
bash competition_submission/training/scripts/sync_codabench.sh
```

Copies `competition_submission/submission/` → `my_submission/` and rebuilds the zip.

`model.py` loads MPNet + IRT **at import** (not inside `predict()`). On Codabench it uses `/app/hf_cache` + `local_files_only=True` per `models.txt`.
