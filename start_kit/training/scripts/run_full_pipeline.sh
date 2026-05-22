#!/usr/bin/env bash
# Smoke pipeline: stratified benchmark sample + row subsample (run each step separately).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export PIPELINE_SAMPLE_MODE="${PIPELINE_SAMPLE_MODE:-stratified}"
export PIPELINE_BENCHMARKS_PER_DOMAIN="${PIPELINE_BENCHMARKS_PER_DOMAIN:-1}"
export PIPELINE_ROW_SAMPLE_FRAC="${PIPELINE_ROW_SAMPLE_FRAC:-0.1}"
unset PIPELINE_MAX_DATASETS

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Export it or run: modal secret create huggingface HF_TOKEN=hf_..."
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
MODAL="${REPO_ROOT}/.venv/bin/modal"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
  MODAL="modal"
fi

echo "=== 1/4 embed (mode=$PIPELINE_SAMPLE_MODE per_domain=$PIPELINE_BENCHMARKS_PER_DOMAIN row_frac=$PIPELINE_ROW_SAMPLE_FRAC) ==="
"$MODAL" run start_kit/training/embed.py

if [[ ! -f start_kit/training/artifacts/item_embs.npy ]]; then
  echo "ERROR: embed did not produce start_kit/training/artifacts/item_embs.npy"
  exit 1
fi

echo "=== 2/4 train ==="
"$MODAL" run start_kit/training/train.py

echo "=== 3/4 fetch weights from Modal volume ==="
bash start_kit/training/scripts/pull_artifacts.sh

echo "=== 4/4 local predict smoke test ==="
"$PYTHON" start_kit/submission/model.py

if [[ -f start_kit/training/artifacts/val_triples.npy ]]; then
  echo "=== optional: val holdout (1000 rows) ==="
  "$PYTHON" start_kit/training/eval_val.py 1000 || true
fi

echo "=== sync Codabench zip (my_submission/) ==="
bash start_kit/training/scripts/sync_codabench.sh

echo "Done."
