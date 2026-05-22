#!/usr/bin/env bash
# Smoke pipeline: stratified benchmark sample + row subsample (run each step separately).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export PIPELINE_SAMPLE_MODE="${PIPELINE_SAMPLE_MODE:-stratified}"
export PIPELINE_BENCHMARKS_PER_DOMAIN="${PIPELINE_BENCHMARKS_PER_DOMAIN:-1}"
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

echo "=== 1/4 embed (mode=$PIPELINE_SAMPLE_MODE per_domain=$PIPELINE_BENCHMARKS_PER_DOMAIN row_frac=0.25) ==="
"$MODAL" run competition_submission/training/embed.py

if [[ ! -f competition_submission/training/artifacts/item_embs.npy ]]; then
  echo "ERROR: embed did not produce competition_submission/training/artifacts/item_embs.npy"
  exit 1
fi

echo "=== 2/4 train ==="
"$MODAL" run competition_submission/training/train.py

echo "=== 3/4 fetch weights from Modal volume ==="
bash competition_submission/training/scripts/pull_artifacts.sh

echo "=== 4/4 local predict smoke test ==="
"$PYTHON" competition_submission/submission/model.py

if [[ -f competition_submission/training/artifacts/val_triples.npy ]]; then
  echo "=== optional: val holdout (1000 rows) ==="
  "$PYTHON" competition_submission/training/eval_val.py 1000 || true
fi

echo "=== sync Codabench zip (my_submission/) ==="
bash competition_submission/training/scripts/sync_codabench.sh

echo "Done."
