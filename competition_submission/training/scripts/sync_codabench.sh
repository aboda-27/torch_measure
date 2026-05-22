#!/usr/bin/env bash
# Build Codabench zip from start_kit/submission/ (single source of truth).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

SUB="start_kit/submission"
TRAIN_ART="start_kit/training/artifacts"
OUT="my_submission"

for f in amortized_irt.pt model_meta.json subject2idx.json; do
  if [[ ! -f "$SUB/artifacts/$f" ]]; then
    if [[ -f "$TRAIN_ART/$f" ]]; then
      mkdir -p "$SUB/artifacts"
      cp "$TRAIN_ART/$f" "$SUB/artifacts/"
    else
      echo "ERROR: missing $SUB/artifacts/$f (run training or pull_artifacts.sh)"
      exit 1
    fi
  fi
done

rm -rf "$OUT"
mkdir -p "$OUT/artifacts"
cp "$SUB/model.py" "$OUT/model.py"
cp "$SUB/models.txt" "$OUT/models.txt"
cp "$SUB/artifacts/amortized_irt.pt" "$SUB/artifacts/model_meta.json" "$SUB/artifacts/subject2idx.json" "$OUT/artifacts/"

rm -rf "$OUT/__pycache__"
rm -f "my_submission.zip"
(cd "$OUT" && zip -r "../my_submission.zip" . -x "__pycache__/*" -x "*.pyc")

echo "Synced → $OUT/ and my_submission.zip (from $SUB/)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "$PYTHON" ]]; then
  "$PYTHON" start_kit/tools/check_submission_zip.py my_submission.zip
fi
