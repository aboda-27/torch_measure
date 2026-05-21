#!/usr/bin/env bash
# Copy pipeline model + submission artifacts into my_submission/ and rebuild zip.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PIPELINE="start_kit/pipeline"
OUT="my_submission"
ART="$PIPELINE/artifacts"

for f in amortized_irt.pt model_meta.json subject2idx.json; do
  if [[ ! -f "$ART/$f" ]]; then
    echo "ERROR: missing $ART/$f (run embed/train or modal volume get)"
    exit 1
  fi
done

mkdir -p "$OUT/artifacts"
cp "$PIPELINE/model.py" "$OUT/model.py"
cp "$PIPELINE/models.txt" "$OUT/models.txt"
cp "$ART/amortized_irt.pt" "$ART/model_meta.json" "$ART/subject2idx.json" "$OUT/artifacts/"

rm -rf "$OUT/__pycache__"
(cd "$OUT" && zip -r "../my_submission.zip" . -x "__pycache__/*" -x "*.pyc")

echo "Synced → $OUT/ and my_submission.zip"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "$PYTHON" ]]; then
  "$PYTHON" start_kit/tools/check_submission_zip.py my_submission.zip
fi
