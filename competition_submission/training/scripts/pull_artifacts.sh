#!/usr/bin/env bash
# Mirror the latest Modal run into start_kit/training/artifacts/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
ART="start_kit/training/artifacts"
SUB_ART="start_kit/submission/artifacts"
VOL="irt-pipeline-artifacts"

FILES=(
  amortized_irt.pt
  model_meta.json
  item_embs.npy
  subject_embs.npy
  item2idx.json
  subject2idx.json
  train_triples.npy
  val_triples.npy
  val_item_indices.json
  unique_items_strings.npy
  unique_subjects_strings.npy
  sampling_meta.json
  selected_benchmarks.json
  embed_summary.json
)

mkdir -p "$ART"
for f in "${FILES[@]}"; do
  echo "get $f ..."
  modal volume get --force "$VOL" "$f" "$ART/"
done

mkdir -p "$SUB_ART"
for f in amortized_irt.pt model_meta.json subject2idx.json; do
  cp "$ART/$f" "$SUB_ART/"
done

echo "Done → $ART/ ($(du -sh "$ART" | cut -f1) total)"
echo "Copied submission weights → $SUB_ART/"
