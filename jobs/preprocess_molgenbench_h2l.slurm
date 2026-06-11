#!/bin/bash
#SBATCH -J molgen-h2l-prep
#SBATCH -p Students
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/molgen-h2l-prep-%j.out
#SBATCH -e logs/molgen-h2l-prep-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/scc/pb22000262/ligand-optimization}"
RAW_ROOT="${RAW_ROOT:-${REPO_DIR}/data/raw/molgenbench_v3/extracted/MolGenBench_Version3}"
OUTDIR="${OUTDIR:-${REPO_DIR}/data/processed_h2l/train}"
MAX_RECORDS_ARG=()
if [ "${MAX_RECORDS:-}" != "" ]; then
  MAX_RECORDS_ARG=(--max-records "$MAX_RECORDS")
fi
PAIR_ARGS=()
if [ "${PAIR_MODE:-}" != "" ]; then
  PAIR_ARGS+=(--pair-mode "$PAIR_MODE")
fi
if [ "${MIN_ACTIVITY_DELTA:-}" != "" ]; then
  PAIR_ARGS+=(--min-activity-delta "$MIN_ACTIVITY_DELTA")
fi
if [ "${MAX_PAIRS_PER_SERIES:-}" != "" ]; then
  PAIR_ARGS+=(--max-pairs-per-series "$MAX_PAIRS_PER_SERIES")
fi
if [ "${AUGMENT_COPIES:-}" != "" ]; then
  PAIR_ARGS+=(--augment-copies "$AUGMENT_COPIES")
fi
if [ "${GLOBAL_RANDOM_ROTATE:-}" = "1" ]; then
  PAIR_ARGS+=(--global-random-rotate)
fi
if [ "${HARD_NEGATIVE_MODE:-}" != "" ]; then
  PAIR_ARGS+=(--hard-negative-mode "$HARD_NEGATIVE_MODE")
fi
if [ "${HARD_NEGATIVE_MIN_TARGET_SIMILARITY:-}" != "" ]; then
  PAIR_ARGS+=(--hard-negative-min-target-similarity "$HARD_NEGATIVE_MIN_TARGET_SIMILARITY")
fi
if [ "${HARD_NEGATIVE_SCORE_SLACK:-}" != "" ]; then
  PAIR_ARGS+=(--hard-negative-score-slack "$HARD_NEGATIVE_SCORE_SLACK")
fi
for arg in GLOBAL_TRANSLATE_SIGMA LIGAND_NOISE_SIGMA SOURCE_NOISE_SIGMA PROTEIN_NOISE_SIGMA NEGATIVE_NOISE_SIGMA; do
  value="${!arg:-}"
  if [ "$value" != "" ]; then
    option="$(echo "$arg" | tr '[:upper:]_' '[:lower:]-')"
    PAIR_ARGS+=(--"$option" "$value")
  fi
done

cd "$REPO_DIR"
mkdir -p logs "$OUTDIR"

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python scripts/ligopt_v3.py preprocess-molgenbench-h2l -- \
  --root "$RAW_ROOT" \
  --outdir "$OUTDIR" \
  --pocket-radius "${POCKET_RADIUS:-10.0}" \
  "${PAIR_ARGS[@]}" \
  "${MAX_RECORDS_ARG[@]}"

python - <<'PY'
from pathlib import Path
import torch

manifest = Path("data/processed_h2l/train/manifest.txt")
paths = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]
print("manifest_records", len(paths))
if not paths:
    raise SystemExit("empty manifest")
rec = torch.load(paths[0], map_location="cpu", weights_only=False)
print("first_record", rec["record_id"])
for key in ["protein_atom_type", "source_atom_type", "ligand_atom_type"]:
    print(key, tuple(rec[key].shape))
PY
