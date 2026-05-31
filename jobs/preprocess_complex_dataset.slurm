#!/bin/bash
#SBATCH -J ligopt-complex-prep
#SBATCH -p Students
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/ligopt-complex-prep-%j.out
#SBATCH -e logs/ligopt-complex-prep-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/scc/pb22000262/ligand-optimization}"
OUTDIR="${OUTDIR:-${REPO_DIR}/data/pretrain_complexes/train}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SOURCE_MODE="${SOURCE_MODE:-self}"
PRESET="${PRESET:-auto}"

INPUT_ARGS=()
if [ "${CSV:-}" != "" ]; then
  INPUT_ARGS=(--csv "$CSV")
elif [ "${JSONL:-}" != "" ]; then
  INPUT_ARGS=(--jsonl "$JSONL")
elif [ "${ROOT:-}" != "" ]; then
  INPUT_ARGS=(--root "$ROOT" --preset "$PRESET")
else
  echo "Set one of CSV, JSONL, or ROOT." >&2
  exit 64
fi

EXTRA_ARGS=()
if [ "${MAX_RECORDS:-}" != "" ]; then
  EXTRA_ARGS+=(--max-records "$MAX_RECORDS")
fi
if [ "${SKIP_EXISTING:-}" = "1" ]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [ "${LIMIT_DIRS:-}" != "" ]; then
  EXTRA_ARGS+=(--limit-dirs "$LIMIT_DIRS")
fi
if [ "${LIGANDS_PER_PROTEIN:-}" != "" ]; then
  EXTRA_ARGS+=(--ligands-per-protein "$LIGANDS_PER_PROTEIN")
fi
if [ "${MAX_LIGANDS_PER_DIR:-}" != "" ]; then
  EXTRA_ARGS+=(--max-ligands-per-dir "$MAX_LIGANDS_PER_DIR")
fi

cd "$REPO_DIR"
mkdir -p logs "$OUTDIR"

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python scripts/preprocess_complex_dataset.py \
  "${INPUT_ARGS[@]}" \
  --outdir "$OUTDIR" \
  --pocket-radius "${POCKET_RADIUS:-10.0}" \
  --num-workers "$NUM_WORKERS" \
  --chunk-size "${CHUNK_SIZE:-64}" \
  --source-mode "$SOURCE_MODE" \
  "${EXTRA_ARGS[@]}"
