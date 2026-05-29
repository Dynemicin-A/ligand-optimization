#!/bin/bash
#SBATCH -J ligopt-pretrain-prep
#SBATCH -p Students
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/ligopt-pretrain-prep-%j.out
#SBATCH -e logs/ligopt-pretrain-prep-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/scc/pb22000262/ligand-optimization}"
CSV="${CSV:?Set CSV to a pretraining complex CSV with protein_path,ligand_path columns.}"
OUTDIR="${OUTDIR:-${REPO_DIR}/data/pretrain_complexes/train}"
MAX_RECORDS_ARG=()
if [ "${MAX_RECORDS:-}" != "" ]; then
  MAX_RECORDS_ARG=(--max-records "$MAX_RECORDS")
fi

cd "$REPO_DIR"
mkdir -p logs "$OUTDIR"

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python scripts/preprocess_complex_pretrain.py \
  --csv "$CSV" \
  --outdir "$OUTDIR" \
  --pocket-radius "${POCKET_RADIUS:-10.0}" \
  "${MAX_RECORDS_ARG[@]}"
