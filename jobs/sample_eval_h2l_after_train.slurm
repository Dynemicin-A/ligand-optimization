#!/bin/bash
#SBATCH -J ligopt-h2l-eval
#SBATCH -p Students
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/ligopt-h2l-eval-%j.out
#SBATCH -e logs/ligopt-h2l-eval-%j.err

set -euo pipefail

REPO_DIR="/home/scc/pb22000262/ligand-optimization"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed training output directory.}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoint_last.pt}"
MANIFEST="${MANIFEST:-data/processed_h2l/train/manifest.txt}"
SAMPLE_DIR="${SAMPLE_DIR:-${RUN_DIR}/generated_sdf}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
NUM_STEPS="${NUM_STEPS:-64}"
EVAL_OUT="${EVAL_OUT:-${RUN_DIR}/eval_metrics.json}"

cd "$REPO_DIR"
mkdir -p logs "$SAMPLE_DIR"

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

if [ ! -s "$CHECKPOINT" ]; then
  if [ "$CHECKPOINT" = "latest" ] || [ "$CHECKPOINT" = "auto" ]; then
    if [ -s "$RUN_DIR/checkpoint_last.pt" ]; then
      CHECKPOINT="$RUN_DIR/checkpoint_last.pt"
    else
      CHECKPOINT="$(ls -1t "$RUN_DIR"/checkpoint_step_*.pt 2>/dev/null | head -n 1 || true)"
    fi
  fi
fi

if [ ! -s "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 66
fi

echo "checkpoint: $CHECKPOINT"
echo "sample_dir: $SAMPLE_DIR"
echo "eval_out: $EVAL_OUT"

python scripts/sample_h2l_manifest.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --outdir "$SAMPLE_DIR" \
  --device cuda \
  --num-samples "$NUM_SAMPLES" \
  --num-steps "$NUM_STEPS"

python scripts/evaluate_molecules.py \
  --generated-sdf "$SAMPLE_DIR/sdf" \
  --reference-smiles "$SAMPLE_DIR/reference.smi" \
  --source-smiles "$SAMPLE_DIR/source.smi" \
  --out "$EVAL_OUT" \
  --expected-total "$NUM_SAMPLES"
