#!/bin/bash
#SBATCH -J ligopt-pretrain-1g
#SBATCH -p Students
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/ligopt-pretrain-1g-%j.out
#SBATCH -e logs/ligopt-pretrain-1g-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/scc/pb22000262/ligand-optimization}"
CONFIG="${CONFIG:-configs/pretrain_complexes_a100_1gpu_4h.yaml}"
RUN_NAME="${RUN_NAME:-pretrain_complexes_a100_1gpu_4h_${SLURM_JOB_ID}}"
OUTDIR="${OUTDIR:-outputs/${RUN_NAME}}"

cd "$REPO_DIR"
mkdir -p logs outputs

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python scripts/train_diffusion.py \
  --config "$CONFIG" \
  --outdir "$OUTDIR" \
  --device cuda \
  --num-workers "${NUM_WORKERS:-0}"
