#!/bin/bash
#SBATCH -J ligopt-h2l-a100-1g
#SBATCH -p Students
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/ligopt-h2l-a100-1g-%j.out
#SBATCH -e logs/ligopt-h2l-a100-1g-%j.err

set -euo pipefail

REPO_DIR="/home/scc/pb22000262/ligand-optimization"
CONFIG="${CONFIG:-configs/train_h2l_a100_1gpu_4h.yaml}"
RUN_NAME="${RUN_NAME:-h2l_a100_1gpu_4h_${SLURM_JOB_ID}}"
OUTDIR="${OUTDIR:-outputs/${RUN_NAME}}"
RESUME_ARG=()
if [ "${RESUME:-}" != "" ]; then
  RESUME_ARG=(--resume "$RESUME")
fi
INIT_ARG=()
if [ "${INIT_MODEL:-}" != "" ]; then
  INIT_ARG=(--init-model "$INIT_MODEL" --init-weights "${INIT_WEIGHTS:-auto}")
fi
DATA_ARG=()
if [ "${DATA_MANIFEST:-}" != "" ]; then
  DATA_ARG=(--data-manifest "$DATA_MANIFEST")
fi

cd "$REPO_DIR"
mkdir -p logs outputs

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    raise SystemExit("Need at least 1 visible CUDA device")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i}", torch.cuda.get_device_name(i))
PY

python scripts/train_diffusion.py \
  --config "$CONFIG" \
  --outdir "$OUTDIR" \
  --device cuda \
  --num-workers "${NUM_WORKERS:-0}" \
  "${DATA_ARG[@]}" \
  "${INIT_ARG[@]}" \
  "${RESUME_ARG[@]}"
