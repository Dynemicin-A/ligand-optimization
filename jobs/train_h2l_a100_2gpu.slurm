#!/bin/bash
#SBATCH -J ligopt-h2l-a100
#SBATCH -p GPU-A100
#SBATCH --gres=gpu:A100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 4-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -o logs/ligopt-h2l-a100-%j.out
#SBATCH -e logs/ligopt-h2l-a100-%j.err

set -euo pipefail

REPO_DIR="/home/scc/pb22000262/ligand-optimization"
CONFIG="${CONFIG:-configs/train_h2l_a100_full.yaml}"
RUN_NAME="${RUN_NAME:-h2l_a100_full_${SLURM_JOB_ID}}"
OUTDIR="${OUTDIR:-outputs/${RUN_NAME}}"

cd "$REPO_DIR"
mkdir -p logs outputs

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

MANIFEST="$(python - <<'PY'
import os
import yaml
from pathlib import Path
cfg_path = os.environ.get("CONFIG", "configs/train_h2l_a100_full.yaml")
cfg = yaml.safe_load(open(cfg_path))
path = Path(cfg["data"]["manifest"])
print(path if path.is_absolute() else Path.cwd() / path)
PY
)"

if [ ! -s "$MANIFEST" ]; then
  echo "Missing H2L manifest: $MANIFEST" >&2
  echo "Run scripts/preprocess_h2l.py first, or set CONFIG to a config with an existing data.manifest." >&2
  exit 66
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if torch.cuda.device_count() < 2:
    raise SystemExit("Need at least 2 visible CUDA devices")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i}", torch.cuda.get_device_name(i))
PY

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_diffusion.py \
  --config "$CONFIG" \
  --outdir "$OUTDIR" \
  --device cuda
