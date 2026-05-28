#!/bin/bash
#SBATCH -J molgen-h2l-prep
#SBATCH -p Students
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

cd "$REPO_DIR"
mkdir -p logs "$OUTDIR"

source /home/scc/pb22000262/miniconda3/etc/profile.d/conda.sh
conda activate dataset-agent

python scripts/preprocess_molgenbench_h2l.py \
  --root "$RAW_ROOT" \
  --outdir "$OUTDIR" \
  --pocket-radius "${POCKET_RADIUS:-10.0}" \
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
