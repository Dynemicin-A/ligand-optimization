#!/bin/bash
#SBATCH -J molgenbench-dl
#SBATCH -p CPU-6530
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/molgenbench-dl-%j.out
#SBATCH -e logs/molgenbench-dl-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/scc/pb22000262/ligand-optimization}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/raw/molgenbench_v3}"
URL="${URL:-https://zenodo.org/records/18183463/files/MolGenBench_Version3.zip?download=1}"
EXPECTED_MD5="${EXPECTED_MD5:-53593636b089332b50c9da3b646e537c}"

cd "$REPO_DIR"
mkdir -p logs "$DATA_DIR"
cd "$DATA_DIR"

echo "data_dir=$DATA_DIR"
echo "url=$URL"

curl -L -C - \
  --retry 20 \
  --retry-delay 30 \
  --connect-timeout 30 \
  -o MolGenBench_Version3.zip \
  "$URL"

echo "${EXPECTED_MD5}  MolGenBench_Version3.zip" | md5sum -c -
sha256sum MolGenBench_Version3.zip > MolGenBench_Version3.zip.sha256

rm -rf extracted
mkdir -p extracted
unzip -q -o MolGenBench_Version3.zip -d extracted

find extracted -maxdepth 5 -type f | sort > extracted_filelist.txt
du -h -d 3 . > size_report.txt

echo "downloaded and extracted MolGenBench v3"
echo "filelist=$DATA_DIR/extracted_filelist.txt"
