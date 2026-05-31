#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/home/zhangxuanhao/zxh/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-50}"
RETRY_SLEEP="${RETRY_SLEEP:-60}"
NUM_DOWNLOADS="${NUM_DOWNLOADS:-4}"

now() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

is_running() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

start_dataset() {
  local dataset="$1"
  local dirname="$2"
  local pattern="$3"
  local outdir="$DATA_ROOT/$dirname"
  mkdir -p "$outdir/logs"

  if [ -f "$outdir/download.done" ]; then
    return 0
  fi
  if is_running "$pattern"; then
    return 0
  fi

  if [ "$dataset" = "crossdocked2020" ]; then
    find "$outdir/raw/.cache" -name "*.lock" -type f -print -delete 2>/dev/null || true
  fi

  local log="$outdir/logs/download_$(date +%Y%m%d_%H%M%S).log"
  nohup "$PYTHON_BIN" "$REPO_DIR/scripts/download_sbdd_datasets.py" \
    --root "$DATA_ROOT" \
    --datasets "$dataset" \
    --hf-endpoint "$HF_ENDPOINT" \
    --max-attempts "$MAX_ATTEMPTS" \
    --retry-sleep "$RETRY_SLEEP" \
    --num-downloads "$NUM_DOWNLOADS" \
    > "$log" 2>&1 &
  echo "[$(now)] restarted $dataset pid=$! log=$log"
}

echo "[$(now)] SBDD dataset manager on $(hostname)"
start_dataset pdbbind_v2020 PDBbind_v2020 "download_sbdd_datasets.py .*pdbbind_v2020|download_pdbbind_v2020.sh|download_zenodo_record.py 7014096"
start_dataset bindingmoad_prepared BindingMOAD_prepared "download_sbdd_datasets.py .*bindingmoad_prepared|download_bindingmoad_prepared.sh|download_zenodo_record.py 11191555"
start_dataset crossdocked2020 CrossDocked2020 "download_sbdd_datasets.py .*crossdocked2020|download_crossdocked2020.sh|hf download kohbanye/crossdocked2020"

echo "---- active downloads ----"
ps -u "$USER" -o pid,etime,pcpu,pmem,cmd \
  | grep -E "download_sbdd_datasets.py|download_(pdbbind|bindingmoad|crossdocked)|download_zenodo_record|wget -c|hf download" \
  | grep -v grep || true

echo "---- dataset status ----"
for dirname in PDBbind_v2020 BindingMOAD_prepared CrossDocked2020; do
  echo "==== $dirname ===="
  if [ -f "$DATA_ROOT/$dirname/download.done" ]; then
    echo "DONE=yes"
  else
    echo "DONE=no"
  fi
  du -sh "$DATA_ROOT/$dirname" 2>/dev/null || true
  latest="$(ls -t "$DATA_ROOT/$dirname"/logs/download_*.log 2>/dev/null | head -1 || true)"
  echo "latest_log=$latest"
  find "$DATA_ROOT/$dirname" -maxdepth 4 -type f \
    \( -name "download.done" -o -name "*.tar.gz" -o -name "*.tar" -o -name "*.parquet" -o -name "archive_manifest.txt" \) \
    -printf "%TY-%Tm-%Td %TH:%TM %12s %p\n" 2>/dev/null | sort | tail -25 || true
  if [ -n "$latest" ]; then
    echo "---- log tail ----"
    tail -25 "$latest" || true
  fi
done

echo "---- disk ----"
df -h "$DATA_ROOT" || true
