#!/bin/bash
# CPU-only collection. One shared POSIX limiter across all four cells/processes.
set -euo pipefail
resume=0
if [[ ${1:-} == --resume ]]; then resume=1; shift; fi
[[ $# == 0 ]] || { echo "Usage: $0 [--resume]" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
: "${ENV_PYTHON:=.venv/bin/python}" "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}" "${ADFTP_ROOT:=reports/runs/adftp-collections}"
: "${ADFTP_RATE_LIMIT_DIR:=data/adftp_rate_limit}" "${LOG_ROOT:=logs}"
cells=(k100_s0 k100_s1 k100_s2 full_train)
# Preflight every source and identity before any billable request.
for cell in "${cells[@]}"; do
    [[ -f "$COLLECTION_ROOT/seeds_$cell/manifest.json" ]] || { echo "Missing verified source: $cell" >&2; exit 2; }
    output="$ADFTP_ROOT/$cell"
    [[ ! -e "$output" || $resume == 1 ]] || { echo "Output exists: $output; use --resume" >&2; exit 2; }
    [[ ! -e "$output" || -f "$output/identity.json" ]] || { echo 'Existing output has no identity' >&2; exit 2; }
done
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_adftp.lock"
flock -n 9 || { echo 'AD-FTP launcher already running' >&2; exit 2; }
# Read-only validation of every VERIFIED source, including the full cell.
"$ENV_PYTHON" - "$COLLECT_CONFIG" "$COLLECTION_ROOT" <<'PY'
import sys
from pathlib import Path
from sed.config import load_config
from sed.manifest import load_frozen
from sed.seed_evidence import load_verified_collection
config = load_config(sys.argv[1])
manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
for cell in ('k100_s0', 'k100_s1', 'k100_s2', 'full_train'):
    collection, selection, _ = load_verified_collection(Path(sys.argv[2]) / ('seeds_' + cell), manifest, freeze)
    if cell == 'full_train':
        if selection['selection_rule'] != 'all_train_v1':
            raise ValueError('Full-training source must cover every eligible TRAIN game')
    elif collection['K'] != 100 or selection['seed'] != int(cell[-1]):
        raise ValueError('K100 support cell mismatch')
PY
for cell in "${cells[@]}"; do
    output="$ADFTP_ROOT/$cell"
    [[ ! -e "$output" || $resume == 1 ]] || { echo "Output exists: $output" >&2; exit 2; }
    subsets=()
    if [[ "$cell" != full_train ]]; then subsets=(--subset-k 5 10 20 50); fi
    log_file=$(mktemp "$LOG_ROOT/collect_adftp_${cell}.XXXXXX.log")
    echo "AD-FTP $cell; log=$log_file"
    "$ENV_PYTHON" -m baselines.adftp.pipeline --config "$COLLECT_CONFIG" \
        --verified "$COLLECTION_ROOT/seeds_$cell" --output "$output" --through collect \
        --rate-limit-dir "$ADFTP_RATE_LIMIT_DIR" "${subsets[@]}" >"$log_file" 2>&1
done
