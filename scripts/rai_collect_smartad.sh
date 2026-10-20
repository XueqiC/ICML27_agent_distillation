#!/bin/bash
# One CPU collection per K100 support set; one shared POSIX limiter directory.
set -euo pipefail
resume=0
if [[ ${1:-} == --resume ]]; then resume=1; shift; fi
[[ $# == 0 ]] || { echo "Usage: $0 [--resume]" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
: "${ENV_PYTHON:=.venv/bin/python}" "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}" "${SMARTAD_ROOT:=reports/runs/smartad-collections}"
: "${SMARTAD_RATE_LIMIT_DIR:=data/smartad_rate_limit}" "${LOG_ROOT:=logs}"
: "${SMARTAD_FULL_SOURCE:=$COLLECTION_ROOT/seeds_full_train_r2}"
cells=(k100_s0 k100_s1 k100_s2)
sources=("$COLLECTION_ROOT/seeds_k100_s0" "$COLLECTION_ROOT/seeds_k100_s1" "$COLLECTION_ROOT/seeds_k100_s2")
if [[ -f "$SMARTAD_FULL_SOURCE/manifest.json" ]]; then
    cells+=(full_train); sources+=("$SMARTAD_FULL_SOURCE")
else
    echo "Full-training source not yet available: $SMARTAD_FULL_SOURCE; collecting K100 cells only"
fi
for i in "${!cells[@]}"; do
    [[ -f "${sources[$i]}/manifest.json" ]] || { echo "Missing source: ${sources[$i]}" >&2; exit 2; }
    output="$SMARTAD_ROOT/${cells[$i]}"
    [[ ! -e "$output" || $resume == 1 ]] || { echo "Output exists: $output; use --resume" >&2; exit 2; }
    [[ ! -e "$output" || -f "$output/identity.json" ]] || { echo 'Existing output has no identity' >&2; exit 2; }
done
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_smartad.lock"
flock -n 9 || { echo 'SmartAD launcher already running' >&2; exit 2; }
"$ENV_PYTHON" - "$COLLECT_CONFIG" "${sources[@]}" <<'PY'
import sys
from sed.config import load_config
from sed.manifest import load_frozen
from sed.seed_evidence import load_verified_collection
config = load_config(sys.argv[1])
manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
for index, path in enumerate(sys.argv[2:]):
    collection, selection, _ = load_verified_collection(path, manifest, freeze)
    if index < 3:
        if collection['K'] != 100 or selection['seed'] != index:
            raise ValueError('K100 support cell mismatch')
    elif selection['selection_rule'] != 'all_train_v1':
        raise ValueError('Full-training source must cover every eligible TRAIN game')
PY
for i in "${!cells[@]}"; do
    cell=${cells[$i]}
    subsets=()
    if [[ "$cell" != full_train ]]; then subsets=(--subset-k 5 10 20 50); fi
    log_file=$(mktemp "$LOG_ROOT/collect_smartad_${cell}.XXXXXX.log")
    echo "SmartAD $cell; log=$log_file"
    "$ENV_PYTHON" -m baselines.smartad.pipeline --config "$COLLECT_CONFIG" \
        --verified "${sources[$i]}" --output "$SMARTAD_ROOT/$cell" --through collect \
        --rate-limit-dir "$SMARTAD_RATE_LIMIT_DIR" "${subsets[@]}" >"$log_file" 2>&1
done
