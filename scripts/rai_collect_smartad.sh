#!/bin/bash
# Candidate-slot CPU shards for every cell; one shared POSIX limiter directory.
set -euo pipefail
resume=0
if [[ ${1:-} == --resume ]]; then resume=1; shift; fi
[[ $# == 0 ]] || { echo "Usage: $0 [--resume]" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
: "${ENV_PYTHON:=.venv/bin/python}" "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}" "${SMARTAD_ROOT:=reports/runs/smartad-collections}"
: "${SMARTAD_RATE_LIMIT_DIR:=data/smartad_rate_limit}" "${LOG_ROOT:=logs}"
: "${SMARTAD_SHARDS:=16}"
: "${SMARTAD_FULL_SOURCE:=$COLLECTION_ROOT/seeds_full_train_r2}"
[[ "$SMARTAD_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "SMARTAD_SHARDS must be positive" >&2; exit 2; }
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_smartad.lock"
flock -n 9 || { echo 'SmartAD launcher already running' >&2; exit 2; }
cells=(k100_s0 k100_s1 k100_s2)
sources=("$COLLECTION_ROOT/seeds_k100_s0" "$COLLECTION_ROOT/seeds_k100_s1" "$COLLECTION_ROOT/seeds_k100_s2")
if [[ -f "$SMARTAD_FULL_SOURCE/manifest.json" ]] && "$ENV_PYTHON" -c \
    'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' \
    "$SMARTAD_FULL_SOURCE/manifest.json"; then
    cells+=(full_train); sources+=("$SMARTAD_FULL_SOURCE")
else
    echo "Skipping full_train: $SMARTAD_FULL_SOURCE/manifest.json is missing or its status is not complete"
fi
for i in "${!cells[@]}"; do
    [[ -f "${sources[$i]}/manifest.json" ]] || { echo "Missing source: ${sources[$i]}" >&2; exit 2; }
    output="$SMARTAD_ROOT/${cells[$i]}"
    [[ ! -e "$output" || $resume == 1 ]] || { echo "Output exists: $output; use --resume" >&2; exit 2; }
    [[ ! -e "$output" || -f "$output/identity.json" ]] || { echo 'Existing output has no identity' >&2; exit 2; }
done
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
pids=()
stop() {
    trap - TERM INT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 143
}
trap stop TERM INT
launch_cell() {
    local i=$1 cell=${cells[$1]} shard log_file
    local -a subsets=()
    if [[ "$cell" != full_train ]]; then subsets=(--subset-k 5 10 20 50); fi
    for ((shard=0; shard<SMARTAD_SHARDS; shard++)); do
        log_file=$(mktemp "$LOG_ROOT/collect_smartad_${cell}_shard${shard}.XXXXXX.log")
        echo "SmartAD $cell shard $shard/$SMARTAD_SHARDS; log=$log_file"
        # The launcher checked fresh/resume above. Workers may join the shared
        # identity created by another shard and reuse already durable slots.
        "$ENV_PYTHON" -m baselines.smartad.pipeline --config "$COLLECT_CONFIG" \
            --verified "${sources[$i]}" --output "$SMARTAD_ROOT/$cell" --through collect \
            --rate-limit-dir "$SMARTAD_RATE_LIMIT_DIR" "${subsets[@]}" \
            --shard-index "$shard" --shard-count "$SMARTAD_SHARDS" --resume >"$log_file" 2>&1 &
        pids+=("$!")
    done
}
for i in "${!cells[@]}"; do launch_cell "$i"; done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
# A non-reusable slot 1 belongs to a deterministic shard too. Its owner may
# have exited before the other nine slots settled. One more sharded pass can
# finish every such dependency; durable slots are read, never executed again.
if [[ $status == 0 ]]; then
    pids=()
    for i in "${!cells[@]}"; do
        if [[ ! -f "$SMARTAD_ROOT/${cells[$i]}/candidates.json" ]]; then
            echo "SmartAD ${cells[$i]}: completing deferred candidate slots"
            launch_cell "$i"
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
fi
# The last finishing shard publishes only after every candidate slot is final.
"$ENV_PYTHON" - "$SMARTAD_ROOT" "${cells[@]}" <<'PY' || status=1
import json, sys
from pathlib import Path
complete = True
for cell in sys.argv[2:]:
    path = Path(sys.argv[1]) / cell / 'candidates.json'
    status = json.loads(path.read_text()).get('status') if path.exists() else 'incomplete'
    print(f'SmartAD {cell}: {status}')
    complete &= status == 'complete'
sys.exit(0 if complete else 1)
PY
exit "$status"
