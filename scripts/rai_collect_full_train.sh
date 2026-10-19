#!/bin/bash
# Explicit live CPU collection only when this launcher is run by the operator.
set -euo pipefail
resume=0
workers=${WORKERS:-8}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) resume=1; shift ;;
        --workers) workers=${2:?Missing worker count}; shift 2 ;;
        *) echo "Usage: $0 [--resume] [--workers N]" >&2; exit 2 ;;
    esac
done
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "Workers must be positive" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
export ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1}
: "${ENV_PYTHON:=.venv/bin/python}"
: "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}"
: "${LOG_ROOT:=logs}"
output="$COLLECTION_ROOT/seeds_full_train"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_full_train.lock"
flock -n 9 || { echo "Full TRAIN launcher already running" >&2; exit 2; }
args=(--config "$COLLECT_CONFIG" --selection data/manifest/seeds_full_train.json
      --attempts 3 --shard-count "$workers" --max-calls 160000
      --max-input-tokens 700000000 --output "$output")
preflight_resume=()
if [[ $resume == 1 ]]; then preflight_resume=(--resume); fi
# Read-only preflight checks the identity and hashes of every game before any call.
"$ENV_PYTHON" teachers/collect_seeds.py "${args[@]}" "${preflight_resume[@]}" --preflight
pids=()
stop() {
    trap - TERM INT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 143
}
trap stop TERM INT
for ((shard=0; shard<workers; shard++)); do
    log_file=$(mktemp "$LOG_ROOT/collect_full_train_shard${shard}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")
    echo "[$(date -u +%FT%TZ)] full TRAIN shard $shard/$workers; log=$log_file"
    # Preflight guarded fresh vs resume. Workers join the same newly initialized
    # collection; shard locks exclude duplicate work and preserve one identity.
    "$ENV_PYTHON" teachers/collect_seeds.py "${args[@]}" --resume --shard-index "$shard" >"$log_file" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$ENV_PYTHON" - "$output/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(json.dumps({key: value[key] for key in
                 ("status", "candidates", "K", "unique_instances", "task_groups", "budget")}))
sys.exit(0 if value["status"] == "complete" else 1)
PY
exit "$status"
