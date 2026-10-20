#!/bin/bash
# Explicit live CPU collection only when this launcher is run by the operator.
set -euo pipefail
preflight_only=0
workers=32
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) shift ;;
        --preflight) preflight_only=1; shift ;;
        --workers) workers=${2:?Missing worker count}; shift 2 ;;
        *) echo "Usage: $0 [--resume] [--preflight] [--workers 32]" >&2; exit 2 ;;
    esac
done
[[ "$workers" == 32 ]] || { echo "Recovery requires the original 32 shards" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
export ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1}
: "${ENV_PYTHON:=.venv/bin/python}"
: "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}"
: "${LOG_ROOT:=logs}"
: "${RECOVERY_SOURCE:=/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train}"
output="$COLLECTION_ROOT/seeds_full_train_r2"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_full_train_r2.lock"
flock -n 9 || { echo "Full TRAIN launcher already running" >&2; exit 2; }
args=(--config "$COLLECT_CONFIG" --selection data/manifest/seeds_full_train.json
      --attempts 3 --shard-count "$workers" --max-calls 160000
      --max-input-tokens 700000000 --output "$output")
# Recovery/resume and preflight construct no teacher and make no API calls.
"$ENV_PYTHON" tools/recover_full_train.py --config "$COLLECT_CONFIG" \
    --source "$RECOVERY_SOURCE" --output "$output" --summary "$LOG_ROOT/full_train_r2_recovery.json"
"$ENV_PYTHON" teachers/collect_seeds.py "${args[@]}" --resume --preflight
if [[ $preflight_only == 1 ]]; then exit 0; fi
pids=()
stop() {
    trap - TERM INT
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 143
}
trap stop TERM INT
for ((shard=0; shard<workers; shard++)); do
    log_file=$(mktemp "$LOG_ROOT/collect_full_train_r2_shard${shard}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")
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
