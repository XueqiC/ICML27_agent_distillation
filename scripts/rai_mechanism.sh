#!/bin/bash
# Later authorized live use only. GPU GAP_JSON VERIFIED OUT NAME=ADAPTER
set -euo pipefail
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
WORKERS=${WORKERS:-8}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be a positive integer' >&2; exit 2; }
case "$BATCH_INVARIANT" in 0|1) ;; *) echo 'BATCH_INVARIANT must be 0 or 1' >&2; exit 2 ;; esac
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
[[ $# == 5 && "$1" =~ ^[0-9]+$ && "$5" == *=* ]] || {
    echo 'Usage: rai_mechanism.sh GPU GAP_JSON VERIFIED OUT NAME=ADAPTER' >&2; exit 2;
}
gpu=$1
GAP_SET=$2
VERIFIED=$3
OUT=$4
adapter=$5
name=${adapter%%=*}
adapter_path=${adapter#*=}
: "${MAX_TEACHER_CALLS:?Set explicit per-episode HTTP-attempt cap}"
: "${MAX_TEACHER_CALLS_RUN:?Set explicit per-run HTTP-attempt cap}"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_ROOT"
ENV_PYTHON=${ENV_PYTHON:-.venv/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/vllm-serve/.venv/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}
export ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
CONFIG=${CONFIG:-configs/base.yaml}
READY_TIMEOUT=${READY_TIMEOUT:-900}
LORA_CONVERSION=${LORA_CONVERSION:-none}
INCLUDE_BASE=${INCLUDE_BASE:-0}
[[ "$INCLUDE_BASE" == 0 || "$INCLUDE_BASE" == 1 ]]
[[ -x "$ENV_PYTHON" && -x "$VLLM_PYTHON" && -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" ]]
# Authorization, checkpoint/seed binding, immutable population and resume identity before GPU work.
launch=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -m analysis.mechanism.launcher \
    --config "$CONFIG" --gap-set "$GAP_SET" --verified "$VERIFIED" --adapter "$adapter" \
    --snapshot "$MODEL_PATH" --output "$OUT" --include-base "$INCLUDE_BASE" \
    --max-teacher-calls "$MAX_TEACHER_CALLS" --max-teacher-calls-run "$MAX_TEACHER_CALLS_RUN")
# Only one active orchestration owns this workspace; a competing launch fails before serving.
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo 'Mechanism workspace already has an active launcher' >&2; exit 2; }
launcher_pid=''
work_pid=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$work_pid" ]]; then kill -TERM -- "-$work_pid" 2>/dev/null || true; wait "$work_pid" 2>/dev/null || true; fi
    if [[ -n "$launcher_pid" ]]; then kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
run_work() {
    setsid "$@" &
    work_pid=$!
    wait "$work_pid"
    work_pid=''
}
run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$adapter_path" --verified "$VERIFIED" --model-name "$name" \
    --hf-only --output "$launch/hf"
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" --lora-adapter "$adapter" --lora-conversion "$LORA_CONVERSION" \
    --port 0 --ready-timeout "$READY_TIMEOUT" --output "$launch/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$launch/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
endpoint=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$launch/server/ready.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["endpoint"])
PY
)
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$adapter_path" --verified "$VERIFIED" --model-name "$name" \
    --hf-reference "$launch/hf/reference.json" --endpoint "$endpoint" \
    --server-provenance "$launch/server/provenance.json" --output "$launch/check"
common=(--config "$CONFIG" --gap-set "$GAP_SET" --max-teacher-calls "$MAX_TEACHER_CALLS"
        --max-teacher-calls-run "$MAX_TEACHER_CALLS_RUN" --budget-dir "$OUT" --workers "$WORKERS")
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.fce "${common[@]}" --output "$OUT/fce"
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.interventions "${common[@]}" \
    --fce "$OUT/fce/fce.json" --verified "$VERIFIED" --output "$OUT/interventions" \
    --endpoint "$endpoint" --model-name "$name" --server-provenance "$launch/server/provenance.json" \
    --lora-check "$launch/check/report.json"
if [[ "$INCLUDE_BASE" == 1 ]]; then
    base_name=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$CONFIG" <<'PY'
import sys
from sed.config import load_config
print(load_config(sys.argv[1])["models"]["student"]["requested_repo"])
PY
)
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.interventions "${common[@]}" \
        --fce "$OUT/fce/fce.json" --verified "$VERIFIED" --output "$OUT/base-interventions" --subject base \
        --endpoint "$endpoint" --model-name "$base_name" --server-provenance "$launch/server/provenance.json"
fi
readout_args=()
if [[ "$INCLUDE_BASE" == 1 ]]; then readout_args+=(--compare-interventions "$OUT/base-interventions/interventions.json"); fi
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.readout \
    --config "$CONFIG" --gap-set "$GAP_SET" --fce "$OUT/fce/fce.json" \
    --interventions "$OUT/interventions/interventions.json" --output "$launch/readout" "${readout_args[@]}"
