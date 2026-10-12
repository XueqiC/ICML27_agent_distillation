#!/bin/bash
# Shared body, sourced by hpg_rollout_adapter.slurm; ADAPTER is one NAME=PATH.
# Serving/check lifecycle mirrors eval_adapters.sh without changing that launcher.
set -euo pipefail
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
case "$BATCH_INVARIANT" in 0|1) ;; *) echo 'BATCH_INVARIANT must be 0 or 1' >&2; exit 2 ;; esac
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
ENABLE_THINKING=${ENABLE_THINKING:-off}
MAX_TOKENS=${MAX_TOKENS:-1024}
READY_TIMEOUT=${READY_TIMEOUT:-900}
LORA_CONVERSION=${LORA_CONVERSION:-none}
LORA_DTYPE=${LORA_DTYPE:-bfloat16}
case "$LORA_DTYPE" in bfloat16|float16) ;; *) echo 'LORA_DTYPE must be bfloat16 or float16' >&2; exit 2 ;; esac
CONFIG=${CONFIG:-configs/base.yaml}
DECODING=${DECODING:-greedy}
[[ "$DECODING" == greedy && "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]] || {
    echo 'Rollouts fix DECODING=greedy, ENABLE_THINKING=off and MAX_TOKENS=1024' >&2; exit 2;
}
: "${ADAPTER:?Set ADAPTER to one NAME=PATH entry}"
: "${VERIFIED:?Set VERIFIED to the training-eligible verified seed collection}"
: "${OUT:?Set OUT to a new rollout run directory}"
[[ -z "${ADAPTERS:-}" && -z "${MERGED_MODEL_PATH:-}" && "${INCLUDE_BASE:-0}" == 0 ]] || {
    echo 'Rollouts require one ADAPTER; ADAPTERS, merged checkpoints and INCLUDE_BASE are unsupported' >&2; exit 2;
}
ROLLOUT_SUBSET=${ROLLOUT_SUBSET:-R2}
case "$ROLLOUT_SUBSET" in all|R1_P2|R1_P3|R2) ;; *) echo 'ROLLOUT_SUBSET must be all, R1_P2, R1_P3 or R2' >&2; exit 2 ;; esac
LABEL_SAMPLES=${LABEL_SAMPLES:-0}
[[ "$LABEL_SAMPLES" =~ ^[0-9]+$ ]] || { echo 'LABEL_SAMPLES must be a nonnegative integer' >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo 'WORKERS and READY_TIMEOUT must be positive integers' >&2; exit 2;
}
TASKS=${TASKS:-data/manifest/r03_pool200.json}
[[ -x "$ENV_PYTHON" && -x "$VLLM_PYTHON" && -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -d "$VERIFIED" && -f "$TASKS" ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Output already exists' >&2; exit 2; }
# Validate the frozen training pool and checkpoint identity before any GPU work.
CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$CONFIG" "$MODEL_PATH" "$TASKS" "$ROLLOUT_SUBSET" "$ADAPTER" <<'PY'
import sys
from sed.config import load_config
from sed.manifest import load_frozen
from harness.lora_adapters import parse_adapters, inspect_adapter
from tools.rollout_pool import pool_tasks
config = load_config(sys.argv[1])
manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
pool_tasks(sys.argv[3], manifest, freeze, sys.argv[4])
student = config["models"]["student"]
name, path = parse_adapters([sys.argv[5]], student["requested_repo"])[0]
adapter = inspect_adapter(name, path, student, sys.argv[2])
if adapter["checkpoint_metadata"]["supervised_tokens"] != 104070:
    raise ValueError("R03 rollouts require the p10 adapter at 104070 supervised tokens")
PY
name=${ADAPTER%%=*}
path=${ADAPTER#*=}
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
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
    local status=0
    setsid "$@" &
    work_pid=$!
    wait "$work_pid" || status=$?
    work_pid=''
    return "$status"
}
mark_check_failed() {
    local stage=$1 status=$2
    mkdir -p "$OUT/check"
    printf '%s failed (exit %s)\n' "$stage" "$status" > "$OUT/check/FAILED"
    echo "Equivalence check failed for $name; see $OUT/check/FAILED" >&2
}
# HF must finish before vLLM reserves GPU memory. No merged checkpoint copy.
if run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
    --hf-only --output "$OUT/hf"; then
    :
else
    mark_check_failed hf-reference "$?"
    exit 1
fi
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" --lora-adapter "$ADAPTER" --lora-conversion "$LORA_CONVERSION" \
    --lora-dtype "$LORA_DTYPE" \
    --port 0 --ready-timeout "$READY_TIMEOUT" --output "$OUT/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
endpoint=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$OUT/server/ready.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["endpoint"])
PY
)
if CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
    --hf-reference "$OUT/hf/reference.json" --endpoint "$endpoint" \
    --server-provenance "$OUT/server/provenance.json" --output "$OUT/check"; then
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/rollout_pool.py run --config "$CONFIG" \
        --pool "$TASKS" --subset "$ROLLOUT_SUBSET" --endpoint "$endpoint" --model-name "$name" \
        --server-provenance "$OUT/server/provenance.json" --lora-check "$OUT/check/report.json" \
        --label-samples "$LABEL_SAMPLES" --workers "$WORKERS" --output "$OUT/rollouts"
else
    mark_check_failed serving-comparison "$?"
    exit 1
fi
