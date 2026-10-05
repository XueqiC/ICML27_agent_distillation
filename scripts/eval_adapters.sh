#!/bin/bash
# Shared body, sourced by the rai/HPG entrypoints; adapters is a Bash array.
set -euo pipefail
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
ENABLE_THINKING=${ENABLE_THINKING:-off}
MAX_TOKENS=${MAX_TOKENS:-1024}
READY_TIMEOUT=${READY_TIMEOUT:-900}
LORA_CONVERSION=${LORA_CONVERSION:-none}
CONFIG=${CONFIG:-configs/base.yaml}
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]] || {
    echo 'Evaluation fixes ENABLE_THINKING=off and MAX_TOKENS=1024' >&2; exit 2;
}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]
[[ "$INCLUDE_BASE" == 0 || "$INCLUDE_BASE" == 1 ]]
[[ -x "$ENV_PYTHON" && -x "$VLLM_PYTHON" && -d "$MODEL_PATH" && -f "$TASKS" ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Output already exists' >&2; exit 2; }
if [[ ${#adapters[@]} -gt 0 ]]; then
    : "${VERIFIED:?Set VERIFIED to the training-eligible verified seed collection}"
    [[ -x "$TRAIN_PYTHON" && -d "$VERIFIED" ]]
fi
# Validate names, checkpoint identities, and the frozen dev manifest before GPU work.
base_name=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$CONFIG" "$MODEL_PATH" "$TASKS" "${adapters[@]}" <<'PY'
import sys
from sed.config import load_config
from sed.manifest import load_frozen
from eval.run_suite import frozen_tasks
from harness.lora_adapters import parse_adapters, inspect_adapter
config = load_config(sys.argv[1])
manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
listing, _ = frozen_tasks(sys.argv[3], manifest, freeze)
if listing.get("name") not in {"dev40", "dev140"}:
    raise ValueError("Launcher requires frozen dev40 or dev140")
student = config["models"]["student"]
for name, path in parse_adapters(sys.argv[4:], student["requested_repo"]):
    inspect_adapter(name, path, student, sys.argv[2])
print(student["requested_repo"])
PY
)
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
    setsid "$@" &
    work_pid=$!
    wait "$work_pid"
    work_pid=''
}
# HF references run sequentially BEFORE vLLM reserves GPU memory. No merged copy.
for pair in "${adapters[@]}"; do
    name=${pair%%=*}
    path=${pair#*=}
    run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
        --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
        --hf-only --output "$OUT/$name/hf"
done
lora_args=()
for pair in "${adapters[@]}"; do lora_args+=(--lora-adapter "$pair"); done
# --port 0 binds loopback port 0 to obtain a free port. The selected port is
# persisted in server provenance and ready.json; no guessed/hardcoded endpoint.
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" "${lora_args[@]}" --lora-conversion "$LORA_CONVERSION" \
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
eval_model() {
    local request_model=$1
    shift
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" eval/run_suite.py --config "$CONFIG" \
        --policy student --endpoint "$endpoint" --model-name "$request_model" \
        --enable-thinking off --max-tokens 1024 --tasks "$TASKS" --workers "$WORKERS" \
        --output "$OUT/$request_model/eval" --server-provenance "$OUT/server/provenance.json" "$@"
}
if [[ "$INCLUDE_BASE" == 1 ]]; then eval_model "$base_name"; fi
for pair in "${adapters[@]}"; do
    name=${pair%%=*}
    path=${pair#*=}
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
        --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
        --hf-reference "$OUT/$name/hf/reference.json" --endpoint "$endpoint" \
        --server-provenance "$OUT/server/provenance.json" --output "$OUT/$name/check"
    eval_model "$name" --lora-check "$OUT/$name/check/report.json"
done
