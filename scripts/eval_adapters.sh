#!/bin/bash
# Shared body, sourced by the rai/HPG entrypoints; adapters is a Bash array.
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
case "$DECODING" in greedy|sag|wkm) ;; *) echo 'DECODING must be greedy, sag or wkm' >&2; exit 2 ;; esac
if [[ "$DECODING" == sag && "$BATCH_INVARIANT" != 1 ]]; then
    echo 'Stage 3b SAG requires BATCH_INVARIANT=1' >&2; exit 2
fi
if [[ "$DECODING" == wkm ]]; then
    : "${WKM_AGENT_NAME:?Set the agent adapter name}" "${WKM_KNOWLEDGE_NAME:?Set the world adapter name}"
    : "${WKM_KNOWLEDGE_BASE:?Set the frozen state knowledge base}"
    [[ "$BATCH_INVARIANT" == 1 && "$INCLUDE_BASE" == 0 && ${#adapters[@]} == 2 &&
       "$WKM_AGENT_NAME" != "$WKM_KNOWLEDGE_NAME" && -f "$WKM_KNOWLEDGE_BASE" ]] || exit 2
fi
MERGED_MODEL_PATH=${MERGED_MODEL_PATH:-}
FULL_CHECKPOINT_NAME=${FULL_CHECKPOINT_NAME:-full-checkpoint}
if [[ -n "$MERGED_MODEL_PATH" ]]; then
    [[ ${#adapters[@]} == 0 && "$INCLUDE_BASE" == 1 && -d "$MERGED_MODEL_PATH" ]] || {
        echo 'Full checkpoint mode requires no adapters and INCLUDE_BASE=1' >&2; exit 2;
    }
    [[ "$FULL_CHECKPOINT_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
fi
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
base_name=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$CONFIG" "$MODEL_PATH" "$TASKS" "$MERGED_MODEL_PATH" "${adapters[@]}" <<'PY'
import sys
from sed.config import load_config
from sed.manifest import load_frozen
from eval.run_suite import frozen_tasks
from harness.lora_adapters import parse_adapters, inspect_adapter
config = load_config(sys.argv[1])
manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
listing, _ = frozen_tasks(sys.argv[3], manifest, freeze)
if listing.get("name") not in {"dev40", "dev140", "r03_dev300"}:
    raise ValueError("Launcher requires frozen dev40, dev140 or r03_dev300")
student = config["models"]["student"]
if sys.argv[4]:
    from baselines.common.fsdp_train import validate_checkpoint
    validate_checkpoint(sys.argv[4], student, base_snapshot=sys.argv[2])
for name, path in parse_adapters(sys.argv[5:], student["requested_repo"]):
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
    local status=0
    setsid "$@" &
    work_pid=$!
    wait "$work_pid" || status=$?
    work_pid=''
    return "$status"
}
failed_checks=0
mark_check_failed() {
    local name=$1 stage=$2 status=$3
    mkdir -p "$OUT/$name/check"
    printf '%s failed (exit %s)\n' "$stage" "$status" > "$OUT/$name/check/FAILED"
    echo "Equivalence check failed for $name; see $OUT/$name/check/FAILED" >&2
    failed_checks=$((failed_checks + 1))
}
# HF references run sequentially BEFORE vLLM reserves GPU memory. No merged copy.
if [[ "${FULL_EQUIVALENCE:-0}" == 1 ]]; then
    : "${VERIFIED:?Full checkpoint gate requires VERIFIED}"
    [[ -n "$MERGED_MODEL_PATH" ]] || exit 2
    run_work "$TRAIN_PYTHON" tools/check_full_serving.py --config "$CONFIG" --snapshot "$MODEL_PATH" \
        --checkpoint "$MERGED_MODEL_PATH" --verified "$VERIFIED" --hf-only --output "$OUT/full-hf"
fi
ready_adapters=()
for pair in "${adapters[@]}"; do
    name=${pair%%=*}
    path=${pair#*=}
    if run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
        --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
        --hf-only --output "$OUT/$name/hf"; then
        ready_adapters+=("$pair")
    else
        mark_check_failed "$name" hf-reference "$?"
    fi
done
lora_args=()
if [[ -n "$MERGED_MODEL_PATH" ]]; then lora_args+=(--merged-model-path "$MERGED_MODEL_PATH"); fi
for pair in "${adapters[@]}"; do lora_args+=(--lora-adapter "$pair"); done
# --port 0 binds loopback port 0 to obtain a free port. The selected port is
# persisted in server provenance and ready.json; no guessed/hardcoded endpoint.
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" "${lora_args[@]}" --lora-conversion "$LORA_CONVERSION" \
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
eval_model() {
    local request_model=$1
    local output_name=$1
    if [[ -n "$MERGED_MODEL_PATH" ]]; then output_name=$FULL_CHECKPOINT_NAME; fi
    shift
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" eval/run_suite.py --config "$CONFIG" \
        --policy student --decoding "$DECODING" --endpoint "$endpoint" --model-name "$request_model" \
        --enable-thinking off --max-tokens 1024 --tasks "$TASKS" --workers "$WORKERS" \
        --output "$OUT/$output_name/eval" --server-provenance "$OUT/server/provenance.json" "$@"
}
if [[ "$INCLUDE_BASE" == 1 ]]; then
    if [[ "${FULL_EQUIVALENCE:-0}" == 1 ]]; then
        CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_full_serving.py --config "$CONFIG" \
            --snapshot "$MODEL_PATH" --checkpoint "$MERGED_MODEL_PATH" --verified "$VERIFIED" \
            --hf-reference "$OUT/full-hf/reference.json" --endpoint "$endpoint" \
            --server-provenance "$OUT/server/provenance.json" --output "$OUT/full-check"
        eval_model "$base_name" --full-check "$OUT/full-check/report.json"
    else
        eval_model "$base_name"
    fi
fi
for pair in "${ready_adapters[@]}"; do
    name=${pair%%=*}
    path=${pair#*=}
    if CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
        --snapshot "$MODEL_PATH" --adapter "$path" --verified "$VERIFIED" --model-name "$name" \
        --hf-reference "$OUT/$name/hf/reference.json" --endpoint "$endpoint" \
        --server-provenance "$OUT/server/provenance.json" --output "$OUT/$name/check"; then
        if [[ "$DECODING" != wkm ]]; then
            eval_model "$name" --lora-check "$OUT/$name/check/report.json"
        fi
    else
        mark_check_failed "$name" serving-comparison "$?"
    fi
done
if (( failed_checks > 0 )); then exit 1; fi
if [[ "$DECODING" == wkm ]]; then
    [[ -f "$OUT/$WKM_AGENT_NAME/check/report.json" && -f "$OUT/$WKM_KNOWLEDGE_NAME/check/report.json" ]] || exit 2
    eval_model "$WKM_AGENT_NAME" --lora-check "$OUT/$WKM_AGENT_NAME/check/report.json" \
        --knowledge-model "$WKM_KNOWLEDGE_NAME" --knowledge-base "$WKM_KNOWLEDGE_BASE" \
        --knowledge-lora-check "$OUT/$WKM_KNOWLEDGE_NAME/check/report.json"
fi
