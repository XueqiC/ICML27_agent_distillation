#!/bin/bash
#SBATCH --job-name=gemma-base-dev40
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60

set -euo pipefail
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
case "$BATCH_INVARIANT" in 0|1) ;; *) echo 'BATCH_INVARIANT must be 0 or 1' >&2; exit 2 ;; esac
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
# Submit from the checkout, or explicitly set PROJECT_ROOT / HPG_CONFIG.
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
case "$ENABLE_THINKING" in off|on) ;; *) echo 'ENABLE_THINKING must be off or on' >&2; exit 2 ;; esac
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]] || {
    echo 'Stage 2b fixes ENABLE_THINKING=off and MAX_TOKENS=1024' >&2; exit 2;
}
[[ -x "$VLLM_PYTHON" && -x "$ENV_PYTHON" && -d "$MODEL_PATH" && -f "$TASKS" ]]
# Refuse reuse to keep the two thinking conditions and failed attempts distinct.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
launcher_pid=''
eval_pid=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$eval_pid" ]]; then kill -TERM -- "-$eval_pid" 2>/dev/null || true; wait "$eval_pid" 2>/dev/null || true; fi
    if [[ -n "$launcher_pid" ]]; then kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
port=$((20000 + SLURM_JOB_ID % 20000))
# GPU visibility inherited only by the vLLM child; environment workers stay on CPU.
merged_args=()
if [[ -n "${MERGED_MODEL_PATH:-}" ]]; then
    merged_args=(--merged-model-path "$MERGED_MODEL_PATH")
fi
"$ENV_PYTHON" harness/serve_student.py --python "$VLLM_PYTHON" --model-path "$MODEL_PATH" "${merged_args[@]}" \
    --port "$port" --ready-timeout "$READY_TIMEOUT" --output "$OUT/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
CUDA_VISIBLE_DEVICES='' setsid "$ENV_PYTHON" eval/run_suite.py \
    --policy student --endpoint "http://127.0.0.1:$port/v1" \
    --enable-thinking "$ENABLE_THINKING" --max-tokens "$MAX_TOKENS" \
    --tasks "$TASKS" --workers "$WORKERS" --output "$OUT/eval" \
    --server-provenance "$OUT/server/provenance.json" &
eval_pid=$!
wait "$eval_pid"
eval_pid=''
