#!/bin/bash
#SBATCH --job-name=p1-dev140
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60

# New wrapper of tree 46b8527's hpg_eval_student.slurm. The historical
# launcher defaults to configs/base.yaml (12B); E4B requires explicit --config.
set -euo pipefail
: "${PROJECT_ROOT:?set the P1 checkout}"
: "${EVAL_ROOT:?set the read-only tree-46b8527 checkout}"
: "${MERGED_MODEL_PATH:?set the exported standalone directory}"
: "${CONDITIONS:?set the complete frozen condition directory}"
source "${HPG_CONFIG:-${EVAL_ROOT}/configs/hpg.env}"
: "${P1_CONFIG:=${PROJECT_ROOT}/configs/base_e4b.yaml}"
: "${BATCH_INVARIANT:=1}"
[[ "$BATCH_INVARIANT" == 1 && "$WORKERS" == 16 && "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]]
[[ -f "$TASKS" && -f "$VERIFIED/manifest.json" && -d "$MODEL_PATH" ]]
[[ "$(git -C "$EVAL_ROOT" rev-parse HEAD)" == "$(git -C "$EVAL_ROOT" rev-parse 46b8527)" ]]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$ENV_PYTHON" -c \
  'import sys; from surgery.export import verify_export; verify_export(sys.argv[1])' "$MERGED_MODEL_PATH"
cd "$EVAL_ROOT"
export PYTHONPATH="$EVAL_ROOT/src:$EVAL_ROOT"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$ENV_PYTHON" -c \
  'import sys; from surgery.export import record_m4_launch; record_m4_launch(*sys.argv[1:])' \
  "$MERGED_MODEL_PATH" "$TASKS" "$VERIFIED" "$OUT" "$CONDITIONS"
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
"$ENV_PYTHON" harness/serve_student.py --config "$P1_CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" --merged-model-path "$MERGED_MODEL_PATH" \
    --port "$port" --ready-timeout "$READY_TIMEOUT" --output "$OUT/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
CUDA_VISIBLE_DEVICES='' setsid "$ENV_PYTHON" eval/run_suite.py --config "$P1_CONFIG" \
    --policy student --decoding greedy --endpoint "http://127.0.0.1:$port/v1" \
    --enable-thinking off --max-tokens 1024 --tasks "$TASKS" --workers 16 --output "$OUT/eval" \
    --server-provenance "$OUT/server/provenance.json" &
eval_pid=$!
wait "$eval_pid"
eval_pid=''
