#!/bin/bash
# rsi_wsp_d_merged_eval.sh GPU EXPORT_DIR TASKS(dev140|t2) OUT — evaluates ONE exported (merged bf16) surgery model on rsi.
# Why merged: the D edits change teacher-forced log-probabilities by less than the vLLM-vs-HF kernel noise, so the
# adapter path's serving check ("visible adapter effect") cannot certify them; the merged checkpoint is exactly the
# weight set the teacher session deployed. Port of rsi_p1_dev140.sh without the P1 M4 bookkeeping (record_m4_launch
# needs P1's full condition set). Same server launcher and suite runner as every other evaluation.
set -euo pipefail
gpu=${1:?GPU}; MERGED_MODEL_PATH=${2:?export dir}; which=${3:?dev140|t2}; OUT=${4:?out dir}
SURGERY=/home/xueqi/hq/projects/ICML27_run_repair_eval_d7b221c
case "$which" in
  dev140) EVAL_ROOT=/home/xueqi/hq/projects/ICML27_runs/wsp_4ba5d0e; TASKS=$EVAL_ROOT/data/manifest/dev140.json;;
  t2)     EVAL_ROOT=/home/xueqi/hq/projects/ICML27_runs/wsp_1f649a2; TASKS=/home/xueqi/hq/projects/ICML27_wsp/queue/lists/t2_k20_train.json;;
  *) echo "TASKS must be dev140 or t2" >&2; exit 2;;
esac
[[ -d "$MERGED_MODEL_PATH" && -f "$TASKS" && ! -e "$OUT" ]]
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [[ "$used" -lt 2000 ]] || { echo "GPU_BUSY $gpu ${used}MiB"; exit 3; }
export PROJECT_ROOT=$EVAL_ROOT
source /home/xueqi/hq/projects/ICML27_eval_46b8527/configs_rsi/rsi_eval.env
CONFIG=$EVAL_ROOT/configs/base_e4b.yaml
[[ "$WORKERS" == 16 && "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/home/xueqi/hq/tools/vllm-cache TRITON_CACHE_DIR=/home/xueqi/hq/tools/triton-cache VLLM_NO_USAGE_STATS=1
export SED_HASH_CACHE_DIR=/home/xueqi/hq/projects/ICML27_wsp/sed-hash-receipts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
echo "== wsp-d merged eval $which $(basename "$MERGED_MODEL_PATH") start $(date -Is) host=$(hostname) GPU $gpu tree $(cat "$EVAL_ROOT/GIT_COMMIT") out=$OUT"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$SURGERY/src:$SURGERY" "$TRAIN_PYTHON" -c \
  'import sys; from surgery.export import verify_export; verify_export(sys.argv[1])' "$MERGED_MODEL_PATH"
cd "$EVAL_ROOT"
export PYTHONPATH="$EVAL_ROOT/src:$EVAL_ROOT"
mkdir -p "$(dirname "$OUT")"; mkdir "$OUT"
launcher_pid=''; eval_pid=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$eval_pid" ]]; then kill -TERM -- "-$eval_pid" 2>/dev/null || true; wait "$eval_pid" 2>/dev/null || true; fi
    if [[ -n "$launcher_pid" ]]; then kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
port=$((23270 + gpu))
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" --merged-model-path "$MERGED_MODEL_PATH" \
    --port "$port" --ready-timeout "$READY_TIMEOUT" --output "$OUT/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
CUDA_VISIBLE_DEVICES='' setsid "$ENV_PYTHON" eval/run_suite.py --config "$CONFIG" \
    --policy student --decoding greedy --endpoint "http://127.0.0.1:$port/v1" \
    --enable-thinking off --max-tokens 1024 --tasks "$TASKS" --workers 16 --output "$OUT/eval" \
    --server-provenance "$OUT/server/provenance.json" &
eval_pid=$!
wait "$eval_pid"
eval_pid=''
echo "== wsp-d merged eval $which $(basename "$MERGED_MODEL_PATH") exit 0 $(date -Is)"
