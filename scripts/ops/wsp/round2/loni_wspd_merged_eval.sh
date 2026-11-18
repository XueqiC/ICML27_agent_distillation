#!/bin/bash
# loni_wspd_merged_eval.sh — evaluates ONE exported (merged bf16) surgery model on one LONI A100: the 20 K20 training tasks
# (t2) once, then dev140 DEV140_RUNS times. Port of hpg_wspd_merged_eval.slurm (prereg section 17 route), free port per server.
# Env: MERGED_MODEL_PATH, NAME, DEV140_RUNS. Outputs /work/xueqic/hq/ICML27_wsp_d/eval/$NAME.merged.{t2.run1,dev140.runN}.
set -euo pipefail
: "${MERGED_MODEL_PATH:?}" "${NAME:?}"
DEV140_RUNS=${DEV140_RUNS:-1}
W=/work/xueqic/hq; OUTROOT=$W/ICML27_wsp_d/eval; SURGERY=$W/ICML27_runs/repair_eval_4b4980f; EVAL_ROOT=$W/ICML27_runs/wsp_27a83f4
[[ -f "$MERGED_MODEL_PATH/surgery_manifest.json" ]] || { echo "export incomplete: $MERGED_MODEL_PATH" >&2; exit 2; }
export PROJECT_ROOT=$EVAL_ROOT
source $W/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env
MODEL_PATH=/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
CONFIG=$EVAL_ROOT/configs/base_e4b.yaml; READY_TIMEOUT=${READY_TIMEOUT:-900}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=$W/tools/vllm-cache TRITON_CACHE_DIR=$W/tools/triton-cache
export SED_HASH_CACHE_DIR=$W/ICML27_v0_runs/sed-hash-receipts CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "== wspd merged eval $NAME start $(date -Is) host=$(hostname) tree $(cat "$EVAL_ROOT/GIT_COMMIT")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$SURGERY/src:$SURGERY" PYTHONDONTWRITEBYTECODE=1 "$W/envs/p1-venv/bin/python" -c \
  'import sys; from surgery.export import verify_export; verify_export(sys.argv[1])' "$MERGED_MODEL_PATH"
cd "$EVAL_ROOT"; export PYTHONPATH="$EVAL_ROOT/src:$EVAL_ROOT"; mkdir -p "$OUTROOT"
launcher_pid=''; eval_pid=''
cleanup() { trap - EXIT TERM INT
  if [[ -n "$eval_pid" ]]; then kill -TERM -- "-$eval_pid" 2>/dev/null || true; wait "$eval_pid" 2>/dev/null || true; fi
  if [[ -n "$launcher_pid" ]]; then kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
one() { local label=$1 tasks=$2 OUT="$OUTROOT/$NAME.merged.$1"
  if [[ -f "$OUT/eval/summary.json" ]]; then echo "== $label already done"; return 0; fi
  if [[ -e "$OUT" ]]; then mkdir -p "$OUTROOT/_incomplete"; mv "$OUT" "$OUTROOT/_incomplete/$NAME.merged.$label.$(date +%s)"; fi
  mkdir "$OUT"
  "$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" --model-path "$MODEL_PATH" \
      --merged-model-path "$MERGED_MODEL_PATH" --port 0 --ready-timeout "$READY_TIMEOUT" --output "$OUT/server" &
  launcher_pid=$!
  local deadline=$((SECONDS + READY_TIMEOUT + 30))
  until [[ -f "$OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
  done
  local endpoint; endpoint=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["endpoint"])' "$OUT/server/ready.json")
  CUDA_VISIBLE_DEVICES='' setsid "$ENV_PYTHON" eval/run_suite.py --config "$CONFIG" --policy student --decoding greedy --endpoint "$endpoint" \
      --enable-thinking off --max-tokens 1024 --tasks "$tasks" --workers 16 --output "$OUT/eval" --server-provenance "$OUT/server/provenance.json" &
  eval_pid=$!; local rc=0; wait "$eval_pid" || rc=$?; eval_pid=''
  kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; launcher_pid=''
  echo "== $label exit $rc $(date -Is)"; [[ -f "$OUT/eval/summary.json" ]] || exit 1
  # training-task rollouts: progress score now (it needs the episode files), then pack the episode logs of this run
  if [[ "$label" == t2.* ]]; then ( SED_HASH_CACHE=0 CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" tools/wsp_progress.py --eval "$OUT/eval" \
      --verified $W/ICML27_data/verified/seeds_full_train_r2/subsets/k20 --out "$OUT/eval/progress.json" > /dev/null ) || echo "PROGRESS_FAIL $label"; fi
  ( cd "$OUT/eval" && n=$(find episodes | wc -l) && tar -cf episodes.tar episodes && tar -df episodes.tar > /dev/null 2>&1 && [ "$(tar -tf episodes.tar | wc -l)" = "$n" ] \
      && rm -rf "$OUT/eval/episodes" && echo "== $label packed $n entries" ) || echo "PACK_FAIL $label"
  sleep 5; }
one t2.run1 "$W/ICML27_wsp/queue/lists/t2_k20_train.json"
for i in $(seq 1 "$DEV140_RUNS"); do one "dev140.run$i" "$EVAL_ROOT/data/manifest/dev140.json"; done
echo "== wspd merged eval $NAME done $(date -Is)"
