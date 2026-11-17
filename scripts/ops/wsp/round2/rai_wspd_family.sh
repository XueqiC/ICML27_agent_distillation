#!/bin/bash
# rai_wspd_family.sh FAMILY(c4|d3|d4|d5) GPU [extra diagnose args] — WSP round 2 on rai (hpg queue is blocked, rsi cards taken):
# runs the surgery steps of one family on ONE rai GPU with the clean checkout of tree 4b4980f (or TREE=...):
#   c4      : additive (fixed-rule additive conditions, K20 selection)
#   d3 / d4 : diagnose --variant (teacher session) -> controls (stopped once B2_4.json is frozen)
# Exports are NOT made here (16 GB each; no bulk writes on rai): the frozen condition files are exported elsewhere.
# The teacher client reads its key from ~/hq/secrets itself; it is never passed on a command line or printed.
set -uo pipefail
fam=${1:?c4|d3|d4|d5}; gpu=${2:?GPU}; shift 2
T=${TREE:-/home/xueqi/hq/projects/ICML27_run_repair_eval_4b4980f}; D=/home/xueqi/hq/projects/ICML27_wsp_d; P=$D/inputs/pairs_B.json
cd "$T" || exit 2
[ -z "$(git status --short)" ] || { echo "tree not clean" >&2; exit 2; }
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || { echo "GPU $gpu is in use ($used MiB): not starting" >&2; exit 3; }
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=$gpu
PY=/home/xueqi/hq/projects/ICML27_agent_distillation/envs/p0-venv/bin/python
COMMON=(--base /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
  --adapter /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
  --round-dir /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1
  --verified /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20
  --alfworld-data /home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1)
echo "== wspd $fam start $(date -Is) host=$(hostname) GPU $gpu $(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader) tree $(git rev-parse --short=7 HEAD) pid $$"
if [ "$fam" = c4 ]; then
  "$PY" tools/p1_surgery.py additive --pairs "$P" --output "$D/conditions_c4" "${COMMON[@]}" "$@"; rc=$?
  echo "== wspd c4 additive exit $rc $(date -Is)"; exit $rc
fi
run=$D/runs/${fam}_s1
if [ ! -f "$run/submission.json" ]; then
  [ ! -e "$run" ] || { echo "run dir exists without a submission: $run" >&2; exit 2; }
  "$PY" tools/p1_surgery.py diagnose --variant "$fam" --pairs "$P" --dossier "$D/inputs/dossier_k20_raw_b4.json" \
    --rate-limit-dir "$D/joint-rate" --allow-teacher-api --output "$run" "${COMMON[@]}" "$@"; rc=$?
  echo "== wspd $fam diagnose exit $rc $(date -Is)"
  [ $rc -eq 0 ] && [ -f "$run/submission.json" ] || exit 1
fi
C=$D/conditions_$fam
"$PY" tools/p1_surgery.py controls --pairs "$P" --submission "$run/submission.json" --output "$C" "${COMMON[@]}" &
child=$!
while kill -0 "$child" 2>/dev/null; do
  if [ -e "$C/B2_4.json" ]; then sleep 10; echo "== $C/B2_4.json frozen: stopping controls (B3 / B5 are P1 baselines, not part of D)"; kill -TERM "$child" 2>/dev/null; break; fi
  sleep 10
done
wait "$child" 2>/dev/null
[ -e "$C/B2_4.json" ] && echo "== wspd $fam controls frozen $(date -Is): $(ls "$C" | grep -c json) condition files" || { echo "== wspd $fam controls FAILED $(date -Is)"; exit 1; }
