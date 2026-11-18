#!/bin/bash
# loni_wspd_common.sh — sourced by the D-lane derive scripts on LONI (WSP round 2, prereg sections 18-20): surgery tool of
# tree 4b4980f (or TREE=...) on ONE A100, env envs/p1-venv. Port of scripts/ops/wsp/round2/hpg_wspd_steps.slurm.
# The teacher key is read from /work/xueqic/hq/secrets/openai.env into the environment and never printed.
set -u
W=/work/xueqic/hq; D=$W/ICML27_wsp_d; T=${TREE:-$W/ICML27_runs/repair_eval_4b4980f}; PY=$W/envs/p1-venv/bin/python; P=$D/inputs/pairs_B.json
cd "$T" || exit 2
[ "$(git rev-parse --short=7 HEAD)" = "${EXPECT:-4b4980f}" ] || { echo "tree mismatch" >&2; exit 2; }
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
  PYTHONDONTWRITEBYTECODE=1 HF_HOME=/work/xueqic/hf-cache SED_HASH_CACHE_DIR=$W/ICML27_v0_runs/sed-hash-receipts
COMMON=(--base /work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
  --adapter $W/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
  --round-dir $W/ICML27_v0_runs/v0_k20_r1
  --verified $W/ICML27_data/verified/seeds_full_train_r2/subsets/k20
  --alfworld-data $W/tc-alignment/envs/alfworld/data/json_2.1.1
  --relocate /home/xueqi/hq/projects/ICML27_agent_distillation/data=$W/ICML27_data
  --relocate /home/xueqi/.cache/huggingface=/work/xueqic/hf-cache)
step() { echo "== $(date -Is) $*"; }
run() { step "p1_surgery $1 start"; "$PY" tools/p1_surgery.py "$@" "${COMMON[@]}"; local rc=$?; step "p1_surgery $1 exit $rc"; return $rc; }
teacher_key() {
  OPENAI_API_KEY=$(sed -nE 's/^(export )?OPENAI_API_KEY=//p' $W/secrets/openai.env | head -1 | sed -E "s/^[\"']//; s/[\"']$//"); export OPENAI_API_KEY
  [ -n "$OPENAI_API_KEY" ] || { echo "no teacher key" >&2; return 2; }
  "$PY" - <<'PY'
import os, sys, urllib.request, urllib.error
req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
try:
    with urllib.request.urlopen(req, timeout=40) as r: print("teacher API preflight: HTTP", r.status)
except urllib.error.HTTPError as e: print("teacher API preflight: HTTP", e.code); sys.exit(1)
except Exception as e: print("teacher API preflight:", type(e).__name__); sys.exit(1)
PY
}
exports() { # exports FAMILY NAME...
  local fam=$1 n; shift
  for n in "$@"; do
    [ -f "$D/export_$fam/$n/surgery_manifest.json" ] && continue
    [ -f "$D/conditions_$fam/$n.json" ] || { echo "missing condition $fam/$n" >&2; return 1; }
    if [ -e "$D/export_$fam/$n" ]; then mkdir -p "$D/_failed"; mv "$D/export_$fam/$n" "$D/_failed/export_${fam}_$n.$(date +%s)"; fi
    run export --pairs "$P" --condition "$D/conditions_$fam/$n.json" --output "$D/export_$fam/$n" || return 1
  done; }
session() { # session FAMILY VARIANT RUNNAME [extra diagnose args]: teacher session, then the frozen controls (stopped after B2_4)
  local fam=$1 variant=$2 name=$3; shift 3; local run=$D/runs/$name C=$D/conditions_$fam child
  if [ ! -f "$run/submission.json" ]; then
    if [ -e "$run" ]; then mkdir -p "$D/_failed"; mv "$run" "$D/_failed/$name.$(date +%s)"; fi
    teacher_key || return 2
    run diagnose --variant "$variant" --pairs "$P" --dossier "$D/inputs/dossier_k20_raw_b4.json" --rate-limit-dir "$D/joint-rate" \
      --allow-teacher-api --output "$run" "$@" || return 1
    [ -f "$run/submission.json" ] || { echo "session ended without a submission" >&2; return 1; }
  fi
  if [ ! -f "$C/B2_4.json" ]; then
    if [ -e "$C" ]; then mkdir -p "$D/_failed"; mv "$C" "$D/_failed/conditions_$fam.$(date +%s)"; fi
    step "controls start"
    "$PY" tools/p1_surgery.py controls --pairs "$P" --submission "$run/submission.json" --output "$C" "${COMMON[@]}" & child=$!
    while kill -0 "$child" 2>/dev/null; do
      if [ -e "$C/B2_4.json" ]; then sleep 10; step "B2_4 frozen: stopping controls (B3 / B5 are P1 baselines, not part of D)"; kill -TERM "$child" 2>/dev/null; break; fi
      sleep 10
    done
    wait "$child" 2>/dev/null
    [ -e "$C/B2_4.json" ] || { echo "controls failed" >&2; return 1; }
  fi; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
