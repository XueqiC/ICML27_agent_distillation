#!/bin/bash
#SBATCH --job-name=wspd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@120
# hpg_wspd_steps.slurm — WSP round 2 (prereg 2026-11-15 section 18): surgery tool steps on ONE GPU of hpg, tree 4b4980f
# (git checkout; provenance needs git). Port of ICML27_runs/p1_scripts/hpg_p1_surgery_29162fc.slurm.
# Env: P1_STEPS = file with one tools/p1_surgery.py subcommand+args per line, run in order, stop at the first failure.
#   line prefix "CPU "          : run with CUDA hidden
#   line prefix "UNTIL <path> " : run the step, and once <path> exists terminate it and continue (counts as success)
#   line prefix "SKIPIF <path> ": skip the step when <path> already exists (makes a resubmitted job continue)
# TEACHER=1 exports the teacher key from $H/secrets/openai.env (never printed) and checks the API is reachable first.
# Partition / gpus / cpus / mem / time / output come from the sbatch command line.
set -euo pipefail
: "${P1_STEPS:?}"
H=/blue/fsu-compsci-dept/xc25.fsu/hq
T=$H/ICML27_runs/repair_eval_4b4980f
cd "$T"
[ "$(git rev-parse --short=7 HEAD)" = 4b4980f ] || { echo "tree mismatch" >&2; exit 2; }
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=$H/tools/hf-cache \
  SED_HASH_CACHE_DIR=$H/.sed-hash-receipts
PY=$H/envs/p1-venv/bin/python
if [[ "${TEACHER:-0}" == 1 ]]; then
  OPENAI_API_KEY=$(sed -nE 's/^(export )?OPENAI_API_KEY=//p' $H/secrets/openai.env | head -1 | sed -E "s/^[\"']//; s/[\"']$//")
  export OPENAI_API_KEY
  [[ -n "$OPENAI_API_KEY" ]] || { echo "no teacher key" >&2; exit 2; }
  "$PY" - <<'PY' || { echo "teacher API preflight failed" >&2; exit 2; }
import os, sys, urllib.request, urllib.error
req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
try:
    with urllib.request.urlopen(req, timeout=40) as r:
        print("teacher API preflight: HTTP", r.status)
except urllib.error.HTTPError as e:
    print("teacher API preflight: HTTP", e.code); sys.exit(1)
except Exception as e:
    print("teacher API preflight:", type(e).__name__); sys.exit(1)
PY
fi
COMMON=(--base $H/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
  --adapter $H/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
  --round-dir $H/ICML27_v0_runs/v0_k20_r1
  --verified $H/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20
  --alfworld-data $H/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
  --relocate /home/xueqi/hq/projects=$H
  --relocate /home/xueqi/.cache/huggingface=$H/tools/hf-cache)
echo "== wspd start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-manual} steps=$P1_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
child=''
forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 143; }
trap forward TERM
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  cvd="${CUDA_VISIBLE_DEVICES:-}"; until_path=''
  if [[ "$line" == "SKIPIF "* ]]; then rest=${line#SKIPIF }; sp=${rest%% *}; line=${rest#* }
    if [[ -e "$sp" ]]; then echo "== step [$line] skipped: $sp exists"; continue; fi; fi
  if [[ "$line" == "UNTIL "* ]]; then rest=${line#UNTIL }; until_path=${rest%% *}; line=${rest#* }; fi
  if [[ "$line" == "CPU "* ]]; then line=${line#CPU }; cvd=''; fi
  read -r -a ARGS <<< "$line"
  echo "== step [$line] start $(date -Is)"
  CUDA_VISIBLE_DEVICES="$cvd" "$PY" tools/p1_surgery.py "${ARGS[@]}" "${COMMON[@]}" &
  child=$!
  if [[ -n "$until_path" ]]; then
    stopped=0
    while kill -0 "$child" 2>/dev/null; do
      if [[ -e "$until_path" ]]; then sleep 10; echo "== $until_path exists: stopping this step"; kill -TERM "$child" 2>/dev/null || true; stopped=1; break; fi
      sleep 10
    done
    set +e; wait "$child"; rc=$?; set -e; child=''
    if [[ "$stopped" == 1 ]]; then rc=0; elif [[ $rc -eq 0 && ! -e "$until_path" ]]; then echo "step ended without $until_path" >&2; rc=1; fi
  else
    set +e; wait "$child"; rc=$?; set -e; child=''
  fi
  echo "== step exit $rc $(date -Is)"
  [[ $rc -eq 0 ]] || exit $rc
done < "$P1_STEPS"
echo "== wspd done $(date -Is)"
