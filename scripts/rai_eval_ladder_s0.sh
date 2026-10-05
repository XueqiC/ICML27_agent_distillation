#!/bin/bash
# 2026-10-04: dev140 eval of the Seed-SFT s0 K-ladder (K=5/10/20 x 3/5/10/20 passes) + base on one server, rai GPU1,
# from a frozen worktree (RUN_TREE). Waits for the K=5 training chain first.
set -u
M=/home/xueqi/hq/projects/ICML27_agent_distillation
RUN_TREE=${RUN_TREE:-/home/xueqi/hq/projects/ICML27_run_57ae004}
while ! grep -q "train k5 s0 exit" $M/logs/rai_train_k10_k5_s0.log; do sleep 30; done
grep -q "train k5 s0 exit 0" $M/logs/rai_train_k10_k5_s0.log || { echo "k5 training failed"; exit 1; }
args=()
for k in 20 10 5; do
  d=$M/reports/runs/train/seedsft-k$k-s0-rai
  for c in $(ls -d $d/checkpoint-tokens-* | sort -t- -k3 -n); do
    t=${c##*-}; per=$(python3 -c "import json;print(json.load(open('$d/preparation.json'))['supervised_tokens_per_pass'])")
    p=$(( (t + per/2) / per )); args+=("k${k}s0-p$(printf %02d $p)=$c")
  done
done
[ "${SKIP_DONE:-1}" = 1 ] && args=("${args[@]/k20s0-p03=*/}")
args=($(printf '%s\n' "${args[@]}" | grep .))
echo "adapters: ${args[*]}"
cd $RUN_TREE
ENV_PYTHON=$M/.venv/bin/python VERIFIED=$M/data/verified/seeds_k20_s0 \
  bash scripts/rai_eval_student.sh 1 data/manifest/dev140.json $M/reports/runs/eval/ladder-s0-rai-dev140 --base --workers 32 "${args[@]}"
echo "eval exit $?"
