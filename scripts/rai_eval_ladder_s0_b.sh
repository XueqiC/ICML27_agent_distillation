#!/bin/bash
M=/home/xueqi/hq/projects/ICML27_agent_distillation; R=$M/reports/runs/train; args=()
for k in 20 10 5; do
  per=$(python3 -c "import json;print(json.load(open('$R/seedsft-k$k-s0-rai/preparation.json'))['supervised_tokens_per_pass'])")
  for c in $R/seedsft-k$k-s0-rai/checkpoint-tokens-*; do
    t=${c##*-}; p=$(( (t + per/2) / per )); n=k${k}s0-p$(printf %02d $p)
    [ "$n" = k20s0-p10 ] && continue; args+=("$n=$c"); done; done
echo "adapters (${#args[@]}): ${args[*]%%=*}"
cd $M/../ICML27_run_aa2fd07
INCLUDE_BASE=0 ENV_PYTHON=$M/.venv/bin/python VERIFIED=$M/data/verified/seeds_k20_s0 \
  bash scripts/rai_eval_student.sh 1 data/manifest/dev140.json $M/reports/runs/eval/ladder-s0-rai-dev140-b --workers 32 "${args[@]}"
echo "eval exit $?"
