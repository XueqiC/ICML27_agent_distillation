#!/bin/bash
# 2026-10-05: batch-invariant dev140 eval on rai GPU1 of hpg-trained adapters, one adapter per server (canonical) — gap sets for the next mechanism runs.
M=/home/xueqi/hq/projects/ICML27_agent_distillation
cd /home/xueqi/hq/projects/ICML27_run_70314a3
for spec in "k5s0-t400k:seedsft-k5s0-hpg/checkpoint-tokens-400172:seeds_k20_s0/subsets/k5" "k10s0-t400k:seedsft-k10s0-hpg/$(ls $M/reports/runs/train/hpg-adapters/seedsft-k10s0-hpg | grep checkpoint):seeds_k20_s0/subsets/k10"; do
  name=${spec%%:*}; rest=${spec#*:}; ad=${rest%%:*}; ver=${rest#*:}
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] $name start"
  BATCH_INVARIANT=1 INCLUDE_BASE=0 ENV_PYTHON=$M/.venv/bin/python VERIFIED=$M/data/verified/$ver     bash scripts/rai_eval_student.sh 1 data/manifest/dev140.json $M/reports/runs/eval/bi-dev140-rai-$name --workers 32 $name=$M/reports/runs/train/hpg-adapters/$ad
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] $name exit $?"
done
