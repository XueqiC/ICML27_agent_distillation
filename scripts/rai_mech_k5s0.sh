#!/bin/bash
# 2026-10-05: mechanism run on k5s0-t400k (hpg-trained, rai BI 74/140, G=43) after the K10 eval frees GPU1.
M=/home/xueqi/hq/projects/ICML27_agent_distillation
while kill -0 1482167 2>/dev/null; do sleep 30; done
cd /home/xueqi/hq/projects/ICML27_run_70314a3
MAX_TEACHER_CALLS=700 MAX_TEACHER_CALLS_RUN=30000 BATCH_INVARIANT=1 WORKERS=8 ENV_PYTHON=$M/.venv/bin/python \
  bash scripts/rai_mechanism.sh 1 $M/reports/runs/mechanism/k5s0-t400k-bi/taxonomy/taxonomy.json $M/data/verified/seeds_k20_s0/subsets/k5 \
  $M/reports/runs/mechanism/k5s0-t400k-bi/run k5s0-t400k=$M/reports/runs/train/hpg-adapters/seedsft-k5s0-hpg/checkpoint-tokens-400172
echo "mech exit $?"
