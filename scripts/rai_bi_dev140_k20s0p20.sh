#!/bin/bash
# 2026-10-05: batch-invariant dev140 eval of base + k20s0-p20 (rai-trained) on rai GPU1 — defines the gap set for the first mechanism run.
M=/home/xueqi/hq/projects/ICML27_agent_distillation
cd /home/xueqi/hq/projects/ICML27_run_3045b60
BATCH_INVARIANT=1 ENV_PYTHON=$M/.venv/bin/python VERIFIED=$M/data/verified/seeds_k20_s0 \
  bash scripts/rai_eval_student.sh 1 data/manifest/dev140.json $M/reports/runs/eval/bi-dev140-rai --base --workers 32 \
  k20s0-p20=$M/reports/runs/train/seedsft-k20-s0-rai/checkpoint-tokens-208140
echo "exit $?"
