#!/bin/bash
# 2026-10-05: is an unmerged LoRA adapter deterministic under VLLM_BATCH_INVARIANT=1? k20s0-p20 on dev40, two server instances.
M=/home/xueqi/hq/projects/ICML27_agent_distillation
cd /home/xueqi/hq/projects/ICML27_run_a0a08b0
for i in 1 2; do
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] instance $i start"
  VLLM_BATCH_INVARIANT=1 INCLUDE_BASE=0 ENV_PYTHON=$M/.venv/bin/python VERIFIED=$M/data/verified/seeds_k20_s0 \
    bash scripts/rai_eval_student.sh 1 data/manifest/dev40.json $M/reports/runs/eval/bi-lora-pilot-dev40-$i --workers 16 \
    k20s0-p20=$M/reports/runs/train/seedsft-k20-s0-rai/checkpoint-tokens-208140
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] instance $i exit $?"
done
