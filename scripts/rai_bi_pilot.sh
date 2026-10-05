#!/bin/bash
# 2026-10-05: pilot vLLM batch-invariant mode: base on dev40 on two separate server instances (rai GPU1).
M=/home/xueqi/hq/projects/ICML27_agent_distillation
cd /home/xueqi/hq/projects/ICML27_run_aa2fd07
for i in 1 2; do
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] instance $i start"
  VLLM_BATCH_INVARIANT=1 INCLUDE_BASE=1 ENV_PYTHON=$M/.venv/bin/python \
    bash scripts/rai_eval_student.sh 1 data/manifest/dev40.json $M/reports/runs/eval/bi-pilot-dev40-$i --base --workers 16
  echo "[$(TZ=America/Chicago date +%H:%M:%S)] instance $i exit $?"
done
