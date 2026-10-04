#!/bin/bash
# 2026-10-04: collect K=20 verified seeds for support seeds 0/1/2 (K=5/10 as nested subsets), then the teacher on dev140.
# Sequential so that a single run at a time uses the teacher rate limit.
set -u
cd /home/xueqi/hq/projects/ICML27_agent_distillation
export ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
log(){ echo "[$(TZ=America/Chicago date '+%H:%M:%S')] $*"; }
for seed in 0 1 2; do
  log "collect seeds_k20_s$seed"
  .venv/bin/python teachers/collect_seeds.py --config configs/base.yaml --selection "data/manifest/seeds_k20_s${seed}.json" \
    --attempts 3 --subset-k 5 10 --output "data/verified/seeds_k20_s${seed}" > "logs/collect_k20_s${seed}.log" 2>&1
  log "collect s$seed exit $?"
done
log "teacher dev140"
.venv/bin/python eval/run_suite.py --config configs/base.yaml --policy teacher --tasks data/manifest/dev140.json \
  --workers 8 --output reports/runs/teacher-dev140 > logs/teacher-dev140.log 2>&1
log "teacher dev140 exit $?"
