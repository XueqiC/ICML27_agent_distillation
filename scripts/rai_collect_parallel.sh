#!/bin/bash
# 2026-10-04 18:05: support seeds 1 and 2 collected in parallel with seed 0 (already running, pid 4165608);
# the teacher dev140 evaluation starts once all three collections have exited.
set -u
cd /home/xueqi/hq/projects/ICML27_agent_distillation
export ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
log(){ echo "[$(TZ=America/Chicago date '+%H:%M:%S')] $*"; }
pids=()
for seed in 1 2; do
  .venv/bin/python teachers/collect_seeds.py --config configs/base.yaml --selection "data/manifest/seeds_k20_s${seed}.json" \
    --attempts 3 --subset-k 5 10 --output "data/verified/seeds_k20_s${seed}" > "logs/collect_k20_s${seed}.log" 2>&1 &
  pids+=($!); log "collect s$seed pid $!"
done
for p in "${pids[@]}"; do wait $p; log "pid $p exit $?"; done
while ps -p 4165608 >/dev/null 2>&1; do sleep 60; done; log "seed 0 collection exited"
log "teacher dev140"
.venv/bin/python eval/run_suite.py --config configs/base.yaml --policy teacher --tasks data/manifest/dev140.json \
  --workers 8 --output reports/runs/teacher-dev140 > logs/teacher-dev140.log 2>&1
log "teacher dev140 exit $?"
