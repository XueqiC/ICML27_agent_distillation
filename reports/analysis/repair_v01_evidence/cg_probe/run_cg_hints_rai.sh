#!/bin/bash
# CG probe hints (teacher, CPU) on rai from the pinned tree (prereg 83b53b4). Usage: run_cg_hints_rai.sh <out> [groups]
set -euo pipefail
P=/home/xueqi/hq/projects/ICML27_agent_distillation
COV=$P/reports/runs/repair_pilot/cov_theta0_umech/coverage
G=${2:-look_at_obj_in_light-Newspaper-None-DeskLamp-224,pick_and_place_simple-Kettle-None-Shelf-1,pick_clean_then_place_in_recep-Kettle-None-DiningTable-7,pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18,pick_cool_then_place_in_recep-Potato-None-SideTable-21,pick_cool_then_place_in_recep-Apple-None-Microwave-19,pick_cool_then_place_in_recep-Egg-None-Microwave-5}
cd /home/xueqi/hq/projects/ICML27_run_repair_eval_${CG_TREE:-9f16962}
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
exec "$P/.venv/bin/python" -m tools.cg_probe hints --failures "$COV/failures.json" --coverage-root "$COV" --tasks "$G" \
  --branch-index 10 --config configs/base_e4b.yaml --max-teacher-input-tokens "${CG_MAX_IN:-200000}" --max-teacher-output-tokens "${CG_MAX_OUT:-60000}" --output "$1"
