#!/bin/bash
# CG probe rollouts on rai: serve theta0 (HF/vLLM LoRA gate as coverage), then A/B/C/D rollouts. Pinned tree.
# Usage: run_cg_rollouts_rai.sh <serve_out_name> <hints.json> <hints_approval.json> [extra cg_probe args]
set -euo pipefail
export PROJECT_ROOT=/home/xueqi/hq/projects/ICML27_run_repair_eval_${CG_TREE:-9f16962}
export HPG_CONFIG=/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/repair_pilot/rai.env
P=/home/xueqi/hq/projects/ICML27_agent_distillation
export REPAIR_OUT=$P/reports/runs/repair_pilot/cg_probe/$1
export CG_HINTS=$2 CG_APPROVAL=$3 CG_EXTRA=${4:-}
export COV=$P/reports/runs/repair_pilot/cov_theta0_umech/coverage
export THETA0=$P/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
export VERIFIED=$P/data/verified/seeds_full_train_r2/subsets/k20
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1} WORKERS=${WORKERS:-8}
cd "$PROJECT_ROOT"
source scripts/repair_stage_env.sh
ADAPTER=$THETA0
MODEL_NAME=e4b-tsft-k20-e32-lr5e-5
repair_work() {
    local extra=()
    [[ -n "$CG_EXTRA" ]] && read -r -a extra <<< "$CG_EXTRA"
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.cg_probe rollouts --hints "$CG_HINTS" --approval "$CG_APPROVAL" \
        --failures "$COV/failures.json" --coverage-root "$COV" --config "$CONFIG" \
        --endpoint "$1" --server-provenance "$2" --lora-check "$3" --model-name "$MODEL_NAME" \
        --data-root "$ALFWORLD_DATA_ROOT" --workers "$WORKERS" "${extra[@]}" --output "$REPAIR_OUT/rollouts"
}
source scripts/repair_serve_stage.sh
