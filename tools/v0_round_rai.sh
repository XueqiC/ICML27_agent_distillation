#!/bin/bash
# Example launcher; launching this script starts a server. Not used by CPU tests.
# Source/cache/model paths come from the same rai.env as run_v2_eval_rai.sh.
set -euo pipefail
: "${V0_ROUND_OUT:?Set the round root, already initialized by synth}"
: "${V0_STAGE:?Set attempt, probe, review or expand}"
: "${ADAPTER:?Set the theta0 checkpoint path}"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export V0_SOURCE_ROOT=${V0_SOURCE_ROOT:-/home/xueqi/hq/projects/ICML27_agent_distillation}
export HPG_CONFIG=${HPG_CONFIG:-$V0_SOURCE_ROOT/reports/runs/repair_pilot/rai.env}
export MODEL_NAME=e4b-tsft-k20-e32-lr5e-5
export VERIFIED=${VERIFIED:-$V0_SOURCE_ROOT/data/verified/seeds_full_train_r2/subsets/k20}
export REPAIR_OUT=${REPAIR_OUT:-$V0_ROUND_OUT/serve-$V0_STAGE${V0_VARIANT:+-$V0_VARIANT}}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
case "$V0_STAGE" in attempt|probe|review|expand) ;; *) echo 'Unknown V0_STAGE' >&2; exit 2 ;; esac
cd "$PROJECT_ROOT"
source scripts/repair_stage_env.sh
repair_work() {
    local extra=()
    [[ "${V0_RESUME:-0}" == 1 ]] && extra+=(--resume)
    if [[ "$V0_STAGE" == expand ]]; then
        : "${V0_VARIANT:?Set D-data or D-param}" "${V0_DATASET:?Set the immutable input dataset}"
        extra+=(--variant "$V0_VARIANT" --dataset "$V0_DATASET"
            --seeds "$VERIFIED/manifest.json" --data-root "$ALFWORLD_DATA_ROOT"
            --main-checkout "$V0_SOURCE_ROOT")
        if [[ "$V0_VARIANT" == D-data ]]; then
            : "${V0_GROUPS:?Set the grouping output directory}"
            extra+=(--groups "$V0_GROUPS/groups.json" --weights "$V0_GROUPS/weights.json")
        fi
    fi
    # repair_serve_stage.sh supplies the actual ready endpoint and both gates.
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.v0_round "$V0_STAGE" \
        --config "$CONFIG" --output "$V0_ROUND_OUT" --seed 0 \
        --endpoint "$1" --server-provenance "$2" --lora-check "$3" \
        --model-name "$MODEL_NAME" "${extra[@]}"
}
source scripts/repair_serve_stage.sh
