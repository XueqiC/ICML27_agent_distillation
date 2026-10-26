#!/bin/bash
#SBATCH --job-name=e4b-repair-coverage
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# E-series output convention: reports/runs/eval/chain/e4b-repair-<run>/coverage
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/repair_stage_env.sh"
REPAIR_BRANCH=${REPAIR_BRANCH:-P0_K20}
task_args=()
case "$REPAIR_BRANCH" in
    P0_K20|P1_U_mech)
        : "${THETA0:?Set e4b-tsft-k20-e32-lr5e-5 checkpoint}"
        ADAPTER=$THETA0
        MODEL_NAME=e4b-tsft-k20-e32-lr5e-5
        SERVE_BASE=0
        if [[ "$REPAIR_BRANCH" == P1_U_mech ]]; then
            REPAIR_SUPPORT_THETA0=$THETA0
            TASKS=${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}
            task_args+=(--task-set U_mech)
        fi ;;
    P0_K0)
        SERVE_BASE=1
        MODEL_NAME=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'from sed.config import load_config; import sys; print(load_config(sys.argv[1])["models"]["student"]["requested_repo"])' "$CONFIG") ;;
    *) echo 'Unknown REPAIR_BRANCH' >&2; exit 2 ;;
esac
: "${TASKS:?Set the main-grid K20 selection}"
repair_work() {
    local check_args=()
    if [[ -n "$3" ]]; then check_args+=(--lora-check "$3"); fi
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.repair_coverage --config "$CONFIG" \
        --tasks "$TASKS" "${task_args[@]}" --repair-splits "${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}" \
        --endpoint "$1" --server-provenance "$2" "${check_args[@]}" --model-name "$MODEL_NAME" \
        --extra-sampled "${EXTRA_SAMPLED:-0}" --workers "${WORKERS:-4}" --output "$REPAIR_OUT/coverage"
}
source scripts/repair_serve_stage.sh
