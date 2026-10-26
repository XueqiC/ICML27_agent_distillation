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
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/scripts/repair_stage_env.sh"
: "${TASKS:?Set the main-grid K20 selection}" "${THETA0:?Set e4b-tsft-k20-e32-lr5e-5 checkpoint}"
ADAPTER=$THETA0
MODEL_NAME=e4b-tsft-k20-e32-lr5e-5
repair_work() {
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.repair_coverage --config "$CONFIG" \
        --tasks "$TASKS" --repair-splits "${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}" \
        --endpoint "$1" --server-provenance "$2" --lora-check "$3" --model-name "$MODEL_NAME" \
        --extra-sampled "${EXTRA_SAMPLED:-0}" --workers "${WORKERS:-4}" --output "$REPAIR_OUT/coverage"
}
source scripts/repair_serve_stage.sh
