#!/bin/bash
#SBATCH --job-name=sad-shared-history-check
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
# SAD_PREPARED accepts the existing sad-k20s0 run root or its prepare stage output.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
: "${SAD_PREPARED:=${SAD_OUT:-${PROJECT_ROOT}/reports/runs/baselines/sad-k20s0}}"
: "${OUTPUT:=${PROJECT_ROOT}/reports/engineering/sad_shared_history_gpu-${SLURM_JOB_ID}.json}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -d "$SAD_PREPARED" ]]
exec "$TRAIN_PYTHON" -m tools.validate_shared_history_gpu \
    --config "${CONFIG:-configs/base.yaml}" --snapshot "$MODEL_PATH" \
    --sad-prepared "$SAD_PREPARED" --num-rows "${NUM_ROWS:-4}" --output "$OUTPUT"
