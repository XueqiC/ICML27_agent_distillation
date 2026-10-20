#!/bin/bash
#SBATCH --job-name=sft-shared-history-check
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
: "${VERIFIED:?Set a verified training collection with at least four rows}"
: "${OUTPUT_PREFIX:=${PROJECT_ROOT}/reports/engineering/sft_shared_history_gpu-${SLURM_JOB_ID}}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -d "$VERIFIED" ]]
status=0
for objective in sft smartad; do
    "$TRAIN_PYTHON" -m tools.validate_shared_history_gpu \
        --config "${CONFIG:-configs/base.yaml}" --snapshot "$MODEL_PATH" \
        --verified "$VERIFIED" --objective "$objective" --num-rows "${NUM_ROWS:-4}" \
        --output "${OUTPUT_PREFIX}-${objective}.json" || status=1
done
exit "$status"
