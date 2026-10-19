#!/bin/bash
#SBATCH --job-name=gemma-sad
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# Smoke: override BOTH --gpus=1 and NGPUS=1. SAD has no teacher stage.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1
: "${SAD_OUT:?Set the persistent output tree}" "${VERIFIED:?Set the verified support collection}"
NGPUS=${NGPUS:-4}
case "$NGPUS" in 1|2|4|8) ;; *) echo 'NGPUS must be 1/2/4/8' >&2; exit 2 ;; esac
exec "$ENV_PYTHON" -m baselines.sad.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$SAD_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus "$NGPUS" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --mode "${SAD_MODE:-train}" --through "${SAD_THROUGH:-all}"
