#!/bin/bash
#SBATCH --job-name=gemma-sft-dev40
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60

set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
: "${ADAPTER:?Set ADAPTER to one checkpoint-tokens-* directory}"
: "${MERGED_MODEL_PATH:?Set MERGED_MODEL_PATH to a fresh standalone destination}"
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]]
[[ "$TASKS" == "$PROJECT_ROOT/data/manifest/dev40.json" ]]
"$TRAIN_PYTHON" train/merge_lora.py --snapshot "$MODEL_PATH" --adapter "$ADAPTER" \
    --output "$MERGED_MODEL_PATH"
# Merge exits nonzero unless all fixed-input logits pass after standalone reload.
export MERGED_MODEL_PATH
exec bash scripts/hpg_eval_student.slurm
