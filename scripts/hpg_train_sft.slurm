#!/bin/bash
#SBATCH --job-name=gemma-seed-sft
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]]
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" ]]
args=(--snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$TRAIN_OUT"
      --tokens-per-update "$TOKENS_PER_UPDATE" --learning-rate "$LEARNING_RATE"
      --seed "$TRAIN_SEED" --max-length "$MAX_LENGTH")
# Set TRAIN_PASSES=0 for a token-only run; with both, the first limit wins.
if [[ "$TRAIN_PASSES" != 0 ]]; then args+=(--passes "$TRAIN_PASSES"); fi
if [[ -n "$TRAIN_TOKENS" ]]; then args+=(--supervised-tokens "$TRAIN_TOKENS"); fi
if [[ -n "$CHECKPOINT_TOKENS" ]]; then
    read -r -a endpoints <<< "$CHECKPOINT_TOKENS"
    args+=(--checkpoint-tokens "${endpoints[@]}")
fi
"$TRAIN_PYTHON" train/sft.py "${args[@]}"
