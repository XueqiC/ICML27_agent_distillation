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
if [[ "${TRAIN_PASSES_LADDER:-0}" == 1 ]]; then
    args+=(--passes-ladder --max-passes "${TRAIN_MAX_PASSES:?Set a power-of-two convergence cap}"
           --converge-delta "${TRAIN_CONVERGE_DELTA:-0.01}" --packing episode)
else
    # Set TRAIN_PASSES=0 for a token-only run; with both, the first limit wins.
    if [[ "$TRAIN_PASSES" != 0 ]]; then args+=(--passes "$TRAIN_PASSES"); fi
    if [[ -n "$TRAIN_TOKENS" ]]; then args+=(--supervised-tokens "$TRAIN_TOKENS"); fi
    if [[ -n "$CHECKPOINT_TOKENS" ]]; then
        read -r -a endpoints <<< "$CHECKPOINT_TOKENS"
        args+=(--checkpoint-tokens "${endpoints[@]}")
    fi
fi
if [[ -n "${MOMENT_WARMUP_STEPS:-}" && "$MOMENT_WARMUP_STEPS" != 0 ]]; then
    args+=(--moment-warmup-steps "$MOMENT_WARMUP_STEPS")
fi
if [[ -n "${LR_WARMUP_STEPS:-}" && "$LR_WARMUP_STEPS" != 0 ]]; then
    args+=(--lr-warmup-steps "$LR_WARMUP_STEPS")
fi
if [[ -n "${SAVE_EVERY_UPDATE_THROUGH:-}" && "$SAVE_EVERY_UPDATE_THROUGH" != 0 ]]; then
    args+=(--save-every-update-through "$SAVE_EVERY_UPDATE_THROUGH")
fi
if [[ "${LOG_WARMTH:-}" == 1 ]]; then args+=(--log-warmth); fi
"$TRAIN_PYTHON" train/sft.py "${args[@]}"
