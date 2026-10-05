#!/bin/bash
#SBATCH --job-name=merged-robust
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-merged-robust-%A_%a.out
#SBATCH --signal=B:TERM@60
# 2026-10-05: serving-precision robustness check. Merge a high-dose Seed-SFT adapter into standalone bf16 weights (train/merge_lora.py
# checks all next-token logits against PEFT) and evaluate the merged model on dev140 (batch-invariant). Compare with the
# adapter-served result of the same checkpoint: k20s0-p40 passed the LoRA gate (115/140), k20s2-p40 failed it on outlier tokens.
# Required: PROJECT_ROOT=<frozen tree>, TRAIN_ROOT, OUT_ROOT.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${TRAIN_ROOT:?}" "${OUT_ROOT:?}"
names=(k20s0 k20s2)
tokens=(416280 400000)
name=${names[$SLURM_ARRAY_TASK_ID]}
run_dir="$TRAIN_ROOT/seedsft-$name-hpg"
ckpt=$(ls -d "$run_dir"/checkpoint-tokens-* | awk -F- -v t="${tokens[$SLURM_ARRAY_TASK_ID]}" '$NF+0 >= t' | sort -t- -k3 -n | head -1)
[[ -d "$ckpt" ]] || { echo "no checkpoint for $name" >&2; exit 2; }
merged="$OUT_ROOT/merged/$name-p40"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "task $SLURM_ARRAY_TASK_ID: $name checkpoint $ckpt -> $merged"
"$TRAIN_PYTHON" train/merge_lora.py --snapshot "$MODEL_PATH" --adapter "$ckpt" --output "$merged"
export MERGED_MODEL_PATH="$merged" TASKS="$PROJECT_ROOT/data/manifest/dev140.json" OUT="$OUT_ROOT/eval/$name-p40-merged"
export WORKERS=16 BATCH_INVARIANT=1 INCLUDE_BASE=1
unset ADAPTERS
exec bash scripts/hpg_eval_adapters.slurm
