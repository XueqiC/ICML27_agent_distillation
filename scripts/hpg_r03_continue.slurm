#!/bin/bash
#SBATCH --job-name=r03-cont
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
# R1/R2 continuation arm (train/continue_from_state.py): restore R03_CHECKPOINT, continue to the p40 budget with a
# fraction R03_FRACTION of the remaining updates replaced by batches from R03_EXTRA (optionally only the states listed
# in R03_EXTRA_STATES). Required: R03_CHECKPOINT, R03_EXTRA, R03_FRACTION, R03_SEED, R03_OUT (new dir).
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# Sizing (sacct 10-01..10-14, 20 jobs): ~1.0 busy core, peak RSS ~23 GB (gemma-4-12B weights). OMP_NUM_THREADS stays 6,
# independent of the allocation, so CPU-side numerics match the earlier 6-CPU runs (R1 CRN pairing, T1 arms).
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${R03_CHECKPOINT:?}" "${R03_EXTRA:?}" "${R03_FRACTION:?}" "${R03_SEED:?}" "${R03_OUT:?}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" && -d "$R03_CHECKPOINT" && -f "$R03_EXTRA/manifest.json" ]]
[[ -z "${R03_EXTRA_STATES:-}" || -f "$R03_EXTRA_STATES" ]]
[[ ! -e "$R03_OUT" ]]
mkdir -p "$(dirname "$R03_OUT")"
states=()
[[ -z "${R03_EXTRA_STATES:-}" ]] || states=(--extra-states "$R03_EXTRA_STATES")
"$TRAIN_PYTHON" -m train.continue_from_state --snapshot "$MODEL_PATH" --checkpoint "$R03_CHECKPOINT" --verified "$VERIFIED" \
    --extra "$R03_EXTRA" "${states[@]}" --fraction "$R03_FRACTION" --seed "$R03_SEED" --device cuda --output "$R03_OUT"
ls -d "$R03_OUT"/checkpoint-tokens-*
