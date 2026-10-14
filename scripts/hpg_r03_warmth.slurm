#!/bin/bash
#SBATCH --job-name=r03-warmth
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --output=slurm-%j.out
# R0a: sensitivity-weighted frozen-denominator error (tools/warmth_dump.py) at several saved states of one run.
# Required: R03_RUN (Seed-SFT run dir), R03_STATES (space-separated dir names under R03_RUN), R03_MSGDIR (dir with
# <state>.json message batches), R03_OUTDIR (new dir). Optional: R03_SURROGATE.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# Sizing (sacct 10-01..10-14, 20 jobs): ~1.0 busy core, peak RSS ~23 GB (gemma-4-12B weights). OMP_NUM_THREADS stays 6,
# independent of the allocation, so CPU-side numerics match the earlier 6-CPU runs (R1 CRN pairing, T1 arms).
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${R03_RUN:?}" "${R03_STATES:?}" "${R03_MSGDIR:?}" "${R03_OUTDIR:?}"
: "${R03_SURROGATE:=${PROJECT_ROOT}/data/manifest/r03_surrogate40.json}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" ]]
[[ ! -e "$R03_OUTDIR" ]]
mkdir -p "$R03_OUTDIR"
read -r -a states <<< "$R03_STATES"
for s in "${states[@]}"; do
    [[ -d "$R03_RUN/$s" && -f "$R03_MSGDIR/$s.json" ]]
    "$TRAIN_PYTHON" -m tools.warmth_dump --snapshot "$MODEL_PATH" --checkpoint "$R03_RUN/$s" --verified "$VERIFIED" \
        --surrogate "$R03_SURROGATE" --message "$R03_MSGDIR/$s.json" --output "$R03_OUTDIR/$s.json"
done
