#!/bin/bash
#SBATCH --job-name=r03-amp
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out
# R3 pre-registered prediction (tools/amplitude_prediction.py, Corollary 7.2(a) summed over coordinates).
# Required: R03_CHECKPOINT, R03_MESSAGE (unscaled episode batch JSON), R03_OUT (new JSON). Optional: R03_SURROGATE, R03_GRID.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${R03_CHECKPOINT:?}" "${R03_MESSAGE:?}" "${R03_OUT:?}"
: "${R03_SURROGATE:=${PROJECT_ROOT}/data/manifest/r03_surrogate40.json}"
: "${R03_GRID:=0 0.25 0.5 0.75 1 1.5 2 3 4 6 8 12 16}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" && -d "$R03_CHECKPOINT" && -f "$R03_MESSAGE" ]]
[[ ! -e "$R03_OUT" ]]
mkdir -p "$(dirname "$R03_OUT")"
read -r -a grid <<< "$R03_GRID"
"$TRAIN_PYTHON" -m tools.amplitude_prediction --snapshot "$MODEL_PATH" --checkpoint "$R03_CHECKPOINT" --verified "$VERIFIED" \
    --surrogate "$R03_SURROGATE" --message "$R03_MESSAGE" --tail-updates 10 --grid "${grid[@]}" --output "$R03_OUT"
