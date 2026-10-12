#!/bin/bash
#SBATCH --job-name=r03-gradstats
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
# R2: per-cluster gradient second moments (student-sampled labels; Fisher-trace estimate) at a restored state.
# Required: R03_CHECKPOINT, R03_ROLLOUTS, R03_OUT.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${R03_CHECKPOINT:?}" "${R03_ROLLOUTS:?}" "${R03_OUT:?}"
[[ ! -e "$R03_OUT" ]]
mkdir -p "$(dirname "$R03_OUT")"
"$TRAIN_PYTHON" -m tools.grad_stats --snapshot "$MODEL_PATH" --checkpoint "$R03_CHECKPOINT" --verified "$VERIFIED" \
    --rollouts "$R03_ROLLOUTS" --pool data/manifest/r03_pool200.json --states-per-cluster 10 --seed 0 --device cuda \
    --output "$R03_OUT"
