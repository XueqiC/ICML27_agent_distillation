#!/bin/bash
#SBATCH --job-name=r03-crn
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
# R0-R3 (docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md): one shard of CRN paired-replacement tails
# (tools/crn_tail.py run). Required: R03_CHECKPOINT (checkpoint-tokens-N or state-update-N of an existing Seed-SFT run),
# R03_ITEMS, R03_OUT (new JSONL path), R03_SHARD, R03_SHARDS. Optional: R03_SURROGATE (default: 40-trajectory listing).
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${R03_CHECKPOINT:?}" "${R03_ITEMS:?}" "${R03_OUT:?}" "${R03_SHARD:?}" "${R03_SHARDS:?}"
: "${R03_SURROGATE:=${PROJECT_ROOT}/data/manifest/r03_surrogate40.json}"
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" && -d "$R03_CHECKPOINT" && -f "$R03_ITEMS" ]]
[[ ! -e "$R03_OUT" ]]
mkdir -p "$(dirname "$R03_OUT")"
"$TRAIN_PYTHON" -m tools.crn_tail run --snapshot "$MODEL_PATH" --checkpoint "$R03_CHECKPOINT" --verified "$VERIFIED" \
    --surrogate "$R03_SURROGATE" --items "$R03_ITEMS" --shard "$R03_SHARD" --num-shards "$R03_SHARDS" --device cuda \
    --output "$R03_OUT"
