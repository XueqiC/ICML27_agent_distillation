#!/bin/bash
#SBATCH --job-name=seedsft-sweep
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --array=0-8%4
#SBATCH --output=logs/slurm-seedsft-%A_%a.out
# 2026-10-05: paper-grade Seed-SFT on hpg B200 — one array task per (K, support seed) in configs/seedsft_dose_grid.json.
# Submit from a frozen tree: sbatch --export=ALL,PROJECT_ROOT=<tree>,OUT_ROOT=<shared runs dir> scripts/hpg_seedsft_sweep.slurm
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${OUT_ROOT:?Set OUT_ROOT to the shared output directory}"
: "${DATA_ROOT:=$PROJECT_ROOT}"
: "${DOSE_GRID:=configs/seedsft_dose_grid.json}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read -r name verified total ckpts < <("$ENV_PYTHON" - "$SLURM_ARRAY_TASK_ID" "$DOSE_GRID" <<'PY'
import sys
from train.dose_grid import load_ready_run
run = load_ready_run(sys.argv[2], int(sys.argv[1]))
print(run["name"], run["verified"], run["total_tokens"], ",".join(str(c["tokens"]) for c in run["checkpoints"]))
PY
)
out="$OUT_ROOT/seedsft-$name-hpg"
[[ ! -e "$out" ]] || { echo "Output exists: $out" >&2; exit 2; }
echo "task $SLURM_ARRAY_TASK_ID: $name verified=$verified total=$total checkpoints=$ckpts host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
IFS=, read -r -a endpoints <<< "$ckpts"
"$TRAIN_PYTHON" train/sft.py --snapshot "$MODEL_PATH" --verified "$DATA_ROOT/$verified" --output "$out" \
    --supervised-tokens "$total" --checkpoint-tokens "${endpoints[@]}" --packing episode \
    --tokens-per-update 512 --learning-rate 1e-5 --seed 0 --max-length 32768
