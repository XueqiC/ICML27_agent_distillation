#!/bin/bash
#SBATCH --job-name=gemma-adam-probe
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 ]]
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" ]]
: "${PROBE_RUN:?Set PROBE_RUN to an existing Seed-SFT output directory}"
: "${PROBE_STATES:=cold warmup:32 update:1}"
: "${PROBE_PAIRS:=4}"
: "${PROBE_CONFIG:=${PROJECT_ROOT}/configs/base.yaml}"
: "${PROBE_OUTPUT:=${PROJECT_ROOT}/reports/adam-probe-${SLURM_JOB_ID:-manual}.json}"
[[ -f "$PROBE_RUN/provenance.json" && -f "$PROBE_RUN/exposure_plan.json" ]]
read -r -a states <<< "$PROBE_STATES"
"$TRAIN_PYTHON" tools/adam_response_probe.py --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
    --config "$PROBE_CONFIG" --run "$PROBE_RUN" --states "${states[@]}" \
    --pairs "$PROBE_PAIRS" --output "$PROBE_OUTPUT"
