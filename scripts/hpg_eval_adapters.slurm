#!/bin/bash
#SBATCH --job-name=gemma-adapters-dev
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60

set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Whitespace-delimited NAME=PATH entries; never evaluate shell code from ADAPTERS.
if [[ -n "${MERGED_MODEL_PATH:-}" ]]; then
    [[ -z "${ADAPTERS:-}" ]] || { echo 'Full checkpoint mode excludes ADAPTERS' >&2; exit 2; }
    adapters=()
    INCLUDE_BASE=${INCLUDE_BASE:-1}
else
    : "${ADAPTERS:?Set ADAPTERS to NAME=PATH entries}"
    read -r -a adapters <<< "${ADAPTERS//$'\n'/ }"
    INCLUDE_BASE=${INCLUDE_BASE:-0}
fi
source scripts/eval_adapters.sh
