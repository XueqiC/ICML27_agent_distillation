#!/bin/bash
#SBATCH --job-name=r03-rollout
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
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
case "$BATCH_INVARIANT" in 0|1) ;; *) echo 'BATCH_INVARIANT must be 0 or 1' >&2; exit 2 ;; esac
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
# hpg.env defaults TASKS/OUT to evaluation destinations; set rollout inputs first.
TASKS=${TASKS:-data/manifest/r03_pool200.json}
: "${OUT:?Set OUT to a new rollout run directory}"
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
LORA_DTYPE=${LORA_DTYPE:-bfloat16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
source scripts/rollout_adapter.sh
