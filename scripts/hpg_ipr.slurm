#!/bin/bash
#SBATCH --job-name=gemma-ipr
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8
export BATCH_INVARIANT=${BATCH_INVARIANT:-1}
[[ "$BATCH_INVARIANT" == 1 ]] || { echo 'IPR requires BATCH_INVARIANT=1' >&2; exit 2; }
export VLLM_BATCH_INVARIANT=1
: "${IPR_OUT:?Set IPR_OUT to a persistent output tree for this K/support seed}"
: "${VERIFIED:?Set VERIFIED to the K verified seeds or nested subset}"
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
[[ -x "$ENV_PYTHON" && -x "$TRAIN_PYTHON" && -x "$VLLM_PYTHON" && -d "$MODEL_PATH" ]]
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$ENV_PYTHON" -m baselines.ipr.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$IPR_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus 4 --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --mode "${IPR_MODE:-train}" --through "${IPR_THROUGH:-all}" &
child=$!
wait "$child"
child=''
