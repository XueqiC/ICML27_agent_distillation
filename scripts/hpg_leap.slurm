#!/bin/bash
#SBATCH --job-name=gemma-leap
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
[[ "$BATCH_INVARIANT" == 1 ]] || { echo 'LEAP requires BATCH_INVARIANT=1' >&2; exit 2; }
export VLLM_BATCH_INVARIANT=1
: "${LEAP_OUT:?Set LEAP_OUT to persistent /blue storage (about 24 GB per merge)}"
: "${LEAP_VARIANT:?Set LEAP_VARIANT=privileged or zero}"
: "${VERIFIED:?Set VERIFIED to the K verified seeds or nested subset}"
[[ "$LEAP_VARIANT" == privileged || "$LEAP_VARIANT" == zero ]]
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
[[ -x "$ENV_PYTHON" && -x "$TRAIN_PYTHON" && -x "$VLLM_PYTHON" && -d "$MODEL_PATH" ]]
init_args=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then init_args+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$ENV_PYTHON" -m baselines.leap.pipeline "${init_args[@]}" --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$LEAP_OUT" --variant "$LEAP_VARIANT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --extraction-tokens "${LEAP_EXTRACTION_TOKENS:-4096}" --correction-tokens "${LEAP_CORRECTION_TOKENS:-16384}" \
    --mode "${LEAP_MODE:-train}" --through "${LEAP_THROUGH:-all}" &
child=$!
wait "$child"
child=''
