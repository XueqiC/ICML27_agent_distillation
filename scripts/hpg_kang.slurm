#!/bin/bash
#SBATCH --job-name=gemma-kang-sft
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8
export BATCH_INVARIANT=${BATCH_INVARIANT:-1}
[[ "$BATCH_INVARIANT" == 1 ]] || { echo 'Stage 3b requires BATCH_INVARIANT=1' >&2; exit 2; }
export VLLM_BATCH_INVARIANT=1
: "${KANG_OUT:?Set KANG_OUT to the persistent K/support-seed output tree}"
: "${VERIFIED:?Set VERIFIED to the K verified seeds or nested subset}"
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
[[ -x "$ENV_PYTHON" && -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" ]]
NGPUS=${NGPUS:-1}
case "$NGPUS" in 1|2|4|8) ;; *) echo 'NGPUS must be 1/2/4/8' >&2; exit 2 ;; esac
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$ENV_PYTHON" -m baselines.kang.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --gpus "$NGPUS" --scoring "${KANG_SCORING:-per_turn}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$KANG_OUT" \
    --training-seed "${KANG_TRAINING_SEED:-0}" \
    --epochs "${KANG_EPOCHS:-2}" \
    --train-python "$TRAIN_PYTHON" --mode "${KANG_MODE:-train}" &
child=$!
wait "$child"
child=''
