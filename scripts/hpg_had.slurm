#!/bin/bash
#SBATCH --job-name=gemma-had
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# Deliberately no GPU request. Submit teacher-r on a CPU partition.
# GPU example: sbatch --partition=hpg-b200 --gpus=4 --export=ALL,HAD_THROUGH=train-0 ...
# CPU example: sbatch --partition=<cpu-partition> --export=ALL,HAD_THROUGH=teacher-0 ...
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${HAD_OUT:?Set a persistent HAD cell output}"
: "${VERIFIED:?Set the verified support collection}"
HAD_MODE=${HAD_MODE:-train}
HAD_WORKER=${HAD_WORKER:-}
if [[ -n "$HAD_WORKER" ]]; then
    : "${HAD_THROUGH:=explore-0}"
else
    : "${HAD_THROUGH:?Set explore-r, teacher-r, or train-r}"
fi
NGPUS=${NGPUS:-4} # Identity/training world size, also retained in CPU jobs.
[[ "$NGPUS" =~ ^(1|2|4|8)$ ]]
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$HAD_WORKER" == teacher || ( "$HAD_MODE" == train && "$HAD_THROUGH" == teacher-* ) ]]; then
    [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || { echo 'HAD teacher stage requires a CPU allocation' >&2; exit 2; }
    for allocation in "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS:-}"; do
        [[ -z "$allocation" || "$allocation" == 0 ]] || { echo 'HAD teacher stage requires a CPU allocation' >&2; exit 2; }
    done
    export CUDA_VISIBLE_DEVICES=''
fi
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
worker_args=()
if [[ -n "$HAD_WORKER" ]]; then
    worker_args=(--worker "$HAD_WORKER" --gpu-wait-seconds "${HAD_GPU_WAIT_SECONDS:-1200}"
        --poll-seconds "${HAD_POLL_SECONDS:-1}" --poll-max-seconds "${HAD_POLL_MAX_SECONDS:-30}")
fi
"$ENV_PYTHON" -m baselines.had.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$HAD_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus "$NGPUS" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --mode "$HAD_MODE" --through "$HAD_THROUGH" "${worker_args[@]}" &
child=$!
wait "$child"
child=''
