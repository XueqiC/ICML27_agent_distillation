#!/bin/bash
#SBATCH --job-name=adftp
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-milan
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# Default CPU collection. GPU stage: sbatch --partition=hpg-b200 --gpus=1
# --export=ALL,ADFTP_THROUGH=train,VERIFIED=<FTP collection or nested subset> ...
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1
: "${ADFTP_OUT:?Set the persistent output tree}" "${VERIFIED:?Set the verified source collection}"
ADFTP_THROUGH=${ADFTP_THROUGH:-collect}
case "$ADFTP_THROUGH" in collect|train) ;; *) echo 'ADFTP_THROUGH must be collect or train; use separate allocations' >&2; exit 2 ;; esac
if [[ "$ADFTP_THROUGH" == collect && "${ADFTP_MODE:-train}" != evaluate ]]; then
    [[ -z "${SLURM_JOB_GPUS:-}" && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || { echo 'Teacher collection must not hold a GPU' >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES=''
    subset_args=()
    if [[ -n "${ADFTP_SUBSET_K:-}" ]]; then read -r -a ks <<< "$ADFTP_SUBSET_K"; subset_args=(--subset-k "${ks[@]}"); fi
    exec "$ENV_PYTHON" -m baselines.adftp.pipeline --config "${CONFIG:-configs/base.yaml}" \
        --verified "$VERIFIED" --output "$ADFTP_OUT" --through collect \
        --rate-limit-dir "${ADFTP_RATE_LIMIT_DIR:-data/adftp_rate_limit}" "${subset_args[@]}"
fi
NGPUS=${NGPUS:-1}
case "$NGPUS" in 1|2|4|8) ;; *) echo 'NGPUS must be 1/2/4/8' >&2; exit 2 ;; esac
exec "$ENV_PYTHON" -m baselines.adftp.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --gpus "$NGPUS" --scoring "${ADFTP_SCORING:-per_turn}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$ADFTP_OUT" --through train \
    --mode "${ADFTP_MODE:-train}" --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" \
    --vllm-python "$VLLM_PYTHON" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT"
