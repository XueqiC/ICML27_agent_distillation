#!/bin/bash
#SBATCH --job-name=smartad
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-milan
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# Separate allocations: collect CPU; select/train/evaluate --partition=hpg-b200 --gpus=1.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1
: "${SMARTAD_OUT:?Set persistent stage output}" "${VERIFIED:?Set stage input}"
SMARTAD_THROUGH=${SMARTAD_THROUGH:-collect}
case "$SMARTAD_THROUGH" in collect|select|train|evaluate) ;; *) echo 'Invalid SmartAD stage' >&2; exit 2 ;; esac
if [[ "$SMARTAD_THROUGH" == collect ]]; then
    [[ -z "${SLURM_JOB_GPUS:-}" && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || { echo 'Teacher collection must not hold a GPU' >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES=''
    subset_args=()
    if [[ -n "${SMARTAD_SUBSET_K:-}" ]]; then read -r -a ks <<< "$SMARTAD_SUBSET_K"; subset_args=(--subset-k "${ks[@]}"); fi
    exec "$ENV_PYTHON" -m baselines.smartad.pipeline --config "${CONFIG:-configs/base.yaml}" \
        --verified "$VERIFIED" --output "$SMARTAD_OUT" --through collect \
        --rate-limit-dir "${SMARTAD_RATE_LIMIT_DIR:-data/smartad_rate_limit}" "${subset_args[@]}"
fi
[[ "${SLURM_GPUS_ON_NODE:-0}" == 1 ]] || { echo 'SmartAD select/train/evaluate require exactly one GPU' >&2; exit 2; }
python_bin=$ENV_PYTHON
if [[ "$SMARTAD_THROUGH" == select ]]; then python_bin=$TRAIN_PYTHON; fi
exec "$python_bin" -m baselines.smartad.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$SMARTAD_OUT" --through "$SMARTAD_THROUGH" \
    --mode "${SMARTAD_MODE:-train}" --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" \
    --vllm-python "$VLLM_PYTHON" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT"
