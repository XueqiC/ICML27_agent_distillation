#!/bin/bash
#SBATCH --job-name=gemma-gad
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# No GPU request here: teacher-r jobs must use a CPU allocation.
# GPU: sbatch --partition=hpg-b200 --gpus=4 --export=ALL,GAD_THROUGH=train-0 ...
# CPU: sbatch --partition=<cpu> --export=ALL,GAD_THROUGH=teacher-0 ...
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${GAD_OUT:?Set a persistent GAD cell output}"
: "${VERIFIED:?Set the VERIFIED support collection}"
: "${GAD_THROUGH:?Set explore-r, teacher-r, or train-r; r=0,1,2}"
GAD_MODE=${GAD_MODE:-train}
NGPUS=${NGPUS:-4} # Keep the same training identity even in CPU teacher jobs.
[[ "$NGPUS" =~ ^(1|2|4|8)$ ]]
[[ "$GAD_THROUGH" =~ ^(explore|teacher|train)-[012]$ ]]
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$GAD_MODE" == train && "$GAD_THROUGH" == teacher-* ]]; then
    [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || { echo 'GAD teacher stage requires a CPU allocation' >&2; exit 2; }
    for allocation in "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS:-}"; do
        [[ -z "$allocation" || "$allocation" == 0 ]] || { echo 'GAD teacher stage requires a CPU allocation' >&2; exit 2; }
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
"$ENV_PYTHON" -m baselines.gad.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$GAD_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus "$NGPUS" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --mode "$GAD_MODE" --through "$GAD_THROUGH" &
child=$!
wait "$child"
child=''
