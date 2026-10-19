#!/bin/bash
#SBATCH --job-name=gemma-coa
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# No implicit GPU reservation. Teacher stages must use a CPU partition.
# sbatch --partition=<cpu> --export=ALL,COA_THROUGH=teacher scripts/hpg_coa.slurm
# sbatch --partition=hpg-b200 --gpus=4 --export=ALL,COA_THROUGH=train-sft scripts/hpg_coa.slurm
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${COA_OUT:?Set a persistent CoA cell output outside reports/runs}"
: "${VERIFIED:?Set the verified support collection}"
: "${COA_THROUGH:?Set teacher, data, train-sft, query-filter, explore-r or train-rl-r}"
COA_MODE=${COA_MODE:-train}
NGPUS=${NGPUS:-4} # Keep identity/world size the same in CPU stage jobs.
[[ "$NGPUS" =~ ^(1|2|4|8)$ ]]
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$COA_MODE" == teacher-evaluate || "$COA_MODE" == prepare || ( "$COA_MODE" == train && "$COA_THROUGH" =~ ^(teacher|data)$ ) ]]; then
    [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || { echo 'CoA CPU stage cannot hold a GPU' >&2; exit 2; }
    for allocation in "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS:-}"; do
        [[ -z "$allocation" || "$allocation" == 0 ]] || { echo 'CoA CPU stage cannot hold a GPU' >&2; exit 2; }
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
"$ENV_PYTHON" -m baselines.coa.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$COA_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus "$NGPUS" --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" \
    --mode "$COA_MODE" --through "$COA_THROUGH" &
child=$!
wait "$child"
child=''
