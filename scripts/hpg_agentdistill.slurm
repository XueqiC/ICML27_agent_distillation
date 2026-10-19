#!/bin/bash
#SBATCH --job-name=gemma-agentdistill
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# No GPU allocation by default. Only evaluate needs --partition=hpg-b200 --gpus=1.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${AGENTDISTILL_OUT:?Set persistent AgentDistill cell output outside reports/runs}"
: "${VERIFIED:?Set VERIFIED support collection or nested view}"
: "${AGENTDISTILL_THROUGH:?Set collect, abstract, cluster, consolidate, publish, or evaluate}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mode=train
through=$AGENTDISTILL_THROUGH
case "$through" in
    evaluate) mode=evaluate; through=publish ;;
    collect|abstract|cluster|consolidate|publish)
        [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || { echo 'Teacher stage requires CPU allocation' >&2; exit 2; }
        for allocation in "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS:-}"; do
            [[ -z "$allocation" || "$allocation" == 0 ]] || { echo 'Teacher stage requires CPU allocation' >&2; exit 2; }
        done
        export CUDA_VISIBLE_DEVICES=''
        ;;
    *) echo 'Invalid AGENTDISTILL_THROUGH' >&2; exit 2 ;;
esac
[[ "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$ENV_PYTHON" -m baselines.agentdistill.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$AGENTDISTILL_OUT" \
    --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus 1 --workers "$WORKERS" --ready-timeout "$READY_TIMEOUT" --mode "$mode" --through "$through" &
child=$!
wait "$child"
child=''
