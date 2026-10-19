#!/bin/bash
#SBATCH --job-name=steca
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# No GPU request here: teacher/data-only jobs must hold no GPU.
# GPU stage: sbatch --partition=hpg-b200 --gpus=4 --export=ALL,STECA_THROUGH=STAGE ...
# CPU stage: sbatch --partition=<CPU partition> --gpus=0 --export=ALL,STECA_THROUGH=STAGE ...
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${STECA_OUT:?Set persistent cell output}" "${VERIFIED:?Set verified support collection}"
: "${EMBEDDING_SNAPSHOT:?Set registered local embedding snapshot}"
through=${STECA_THROUGH:-all}
mode=${STECA_MODE:-train}
stage_kind=$through
if [[ "$mode" == evaluate ]]; then stage_kind=evaluate; fi
case "$stage_kind" in
    detect|reflect|datasets)
        [[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || {
            echo 'Teacher/data-only stages must be submitted without GPUs' >&2; exit 2;
        }
        export CUDA_VISIBLE_DEVICES=''
        ;;
    *)
        if [[ "$mode" != prepare ]]; then
            [[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]] || {
                echo 'This stage needs an explicit GPU allocation' >&2; exit 2;
            }
        fi
        ;;
esac
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1
NGPUS=${NGPUS:-4}
pipeline_python=$ENV_PYTHON
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$pipeline_python" -m baselines.steca.pipeline --config "${CONFIG:-configs/base.yaml}" \
    --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$STECA_OUT" \
    --embedding-snapshot "$EMBEDDING_SNAPSHOT" --env-python "$ENV_PYTHON" \
    --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON" \
    --gpus "$NGPUS" --workers "${WORKERS:-8}" --ready-timeout "${READY_TIMEOUT:-900}" \
    --mode "$mode" --through "$through" &
child=$!
wait "$child"
child=''
