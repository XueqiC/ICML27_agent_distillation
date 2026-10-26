#!/bin/bash
#SBATCH --job-name=e4b-repair-train
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
set -euo pipefail
# Same torchrun/replicated-LoRA path as hpg_kang; E4B grid resource line.
# Require caller's OFF_HOME locations before hpg.env can supply defaults.
: "${PROJECT_ROOT:?Set this worktree}" "${MODEL_PATH:?Set the pinned E4B snapshot}"
: "${TRAIN_PYTHON:?Set the grid training runtime}" "${CONFIG:?Set configs/base_e4b.yaml}"
: "${ALFWORLD_DATA:?Set OFF_HOME cache}" "${VLLM_CACHE_ROOT:?Set OFF_HOME cache}"
: "${TRITON_CACHE_DIR:?Set OFF_HOME cache}" "${HF_HOME:?Set OFF_HOME cache}"
: "${XDG_CACHE_HOME:?Set OFF_HOME cache}" "${SED_HASH_CACHE_DIR:?Set OFF_HOME cache}"
: "${THETA0:?Set theta0}" "${REPAIR_COLLECTION:?Set e2_training_collection.json}"
: "${ARM:?N or F}" "${REPAIR_OUT:?Set a fresh output directory}"
[[ "$ARM" == N || "$ARM" == F ]]
[[ ! -e "$REPAIR_OUT" && ! -L "$REPAIR_OUT" ]] || { echo 'Output already exists' >&2; exit 2; }
source "${HPG_CONFIG:-${PROJECT_ROOT}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8
export BATCH_INVARIANT=${BATCH_INVARIANT:-1} VLLM_BATCH_INVARIANT=1
[[ "$BATCH_INVARIANT" == 1 && "$ENABLE_THINKING" == off && "$MAX_TOKENS" == 1024 && "$MAX_LENGTH" == 32768 ]]
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" ]]
extra=()
if [[ -n "${REPAIR_PATH_MAP:-}" ]]; then extra+=(--path-map "$REPAIR_PATH_MAP"); fi
child=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
setsid "$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node 2 -m tools.repair_train_arm \
    --config "$CONFIG" --snapshot "$MODEL_PATH" --theta0 "$THETA0" --collection "$REPAIR_COLLECTION" \
    --arm "$ARM" --output "$REPAIR_OUT" --repair-splits "${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}" \
    "${extra[@]}" &
child=$!
wait "$child"
child=''
