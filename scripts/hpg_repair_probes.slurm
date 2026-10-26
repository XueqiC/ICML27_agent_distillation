#!/bin/bash
#SBATCH --job-name=e4b-repair-probes
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@60
# reports/runs/eval/chain/e4b-repair-<run>/<arm>-u<checkpoint>, fresh path per stage.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/repair_stage_env.sh"
: "${REPAIR_COLLECTION:?Set collection.json}" "${ARM:?shared/N/F}" "${CHECKPOINT:?0/1/5/20}"
extra=()
collection_branch=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'from repair.artifacts import checked_document; from repair.branches import collection_branch; import sys; print(collection_branch(checked_document(sys.argv[1]))["branch"])' "$REPAIR_COLLECTION")
if [[ "$collection_branch" == P1_U_mech ]]; then
    REPAIR_SUPPORT_THETA0=${THETA0:-$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'from repair.artifacts import checked_document; import sys; print(checked_document(sys.argv[1])["checkpoint"]["identity"]["path"])' "$REPAIR_COLLECTION")}
fi
SERVE_BASE=0
if [[ "$CHECKPOINT" == 0 ]]; then
    theta0_kind=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'from repair.artifacts import checked_document; from repair.branches import collection_branch; import sys; print(collection_branch(checked_document(sys.argv[1]))["theta0_kind"])' "$REPAIR_COLLECTION")
    if [[ "$theta0_kind" == base_model ]]; then
        SERVE_BASE=1
        MODEL_NAME=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -c 'from sed.config import load_config; import sys; print(load_config(sys.argv[1])["models"]["student"]["requested_repo"])' "$CONFIG")
    fi
fi
if [[ "$CHECKPOINT" != 0 ]]; then
    : "${TRAINING_MANIFEST:?Set the arm manifest.json}"
    extra+=(--training-manifest "$TRAINING_MANIFEST")
    if [[ -n "${TRAINING_COLLECTION:-}" ]]; then extra+=(--training-collection "$TRAINING_COLLECTION"); fi
fi
repair_work() {
    local check_args=()
    if [[ -n "$3" ]]; then check_args+=(--lora-check "$3"); fi
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.repair_probes --config "$CONFIG" \
        --collection "$REPAIR_COLLECTION" --arm "$ARM" --checkpoint "$CHECKPOINT" "${extra[@]}" \
        --repair-splits "${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}" \
        --endpoint "$1" --server-provenance "$2" "${check_args[@]}" --model-name "$MODEL_NAME" \
        --workers "${WORKERS:-4}" --output "$REPAIR_OUT/probes"
}
source scripts/repair_serve_stage.sh
