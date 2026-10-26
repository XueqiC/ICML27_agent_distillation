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
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/scripts/repair_stage_env.sh"
: "${REPAIR_COLLECTION:?Set collection.json}" "${ARM:?shared/N/F}" "${CHECKPOINT:?0/1/5/20}"
extra=()
if [[ "$CHECKPOINT" != 0 ]]; then
    : "${TRAINING_MANIFEST:?Set the arm manifest.json}"
    extra+=(--training-manifest "$TRAINING_MANIFEST")
    if [[ -n "${TRAINING_COLLECTION:-}" ]]; then extra+=(--training-collection "$TRAINING_COLLECTION"); fi
fi
repair_work() {
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.repair_probes --config "$CONFIG" \
        --collection "$REPAIR_COLLECTION" --arm "$ARM" --checkpoint "$CHECKPOINT" "${extra[@]}" \
        --repair-splits "${REPAIR_SPLITS:-data/manifest/repair_v01_splits.json}" \
        --endpoint "$1" --server-provenance "$2" --lora-check "$3" --model-name "$MODEL_NAME" \
        --workers "${WORKERS:-4}" --output "$REPAIR_OUT/probes"
}
source scripts/repair_serve_stage.sh
