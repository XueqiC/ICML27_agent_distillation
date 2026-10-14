#!/bin/bash
#SBATCH --job-name=t1-smoke
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=slurm-%j.out
# T1 GPU smoke test (prereg 2026-10-14): every train.transplant subcommand on the real model with tiny budgets.
# Blocks of 4 updates (B1 contains one replacement update), compose all three arms, a 2+-update post on the
# composed state, scores of a composed and a natural full state. Writes only under T1_SMOKE_OUT.
set -euo pipefail
: "${T1_Z10:?}" "${T1_EXTRA:?}" "${T1_EXTRA_STATES:?}" "${T1_SURR:?}" "${T1_SMOKE_OUT:?}" "${VERIFIED:?}"
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# Sizing (sacct 10-01..10-14, 20 jobs): ~1.0 busy core, peak RSS ~23 GB (gemma-4-12B weights). OMP_NUM_THREADS stays 6,
# independent of the allocation, so CPU-side numerics match the earlier 6-CPU runs (R1 CRN pairing, T1 arms).
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -d "$T1_Z10" && -f "$T1_EXTRA_STATES" && -f "$T1_SURR" ]]
[[ ! -e "$T1_SMOKE_OUT" ]]
mkdir -p "$T1_SMOKE_OUT"
O=$T1_SMOKE_OUT
common=(--snapshot "$MODEL_PATH" --verified "$VERIFIED" --device cuda)
t() { local s=$SECONDS; "$@"; echo "elapsed_seconds=$((SECONDS - s)) :: $2 $3"; }
t "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" --fraction 0 --updates 4 --output "$O/B0"
t "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" --extra "$T1_EXTRA" \
    --extra-states "$T1_EXTRA_STATES" --fraction 0.25 --seed 1202 --updates 4 --output "$O/B1"
"$TRAIN_PYTHON" -c "import json,sys; b=json.load(open('$O/B1/block.json')); assert b['block_replacement_updates']==1, b"
B0=$O/B0/state-update-119; B1=$O/B1/state-update-119
for arm in T10:B1:B0 T01:B0:B1 T11:B1:B1; do
    IFS=: read -r name m v <<< "$arm"
    mm=$B0; [[ $m == B1 ]] && mm=$B1
    vv=$B0; [[ $v == B1 ]] && vv=$B1
    t "$TRAIN_PYTHON" -m train.transplant compose --theta "$B0" --m "$mm" --v "$vv" --output "$O/S_$name"
done
t "$TRAIN_PYTHON" -m train.transplant post "${common[@]}" --checkpoint "$O/S_T11" --plan-from "$T1_Z10" \
    --target-tokens 112000 --output "$O/T11"
t "$TRAIN_PYTHON" -m train.transplant score "${common[@]}" --checkpoint "$O/S_T11" --surrogate "$T1_SURR" --output "$O/score_S_T11"
t "$TRAIN_PYTHON" -m train.transplant score "${common[@]}" --checkpoint "$B1" --surrogate "$T1_SURR" --output "$O/score_B1"
cat "$O/T11/post.json"
echo SMOKE_OK
