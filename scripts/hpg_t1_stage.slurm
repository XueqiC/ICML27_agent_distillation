#!/bin/bash
#SBATCH --job-name=t1-stage
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --output=slurm-%j.out
# T1 (prereg 776819b, addendum 2): one stage of hpg_t1_transplant.slurm as its own job, so stages run in parallel.
#   T1_STAGE=block_B0 | block_B1 | score_A0 | arm (with T1_ARM = T00 T00p T10 T01 T11 N1)
# Commands and arguments are those of hpg_t1_transplant.slurm; the summary is hpg_t1_summary.slurm.
# Sizing (sacct 10-01..10-14, 20 jobs): ~1.0 busy core, peak RSS ~23 GB (gemma-4-12B weights). OMP_NUM_THREADS stays 6,
# independent of the allocation, so CPU-side numerics match the earlier 6-CPU runs (R1 CRN pairing, T1 arms).
# Time: block_B0 ~12 min, block_B1 ~80 min (25 extra-pool updates at ~165 s), arm ~30 min, score_A0 ~5 min;
# pass --time per stage at submission.
set -euo pipefail
: "${T1_STAGE:?}" "${T1_Z10:?}" "${T1_SURR:?}" "${T1_OUT:?}" "${VERIFIED:?}"
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" ]]
[[ -d "$T1_Z10" && -f "$T1_SURR" ]]
"$TRAIN_PYTHON" - "$T1_Z10" <<'PY'
import json, sys
from pathlib import Path
state = json.loads((Path(sys.argv[1]) / 'training_state.json').read_text())
assert state['updates'] == 115 and state['supervised_tokens'] == 104070, 'T1 requires the preregistered z10'
PY
mkdir -p "$T1_OUT"
stage() {
    local name=$1 started=$SECONDS status=0
    shift
    "$@" || status=$?
    printf '%s elapsed_seconds=%s status=%s\n' "$name" "$((SECONDS - started))" "$status"
    return "$status"
}
check_block() {  # name, replacement updates expected by the preregistration
    "$TRAIN_PYTHON" - "$T1_OUT" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
block = json.loads((Path(sys.argv[1]) / sys.argv[2] / 'block.json').read_text())
assert block['block_end_update'] == 215 and block['block_replacement_updates'] == int(sys.argv[3]), block
PY
}
common=(--snapshot "$MODEL_PATH" --verified "$VERIFIED" --device cuda)
B0="$T1_OUT/B0/state-update-215"
B1="$T1_OUT/B1/state-update-215"
score() {  # arm, checkpoint
    stage "score_$1" "$TRAIN_PYTHON" -m train.transplant score "${common[@]}" \
        --checkpoint "$2" --surrogate "$T1_SURR" --output "$T1_OUT/scores/$1"
}
case "$T1_STAGE" in
    block_B0)
        stage B0 "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" \
            --fraction 0 --updates 100 --target-tokens 416280 --output "$T1_OUT/B0"
        check_block B0 0 ;;
    block_B1)
        : "${T1_EXTRA:?}" "${T1_EXTRA_STATES:?}"
        [[ -f "$T1_EXTRA/manifest.json" && -f "$T1_EXTRA_STATES" ]]
        stage B1 "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" \
            --extra "$T1_EXTRA" --extra-states "$T1_EXTRA_STATES" --fraction 0.25 --seed 1202 \
            --updates 100 --target-tokens 416280 --output "$T1_OUT/B1"
        check_block B1 25 ;;
    score_A0)
        : "${T1_A0:?}"
        [[ -d "$T1_A0" ]]
        score A0 "$T1_A0" ;;
    arm)
        : "${T1_ARM:?}"
        case "$T1_ARM" in
            T00|T00p) check_block B0 0; state=$B0 ;;
            N1) check_block B1 25; state=$B1 ;;
            T10|T01|T11)
                check_block B0 0; check_block B1 25
                m=$B0; [[ "$T1_ARM" == T10 || "$T1_ARM" == T11 ]] && m=$B1
                v=$B0; [[ "$T1_ARM" == T01 || "$T1_ARM" == T11 ]] && v=$B1
                state="$T1_OUT/S_$T1_ARM"
                stage "compose_$T1_ARM" "$TRAIN_PYTHON" -m train.transplant compose \
                    --theta "$B0" --m "$m" --v "$v" --output "$state" ;;
            *) echo "unknown T1_ARM=$T1_ARM" >&2; exit 2 ;;
        esac
        stage "post_$T1_ARM" "$TRAIN_PYTHON" -m train.transplant post "${common[@]}" \
            --checkpoint "$state" --plan-from "$T1_Z10" --target-tokens 416280 --output "$T1_OUT/$T1_ARM"
        checkpoint=$("$TRAIN_PYTHON" - "$T1_OUT/$T1_ARM" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(root / json.loads((root / 'profiling_report.json').read_text())['checkpoints'][-1]['path'])
PY
)
        score "$T1_ARM" "$checkpoint" ;;
    *) echo "unknown T1_STAGE=$T1_STAGE" >&2; exit 2 ;;
esac
