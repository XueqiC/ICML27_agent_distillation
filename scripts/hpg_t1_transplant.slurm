#!/bin/bash
#SBATCH --job-name=t1-transplant
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:30:00
#SBATCH --output=slurm-%j.out
# Deploy to a new tree. Submit only after CPU and authorized GPU smoke checks.
# No automatic resubmission. T1_EXTRA=r2_union; T1_EXTRA_STATES=allocations_scalefree/A3_n100.json.
set -euo pipefail
: "${T1_Z10:?}" "${T1_EXTRA:?}" "${T1_EXTRA_STATES:?}" "${T1_SURR:?}" "${T1_A0:?}" "${T1_OUT:?}" "${VERIFIED:?}"
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" && -f "$VERIFIED/manifest.json" ]]
[[ -d "$T1_Z10" && -d "$T1_A0" && -f "$T1_EXTRA/manifest.json" && -f "$T1_EXTRA_STATES" && -f "$T1_SURR" ]]
[[ ! -e "$T1_OUT" && ! -L "$T1_OUT" ]]
"$TRAIN_PYTHON" - "$T1_Z10" <<'PY'
import json, sys
from pathlib import Path
state = json.loads((Path(sys.argv[1]) / 'training_state.json').read_text())
assert state['updates'] == 115 and state['supervised_tokens'] == 104070, 'T1 requires the preregistered z10'
PY
mkdir -p "$(dirname "$T1_OUT")"
mkdir "$T1_OUT"
stage() {
    local name=$1 started=$SECONDS status=0
    shift
    "$@" || status=$?
    printf '%s elapsed_seconds=%s status=%s\n' "$name" "$((SECONDS - started))" "$status"
    return "$status"
}
common=(--snapshot "$MODEL_PATH" --verified "$VERIFIED" --device cuda)
stage B0 "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" \
    --fraction 0 --updates 100 --target-tokens 416280 --output "$T1_OUT/B0"
stage B1 "$TRAIN_PYTHON" -m train.transplant block "${common[@]}" --checkpoint "$T1_Z10" \
    --extra "$T1_EXTRA" --extra-states "$T1_EXTRA_STATES" --fraction 0.25 --seed 1202 \
    --updates 100 --target-tokens 416280 --output "$T1_OUT/B1"
"$TRAIN_PYTHON" - "$T1_OUT" <<'PY'
import json, sys
from pathlib import Path
for name, count in [('B0', 0), ('B1', 25)]:
    block = json.loads((Path(sys.argv[1]) / name / 'block.json').read_text())
    assert block['block_end_update'] == 215 and block['block_replacement_updates'] == count
PY
B0="$T1_OUT/B0/state-update-215"
B1="$T1_OUT/B1/state-update-215"
stage compose_T10 "$TRAIN_PYTHON" -m train.transplant compose --theta "$B0" --m "$B1" --v "$B0" --output "$T1_OUT/S_T10"
stage compose_T01 "$TRAIN_PYTHON" -m train.transplant compose --theta "$B0" --m "$B0" --v "$B1" --output "$T1_OUT/S_T01"
stage compose_T11 "$TRAIN_PYTHON" -m train.transplant compose --theta "$B0" --m "$B1" --v "$B1" --output "$T1_OUT/S_T11"
for arm in T00 T00p T10 T01 T11 N1; do
    case "$arm" in
        T00|T00p) state=$B0 ;;
        N1) state=$B1 ;;
        *) state="$T1_OUT/S_$arm" ;;
    esac
    stage "post_$arm" "$TRAIN_PYTHON" -m train.transplant post "${common[@]}" \
        --checkpoint "$state" --plan-from "$T1_Z10" --target-tokens 416280 --output "$T1_OUT/$arm"
done
stage verify_post_plans "$TRAIN_PYTHON" - "$T1_OUT" <<'PY'
import json, sys
from pathlib import Path
from sed.storage import file_hash
root = Path(sys.argv[1])
hashes = set()
for arm in ('T00', 'T00p', 'T10', 'T01', 'T11', 'N1'):
    sha = file_hash(root / arm / 'post_plan.json')
    assert sha == json.loads((root / arm / 'post.json').read_text())['post_plan_file_sha256']
    hashes.add(sha)
assert len(hashes) == 1, 'T1 post plans differ across arms'
PY
for arm in T00 T00p T10 T01 T11 N1 A0; do
    if [[ "$arm" == A0 ]]; then
        checkpoint=$T1_A0
    else
        checkpoint=$("$TRAIN_PYTHON" - "$T1_OUT/$arm" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(root / json.loads((root / 'profiling_report.json').read_text())['checkpoints'][-1]['path'])
PY
)
    fi
    stage "score_$arm" "$TRAIN_PYTHON" -m train.transplant score "${common[@]}" \
        --checkpoint "$checkpoint" --surrogate "$T1_SURR" --output "$T1_OUT/scores/$arm"
done
stage summary "$TRAIN_PYTHON" - "$T1_OUT" <<'PY'
import json, sys
from pathlib import Path
from sed.storage import file_hash, write_json
from train.transplant import provenance_for, summarize_scores
root = Path(sys.argv[1])
arms = ('T00', 'T00p', 'T10', 'T01', 'T11', 'N1', 'A0')
scores = {arm: json.loads((root / 'scores' / arm / 'score.json').read_text()) for arm in arms}
summary = summarize_scores(scores)
summary['post_plan_file_sha256'] = file_hash(root / 'T00' / 'post_plan.json')
summary['score_sha256'] = {arm: file_hash(root / 'scores' / arm / 'score.json') for arm in arms}
provenance_for(root, 'transplant summary', {'arms': list(arms), 'threshold_floor': 0.0012},
               [root / 'scores', *[root / arm / 'post.json' for arm in arms if arm != 'A0'],
                *[root / arm / 'post_plan.json' for arm in arms if arm != 'A0']])
write_json(root / 'summary.json', summary, exclusive=True)
print(json.dumps(summary, allow_nan=False))
PY
