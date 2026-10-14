#!/bin/bash
#SBATCH --job-name=t1-summary
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-milan
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
# T1 (prereg 776819b, addendum 2): the last stage of hpg_t1_transplant.slurm (post-plan check and summary) after the
# parallel hpg_t1_stage.slurm jobs. CPU only: it reads JSON files and hashes them.
set -euo pipefail
: "${T1_OUT:?}"
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
[[ -x "$TRAIN_PYTHON" && -d "$T1_OUT" && ! -e "$T1_OUT/summary.json" ]]
"$TRAIN_PYTHON" - "$T1_OUT" <<'PY'
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
print('verify_post_plans ok')
PY
"$TRAIN_PYTHON" - "$T1_OUT" <<'PY'
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
