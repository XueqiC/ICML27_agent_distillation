#!/bin/bash
#SBATCH --job-name=seedsft-eval
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48gb
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --output=logs/slurm-seedsft-eval-%A_%a.out
#SBATCH --signal=B:TERM@60
# 2026-10-05: batch-invariant dev140 eval of every checkpoint of one Seed-SFT sweep run (configs/seedsft_dose_grid.json),
# array task i <-> training task i. Submit with --dependency=aftercorr:<training array id>.
# Required: PROJECT_ROOT=<frozen tree>, TRAIN_ROOT=<dir with seedsft-<name>-hpg>, EVAL_ROOT=<output dir>, DATA_ROOT=<base tree>.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${TRAIN_ROOT:?}" "${EVAL_ROOT:?}" "${DATA_ROOT:?}"
read -r name verified < <("$ENV_PYTHON" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
run = json.load(open("configs/seedsft_dose_grid.json"))["runs"][int(sys.argv[1])]
print(run["name"], run["verified"])
PY
)
run_dir="$TRAIN_ROOT/seedsft-$name-hpg"
adapters=""
while read -r label tokens; do
    [[ -d "$run_dir/checkpoint-tokens-$tokens" ]] || { echo "missing checkpoint $tokens" >&2; exit 2; }
    adapters+="$name-$label=$run_dir/checkpoint-tokens-$tokens "
done < <("$ENV_PYTHON" - "$SLURM_ARRAY_TASK_ID" "$run_dir" <<'PY'
import json, sys, pathlib
run = json.load(open("configs/seedsft_dose_grid.json"))["runs"][int(sys.argv[1])]
saved = sorted(int(p.name.rsplit("-", 1)[1]) for p in pathlib.Path(sys.argv[2]).glob("checkpoint-tokens-*"))
for c in run["checkpoints"]:
    # Checkpoints land on the first complete episode at/above each threshold.
    actual = min(t for t in saved if t >= c["tokens"])
    print(c["label"], actual)
PY
)
export ADAPTERS="$adapters" VERIFIED="$DATA_ROOT/$verified" TASKS="$PROJECT_ROOT/data/manifest/dev140.json"
export OUT="$EVAL_ROOT/$name" WORKERS=16 BATCH_INVARIANT=1
export INCLUDE_BASE=$([[ "$SLURM_ARRAY_TASK_ID" == 0 ]] && echo 1 || echo 0)
echo "task $SLURM_ARRAY_TASK_ID $name base=$INCLUDE_BASE adapters: $ADAPTERS"
exec bash scripts/hpg_eval_adapters.slurm
