#!/bin/bash
#SBATCH --job-name=r03-teacher
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-milan
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# Sizing (sacct): CPU-only API client, 0.03-0.07 busy cores, peak RSS 0.9 GB.
# R1/R2 teacher collection (capped, cached, stop on non-200). CPU only.
# Required: R03_MODE (r1|r2), R03_ROLLOUTS, R03_OUTDIR, SED_SECRETS_FILE. r1 also: R03_SELECTION, R03_K20. r2: R03_SELECTION.
# Optional: R03_RUN_TAG, appended to every run ID. A run whose ledger holds a failed call stays halted by design
# (teachers/capped.py); a retry after fixing the cause (e.g. a rotated key) gets a new tag, never a reset counter.
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
: "${R03_RUN_TAG:=}"
: "${R03_MODE:?}" "${R03_ROLLOUTS:?}" "${R03_OUTDIR:?}" "${R03_SELECTION:?}" "${SED_SECRETS_FILE:?}"
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
CACHE=$B/data/raw/r03_teacher_cache; LEDGER=$B/data/raw/r03_teacher_usage.jsonl
mkdir -p "$R03_OUTDIR"
if [[ "$R03_MODE" == r1 ]]; then
  : "${R03_K20:?}"
  [[ -e "$R03_OUTDIR/p2" ]] || "$ENV_PYTHON" tools/collect_corrections.py --source R1_P2 --run-id r03-r1-p2${R03_RUN_TAG} --cap 600 \
      --rollouts "$R03_ROLLOUTS" --selection "$R03_SELECTION" --cache "$CACHE" --usage-ledger "$LEDGER" --output "$R03_OUTDIR/p2"
  [[ -e "$R03_OUTDIR/p4" ]] || "$ENV_PYTHON" tools/collect_rewrites.py --run-id r03-r1-p4${R03_RUN_TAG} --cap 40 --k20 "$R03_K20" \
      --cache "$CACHE" --usage-ledger "$LEDGER" --output "$R03_OUTDIR/p4"
elif [[ "$R03_MODE" == r2 ]]; then
  [[ -e "$R03_OUTDIR/r2_union" ]] || "$ENV_PYTHON" tools/collect_corrections.py --source R2 --run-id r03-r2-union${R03_RUN_TAG} --cap 1500 \
      --rollouts "$R03_ROLLOUTS" --selection "$R03_SELECTION" --cache "$CACHE" --usage-ledger "$LEDGER" --output "$R03_OUTDIR/r2_union"
else
  echo "bad R03_MODE" >&2; exit 2
fi
