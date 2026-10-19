#!/bin/bash
#SBATCH --job-name=multi-stage
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@120
# 2026-10-19: run several consecutive stages of ONE method in ONE allocation, so a cell needs a few SLURM jobs
# instead of one per stage (queue waits on hpg-b200 are hours per job). Every stage is still invoked through the
# method's own wrapper exactly as a separate job would be (same environment variables, same hash-bound stage
# transactions), so stage outputs are identical; only scheduling changes. Resources come from the sbatch command line.
# Required: PROJECT_ROOT, WRAPPER (e.g. scripts/hpg_wkm.slurm), STAGE_VAR (e.g. WKM_THROUGH), STAGES ("a b c").
# Optional: MODE_VAR + MODE (e.g. WKM_MODE=evaluate for an evaluate-only job), STOP_JSON (stop the loop once the
# method has written this run-level result file, whatever its status).
set -uo pipefail
: "${PROJECT_ROOT:?}" "${WRAPPER:?}" "${STAGE_VAR:?}" "${STAGES:?}"
cd "$PROJECT_ROOT" || exit 1
child=''
forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 143; }
trap forward TERM
for stage in $STAGES; do
  if [[ -n "${STOP_JSON:-}" && -f "$STOP_JSON" ]]; then  # the method writes its run-level result only when it ends
    echo "[$(date -u +%FT%TZ)] $STOP_JSON exists ($(head -c 200 "$STOP_JSON")); stopping before $stage"; break
  fi
  echo "[$(date -u +%FT%TZ)] stage $stage"
  if [[ -n "${MODE_VAR:-}" ]]; then
    env "$STAGE_VAR=$stage" "$MODE_VAR=${MODE:?}" bash "$WRAPPER" & child=$!
  else
    env "$STAGE_VAR=$stage" bash "$WRAPPER" & child=$!
  fi
  status=0; wait "$child" || status=$?; child=''
  if [[ $status != 0 ]]; then echo "[$(date -u +%FT%TZ)] stage $stage failed with $status"; exit "$status"; fi
done
echo "[$(date -u +%FT%TZ)] all stages done"
