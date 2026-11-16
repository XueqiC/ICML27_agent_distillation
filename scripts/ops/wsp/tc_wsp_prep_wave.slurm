#!/bin/bash
#SBATCH --job-name=wsp-prep
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# tc_wsp_prep_wave.slurm — CPU preparation (sealed encoding) for a list of Kang-SFT cells, one after another.
# Each cell runs scripts/hpg_kang_wsp.slurm with KANG_MODE=prepare and exactly the variables its train job will use.
# Needs: WSP_QUEUE (queue dir), PREP_LIST (file with cell names), PROJECT_ROOT (code tree). Marker per cell: done/prep.<cell>.
set -u
: "${WSP_QUEUE:?}" "${PREP_LIST:?}" "${PROJECT_ROOT:?}"
cd "$PROJECT_ROOT" || exit 2
mkdir -p "$WSP_QUEUE/done" "$WSP_QUEUE/logs"
fail=0
while IFS= read -r name <&7; do
  [ -n "$name" ] || continue
  envf="$WSP_QUEUE/jobs/prep.$name.env"
  [ -f "$envf" ] || { echo "PREP_NOENV $name"; fail=1; continue; }
  if [ -e "$WSP_QUEUE/done/prep.$name" ]; then echo "PREP_SKIP $name"; continue; fi
  echo "$(date '+%F %T') PREP_START $name"
  ( set -a; . "$envf"; set +a; export KANG_MODE=prepare; exec bash scripts/hpg_kang_wsp.slurm ) > "$WSP_QUEUE/logs/prep.$name.${SLURM_JOB_ID:-manual}.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" -eq 0 ]; then : > "$WSP_QUEUE/done/prep.$name"; echo "$(date '+%F %T') PREP_OK $name"
  else echo "$(date '+%F %T') PREP_FAIL $name rc=$rc"; fail=1; fi
done 7< "$PREP_LIST"
if [ "$fail" -eq 0 ]; then echo "PREP_WAVE_DONE"; else echo "PREP_WAVE_FAILED"; exit 1; fi
