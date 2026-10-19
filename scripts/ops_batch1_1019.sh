#!/bin/bash
# 2026-10-19 batch 1 of the full baseline grid (docs/direction/2026-10-19_full_baseline_grid_plan_ZH.md; user grid
# K in {5,10,20,50,100} x s0-s2 plus full training, 10-19 13:30 CDT): cells that need no new code.
#   AgentGym BC/AgentEvol: K50/K100 x s1,s2 (K5-K20 x s0-s2 and K50/K100 s0 are done).
#   LEAP privileged:       K5-K100 x s1,s2 (s0 is done).
#   LEAP 0% privileged:    K5-K100 x s0-s2 except k20s0 (done).
# Stage-split chains (afterok): GPU stages request only the GPUs they use; teacher-only LEAP stages run CPU-only on
# hpg-milan, so no GPU idles while the teacher answers. Sizes from sacct of the same stages (10-07/10-10 runs):
#   AgentGym train 2 GPU, RSS <= 137 GB, ~2 cores -> 4 CPU / 160G; explore+filter 1 GPU, RSS 8 GB -> 4 CPU / 24G.
#   LEAP GPU stage RSS 22-26 GB, ~1 core -> 4 CPU / 32G; LEAP teacher stage RSS 0.3 GB -> 2 CPU / 8G; eval 8 GB -> 4/24G.
# (10-19 13:5x CDT: pending batch-1 jobs were shrunk to these values with scontrol; the defaults below match.)
# Every stage resumes from its hash-bound completed stages, so a failed or timed-out link is resubmitted unchanged.
# Run from rai: ssh hpg bash -s -- <tree sha> [agentgym|leap-privileged|leap-zero|all] < scripts/ops_batch1_1019.sh
set -u
SHA=${1:?tree sha}
WHAT=${2:-all}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
BL=$B/reports/runs/baselines
SECRETS=/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env
LOG=$HOME/batch1_1019.log
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = "$SHA" ] || { echo "tree $R is not $SHA" >&2; exit 1; }
[ -d third_party/leap_llm/prompts/alfworld ] || { echo "LEAP templates missing in $R" >&2; exit 1; }
[ -r "$SECRETS" ] || { echo "no teacher secrets file" >&2; exit 1; }
ACC="--account=fsu-compsci-dept --qos=fsu-compsci-dept"
verified() {  # nested seed ladder, same as scripts/hpg_baseline_cells.slurm
  case "$1" in
    20|100) echo "$B/data/verified/seeds_k$1_s$2" ;;
    5|10) echo "$B/data/verified/seeds_k20_s$2/subsets/k$1" ;;
    50) echo "$B/data/verified/seeds_k100_s$2/subsets/k50" ;;
  esac
}
sub() {  # name dependency resources walltime script exports -> job id
  local d=""
  [ -n "$2" ] && d="--dependency=afterok:$2"
  sbatch --parsable $ACC $d $3 --time="$4" --job-name="$1" --output="logs/slurm-$1-%j.out" \
    --export="ALL,PROJECT_ROOT=$R,$6" "$5"
}
gpu_hours() {  # walltime of one LEAP GPU link by K (measured: K100 longest link 8.4 h, K50 5.5 h, K20 4.8 h whole run)
  case "$1" in 5|10) echo 01:30:00 ;; 20) echo 02:30:00 ;; 50) echo 07:00:00 ;; 100) echo 12:00:00 ;; esac
}
ag_train_hours() {  # AgentGym train link (measured K100 t2 0:51, t3 0:57)
  case "$1" in 50) echo 02:30:00 ;; 100) echo 03:30:00 ;; *) echo 02:00:00 ;; esac
}
agentgym_chain() {
  local K=$1 S=$2 cell="k$1s$2" V O e j="" m
  V=$(verified "$K" "$S"); O=$BL/agentgym-$cell
  [ -f "$V/manifest.json" ] || { echo "missing $V" >> "$LOG"; return; }
  [ ! -e "$O" ] || { echo "exists $O, skipped" >> "$LOG"; return; }
  e="VERIFIED=$V,AGENTGYM_OUT=$O,NGPUS=2,WORKERS=16,BATCH_INVARIANT=1"
  for m in 0 1 2 3; do
    j=$(sub "ag-$cell-t$m" "$j" "--gpus=2 --cpus-per-task=4 --mem=160G" "$(ag_train_hours "$K")" scripts/hpg_agentgym.slurm \
        "$e,AGENTGYM_THROUGH=train-$m") || return
    j=$(sub "ag-$cell-x$m" "$j" "--gpus=1 --cpus-per-task=4 --mem=24G" 01:30:00 scripts/hpg_agentgym.slurm \
        "$e,AGENTGYM_THROUGH=filter-$m") || return
  done
  j=$(sub "ag-$cell-eval" "$j" "--gpus=1 --cpus-per-task=4 --mem=24G" 04:00:00 scripts/hpg_agentgym.slurm \
      "$e,AGENTGYM_MODE=evaluate") || return
  echo "agentgym $cell submitted, eval $j" >> "$LOG"
}
leap_chain() {
  local var=$1 K=$2 S=$3 cell="k$2s$3" V O e j first i
  V=$(verified "$K" "$S"); O=$BL/leap-$var-$cell
  [ -f "$V/manifest.json" ] || { echo "missing $V" >> "$LOG"; return; }
  [ ! -e "$O" ] || { echo "exists $O, skipped" >> "$LOG"; return; }
  e="LEAP_OUT=$O,LEAP_VARIANT=$var,VERIFIED=$V,SED_SECRETS_FILE=$SECRETS,WORKERS=16,BATCH_INVARIANT=1"
  local cpu="--partition=hpg-milan --gpus=0 --cpus-per-task=2 --mem=8G" gpu="--gpus=1 --cpus-per-task=4 --mem=32G"
  first=data-0; [ "$var" = privileged ] && first=extract
  j=$(sub "lp${var:0:1}-$cell-c0" "" "$cpu" 04:00:00 scripts/hpg_leap.slurm "$e,LEAP_THROUGH=$first") || return
  for i in 1 2 3; do
    j=$(sub "lp${var:0:1}-$cell-g$i" "$j" "$gpu" "$(gpu_hours "$K")" scripts/hpg_leap.slurm "$e,LEAP_THROUGH=explore-$i") || return
    j=$(sub "lp${var:0:1}-$cell-c$i" "$j" "$cpu" 02:00:00 scripts/hpg_leap.slurm "$e,LEAP_THROUGH=data-$i") || return
  done
  j=$(sub "lp${var:0:1}-$cell-g4" "$j" "$gpu" "$(gpu_hours "$K")" scripts/hpg_leap.slurm "$e,LEAP_THROUGH=all") || return
  j=$(sub "lp${var:0:1}-$cell-eval" "$j" "--gpus=1 --cpus-per-task=4 --mem=24G" 03:00:00 scripts/hpg_leap.slurm \
      "$e,LEAP_MODE=evaluate") || return
  echo "leap-$var $cell submitted, eval $j" >> "$LOG"
}
echo "=== batch1 $(date) tree $SHA what=$WHAT" >> "$LOG"
if [ "$WHAT" = agentgym ] || [ "$WHAT" = all ]; then
  for K in 50 100; do for S in 1 2; do agentgym_chain "$K" "$S"; done; done
fi
if [ "$WHAT" = leap-privileged ] || [ "$WHAT" = all ]; then
  for K in 5 10 20 50 100; do for S in 1 2; do leap_chain privileged "$K" "$S"; done; done
fi
if [ "$WHAT" = leap-zero ] || [ "$WHAT" = all ]; then
  for K in 5 10 20 50 100; do for S in 0 1 2; do
    [ "$K$S" = 200 ] && continue
    leap_chain zero "$K" "$S"
  done; done
fi
tail -n 40 "$LOG"
