#!/bin/bash
# 2026-10-19 batch 2 of the full baseline grid: the first cell (K20 s0) of each newly implemented baseline
# (docs/baselines/NEW_METHOD_SPECS_1019.md): SAD, WKM, STeCa, GAD, AgentDistill, Chain-of-Agents. HAD waits for its
# co-scheduled runner; AD-FTP waits for its teacher collection. Consecutive same-resource stages share one allocation
# through scripts/hpg_multi_stage.slurm (scheduling only; each stage runs through the method's own wrapper).
# These methods have no sacct history yet, so sizes are first estimates (to be tightened from this batch's sacct):
#   4-GPU full fine-tuning 8 CPU / 256G; 1-GPU serving/LoRA stages 4 CPU / 48G; teacher/data CPU stages 2 CPU / 8G.
# 4-GPU jobs use the dept account; 1-GPU and CPU jobs use the group account (fairshare 0.32 vs 0.22).
# Run from rai: ssh hpg bash -s -- <tree sha> [methods...] < scripts/ops_batch2_1019.sh
set -u
SHA=${1:?tree sha}; shift
METHODS=${*:-sad wkm steca gad agentdistill coa}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
BL=$B/reports/runs/baselines
NB=$B/reports/new_baselines   # AgentDistill/CoA require outputs outside reports/runs
SECRETS=/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env
EMB=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
LOG=$HOME/batch2_1019.log
V=$B/data/verified/seeds_k20_s0
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = "$SHA" ] || { echo "tree $R is not $SHA" >&2; exit 1; }
[ -d "$EMB" ] && [ -f "$V/manifest.json" ] && [ -r "$SECRETS" ] || { echo "missing inputs" >&2; exit 1; }
mkdir -p "$NB"
DEPT="--account=fsu-compsci-dept --qos=fsu-compsci-dept"
GRP="--account=yd24f.fsu --qos=yd24f.fsu"
G4="$DEPT --partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=256G"
G1="$GRP --partition=hpg-b200 --gpus=1 --cpus-per-task=4 --mem=48G"
C1="$GRP --partition=hpg-milan --gpus=0 --cpus-per-task=2 --mem=8G"
C4="$GRP --partition=hpg-milan --gpus=0 --cpus-per-task=4 --mem=16G"
# job name dependency resources walltime wrapper stage_var "stages" extra_exports [mode_var mode]
job() {
  local name=$1 dep=$2 res=$3 time=$4 wrapper=$5 var=$6 stages=$7 extra=$8 mvar=${9:-} mode=${10:-} d=""
  [ -n "$dep" ] && d="--dependency=afterok:$dep"
  local mexp=""; [ -n "$mvar" ] && mexp=",MODE_VAR=$mvar,MODE=$mode"
  sbatch --parsable $d $res --time="$time" --job-name="$name" --output="logs/slurm-$name-%j.out" \
    --export="ALL,PROJECT_ROOT=$R,WRAPPER=$wrapper,STAGE_VAR=$var,STAGES=$stages,VERIFIED=$V,SED_SECRETS_FILE=$SECRETS,WORKERS=16,BATCH_INVARIANT=1,$extra$mexp" \
    scripts/hpg_multi_stage.slurm
}
fresh() { [ ! -e "$1" ] || { echo "exists $1, skipped" >> "$LOG"; return 1; }; }
echo "=== batch2 $(date) tree $SHA methods=$METHODS" >> "$LOG"
for m in $METHODS; do case $m in
sad)
  O=$BL/sad-k20s0; fresh "$O" || continue; e="SAD_OUT=$O,NGPUS=4"
  j=$(job b2-sad-k20s0-prep "" "$C4" 02:00:00 scripts/hpg_sad.slurm SAD_THROUGH prepare "$e") &&
  j=$(job b2-sad-k20s0-train "$j" "$G4" 24:00:00 scripts/hpg_sad.slurm SAD_THROUGH train "$e") &&
  j=$(job b2-sad-k20s0-eval "$j" "$G1" 10:00:00 scripts/hpg_sad.slurm SAD_THROUGH all "$e" SAD_MODE evaluate) &&
  echo "sad k20s0 eval $j" >> "$LOG" ;;
wkm)
  O=$BL/wkm-k20s0; fresh "$O" || continue; e="WKM_OUT=$O,EMBEDDING_SNAPSHOT=$EMB,NGPUS=1"
  j=$(job b2-wkm-k20s0-g1 "" "$G1" 08:00:00 scripts/hpg_wkm.slurm WKM_THROUGH "experienced explore synthesize" "$e") &&
  j=$(job b2-wkm-k20s0-c1 "$j" "$C4" 02:00:00 scripts/hpg_wkm.slurm WKM_THROUGH "merge knowledge-base" "$e") &&
  j=$(job b2-wkm-k20s0-g2 "$j" "$G1" 08:00:00 scripts/hpg_wkm.slurm WKM_THROUGH "world agent" "$e") &&
  j=$(job b2-wkm-k20s0-eval "$j" "$G1" 08:00:00 scripts/hpg_wkm.slurm WKM_THROUGH all "$e" WKM_MODE evaluate) &&
  echo "wkm k20s0 eval $j" >> "$LOG" ;;
steca)
  O=$BL/steca-k20s0; fresh "$O" || continue; e="STECA_OUT=$O,EMBEDDING_SNAPSHOT=$EMB,NGPUS=4"
  j=$(job b2-steca-k20s0-sft "" "$G4" 04:00:00 scripts/hpg_steca.slurm STECA_THROUGH sft "$e") &&
  j=$(job b2-steca-k20s0-mc "$j" "$G1" 12:00:00 scripts/hpg_steca.slurm STECA_THROUGH "expert-mc explore agent-mc" "$e") &&
  j=$(job b2-steca-k20s0-c1 "$j" "$C1" 04:00:00 scripts/hpg_steca.slurm STECA_THROUGH "detect reflect datasets" "$e") &&
  j=$(job b2-steca-k20s0-rl "$j" "$G4" 06:00:00 scripts/hpg_steca.slurm STECA_THROUGH reinforced "$e") &&
  j=$(job b2-steca-k20s0-eval "$j" "$G1" 06:00:00 scripts/hpg_steca.slurm STECA_THROUGH all "$e" STECA_MODE evaluate) &&
  echo "steca k20s0 eval $j" >> "$LOG" ;;
gad)
  O=$BL/gad-k20s0; fresh "$O" || continue; e="GAD_OUT=$O,NGPUS=4"
  j=$(job b2-gad-k20s0-x0 "" "$G1" 03:00:00 scripts/hpg_gad.slurm GAD_THROUGH explore-0 "$e") &&
  j=$(job b2-gad-k20s0-t0 "$j" "$C1" 04:00:00 scripts/hpg_gad.slurm GAD_THROUGH teacher-0 "$e") &&
  j=$(job b2-gad-k20s0-g0 "$j" "$G4" 08:00:00 scripts/hpg_gad.slurm GAD_THROUGH "train-0 explore-1" "$e") &&
  j=$(job b2-gad-k20s0-t1 "$j" "$C1" 04:00:00 scripts/hpg_gad.slurm GAD_THROUGH teacher-1 "$e") &&
  j=$(job b2-gad-k20s0-g1 "$j" "$G4" 08:00:00 scripts/hpg_gad.slurm GAD_THROUGH "train-1 explore-2" "$e") &&
  j=$(job b2-gad-k20s0-t2 "$j" "$C1" 04:00:00 scripts/hpg_gad.slurm GAD_THROUGH teacher-2 "$e") &&
  j=$(job b2-gad-k20s0-g2 "$j" "$G4" 08:00:00 scripts/hpg_gad.slurm GAD_THROUGH train-2 "$e") &&
  j=$(job b2-gad-k20s0-eval "$j" "$G1" 10:00:00 scripts/hpg_gad.slurm GAD_THROUGH train-2 "$e" GAD_MODE evaluate) &&
  echo "gad k20s0 eval $j" >> "$LOG" ;;
agentdistill)
  O=$NB/agentdistill-k20s0; fresh "$O" || continue; e="AGENTDISTILL_OUT=$O"
  j=$(job b2-ad-k20s0-c "" "$C1" 08:00:00 scripts/hpg_agentdistill.slurm AGENTDISTILL_THROUGH "collect abstract cluster consolidate publish" "$e") &&
  j=$(job b2-ad-k20s0-eval "$j" "$G1" 06:00:00 scripts/hpg_agentdistill.slurm AGENTDISTILL_THROUGH evaluate "$e") &&
  echo "agentdistill k20s0 eval $j" >> "$LOG" ;;
coa)
  O=$NB/coa-k20s0; fresh "$O" || continue; e="COA_OUT=$O,NGPUS=4"
  rl=""; for r in $(seq 0 59); do rl="$rl explore-$r train-rl-$r"; done
  j=$(job b2-coa-k20s0-c "" "$C1" 12:00:00 scripts/hpg_coa.slurm COA_THROUGH "teacher data" "$e") &&
  j=$(job b2-coa-k20s0-g "$j" "$G4" 48:00:00 scripts/hpg_coa.slurm COA_THROUGH "train-sft query-filter$rl" "$e,STOP_JSON=$O/result.json") &&
  j=$(job b2-coa-k20s0-eval "$j" "$G1" 12:00:00 scripts/hpg_coa.slurm COA_THROUGH data "$e" COA_MODE evaluate) &&
  t=$(job b2-coa-k20s0-teval "" "$C1" 12:00:00 scripts/hpg_coa.slurm COA_THROUGH teacher "$e" COA_MODE teacher-evaluate) &&
  echo "coa k20s0 eval $j teacher-eval $t" >> "$LOG" ;;
*) echo "unknown method $m" >> "$LOG" ;;
esac; done
tail -n 12 "$LOG"
