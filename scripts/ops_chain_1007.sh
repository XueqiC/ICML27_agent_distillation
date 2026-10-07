#!/bin/bash
# 2026-10-07: stage-split chains so no job holds an idle GPU (UFRC policy). Same run identity (NGPUS=2 passed to every
# segment); FSDP stages get 2 GPUs, single-model stages (explore / MC / eval) get 1 GPU. Each segment resumes completed
# stages and stops with --through. Segments are chained with afterok.
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/6bac2da
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
LOG=$HOME/chain_1007.log
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = 6bac2da ] || exit 1
two="--gpus=2 --cpus-per-task=16 --mem=160G"
one="--gpus=1 --cpus-per-task=14 --mem=96G"
sub() {  # name dep res time script exports
  local name=$1 dep=$2 res=$3 time=$4 script=$5 exports=$6 d=""
  [ -n "$dep" ] && d="--dependency=afterok:$dep"
  sbatch --parsable $d $res --time=$time --job-name=$name --output=logs/slurm-$name-%j.out \
         --export=ALL,PROJECT_ROOT=$R,NGPUS=2,VERIFIED=$V,WORKERS=16,BATCH_INVARIANT=1,$exports $script
}
# ETO K20 s0 (fresh: eto-k20s0-tp6)
EO=$B/reports/runs/baselines/eto-k20s0-tp6; [ ! -e "$EO" ] || { echo "exists $EO" >> $LOG; exit 1; }
E="ETO_OUT=$EO"
j=$(sub eto-tp6-sft "" "$two" 03:00:00 scripts/hpg_eto.slurm "$E,ETO_THROUGH=sft"); echo "eto sft $j" >> $LOG
for i in 1 2 3; do
  j=$(sub eto-tp6-x$i "$j" "$one" 03:00:00 scripts/hpg_eto.slurm "$E,ETO_THROUGH=pairs-$i"); echo "eto explore+pairs-$i $j" >> $LOG
  j=$(sub eto-tp6-dpo$i "$j" "$two" 08:00:00 scripts/hpg_eto.slurm "$E,ETO_THROUGH=dpo-$i"); echo "eto dpo-$i $j" >> $LOG
done
j=$(sub eto-tp6-eval "$j" "$one" 08:00:00 scripts/hpg_eto.slurm "$E,ETO_MODE=evaluate"); echo "eto eval $j" >> $LOG
# IPR K20 s0 (fresh: ipr-k20s0-r3)
IO=$B/reports/runs/baselines/ipr-k20s0-r3; [ ! -e "$IO" ] || { echo "exists $IO" >> $LOG; exit 1; }
I="IPR_OUT=$IO"
j=$(sub ipr-r3-sft "" "$two" 03:00:00 scripts/hpg_ipr.slurm "$I,IPR_THROUGH=sft"); echo "ipr sft $j" >> $LOG
for i in 1 2 3 4 5 6; do
  j=$(sub ipr-r3-x$i "$j" "$one" 10:00:00 scripts/hpg_ipr.slurm "$I,IPR_THROUGH=pairs-$i"); echo "ipr explore/mc/pairs-$i $j" >> $LOG
  j=$(sub ipr-r3-dpo$i "$j" "$two" 08:00:00 scripts/hpg_ipr.slurm "$I,IPR_THROUGH=dpo-$i"); echo "ipr dpo-$i $j" >> $LOG
done
j=$(sub ipr-r3-eval "$j" "$one" 10:00:00 scripts/hpg_ipr.slurm "$I,IPR_MODE=evaluate"); echo "ipr eval $j" >> $LOG
echo DONE >> $LOG
