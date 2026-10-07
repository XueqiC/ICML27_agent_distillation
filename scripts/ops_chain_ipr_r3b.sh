#!/bin/bash
# 2026-10-07 ~12:4x CDT: IPR r3 dpo segments need >8 h (iteration 1 P=95 → 285 pair-slots; ETO measured 168 s/pair-slot on
# 2×B200). Cancel the 8 h dpo-1 (minutes in) and the pending tail, resubmit from dpo-1 with 24 h dpo limits.
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/6bac2da
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
LOG=$HOME/chain_1007.log
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = 6bac2da ] || exit 1
for j in 45058908 45058909 45058910 45058911 45058912 45058913 45058914 45058915 45058916 45058917 45058918 45058919; do
  n=$(squeue -h -j $j -o %j 2>/dev/null); case "$n" in ipr-r3-*) scancel $j && echo "cancel $j $n" >> $LOG;; *) echo "skip $j '$n'" >> $LOG;; esac
done
sleep 20
two="--gpus=2 --cpus-per-task=16 --mem=160G"
one="--gpus=1 --cpus-per-task=14 --mem=96G"
sub() { local name=$1 dep=$2 res=$3 time=$4 exports=$5 d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  sbatch --parsable $d $res --time=$time --job-name=$name --output=logs/slurm-$name-%j.out \
    --export=ALL,PROJECT_ROOT=$R,NGPUS=2,VERIFIED=$V,WORKERS=16,BATCH_INVARIANT=1,IPR_OUT=$B/reports/runs/baselines/ipr-k20s0-r3,$exports scripts/hpg_ipr.slurm; }
j=$(sub ipr-r3-dpo1 "" "$two" 24:00:00 "IPR_THROUGH=dpo-1"); echo "ipr(b) dpo-1 $j" >> $LOG
for i in 2 3 4 5 6; do
  j=$(sub ipr-r3-x$i "$j" "$one" 10:00:00 "IPR_THROUGH=pairs-$i"); echo "ipr(b) x$i $j" >> $LOG
  j=$(sub ipr-r3-dpo$i "$j" "$two" 24:00:00 "IPR_THROUGH=dpo-$i"); echo "ipr(b) dpo-$i $j" >> $LOG
done
j=$(sub ipr-r3-eval "$j" "$one" 10:00:00 "IPR_MODE=evaluate"); echo "ipr(b) eval $j" >> $LOG
echo DONE-b >> $LOG
