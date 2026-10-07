#!/bin/bash
# 2026-10-07 ~12:4x CDT (user: "尽量快"): IPR K20 s0 restarted fresh with NGPUS=4 for the FSDP segments (DPO ≈ 2× faster than
# 2×B200; r3 would need ~3 days). Identity binds --gpus, so this is a new run (ipr-k20s0-r4); r3 chain cancelled, its
# completed sft/x1 stages stay on disk. Single-model segments keep 1 GPU.
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/6bac2da
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
O=$B/reports/runs/baselines/ipr-k20s0-r4
LOG=$HOME/chain_1007.log
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = 6bac2da ] || exit 1
[ ! -e "$O" ] || { echo "exists $O" >> $LOG; exit 1; }
for j in 45078017 45078018 45078019 45078020 45078021 45078022 45078023 45078024 45078025 45078026 45078027 45078028; do
  n=$(squeue -h -j $j -o %j 2>/dev/null); case "$n" in ipr-r3-*) scancel $j && echo "cancel $j $n" >> $LOG;; *) echo "skip $j '$n'" >> $LOG;; esac
done
four="--gpus=4 --cpus-per-task=16 --mem=320G"
one="--gpus=1 --cpus-per-task=14 --mem=96G"
sub() { local name=$1 dep=$2 res=$3 time=$4 exports=$5 d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  sbatch --parsable $d $res --time=$time --job-name=$name --output=logs/slurm-$name-%j.out \
    --export=ALL,PROJECT_ROOT=$R,NGPUS=4,VERIFIED=$V,WORKERS=16,BATCH_INVARIANT=1,IPR_OUT=$O,$exports scripts/hpg_ipr.slurm; }
j=$(sub ipr-r4-sft "" "$four" 03:00:00 "IPR_THROUGH=sft"); echo "ipr r4 sft $j" >> $LOG
for i in 1 2 3 4 5 6; do
  j=$(sub ipr-r4-x$i "$j" "$one" 10:00:00 "IPR_THROUGH=pairs-$i"); echo "ipr r4 x$i $j" >> $LOG
  j=$(sub ipr-r4-dpo$i "$j" "$four" 16:00:00 "IPR_THROUGH=dpo-$i"); echo "ipr r4 dpo-$i $j" >> $LOG
done
j=$(sub ipr-r4-eval "$j" "$one" 10:00:00 "IPR_MODE=evaluate"); echo "ipr r4 eval $j" >> $LOG
echo DONE-r4 >> $LOG
