#!/bin/bash
# 2026-10-10 (user chose option a): IPR K20 s0 restarted fresh as r5 on tree 5e8443c (parallel evidence hashing and listing +
# rank-0 startup verification). r4 (tree 6bac2da) is withdrawn: its dpo-1 OOM-ed at 160G and every resumed segment
# idled its GPUs ~55 min re-verifying evidence; identity binds code_sha256, so the fix cannot resume r4.
# Memory from sacct of the same objective and rank count: 4-GPU SFT 25G peak, 1-GPU explore/MC 15G, 4-GPU DPO ~210G.
# Run from rai: ssh hpg bash -s < scripts/ops_chain_ipr_r5.sh   (this file is NOT copied into the pinned tree,
# because scripts/*.sh are part of the identity code hash).
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/5e8443c
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
O=$B/reports/runs/baselines/ipr-k20s0-r5
LOG=$HOME/chain_1010.log
cd "$R" || exit 1
[ "$(cat GIT_COMMIT)" = 5e8443c ] || exit 1
[ ! -e "$O" ] || { echo "exists $O" >> $LOG; exit 1; }
sft="--gpus=4 --cpus-per-task=8 --mem=96G"
dpo="--gpus=4 --cpus-per-task=8 --mem=256G"
one="--gpus=1 --cpus-per-task=6 --mem=32G"
sub() { local name=$1 dep=$2 res=$3 time=$4 exports=$5 d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  sbatch --parsable $d $res --time=$time --job-name=$name --output=logs/slurm-$name-%j.out \
    --export=ALL,PROJECT_ROOT=$R,NGPUS=4,VERIFIED=$V,WORKERS=16,BATCH_INVARIANT=1,IPR_OUT=$O,$exports scripts/hpg_ipr.slurm; }
j=$(sub ipr-r5-sft "" "$sft" 03:00:00 "IPR_THROUGH=sft") || exit 1; echo "ipr r5 sft $j" >> $LOG; echo "sft $j"
for i in 1 2 3 4 5 6; do
  j=$(sub ipr-r5-x$i "$j" "$one" 10:00:00 "IPR_THROUGH=pairs-$i") || exit 1; echo "ipr r5 x$i $j" >> $LOG; echo "x$i $j"
  j=$(sub ipr-r5-dpo$i "$j" "$dpo" 16:00:00 "IPR_THROUGH=dpo-$i") || exit 1; echo "ipr r5 dpo-$i $j" >> $LOG; echo "dpo$i $j"
done
j=$(sub ipr-r5-eval "$j" "$one" 10:00:00 "IPR_MODE=evaluate") || exit 1; echo "ipr r5 eval $j" >> $LOG; echo "eval $j"
echo DONE-r5 >> $LOG
