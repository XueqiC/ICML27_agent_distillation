#!/bin/bash
# R3 (prereg R3): amplitude prediction first, then CRN tails (afterok), 6 shards.
# Run from rai: ssh hpg bash -s -- <tree sha> < scripts/ops_r03_r3.sh   (inputs in $B/reports/runs/r03/inputs/r3)
set -u
SHA=${1:?tree sha}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
CK=$B/reports/runs/pilot-warm-1010/train/cold/checkpoint-tokens-104070
IN=$B/reports/runs/r03/inputs/r3
O=$B/reports/runs/r03/r3
LOG=$HOME/chain_r03.log
cd "$R" || exit 1; mkdir -p logs
[ "$(cat GIT_COMMIT)" = "$SHA" ] || exit 1
[ -d "$CK" -a -f "$IN/items.json" -a -f "$IN/batch.json" ] || exit 1
[ ! -e "$O" ] || { echo "exists $O" >> $LOG; exit 1; }
mkdir -p "$O"
common="ALL,PROJECT_ROOT=$R,VERIFIED=$V"
a=$(sbatch --parsable --job-name=r03-r3-amp --output=logs/slurm-r03-r3-amp-%j.out \
    --export="$common,R03_CHECKPOINT=$CK,R03_MESSAGE=$IN/batch.json,R03_OUT=$O/prediction.json" \
    scripts/hpg_r03_amplitude.slurm) || exit 1; echo "r03 r3-amp $a" >> $LOG; echo "r3-amp $a"
for sh in 0 1 2 3 4 5; do
  j=$(sbatch --parsable --dependency=afterok:$a --job-name=r03-r3-$sh --output=logs/slurm-r03-r3-$sh-%j.out \
      --export="$common,R03_CHECKPOINT=$CK,R03_ITEMS=$IN/items.json,R03_OUT=$O/crn-$sh.jsonl,R03_SHARD=$sh,R03_SHARDS=6" \
      scripts/hpg_r03_crn.slurm) || exit 1; echo "r03 r3-$sh $j" >> $LOG; echo "r3-$sh $j"
done
echo DONE-r03-r3 >> $LOG
