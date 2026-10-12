#!/bin/bash
# R0 + noise reference (docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md, R0 and §0).
# Run from rai: ssh hpg bash -s -- <tree sha> < scripts/ops_r03_r0.sh
# Inputs must already be on hpg under $B/reports/runs/r03/inputs/{r0a,r0b} (copied from rai).
set -u
SHA=${1:?tree sha}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
RUN=$B/reports/runs/pilot-warm-1010/train/cold
IN=$B/reports/runs/r03/inputs
O=$B/reports/runs/r03
LOG=$HOME/chain_r03.log
cd "$R" || exit 1; mkdir -p logs
[ "$(cat GIT_COMMIT)" = "$SHA" ] || exit 1
[ -d "$RUN/checkpoint-tokens-104070" ] || exit 1
for d in r0a r0b noise; do [ ! -e "$O/$d" ] || { echo "exists $O/$d" >> $LOG; exit 1; }; done
mkdir -p "$O/r0b" "$O/noise"
common="ALL,PROJECT_ROOT=$R,VERIFIED=$V,SURR=$R/data/manifest/r03_surrogate40.json"
# Noise reference: the cold p40 adapter on dev300, two separate servers (one adapter per server).
for s in a b; do
  j=$(sbatch --parsable --gpus=1 --cpus-per-task=14 --mem=64G --time=01:00:00 --job-name=r03-noise-$s \
      --output=logs/slurm-r03-noise-$s-%j.out \
      --export="$common,TASKS=$R/data/manifest/r03_dev300.json,OUT=$O/noise/$s,WORKERS=16,BATCH_INVARIANT=1,ADAPTERS=cold-p40=$RUN/checkpoint-tokens-416280" \
      scripts/hpg_eval_adapters.slurm) || exit 1; echo "r03 noise-$s $j" >> $LOG; echo "noise-$s $j"
done
# R0a: warmth at saved states.
j=$(sbatch --parsable --job-name=r03-r0a --output=logs/slurm-r03-r0a-%j.out \
    --export="$common,R03_RUN=$RUN,R03_STATES=state-update-1 state-update-4 state-update-8 checkpoint-tokens-31221 checkpoint-tokens-104070 checkpoint-tokens-416280,R03_MSGDIR=$IN/r0a,R03_OUTDIR=$O/r0a" \
    scripts/hpg_r03_warmth.slurm) || exit 1; echo "r03 r0a $j" >> $LOG; echo "r0a $j"
# R0b: 52 items per state, 2 shards per state.
for st in update8:state-update-8 p10:checkpoint-tokens-104070; do
  name=${st%%:*}; ck=$RUN/${st#*:}
  for sh in 0 1; do
    j=$(sbatch --parsable --job-name=r03-r0b-$name-$sh --output=logs/slurm-r03-r0b-$name-$sh-%j.out \
        --export="$common,R03_CHECKPOINT=$ck,R03_ITEMS=$IN/r0b/items_$name.json,R03_OUT=$O/r0b/$name-$sh.jsonl,R03_SHARD=$sh,R03_SHARDS=2" \
        scripts/hpg_r03_crn.slurm) || exit 1; echo "r03 r0b-$name-$sh $j" >> $LOG; echo "r0b-$name-$sh $j"
  done
done
echo DONE-r03-r0 >> $LOG
