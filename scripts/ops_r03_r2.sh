#!/bin/bash
# R2 (prereg R2 + addendum B; protocol fixed in PROJECT_STATE 10-12 23:05 CDT): six continuation arms A{1,2,3} x n{100,400}
# from z10 to p40, fraction 0.25 of the remaining updates from the arm's teacher labels (r2_union filtered to its states),
# seed 1202; each followed (afterok) by a dev300 eval on the noise-reference eval tree. A0 = existing cold p40.
# Run from rai: ssh hpg bash -s -- <tree sha> < scripts/ops_r03_r2.sh
set -u
SHA=${1:?tree sha}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
EVAL_TREE=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/a943c21
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
CK=$B/reports/runs/pilot-warm-1010/train/cold/checkpoint-tokens-104070
EXTRA=$B/data/verified/r03_r2/r2_union
AL=$B/reports/runs/r03/allocations_scalefree
O=$B/reports/runs/r03/r2
cd "$R" || exit 1; mkdir -p logs
[ "$(cat GIT_COMMIT)" = "$SHA" ] || exit 1
[ -d "$CK" -a -f "$EXTRA/manifest.json" -a -d "$EVAL_TREE" ] || exit 1
[ ! -e "$O" ] || { echo "exists $O"; exit 1; }
mkdir -p "$O"
for arm in A1 A2 A3; do for n in 100 400; do
  name=${arm}_n$n
  c=$(sbatch --parsable --job-name=r03-r2-$name --output=logs/slurm-r03-r2-$name-%j.out \
      --export="ALL,PROJECT_ROOT=$R,VERIFIED=$V,R03_CHECKPOINT=$CK,R03_EXTRA=$EXTRA,R03_EXTRA_STATES=$AL/$name.json,R03_FRACTION=0.25,R03_SEED=1202,R03_OUT=$O/$name" \
      scripts/hpg_r03_continue.slurm) || exit 1
  e=$(sbatch --parsable --dependency=afterok:$c --job-name=r03-r2-eval-$name --output=logs/slurm-r03-r2-eval-$name-%j.out \
      --export="ALL,R03_CONT=$O/$name,R03_NAME=$name,EVAL_TREE=$EVAL_TREE,OUT=$O/eval_$name,VERIFIED=$V,SURR=$EVAL_TREE/data/manifest/r03_surrogate40.json" \
      scripts/hpg_r03_eval_continuation.slurm) || exit 1
  echo "$name cont=$c eval=$e"
done; done
