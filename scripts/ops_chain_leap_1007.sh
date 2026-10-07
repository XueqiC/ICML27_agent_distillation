#!/bin/bash
# 2026-10-07 ~12:4x CDT, after the OpenAI top-up: LEAP K50/K100 privileged as stage-split chains. Teacher-only stages
# (correct-N + data-N) run CPU-only on hpg-milan (no idle GPU); training/merge/explore and evaluation get 1 GPU (group QOS,
# as the original LEAP submissions). K100: iteration 2 is invalid (all 44 correction requests failed with HTTP 429
# credit_balance_exhausted) → its correct-2/data-2 stages, teacher request records and run-level iteration-2/result json
# are moved (not deleted) to $B/_trash/leap_k100_credit_exhausted_1007 and re-run under the same identity.
set -u
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
BL=$B/reports/runs/baselines
S=/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env
LOG=$HOME/chain_leap_1007.log
gpu="--account=yd24f.fsu --qos=yd24f.fsu --mem=64G --cpus-per-task=8"
cpu="--account=yd24f.fsu --qos=yd24f.fsu --partition=hpg-milan --gpus=0 --mem=64G --cpus-per-task=4"
sub() {  # tree name dep res time out verified extra
  local R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$1 name=$2 dep=$3 res=$4 time=$5 out=$6 ver=$7 extra=$8 d=""
  [ -n "$dep" ] && d="--dependency=afterok:$dep"
  (cd "$R" && sbatch --parsable $d $res --time=$time --job-name=$name --output=logs/slurm-$name-%j.out \
     --export=ALL,PROJECT_ROOT=$R,LEAP_OUT=$out,LEAP_VARIANT=privileged,VERIFIED=$ver,SED_SECRETS_FILE=$S,WORKERS=16,BATCH_INVARIANT=1,$extra \
     scripts/hpg_leap.slurm)
}
# --- K50 (tree def44a9): next stages train-2, merge-2, explore-3 | correct-3, data-3 | train-3, merge-3 | eval
O50=$BL/leap-privileged-k50s0; V50=$B/data/verified/seeds_k100_s0/subsets/k50
n=$(squeue -h -j 45027583 -o %j 2>/dev/null); [ "$n" = leap-privileged-k50s0 ] && scancel 45027583 && echo "cancel held 45027583" >> $LOG
j=$(sub def44a9 leap-k50-g3 "" "$gpu" 12:00:00 $O50 $V50 LEAP_THROUGH=explore-3); echo "k50 train-2..explore-3 $j" >> $LOG
j=$(sub def44a9 leap-k50-c3 "$j" "$cpu" 04:00:00 $O50 $V50 LEAP_THROUGH=data-3); echo "k50 correct-3,data-3 (cpu) $j" >> $LOG
j=$(sub def44a9 leap-k50-g4 "$j" "$gpu" 12:00:00 $O50 $V50 LEAP_THROUGH=all); echo "k50 train-3,merge-3 $j" >> $LOG
j=$(sub def44a9 leap-k50-eval "$j" "$gpu" 08:00:00 $O50 $V50 LEAP_MODE=evaluate); echo "k50 eval $j" >> $LOG
# --- K100 (tree 61001ec): invalidate iteration 2, then correct-2,data-2 | train-2..explore-3 | correct-3,data-3 | all | eval
O=$BL/leap-privileged-k100s0; V=$B/data/verified/seeds_k100_s0; T=$B/_trash/leap_k100_credit_exhausted_1007
grep -q stopped_empty_min_aggregate "$O/result.json" || { echo "K100 result.json not the invalid stop; refuse" >> $LOG; exit 1; }
grep -q '"rejected_corrections":44' "$O/stages/correct-2/attempt-000/output/report.json" || { echo "K100 correct-2 report unexpected; refuse" >> $LOG; exit 1; }
[ ! -e "$T" ] || { echo "trash exists; refuse" >> $LOG; exit 1; }
mkdir -p "$T/stages" "$T/audit_teacher"
mv "${O:?}/stages/correct-2" "$T/stages/" && mv "${O:?}/stages/data-2" "$T/stages/" && mv "${O:?}/audit/teacher/correction-2" "$T/audit_teacher/" \
  && mv "${O:?}/iteration-2.json" "$T/" && mv "${O:?}/result.json" "$T/" && echo "k100 iteration-2 moved to $T" >> $LOG || { echo "K100 move FAILED" >> $LOG; exit 1; }
echo "Invalidated 2026-10-07: every correction-2 request failed with HTTP 429 insufficient_quota/credit_balance_exhausted (OpenAI credits ran out ~00:50 CDT); re-run under the same run identity after top-up." > "$T/README.txt"
j=$(sub 61001ec leap-k100-c2 "" "$cpu" 04:00:00 $O $V LEAP_THROUGH=data-2); echo "k100 correct-2,data-2 (cpu) $j" >> $LOG
j=$(sub 61001ec leap-k100-g3 "$j" "$gpu" 14:00:00 $O $V LEAP_THROUGH=explore-3); echo "k100 train-2..explore-3 $j" >> $LOG
j=$(sub 61001ec leap-k100-c3 "$j" "$cpu" 04:00:00 $O $V LEAP_THROUGH=data-3); echo "k100 correct-3,data-3 (cpu) $j" >> $LOG
j=$(sub 61001ec leap-k100-g4 "$j" "$gpu" 14:00:00 $O $V LEAP_THROUGH=all); echo "k100 train-3,merge-3 $j" >> $LOG
j=$(sub 61001ec leap-k100-eval "$j" "$gpu" 08:00:00 $O $V LEAP_MODE=evaluate); echo "k100 eval $j" >> $LOG
echo DONE >> $LOG
