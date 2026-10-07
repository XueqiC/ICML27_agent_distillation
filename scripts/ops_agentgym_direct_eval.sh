#!/bin/bash
# Usage: bash direct_eval.sh CELL [CELL...]  — direct full-checkpoint dev140 evals (tree 8b9230a) of trained AgentGym iterates
# whose pipeline eval stage failed or never ran. Log: ~/direct_eval.log
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/8b9230a; B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation; BL=$B/reports/runs/baselines
LOG=$HOME/direct_eval.log; cd "$R" || exit 1
verified() { case $1 in k20s*|k100s*) K=${1%s*}; echo $B/data/verified/seeds_${K}_s${1#*s};; k5s*|k10s*) K=${1%s*}; echo $B/data/verified/seeds_k20_s${1#*s}/subsets/$K;; k1s*|k2s*|k50s*) K=${1%s*}; echo $B/data/verified/seeds_k100_s${1#*s}/subsets/$K;; esac; }
for c in "$@"; do
  m=$BL/agentgym-$c/models.json; [ -f "$m" ] || { echo "$c no models.json" >> $LOG; continue; }
  python3 -c "import json,sys; [print(x['name'], x.get('checkpoint') or x.get('adapter')) for x in json.load(open(sys.argv[1]))]" "$m" > /tmp/agm_$c.txt
  while read -r label ck; do
    st=$BL/agentgym-$c/stages/eval-$label
    if [ -d "$st" ] && [ -z "$(find $st -name failure.json)" ] && [ -n "$(find $st -path '*eval/summary.json')" ]; then echo "$c $label done" >> $LOG; continue; fi
    case "$ck" in $BL/agentgym-$c/stages/train-*/attempt-*/output/checkpoint) ;; *) echo "REFUSE $c $label $ck" >> $LOG; continue;; esac
    O=$B/reports/runs/eval/baselines/agentgym-$c-$label; [ -e "$O" ] && { echo "exists $O" >> $LOG; continue; }
    j=$(sbatch --parsable --gpus=1 --cpus-per-task=14 --mem=64G --time=04:00:00 --job-name=agentgym-$c-$label-eval --output=logs/slurm-agentgym-$c-$label-eval-%j.out --export=ALL,PROJECT_ROOT=$R,MERGED_MODEL_PATH=$ck,FULL_CHECKPOINT_NAME=agentgym-$c-$label,INCLUDE_BASE=1,VERIFIED=$(verified $c),TASKS=$R/data/manifest/dev140.json,OUT=$O,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_eval_adapters.slurm)
    echo "eval $c $label $j" >> $LOG
  done < /tmp/agm_$c.txt
done
echo "DONE $*" >> $LOG
