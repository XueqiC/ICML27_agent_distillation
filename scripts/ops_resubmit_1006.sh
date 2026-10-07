#!/bin/bash
# 2026-10-06 ~21:30 CDT: resubmit after the context-overflow / init-validator fix (tree 8b9230a).
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/8b9230a
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
BL=$B/reports/runs/baselines
T=$B/_trash/overflow_resubmit_1006
LOG=$HOME/resubmit_1006.log; : > $LOG
log() { echo "$*" >> $LOG; }
cd "$R" || { log "no tree"; exit 1; }
mkdir -p "$T"
# 0. cancel old-tree resumes that would hit the same overflow
scancel 44948828 45003751 45003752 && log "cancelled 44948828 45003751 45003752"
verified() { case $1 in k20s*) echo $B/data/verified/seeds_k20_s${1#k20s};; k5s*|k10s*) K=${1%s*}; echo $B/data/verified/seeds_k20_s${1#*s}/subsets/$K;; esac; }
# 1. direct evals of trained AgentGym checkpoints whose eval stage failed or never ran
for c in k5s1 k10s0 k10s1 k5s2 k20s1; do
  m=$BL/agentgym-$c/models.json; [ -f "$m" ] || { log "missing $m"; continue; }
  python3 - "$m" <<'PY' > /tmp/agm_$c.txt
import json,sys
for x in json.load(open(sys.argv[1])): print(x["name"], x.get("checkpoint") or x.get("adapter"))
PY
  while read -r label ck; do
    st=$BL/agentgym-$c/stages/eval-$label
    if [ -d "$st" ] && [ -z "$(find $st -name failure.json)" ] && [ -n "$(find $st -path '*eval/summary.json')" ]; then log "$c $label done"; continue; fi
    case "$ck" in $BL/agentgym-$c/stages/train-*/attempt-*/output/checkpoint) ;; *) log "REFUSE $c $label ck=$ck"; continue;; esac
    O=$B/reports/runs/eval/baselines/agentgym-$c-$label; [ -e "$O" ] && { log "exists $O"; continue; }
    j=$(sbatch --parsable --gpus=1 --cpus-per-task=14 --mem=64G --time=04:00:00 --job-name=agentgym-$c-$label-eval --output=logs/slurm-agentgym-$c-$label-eval-%j.out --export=ALL,PROJECT_ROOT=$R,MERGED_MODEL_PATH=$ck,FULL_CHECKPOINT_NAME=agentgym-$c-$label,INCLUDE_BASE=1,VERIFIED=$(verified $c),TASKS=$R/data/manifest/dev140.json,OUT=$O,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_eval_adapters.slurm)
    log "eval $c $label $j"
  done < /tmp/agm_$c.txt
done
# 2. full re-runs that must re-explore: AgentGym k5s0, Kang k100s1 (wrapper names fixed -> move old dirs), ETO, IPR (new names)
for d in agentgym-k5s0 kang-k100s1-tp leap-privileged-k20s0-init; do [ -e "$BL/$d" ] && mv "$BL/$d" "$T/" && log "moved $d"; done
j=$(sbatch --parsable --array=0 --gpus=2 --cpus-per-task=16 --mem=160G --time=08:00:00 --job-name=agentgym-k5s0-r2 --export=ALL,PROJECT_ROOT=$R,DATA_ROOT=$B,METHOD=agentgym,NGPUS=2,CELLS=k5s0 scripts/hpg_baseline_cells.slurm); log "agentgym k5s0 $j"
j=$(sbatch --parsable --array=0 --mem=32G --time=06:00:00 --job-name=kang-k100s1-r2 --export=ALL,PROJECT_ROOT=$R,DATA_ROOT=$B,METHOD=kang,CELLS=k100s1 scripts/hpg_baseline_cells.slurm); log "kang k100s1 $j"
V=$B/data/verified/seeds_k20_s0
[ ! -e $BL/eto-k20s0-tp5 ] && j=$(sbatch --parsable --gpus=2 --cpus-per-task=16 --mem=160G --time=24:00:00 --job-name=eto-k20s0-tp5 --output=logs/slurm-eto-k20s0-tp5-%j.out --export=ALL,PROJECT_ROOT=$R,NGPUS=2,ETO_OUT=$BL/eto-k20s0-tp5,VERIFIED=$V,ETO_THROUGH=all,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_eto.slurm) && log "eto $j"
[ ! -e $BL/ipr-k20s0-r2 ] && j=$(sbatch --parsable --gpus=2 --cpus-per-task=16 --mem=160G --time=24:00:00 --job-name=ipr-k20s0-r2 --output=logs/slurm-ipr-k20s0-r2-%j.out --export=ALL,PROJECT_ROOT=$R,NGPUS=2,IPR_OUT=$BL/ipr-k20s0-r2,VERIFIED=$V,IPR_THROUGH=all,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_ipr.slurm) && log "ipr $j"
# 3. dose controls (fresh)
I=$B/reports/runs/train/seedsft-k20s0-hpg/checkpoint-tokens-416280
j=$(sbatch --parsable --gpus=2 --cpus-per-task=16 --mem=160G --time=12:00:00 --job-name=agentgym-k20s0-init --output=logs/slurm-agentgym-k20s0-init-%j.out --export=ALL,PROJECT_ROOT=$R,NGPUS=2,AGENTGYM_OUT=$BL/agentgym-k20s0-init,VERIFIED=$V,INIT_CHECKPOINT=$I,AGENTGYM_MODE=train,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_agentgym.slurm); log "ctl agentgym $j"
j=$(sbatch --parsable --gpus=2 --cpus-per-task=16 --mem=160G --time=24:00:00 --job-name=eto-k20s0-init --output=logs/slurm-eto-k20s0-init-%j.out --export=ALL,PROJECT_ROOT=$R,NGPUS=2,ETO_OUT=$BL/eto-k20s0-init,VERIFIED=$V,INIT_CHECKPOINT=$I,ETO_THROUGH=all,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_eto.slurm); log "ctl eto $j"
j=$(sbatch --parsable --gpus=2 --cpus-per-task=16 --mem=160G --time=24:00:00 --job-name=ipr-k20s0-init --output=logs/slurm-ipr-k20s0-init-%j.out --export=ALL,PROJECT_ROOT=$R,NGPUS=2,IPR_OUT=$BL/ipr-k20s0-init,VERIFIED=$V,INIT_CHECKPOINT=$I,IPR_THROUGH=all,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_ipr.slurm); log "ctl ipr $j"
j=$(sbatch --parsable --account=yd24f.fsu --qos=yd24f.fsu --mem=64G --time=24:00:00 --job-name=leap-privileged-k20s0-init --output=logs/slurm-leap-privileged-k20s0-init-%j.out --export=ALL,PROJECT_ROOT=$R,LEAP_OUT=$BL/leap-privileged-k20s0-init,LEAP_VARIANT=privileged,VERIFIED=$V,INIT_CHECKPOINT=$I,SED_SECRETS_FILE=/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env,WORKERS=16,BATCH_INVARIANT=1 scripts/hpg_leap.slurm); log "ctl leap $j"
log DONE
