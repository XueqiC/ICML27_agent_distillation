#!/bin/bash
# 2026-10-10 v0.15 A pilot (user: "关于AB你可以按照你想的来"): does Adam's cold window matter for few-shot LoRA?
# Three K20 s0 Seed-SFT arms on the main-table recipe (416,280 supervised tokens, checkpoints p03..p40, lr 1e-5
# constant, 1xB200): cold (= main-table recipe), moment warm-up W=32 (lr=0 AdamW steps on the same frozen plan;
# parameters bitwise unchanged), linear LR warm-up S=32. All arms log per-update warmth (kappa quantiles).
# Probe (Warm_Adam_Response_EN.md W1-W3) on the cold run: cold / moment-warmed / saved-update / checkpoint states.
# Eval: one BI dev140 server with all six adapters ({cold,mw32,lw32} x {p10,p40}) so every arm shares one config.
# Memory from sacct of the same objective: Seed-SFT 1 GPU MaxRSS <= 7.6G (32G requested), adapter eval 48-64G.
# Short walltimes so the 1-GPU jobs can backfill (hpg-b200 had 852 pending jobs at 16:4x CDT).
# Run from rai: ssh hpg bash -s -- <tree sha> < scripts/ops_pilot_warm_1010.sh   (not copied into the pinned tree:
# scripts/*.sh are part of the identity code hash).
set -u
SHA=${1:?tree sha}
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/$SHA
B=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
V=$B/data/verified/seeds_k20_s0
O=$B/reports/runs/pilot-warm-1010
LOG=$HOME/chain_1010.log
cd "$R" || exit 1; mkdir -p logs
[ "$(cat GIT_COMMIT)" = "$SHA" ] || exit 1
[ ! -e "$O" ] || { echo "exists $O" >> $LOG; exit 1; }
mkdir -p "$O/train" "$O/probe"
common="ALL,PROJECT_ROOT=$R,VERIFIED=$V,TRAIN_PASSES=0,TRAIN_TOKENS=416280,CHECKPOINT_TOKENS=31221 52035 104070 208140 312210 416280,TOKENS_PER_UPDATE=512,LEARNING_RATE=1e-5,TRAIN_SEED=0,MAX_LENGTH=32768,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,LOG_WARMTH=1"
train() { local arm=$1 extra=$2
  sbatch --parsable --gpus=1 --cpus-per-task=4 --mem=32G --time=01:15:00 --job-name=pilot-$arm \
    --output=logs/slurm-pilot-$arm-%j.out --export="$common,TRAIN_OUT=$O/train/$arm$extra" scripts/hpg_train_sft.slurm; }
c=$(train cold ",SAVE_EVERY_UPDATE_THROUGH=9") || exit 1; echo "pilot cold $c" >> $LOG; echo "cold $c"
m=$(train mw32 ",MOMENT_WARMUP_STEPS=32,SAVE_EVERY_UPDATE_THROUGH=2") || exit 1; echo "pilot mw32 $m" >> $LOG; echo "mw32 $m"
l=$(train lw32 ",LR_WARMUP_STEPS=32") || exit 1; echo "pilot lw32 $l" >> $LOG; echo "lw32 $l"
p=$(sbatch --parsable --dependency=afterok:$c --gpus=1 --cpus-per-task=6 --mem=96G --time=02:00:00 --job-name=pilot-probe \
    --output=logs/slurm-pilot-probe-%j.out \
    --export="ALL,PROJECT_ROOT=$R,VERIFIED=$V,PROBE_RUN=$O/train/cold,PROBE_PAIRS=4,PROBE_OUTPUT=$O/probe/cold.json,PROBE_STATES=cold warmup:8 warmup:32 update:1 update:4 update:8 checkpoint:31221 checkpoint:104070 checkpoint:416280" \
    scripts/hpg_adam_probe.slurm) || exit 1; echo "pilot probe $p" >> $LOG; echo "probe $p"
ad=""; for arm in cold mw32 lw32; do ad="$ad $arm-p10=$O/train/$arm/checkpoint-tokens-104070 $arm-p40=$O/train/$arm/checkpoint-tokens-416280"; done
e=$(sbatch --parsable --dependency=afterok:$c:$m:$l --gpus=1 --cpus-per-task=14 --mem=64G --time=03:00:00 --job-name=pilot-eval \
    --output=logs/slurm-pilot-eval-%j.out \
    --export="ALL,PROJECT_ROOT=$R,VERIFIED=$V,TASKS=$R/data/manifest/dev140.json,OUT=$O/eval,WORKERS=16,BATCH_INVARIANT=1,ADAPTERS=${ad# }" \
    scripts/hpg_eval_adapters.slurm) || exit 1; echo "pilot eval $e" >> $LOG; echo "eval $e"
echo DONE-pilot >> $LOG
