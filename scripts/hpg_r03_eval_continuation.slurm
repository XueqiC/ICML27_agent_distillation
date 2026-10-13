#!/bin/bash
#SBATCH --job-name=r03-eval
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
# Evaluate the final checkpoint of one continuation arm with the SAME eval tree and settings as the R0-R3 noise
# reference (scripts/hpg_eval_adapters.slurm, r03_dev300, WORKERS=16, BATCH_INVARIANT=1).
# Required: R03_CONT (continuation output dir), R03_NAME, EVAL_TREE (pinned code tree of the noise reference), OUT,
# VERIFIED, SURR.
set -euo pipefail
: "${R03_CONT:?}" "${R03_NAME:?}" "${EVAL_TREE:?}" "${OUT:?}" "${VERIFIED:?}" "${SURR:?}"
shopt -s nullglob
cks=("$R03_CONT"/checkpoint-tokens-*)
[[ ${#cks[@]} -ge 1 ]]
ck=$(printf '%s\n' "${cks[@]}" | awk -F'checkpoint-tokens-' '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
[[ -f "$ck/adapter_model.safetensors" && ! -e "$OUT" ]]
echo "eval $R03_NAME = $ck"
export PROJECT_ROOT="$EVAL_TREE" ADAPTERS="$R03_NAME=$ck"
export TASKS="${TASKS:-$EVAL_TREE/data/manifest/r03_dev300.json}" WORKERS="${WORKERS:-16}" BATCH_INVARIANT="${BATCH_INVARIANT:-1}"
cd "$EVAL_TREE"
exec bash scripts/hpg_eval_adapters.slurm
