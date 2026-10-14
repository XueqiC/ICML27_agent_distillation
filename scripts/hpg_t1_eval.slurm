#!/bin/bash
#SBATCH --job-name=t1-eval
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
# Sizing (sacct, 53 evals): 1.8-2.2 busy cores, peak RSS 23 GB. CPUs stay 14 so pre-registered comparisons keep the
# settings of the 14-CPU noise reference; a new eval protocol with its own reference should request 6.
# Submit with: sbatch --dependency=afterok:<T1 job id> scripts/hpg_t1_eval.slurm
# EVAL_TREE must be the pinned noise-reference tree, as in ops_r03_r2.sh.
# Its hpg_eval_adapters.slurm accepts whitespace-delimited NAME=PATH entries.
set -euo pipefail
: "${T1_OUT:?}" "${EVAL_TREE:?}" "${OUT:?}" "${VERIFIED:?}"
[[ -f "$T1_OUT/summary.json" && -f "$EVAL_TREE/scripts/hpg_eval_adapters.slurm" ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]]
shopt -s nullglob
adapters=()
for arm in T00 T11; do
    cks=("$T1_OUT/$arm"/checkpoint-tokens-*)
    [[ ${#cks[@]} -ge 1 ]]
    ck=$(printf '%s\n' "${cks[@]}" | awk -F'checkpoint-tokens-' '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
    [[ -f "$ck/adapter_model.safetensors" && "$ck" != *[[:space:]]* ]]
    adapters+=("$arm=$ck")
    echo "eval $arm = $ck"
done
export PROJECT_ROOT="$EVAL_TREE" ADAPTERS="${adapters[*]}"
export TASKS="$EVAL_TREE/data/manifest/r03_dev300.json" SURR="$EVAL_TREE/data/manifest/r03_surrogate40.json"
export WORKERS=16 BATCH_INVARIANT=1 INCLUDE_BASE=0
[[ -f "$TASKS" ]]
cd "$EVAL_TREE"
started=$SECONDS
status=0
bash scripts/hpg_eval_adapters.slurm || status=$?
printf 'eval_T00_T11 elapsed_seconds=%s status=%s\n' "$((SECONDS - started))" "$status"
exit "$status"
