#!/bin/bash
#SBATCH --job-name=baseline-cells
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --signal=B:TERM@60
# 2026-10-05: K x support-seed sweep of one baseline at its official recipe. One array task = one cell: train, then the
# dev140 batch-invariant evaluation the method's paper uses (Kang: SAG decoding; AgentGym: every AgentEvol iteration).
# Required: PROJECT_ROOT=<frozen tree>, DATA_ROOT=<base tree holding data/verified and reports/>, METHOD=kang|agentgym,
# CELLS="k5s0 k5s1 ... full" (array index i -> i-th cell). Resources per method at submission:
#   kang:     sbatch --array=0-N%4 --export=ALL,METHOD=kang,...                     (defaults: 1 GPU, 96G)
#   agentgym: sbatch --array=0-N%2 --gpus=2 --cpus-per-task=16 --mem=240G --export=ALL,METHOD=agentgym,NGPUS=2,...
set -euo pipefail
source "${HPG_CONFIG:-${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${METHOD:?}" "${CELLS:?}" "${DATA_ROOT:?}"
read -r -a cells <<< "$CELLS"
cell=${cells[$SLURM_ARRAY_TASK_ID]:-}
if [[ "$cell" == full ]]; then
    verified="data/verified/seeds_full_train_r2"
else
    [[ "$cell" =~ ^k([0-9]+)s([0-2])$ ]] || { echo "No valid cell for array index $SLURM_ARRAY_TASK_ID: '$cell'" >&2; exit 2; }
    K=${BASH_REMATCH[1]} S=${BASH_REMATCH[2]}
    # Nested seed ladder: K=5/10 are views of the K=20 collection, K=1/2/50 views of the K=100 collection (same as Seed-SFT).
    # Never add views inside seeds_k20_s*: the K=100 extension binds the K=20 tree hash and refuses a changed source.
    case "$K" in
        20|100) verified="data/verified/seeds_k${K}_s$S" ;;
        5|10) verified="data/verified/seeds_k20_s$S/subsets/k$K" ;;
        1|2|50) verified="data/verified/seeds_k100_s$S/subsets/k$K" ;;
        *) echo "K=$K is not on the seed ladder" >&2; exit 2 ;;
    esac
fi
export VERIFIED="$DATA_ROOT/$verified" WORKERS=${WORKERS:-16} BATCH_INVARIANT=1
[[ -f "$VERIFIED/manifest.json" ]] || { echo "Missing collection: $VERIFIED" >&2; exit 2; }
echo "cell $cell method=$METHOD verified=$VERIFIED host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
child=''
forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 143; }
trap forward TERM
run() { "$@" & child=$!; local status=0; wait "$child" || status=$?; child=''; return "$status"; }
case "$METHOD" in
    kang)
        # Forward the caller's training seed unchanged; hpg_kang supplies default 0.
        export KANG_TRAINING_SEED
        export KANG_OUT="$DATA_ROOT/reports/runs/baselines/kang-$cell-tp"
        KANG_MODE=train run bash scripts/hpg_kang.slurm
        adapter=$("$ENV_PYTHON" -c 'import json, sys; m = json.load(open(sys.argv[1])); assert len(m) == 1, m; print(m[0]["adapter"])' \
                  "$KANG_OUT/models.json")
        ADAPTERS="kang-$cell-tp=$adapter" TASKS="$PROJECT_ROOT/data/manifest/dev140.json" DECODING=sag INCLUDE_BASE=0 \
            OUT="$DATA_ROOT/reports/runs/eval/baselines/kang-$cell-tp-sag" run bash scripts/hpg_eval_adapters.slurm
        ;;
    agentgym)
        export AGENTGYM_OUT="$DATA_ROOT/reports/runs/baselines/agentgym-$cell" NGPUS=${NGPUS:-2}
        AGENTGYM_MODE=train run bash scripts/hpg_agentgym.slurm
        AGENTGYM_MODE=evaluate run bash scripts/hpg_agentgym.slurm
        ;;
    *) echo "METHOD must be kang or agentgym" >&2; exit 2 ;;
esac
echo "cell $cell done"
