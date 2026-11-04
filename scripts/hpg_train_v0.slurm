#!/bin/bash
#SBATCH --job-name=v0-train
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --signal=B:TERM@120
# Sizing/environment adapted from main checkout reports/runs/method_v1/hpg_train_v2.slurm.
# Required: TRAIN_TREE, DATASET (full directory path), OUT, UPDATES, CHECKPOINTS.
# Optional: WEIGHTS, MODULES, SYNTHETIC_MANIFEST (required for synthetic rows),
# THETA0, SNAPSHOT, SEED. B=16; checkpoints are update counts, no selection.
set -euo pipefail
: "${TRAIN_TREE:?}" "${DATASET:?}" "${OUT:?}" "${UPDATES:?}" "${CHECKPOINTS:?}"
H=/blue/fsu-compsci-dept/xc25.fsu/hq
D=$H/ICML27_agent_distillation
cd "$TRAIN_TREE"
export PROJECT_ROOT=$TRAIN_TREE
source configs/hpg.env
export SED_HASH_CACHE_DIR=${SED_HASH_CACHE_DIR:-$H/ICML27_runs/sed-hash-receipts}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$SED_HASH_CACHE_DIR"
[[ -x "$TRAIN_PYTHON" ]] || { echo "TRAIN_PYTHON is not executable" >&2; exit 2; }
read -r -a CKPTS <<< "$CHECKPOINTS"
OPTIONAL=()
[[ -z "${WEIGHTS:-}" ]] || OPTIONAL+=(--weights "$WEIGHTS")
[[ -z "${MODULES:-}" ]] || OPTIONAL+=(--trainable-modules "$MODULES")
[[ -z "${SYNTHETIC_MANIFEST:-}" ]] || OPTIONAL+=(--synthetic-manifest "$SYNTHETIC_MANIFEST")
THETA0=${THETA0:-$D/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint}
SNAPSHOT=${SNAPSHOT:-$H/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2}
echo "dataset=$DATASET out=$OUT host=$(hostname) start=$(date -Is)"
child=''
forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 143; }
trap forward TERM
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node 2 tools/train_v0_arm.py run \
    --dataset "$DATASET" --theta0 "$THETA0" --snapshot "$SNAPSHOT" \
    --config configs/base_e4b.yaml \
    --relocate /home/xueqi/hq/projects/ICML27_agent_distillation=$D \
    --relocate /home/xueqi/.cache/huggingface=$H/tools/hf-cache \
    --relocate /home/xueqi/hq/projects/ICML27_run_repair_eval_7748da5=$H/ICML27_runs/repair_eval_7748da5 \
    --relocate /ddnB/work/xueqic/hq/ICML27_data=$D/data \
    --updates "$UPDATES" --batch-examples 16 --checkpoints "${CKPTS[@]}" \
    --seed "${SEED:-0}" --gpus 2 --output "$OUT" "${OPTIONAL[@]}" &
child=$!
wait "$child"
echo "done $(date -Is)"
