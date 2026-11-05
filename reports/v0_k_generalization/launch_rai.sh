#!/usr/bin/env bash
# Reviewable launch commands only. This file was NOT used to start any model/API.
# Usage: bash reports/v0_k_generalization/launch_rai.sh {10|30} STAGE
# STAGE: synth, synth-dryrun, attempt, probe, review-dryrun, review,
#        label, build, expand-D-data, expand-D-param
set -euo pipefail
k=${1:?Set K to 10 or 30}
stage=${2:?Set a stage}
export PROJECT_ROOT=/home/xueqi/hq/projects/ICML27_wt_repair_eval
export V0_SOURCE_ROOT=/home/xueqi/hq/projects/ICML27_agent_distillation
export HPG_CONFIG="$V0_SOURCE_ROOT/reports/runs/repair_pilot/rai.env"
export V0_ROUND_OUT="/home/xueqi/hq/projects/ICML27_v0_runs/v0_k${k}_r1"
case "$k" in
    10) export MODEL_NAME=e4b-tsft-k10-e32-lr5e-5 ;;
    30) export MODEL_NAME=e4b-tsft-k30-e64-lr5e-5 ;;
    *) echo 'K must be 10 or 30' >&2; exit 2 ;;
esac
export VERIFIED="$V0_SOURCE_ROOT/data/verified/seeds_full_train_r2/subsets/k$k"
export ADAPTER="$V0_SOURCE_ROOT/reports/runs/baselines_hpg/$MODEL_NAME/stages/train-sft/attempt-000/output/checkpoint"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The test venv has no ALFWorld/TextWorld. Use the existing native environment.
cpu_python="$V0_SOURCE_ROOT/.venv/bin/python"
cd "$PROJECT_ROOT"

# Synthesis is always CPU, offline, planner-verified; no serving env is needed.
if [[ "$stage" == synth || "$stage" == synth-dryrun ]]; then
    target="$V0_ROUND_OUT"
    [[ "$stage" != synth-dryrun ]] || target="/home/xueqi/hq/projects/ICML27_v0_runs/_dryrun_k$k"
    CUDA_VISIBLE_DEVICES='' "$cpu_python" -m tools.v0_round synth \
        --output "$target" --seeds "$VERIFIED/manifest.json" \
        --data-root /home/xueqi/hq/projects/tc-alignment/envs/alfworld/data/json_2.1.1 \
        --split-freeze /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1/synth/split_freeze.json \
        --main-checkout "$V0_SOURCE_ROOT" --seed 0 --workers 4
    exit
fi

# Reuse the established rai model/interpreter/cache setup, without changing HOME.
source "$HPG_CONFIG"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export GPU=${GPU:-1} WORKERS=${WORKERS:-8} LABEL_WORKERS=${LABEL_WORKERS:-16}
export V0_SNAPSHOT="$MODEL_PATH"
export V0_RESUME=0
unset V0_ATTEMPT_DIR REPAIR_OUT V0_VARIANT
case "$stage" in
    attempt|probe|review)
        export V0_STAGE="$stage"
        bash tools/v0_round_rai.sh
        ;;
    review-dryrun)
        CUDA_VISIBLE_DEVICES='' "$cpu_python" -m tools.v0_round review \
            --output "$V0_ROUND_OUT" --config configs/base_e4b.yaml --seed 0 --dry-run
        ;;
    label)
        CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -m tools.v0_round label \
            --output "$V0_ROUND_OUT" --config configs/base_e4b.yaml --seed 0 \
            --snapshot "$MODEL_PATH" --label-workers "$LABEL_WORKERS"
        ;;
    build)
        CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -m tools.v0_round build \
            --output "$V0_ROUND_OUT" --config configs/base_e4b.yaml --seed 0 \
            --splits data/manifest/method_v1_splits.json \
            --verified-root "$V0_SOURCE_ROOT/data/verified/seeds_full_train_r2" \
            --theta0-training-dir "${ADAPTER%/checkpoint}" --snapshot "$MODEL_PATH"
        ;;
    expand-D-data|expand-D-param)
        export V0_STAGE=expand V0_VARIANT="${stage#expand-}"
        export V0_DATASET="$V0_ROUND_OUT/dataset"
        export V0_GROUPS="$V0_ROUND_OUT/grad_groups"
        bash tools/v0_round_rai.sh
        ;;
    *) echo 'Unknown stage' >&2; exit 2 ;;
esac
