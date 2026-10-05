#!/bin/bash
# Usage: rai_eval_student.sh GPU TASKS OUT [--base] [--workers N] [NAME=PATH ...]
set -euo pipefail
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
case "$BATCH_INVARIANT" in 0|1) ;; *) echo 'BATCH_INVARIANT must be 0 or 1' >&2; exit 2 ;; esac
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"
[[ $# -ge 3 && "$1" =~ ^[0-9]+$ ]] || {
    echo 'Usage: rai_eval_student.sh GPU TASKS OUT [--base] [--workers N] [NAME=PATH ...]' >&2; exit 2;
}
gpu=$1
TASKS=$2
OUT=$3
shift 3
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_ROOT"
ENV_PYTHON=${ENV_PYTHON:-.venv/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/vllm-serve/.venv/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}
export ALFWORLD_DATA_ROOT=${ALFWORLD_DATA_ROOT:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
WORKERS=${WORKERS:-8}
LORA_DTYPE=${LORA_DTYPE:-bfloat16}
INCLUDE_BASE=${INCLUDE_BASE:-0}
adapters=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) INCLUDE_BASE=1; shift ;;
        --workers) [[ $# -ge 2 ]] || exit 2; WORKERS=$2; shift 2 ;;
        *=*) adapters+=("$1"); shift ;;
        *) echo 'Unknown launcher argument' >&2; exit 2 ;;
    esac
done
[[ ${#adapters[@]} -gt 0 ]] || INCLUDE_BASE=1
source scripts/eval_adapters.sh
