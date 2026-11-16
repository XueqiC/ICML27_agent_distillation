#!/bin/bash
# rsi_wsp_d_stage.sh TREE GPU|cpu COMMAND [args] — one p1_surgery.py stage for the D group on rsi (setting B, raw student).
# TREE = short commit of /home/xueqi/hq/projects/ICML27_run_repair_eval_<TREE>. GPU "cpu" hides all GPUs (export-delta).
set -uo pipefail
tree=${1:?tree}; gpu=${2:?gpu|cpu}; shift 2
T=/home/xueqi/hq/projects/ICML27_run_repair_eval_$tree
cd "$T" || exit 2
[ "$(git rev-parse --short=7 HEAD)" = "$tree" ] && [ -z "$(git status --short)" ] || { echo "tree mismatch" >&2; exit 2; }
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
if [ "$gpu" = cpu ]; then export CUDA_VISIBLE_DEVICES=; else
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 2000 ] || { echo "GPU $gpu is in use ($used MiB)" >&2; exit 3; }
  export CUDA_VISIBLE_DEVICES=$gpu; fi
PY=/home/xueqi/hq/envs/p1-venv/bin/python
echo "== wsp-d stage [$*] tree $tree start $(date -Is) host=$(hostname) gpu=$gpu"
"$PY" tools/p1_surgery.py "$@" \
  --base /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --adapter /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint \
  --round-dir /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1 \
  --verified /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20 \
  --alfworld-data /home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
echo "== wsp-d stage [$1] exit $? $(date -Is)"
