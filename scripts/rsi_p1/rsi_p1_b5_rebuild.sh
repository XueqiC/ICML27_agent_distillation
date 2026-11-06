#!/bin/bash
# B5-only rebuild after the 11-06 rsi reboot (prereg §12). Same env/args as rsi_p1_stage.sh; runs ~/hq/rsi_p1_b5_rebuild.py. Args: GPU
set -uo pipefail
gpu=${1:?GPU}
T=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a
P=$T/reports/p1_runs
W=$P/B/conditions_b5r
cd "$T"
[ "$(git rev-parse --short HEAD)" = d082a6a ] || { echo "tree mismatch" >&2; exit 2; }
[ -e "$W" ] && { echo "work dir exists: $W" >&2; exit 2; }
[ -e "$P/B/conditions/B5.json" ] && { echo "B5.json exists" >&2; exit 2; }
mkdir "$W" || exit 2
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 B5_CONDITIONS=$P/B/conditions
PY=/home/xueqi/hq/envs/p1-venv/bin/python
echo "== p1 B5 rebuild start $(date -Is) host=$(hostname) GPU $gpu work=$W"
"$PY" /home/xueqi/hq/rsi_p1_b5_rebuild.py \
  --pairs $P/B/data/pairs.json --submission $P/B/global/submission.json --output "$W" \
  --base /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --adapter /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint \
  --round-dir /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1 \
  --verified /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20 \
  --alfworld-data /home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
echo "== p1 B5 rebuild exit $? $(date -Is)"
