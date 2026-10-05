#!/bin/bash
# 2026-10-04: Seed-SFT K=10 and K=5 (support seed 0) on rai GPU1, run from a frozen worktree (RUN_TREE).
set -u
M=/home/xueqi/hq/projects/ICML27_agent_distillation
cd ${RUN_TREE:-/home/xueqi/hq/projects/ICML27_run_e70fa64}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
PY=/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python
log(){ echo "[$(TZ=America/Chicago date '+%H:%M:%S')] $*"; }
run(){ k=$1; per=$2
  log "train k$k s0 start"
  $PY train/sft.py --snapshot $SNAP --verified $M/data/verified/seeds_k20_s0/subsets/k$k --output $M/reports/runs/train/seedsft-k$k-s0-rai \
    --supervised-tokens $((per*20)) --checkpoint-tokens $((per*3)) $((per*5)) $((per*10)) $((per*20)) --packing episode --seed 0 \
    > $M/logs/train_seedsft_k${k}_s0_rai.log 2>&1
  log "train k$k s0 exit $?"; }
run 10 5311
run 5 2146
