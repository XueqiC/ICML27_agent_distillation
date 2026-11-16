#!/bin/bash
# rsi_wsp_d_session.sh VARIANT(d1|d2) GPU NAME [extra diagnose args]   — D teacher session on rsi
# (prereg 2026-11-15_WSP_vs_SFT section 16). Tree 9d897cc (branch wsp-d), env ~/hq/envs/p1-venv; the teacher client
# reads its key from ~/hq/secrets itself (never passed on the command line or printed). Raw student = setting B.
set -uo pipefail
variant=${1:?d1|d2}; gpu=${2:?GPU}; name=${3:?run name}; shift 3
case "$variant" in d1|d2) ;; *) echo "variant must be d1 or d2" >&2; exit 2;; esac
T=/home/xueqi/hq/projects/ICML27_run_repair_eval_9d897cc
D=/home/xueqi/hq/projects/ICML27_wsp_d
cd "$T" || exit 2
[ "$(git rev-parse --short=7 HEAD)" = 9d897cc ] && [ -z "$(git status --short)" ] || { echo "tree mismatch" >&2; exit 2; }
[ ! -e "$D/runs/$name" ] || { echo "run dir exists: $D/runs/$name" >&2; exit 2; }
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2000 ] || { echo "GPU $gpu is in use ($used MiB): not starting" >&2; exit 3; }
mkdir -p "$D/runs" "$D/joint-rate"
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=$gpu
PY=/home/xueqi/hq/envs/p1-venv/bin/python
echo "== wsp-d $variant $name start $(date -Is) host=$(hostname) GPU $gpu $(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader) dossier $(sha256sum "$D/inputs/dossier_k20_raw.json" | cut -c1-16)"
"$PY" tools/p1_surgery.py diagnose --variant "$variant" \
  --pairs /home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs/B/data/pairs.json \
  --dossier "$D/inputs/dossier_k20_raw.json" --rate-limit-dir "$D/joint-rate" --allow-teacher-api --output "$D/runs/$name" \
  --base /home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --adapter /home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint \
  --round-dir /home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1 \
  --verified /home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20 \
  --alfworld-data /home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 "$@"
echo "== wsp-d $variant $name exit $? $(date -Is)"
