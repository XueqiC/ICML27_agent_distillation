#!/bin/bash
# c5_bf16_check.sh (LONI, CPU job, derive/ script) — prereg section 18, C5: before sending a search end point through
# the merged-checkpoint route, measure whether bf16 rounding can represent the edit at all. Reference rows: a normal-size
# adapter (w2-r1-equal-x0.5, dev140 58). Output: derived/c5-bf16-check/report.jsonl. Ends C5CHK_DONE.
set -u
R=/work/xueqic/hq/ICML27_wsp; PY=/work/xueqic/hq/envs/p1-venv/bin/python
BASE=/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2/model.safetensors
point() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["eval_dir"].rstrip("/").split("/")[-1])' "$R/search/$1/state.json" "$2"; }
specs=(); for sp in c5-k20-rand0:current c5-k20:current c5-k20:best_so_far; do s=${sp%%:*}; k=${sp#*:}; id=$(point "$s" "$k") || exit 1
  specs+=("$s-$k=$R/search/$s/adapters/$id/combined/checkpoint"); done
specs+=("w2-r1-equal-x0.5=$R/derived/w2-r1-equal-x0.5/checkpoint")
mkdir -p "$R/derived/c5-bf16-check"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" /work/xueqic/hq/ICML27_wsp/queue/c5_bf16_check.py "$BASE" "${specs[@]}" | tee "$R/derived/c5-bf16-check/report.jsonl.tmp"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "C5CHK_FAIL"; exit 1; }
mv "$R/derived/c5-bf16-check/report.jsonl.tmp" "$R/derived/c5-bf16-check/report.jsonl"; echo "C5CHK_DONE"
