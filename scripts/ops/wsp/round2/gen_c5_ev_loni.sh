#!/bin/bash
# gen_c5_ev_loni.sh — runs ON LONI. Prereg section 18, C5 "评测": writes the final dev140 evaluation jobs (ordinary adapter
# path, two runs) for one point of a finished C5 search and appends them to lists/r2d.list, which the pass script reads.
# Usage: gen_c5_ev_loni.sh SEARCH_NAME end|best      (end = the search's current point, best = its best-so-far point)
# The jobs reuse the environment of an existing K20 adapter evaluation (same tree, same VERIFIED view = K20, the view the
# gradient basis was computed on). If the edit is too small for the serving check, the job fails there and the point has
# to go through the merged-checkpoint route instead (prereg section 17); nothing here loosens the check.
set -euo pipefail
R=/work/xueqic/hq/ICML27_wsp; Q=$R/queue; S=${1:?search name}; P=${2:?end|best}
case "$P" in end) key=current;; best) key=best_so_far;; *) echo "REFUSE point=$P"; exit 2;; esac
TPL=$Q/jobs/ev.w2-r1-equal-x0.5.run1
[ -f "$TPL.env" ] || { echo "NO_TEMPLATE $TPL.env"; exit 1; }
ev=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["eval_dir"])' "$R/search/$S/state.json" "$key")
cid=$(basename "$ev"); A=$R/search/$S/adapters/$cid/combined/checkpoint
[ -f "$A/adapter_model.safetensors" ] || { echo "NO_ADAPTER $A"; exit 1; }
name=$S-$P
for i in 1 2; do
  j=ev.$name.run$i
  sed -e "s#^ADAPTERS=.*#ADAPTERS='$name=$A'#" -e "s#^OUT=.*#OUT='$R/eval/$name.run$i'#" "$TPL.env" > "$Q/jobs/$j.env"
  grep -q -F "ADAPTERS='$name=$A'" "$Q/jobs/$j.env" && grep -q -F "OUT='$R/eval/$name.run$i'" "$Q/jobs/$j.env" || { echo "BAD_ENV $j"; exit 1; }
  printf 'KIND=gpu\nGPUS=1\nCPUS=8\nSTAGE_SCRIPT=scripts/hpg_eval_adapters.slurm\nTIME=00:50:00\nREQUIRE=%s\nFORBID=%s\n' \
    "$A/adapter_model.safetensors" "$R/eval/$name.run$i" > "$Q/jobs/$j.res"
  grep -q -x -F "$j" "$Q/lists/r2d.list" 2>/dev/null || echo "$j" >> "$Q/lists/r2d.list"
  echo "WROTE $j -> $A"
done
