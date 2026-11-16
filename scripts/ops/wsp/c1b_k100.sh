#!/bin/bash
# c1b_k100.sh (LONI, ONE GPU) — prereg section 10: the C1 step-size minimum sat on the lower edge of the grid, so the
# grid is extended downward / refined (target norms 0.125 0.1875 0.25 0.3125 0.375). For every rank the step with the lowest TRAINING
# objective over BOTH grids becomes derived/c1-k100-r<rank>-opt. Idempotent. Ends C1B_DONE + marker done/c1b.k100.
set -u
R=/work/xueqic/hq/ICML27_wsp; Q=/work/xueqic/hq/ICML27_wsp/queue; PY=/work/xueqic/hq/tc-alignment/.venv/bin/python; CELL=/work/xueqic/hq/ICML27_wsp/cells/sc-k100
set -a; . "$Q/jobs/prep.sc-k100.env"; set +a
source "$HPG_CONFIG"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
[ -f "$Q/done/c1.k100" ] || { echo "C1B_FAIL c1 not finished"; exit 1; }
step() { echo "$(date '+%F %T') $*"; }
fresh() { if [ -e "$1" ]; then mv -- "$1" "$R/derived/_failed/$(basename "$1").$(date +%s)"; fi; }
for r in 1 16 64; do D="$R/derived/c1-k100-g$r"; S="$R/analysis/c1-k100.norm.g$r.json"; L1="$R/analysis/c1-k100.linesearch.r$r.json"; L2="$R/analysis/c1-k100.linesearch2.r$r.json"; F="$R/derived/c1-k100-r$r-opt"
  scales=$("$PY" -c 'import json,sys; n=json.load(open(sys.argv[1]))["summary"]["squared_frobenius_norm"]**0.5; print(" ".join(repr(float(t)/n) for t in sys.argv[2:]))' "$S" 0.125 0.1875 0.25 0.3125 0.375) || exit 1
  if [ ! -f "$L2" ]; then step "line search 2 rank $r: scales $scales"
    "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$L2" --adapter "$D/checkpoint" --scales $scales || { echo "C1B_FAIL linesearch2 $r"; exit 1; }; fi
  eta=$("$PY" -c 'import json,sys; rows=[x for f in sys.argv[1:] for x in json.load(open(f))["rows"] if x.get("scale", 0) > 0]; b=min(rows,key=lambda x:x["loss"]); print(repr(b["scale"]))' "$L1" "$L2") || exit 1
  if [ ! -f "$F/checkpoint/training_state.json" ]; then fresh "$F"; step "opt rank $r eta $eta"
    CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops scale "$D/checkpoint" --out "$F" --factor "$eta" || { echo "C1B_FAIL scale $r"; exit 1; }; fi
done
: > "$Q/done/c1b.k100"; step C1B_DONE
