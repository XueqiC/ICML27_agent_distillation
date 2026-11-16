#!/bin/bash
# shared_directions.sh (hpg, CPU) — shared output directions of the three seeds (K20, K100) and their logit lens.
set -u
H=/blue/fsu-compsci-dept/xc25.fsu/hq; R=$H/ICML27_wsp; Q=$R/queue; PY=$H/tc-alignment/.venv/bin/python
SNAP=$H/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
OLD=$H/ICML27_agent_distillation/reports/runs/baselines; C=stages/train-sft/attempt-000/output/checkpoint
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$R/analysis" "$Q/logs" "$Q/done"; fail=0
run() { local name="$1"; shift
  if [ -f "$R/analysis/$name.json" ]; then echo "SKIP $name"; return; fi
  if "$PY" -I "$Q/derive/wsp_shared_directions.py" "$SNAP" "$R/analysis/$name" "$@" > "$Q/logs/analysis.$name.log" 2>&1; then echo "$(date +%T) ANALYSED $name"
  else echo "$(date +%T) ANALYSIS_FAIL $name"; fail=1; fi; }
run shared.k20 s0=$OLD/e4b-tsft-k20-e32-lr5e-5/$C s1=$OLD/e4b-tsft-k20-e32-lr5e-5-s1/$C s2=$OLD/e4b-tsft-k20-e32-lr5e-5-s2/$C
run shared.k100 s0=$OLD/e4b-tsft-k100-e64-lr1e-4/$C s1=$OLD/e4b-tsft-k100-e64-lr1e-4-s1/$C s2=$OLD/e4b-tsft-k100-e64-lr1e-4-s2/$C
run shared.k20-k100.s0 k20=$OLD/e4b-tsft-k20-e32-lr5e-5/$C k100=$OLD/e4b-tsft-k100-e64-lr1e-4/$C
run shared.all6 k20s0=$OLD/e4b-tsft-k20-e32-lr5e-5/$C k20s1=$OLD/e4b-tsft-k20-e32-lr5e-5-s1/$C k20s2=$OLD/e4b-tsft-k20-e32-lr5e-5-s2/$C k100s0=$OLD/e4b-tsft-k100-e64-lr1e-4/$C k100s1=$OLD/e4b-tsft-k100-e64-lr1e-4-s1/$C k100s2=$OLD/e4b-tsft-k100-e64-lr1e-4-s2/$C
if [ "$fail" = 0 ]; then : > "$Q/done/shared_directions"; echo "SHARED_DONE"; else echo "SHARED_FAILED"; exit 1; fi
