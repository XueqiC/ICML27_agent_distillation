#!/bin/bash
# derive_ladder.sh (LONI, CPU) — spectra and overlaps of the existing E4B trajectory-SFT ladder adapters (seed 0;
# K10/K20 at 16-128 epochs, K30, K100). Read-only on the adapters; outputs analysis/ladder.*.json. Idempotent.
set -u
TT=/work/xueqic/hq/ICML27_runs/wsp_a788e7e; R=/work/xueqic/hq/ICML27_wsp; Q=/work/xueqic/hq/ICML27_wsp/queue; PY=/work/xueqic/hq/tc-alignment/.venv/bin/python
cd "$TT" || exit 2
export PYTHONPATH="$TT/src:$TT" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$R/analysis" "$Q/done" "$Q/logs"; fail=0
an() { local name="$1"; shift
  if [ -f "$R/analysis/$name.json" ]; then echo "SKIP $name"; return; fi
  if "$PY" -m tools.wsp_adapter_ops "$@" --out "$R/analysis/$name.json" > "$Q/logs/analysis.$name.log" 2>&1; then echo "$(date +%T) ANALYSED $name"
  else echo "$(date +%T) ANALYSIS_FAIL $name"; fail=1; fi; }
an ladder.spectrum.k10-e16 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k10-e32 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k10-e64 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k10-e128 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k20-e16 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k20-e32 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k20-e64 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k20-e128 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k30-e64 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k30-e64-s2 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5-s2/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k100-e64 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k100-e64-lr1e-4/stages/train-sft/attempt-000/output/checkpoint
an ladder.spectrum.k100-e64-s2 spectrum /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k100-e64-lr1e-4-s2/stages/train-sft/attempt-000/output/checkpoint
an ladder.overlap.k10-e16.k10-e32 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e32.k10-e64 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e64.k10-e128 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e16.k10-e128 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k20-e16.k20-e32 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k20-e32.k20-e64 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k20-e64.k20-e128 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k20-e16.k20-e128 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e16-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e32.k20-e32 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e64.k20-e64 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k20-e64.k30-e64 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k30-e64.k100-e64 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k100-e64-lr1e-4/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k30-e64.k30-e64-s2 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k30-e64-lr5e-5-s2/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
an ladder.overlap.k10-e128.k20-e128 overlap /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k10-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint /work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e128-lr5e-5/stages/train-sft/attempt-000/output/checkpoint --k 1 4 16
if [ "$fail" = 0 ]; then : > "$Q/done/derive.ladder"; echo "DERIVE_LADDER_DONE"; else echo "DERIVE_LADDER_FAILED"; exit 1; fi
