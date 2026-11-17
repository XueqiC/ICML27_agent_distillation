#!/bin/bash
# w1_overlap_k20.sh (LONI, CPU) — prereg section 19, W1 first measurement with the existing `wsp_adapter_ops overlap`:
# per-module subspace overlap (k = 1 4 16 64) between each readable signal (one gradient of the SFT loss: all tokens /
# Action tokens / Thought tokens, top-64 directions per module) and each working K20 delta (rank 1, rank 16, rank 64),
# plus working-vs-working pairs as the reference for what a real match looks like. Idempotent. Ends W1_DONE.
set -u
R=/work/xueqic/hq/ICML27_wsp; Q=$R/queue; PY=/work/xueqic/hq/tc-alignment/.venv/bin/python; T=/work/xueqic/hq/ICML27_runs/wsp_d6b5199
cd "$T" || exit 2
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=
ck() { echo "$R/cells/$1/stages/train-sft/attempt-000/output/checkpoint"; }
declare -A A=( [grad-all]="$R/derived/c1-k20-g64/checkpoint" [grad-action]="$R/derived/c1a-k20-g64/checkpoint" [grad-thought]="$R/derived/c1t-k20-g64/checkpoint"
               [sft-r1]="$(ck a1-k20-r1-lr2e-4)" [sft-r16]="$(ck a1-k20-r16-lr2e-4)" [sft-r64]="$(ck a0-k20-r64)" [sft-r4]="$(ck a1-k20-r4-lr2e-4)" )
for k in "${!A[@]}"; do [ -f "${A[$k]}/adapter_model.safetensors" ] || { echo "W1_FAIL missing $k ${A[$k]}"; exit 1; }; done
mkdir -p "$R/analysis"
pairs="grad-all:sft-r1 grad-all:sft-r16 grad-all:sft-r64 grad-action:sft-r1 grad-action:sft-r16 grad-action:sft-r64 grad-thought:sft-r1 grad-thought:sft-r16 grad-thought:sft-r64 grad-action:grad-thought sft-r16:sft-r64 sft-r1:sft-r16 sft-r4:sft-r16"
for p in $pairs; do x=${p%%:*}; y=${p##*:}; out="$R/analysis/w1.overlap.$x.$y.json"
  if [ ! -s "$out" ]; then echo "$(date '+%F %T') overlap $x $y"
    "$PY" -m tools.wsp_adapter_ops overlap "${A[$x]}" "${A[$y]}" --k 1 4 16 64 --out "$out" > /dev/null || { echo "W1_FAIL overlap $p"; exit 1; }; fi
  "$PY" - "$out" "$x" "$y" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); s = j["summary"]
def row(tag, d):
    sub = d["subspaces"]
    return "%-22s cos=%+.3f " % (tag, d["frobenius_cosine"] or 0) + " ".join(
        "k%s L=%.3f(ch %.4f) R=%.3f(ch %.4f)" % (k, sub[k]["left"], sub[k]["chance_left"], sub[k]["right"], sub[k]["chance_right"]) for k in sorted(sub, key=int))
print("W1 %s vs %s" % (sys.argv[2], sys.argv[3]))
print("   " + row("mean", s["mean"])); print("   " + row("norm-weighted", s["squared_norm_weighted_mean"]))
for t, d in sorted(s["by_module_type"].items()): print("   " + row(t, d["mean"]))
PY
done
: > "$Q/done/w1.overlap.k20"; echo "$(date '+%F %T') W1_DONE"
