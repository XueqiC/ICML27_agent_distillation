#!/bin/bash
# c1s_k20.sh (LONI, ONE GPU, tree wsp_a5e1a76) — prereg section 15: where the SFT objective sits (thought vs action
# tokens) for raw / SFT / the all-token one-step edit, and the one-step edits from the ACTION tokens only and from the
# THOUGHT tokens only (gradient, norms, training objective by part, overlaps, scaled copies for the closed-loop check).
# Idempotent. Ends C1S_DONE + marker done/c1s.k20.
set -u
R=/work/xueqic/hq/ICML27_wsp; Q=/work/xueqic/hq/ICML27_wsp/queue; PY=/work/xueqic/hq/tc-alignment/.venv/bin/python; CELL=/work/xueqic/hq/ICML27_wsp/cells/sd-k20; LIKE=/work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
set -a; . "$Q/jobs/prep.sd-k20.env"; set +a
source "$HPG_CONFIG"
cd "$PROJECT_ROOT" || exit 2
[ "$(cat GIT_COMMIT)" = a5e1a76 ] || { echo "C1S_FAIL wrong tree"; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$R/analysis" "$R/derived/_failed" "$Q/done" "$Q/logs"
[ -f "$Q/done/prep.sd-k20" ] || { echo "C1S_FAIL score cell not prepared"; exit 1; }
[ -f "$Q/done/c1.k20" ] || { echo "C1S_FAIL c1 not finished"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
step() { echo "$(date '+%F %T') $*"; }
fresh() { if [ -e "$1" ]; then mv -- "$1" "$R/derived/_failed/$(basename "$1").$(date +%s)"; fi; }
A="$R/analysis"
scales() { "$PY" -c 'import json,sys; n=json.load(open(sys.argv[1]))["summary"]["squared_frobenius_norm"]**0.5; print(" ".join(repr(float(t)/n) for t in sys.argv[2:]))' "$@"; }
step "guard: token classes and label masks on the real examples (CPU)"
CUDA_VISIBLE_DEVICES= "$PY" - "$CELL" <<'PYEOF' || { echo "C1S_FAIL guard"; exit 1; }
import collections, sys
from baselines.common.shared_history import pack_shared_history
from tools import wsp_onestep as one
from train.dataset import load_tokenizer
cell = one.load_cell(sys.argv[1]); tok = load_tokenizer(cell["identity"]["snapshot"])
total, first, per_turn = collections.Counter(), collections.Counter(), collections.Counter()
for i, row in enumerate(cell["examples"]):
    packed = pack_shared_history(row); cls = one.classified_tokens(packed, tok)
    total.update(t["class"] for t in cls)
    turns = collections.defaultdict(set)
    for t in cls:
        turns[t["turn"]].add(t["class"])
        if t["thought_first"]:
            first[tok.decode([t["label_id"]], skip_special_tokens=False)] += 1
    per_turn.update("+".join(sorted(v)) for v in turns.values())
    for span in ("thought", "action"):
        want = sorted(t["position"] for t in cls if t["class"] == span)
        got = [p for p, v in enumerate(pack_shared_history(one.restrict_span(row, tok, span))["labels"]) if v != -100]
        assert got == want, (i, span, len(got), len(want))
    if i == 0:
        print("example 0, turn 0:", [(tok.decode([t["label_id"]], skip_special_tokens=False), t["class"][0], int(t["thought_first"])) for t in cls if t["turn"] == 0])
print("GUARD_OK tokens by class", dict(total), "| turns by classes present", dict(per_turn))
print("first thought tokens (top 12)", first.most_common(12))
PYEOF
if [ ! -f "$A/c1s-k20.score.ref.json" ]; then step "score raw + SFT by part"
  "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$A/c1s-k20.score.ref.json" --adapter "$LIKE" --scales 1 --per-token-out "$A/c1s-k20.tokens.ref.jsonl" || { echo "C1S_FAIL score ref"; exit 1; }; fi
if [ ! -f "$A/c1s-k20.score.c1.json" ]; then s=$(scales "$A/c1-k20.norm.g64.json" 0.0625 0.125 0.25) || exit 1; step "score all-token one-step edit by part: $s"
  "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$A/c1s-k20.score.c1.json" --adapter "$R/derived/c1-k20-g64/checkpoint" --scales $s --per-token-out "$A/c1s-k20.tokens.c1.jsonl" || { echo "C1S_FAIL score c1"; exit 1; }; fi
grad() { span=$1; name=$2; shift 2; G="$R/derived/$name-k20-g64"
  if [ ! -f "$G/checkpoint/training_state.json" ]; then fresh "$G"; step "gradient --span $span"
    "$PY" -m tools.wsp_onestep gradient --cell "$CELL" --like "$LIKE" --out "$G" --rank 64 --span "$span" || return 1; fi
  [ -f "$A/$name-k20.norm.g64.json" ] || CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops spectrum "$G/checkpoint" --out "$A/$name-k20.norm.g64.json" || return 1
  if [ ! -f "$A/$name-k20.linesearch.r64.json" ]; then s=$(scales "$A/$name-k20.norm.g64.json" "$@") || return 1; step "$name line search by part: $s"
    "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$A/$name-k20.linesearch.r64.json" --adapter "$G/checkpoint" --scales $s || return 1; fi; }
grad action c1a 0.0625 0.125 0.25 0.5 1 2 || { echo "C1S_FAIL action"; exit 1; }
grad thought c1t 0.0625 0.125 0.25 0.5 || { echo "C1S_FAIL thought"; exit 1; }
export CUDA_VISIBLE_DEVICES=
pair() { o="$A/c1s-k20.overlap.$1.json"; [ -f "$o" ] || "$PY" -m tools.wsp_adapter_ops overlap "$2" "$3" --out "$o" || return 1; }
step overlaps
pair all__action "$R/derived/c1-k20-g64/checkpoint" "$R/derived/c1a-k20-g64/checkpoint" || { echo "C1S_FAIL overlap"; exit 1; }
pair all__thought "$R/derived/c1-k20-g64/checkpoint" "$R/derived/c1t-k20-g64/checkpoint" || { echo "C1S_FAIL overlap"; exit 1; }
pair action__thought "$R/derived/c1a-k20-g64/checkpoint" "$R/derived/c1t-k20-g64/checkpoint" || { echo "C1S_FAIL overlap"; exit 1; }
pair all__sft "$R/derived/c1-k20-g64/checkpoint" "$LIKE" || { echo "C1S_FAIL overlap"; exit 1; }
pair action__sft "$R/derived/c1a-k20-g64/checkpoint" "$LIKE" || { echo "C1S_FAIL overlap"; exit 1; }
pair thought__sft "$R/derived/c1t-k20-g64/checkpoint" "$LIKE" || { echo "C1S_FAIL overlap"; exit 1; }
one() { name=$1; norm=$2; tag=$3; F="$R/derived/$name-k20-r64-$tag"
  [ -f "$F/checkpoint/training_state.json" ] && return 0
  eta=$(scales "$A/$name-k20.norm.g64.json" "$norm") || return 1
  fresh "$F"; step "$name norm $norm eta $eta"
  "$PY" -m tools.wsp_adapter_ops scale "$R/derived/$name-k20-g64/checkpoint" --out "$F" --factor "$eta" || return 1; }
one c1a 0.125 n125 || { echo "C1S_FAIL scale c1a 0.125"; exit 1; }
one c1a 0.25 n250 || { echo "C1S_FAIL scale c1a 0.25"; exit 1; }
one c1a 0.5 n500 || { echo "C1S_FAIL scale c1a 0.5"; exit 1; }
one c1a 1 n1000 || { echo "C1S_FAIL scale c1a 1"; exit 1; }
one c1a 2 n2000 || { echo "C1S_FAIL scale c1a 2"; exit 1; }
one c1t 0.125 n125 || { echo "C1S_FAIL scale c1t 0.125"; exit 1; }
one c1t 0.25 n250 || { echo "C1S_FAIL scale c1t 0.25"; exit 1; }
: > "$Q/done/c1s.k20"; step C1S_DONE
