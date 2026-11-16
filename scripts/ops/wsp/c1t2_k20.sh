#!/bin/bash
# c1t2_k20.sh (hpg, CPU) — prereg section 14: scaled copies of the K20 one-step-gradient edit for the closed-loop
# step-size check on the 20 TRAINING tasks. derived/c1-k20-r<rank>-n<1000*norm> = (norm / ||G_rank||_F) * G_rank.
# Idempotent. Ends C1T2_DONE + marker done/c1t2.k20.
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp; Q=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue; PY=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/.venv/bin/python
set -a; . "$Q/jobs/prep.sc-k20.env"; set +a
source "$PROJECT_ROOT/configs/hpg.env"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
[ -f "$Q/done/c1.k20" ] || { echo "C1T2_FAIL c1 not finished"; exit 1; }
mkdir -p "$R/derived/_failed"
step() { echo "$(date '+%F %T') $*"; }
fresh() { if [ -e "$1" ]; then mv -- "$1" "$R/derived/_failed/$(basename "$1").$(date +%s)"; fi; }
one() { r=$1; norm=$2; tag=$3; D="$R/derived/c1-k20-g$r"; S="$R/analysis/c1-k20.norm.g$r.json"; F="$R/derived/c1-k20-r$r-$tag"
  [ -f "$F/checkpoint/training_state.json" ] && return 0
  eta=$("$PY" -c 'import json,sys; n=json.load(open(sys.argv[1]))["summary"]["squared_frobenius_norm"]**0.5; print(repr(float(sys.argv[2])/n))' "$S" "$norm") || return 1
  fresh "$F"; step "rank $r norm $norm eta $eta"
  "$PY" -m tools.wsp_adapter_ops scale "$D/checkpoint" --out "$F" --factor "$eta" || return 1; }
one 64 0.03125 n031 || { echo "C1T2_FAIL 64 0.03125"; exit 1; }
one 64 0.0625 n062 || { echo "C1T2_FAIL 64 0.0625"; exit 1; }
one 64 0.125 n125 || { echo "C1T2_FAIL 64 0.125"; exit 1; }
one 64 0.1875 n188 || { echo "C1T2_FAIL 64 0.1875"; exit 1; }
one 64 0.375 n375 || { echo "C1T2_FAIL 64 0.375"; exit 1; }
one 1 0.0625 n062 || { echo "C1T2_FAIL 1 0.0625"; exit 1; }
one 1 0.125 n125 || { echo "C1T2_FAIL 1 0.125"; exit 1; }
one 1 0.375 n375 || { echo "C1T2_FAIL 1 0.375"; exit 1; }
ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
miss=$("$PY" -c 'import json,sys,os; ids=json.load(open(sys.argv[1]))["task_ids"]; print(sum(not os.path.isdir(os.path.join(sys.argv[2],i)) for i in ids))' "$Q/lists/t2_k20_train.json" "$ROOT") || exit 1
echo "training task directories missing under the data root: $miss"
[ "$miss" = 0 ] || { echo "C1T2_FAIL task files missing"; exit 1; }
: > "$Q/done/c1t2.k20"; step C1T2_DONE
