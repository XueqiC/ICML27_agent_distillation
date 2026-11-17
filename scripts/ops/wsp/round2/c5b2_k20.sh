#!/bin/bash
# c5b2_k20.sh (LONI, ONE GPU, tree 27a83f4) — C5 control: the same search with the random 6-direction basis (same module
# sets and per-module norms). v2 with slimming, see c5a2_k20.sh. Resumable. Ends C5B_DONE.
set -u
R=/work/xueqic/hq/ICML27_wsp; Q=/work/xueqic/hq/ICML27_wsp/queue; PY=/work/xueqic/hq/tc-alignment/.venv/bin/python; T=/work/xueqic/hq/ICML27_runs/wsp_27a83f4
LIKE=/work/xueqic/hq/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
step() { echo "$(date '+%F %T') $*"; }
# slim DIR: once a candidate evaluation is finished, tar its per-request logs (episodes/*/blobs, episodes/*/student_calls),
# verify the tar and remove them. The scorer only needs events.jsonl / episode.json / per_task.tsv, so resume still works;
# this cuts about 5,600 files per candidate to about 120 (the /work file quota is 4,000,000 for the whole account).
slim() { local d=$1 n; case "$d" in /work/xueqic/hq/ICML27_wsp/search/*/evaluations/*) ;; *) return 0;; esac
  [ -f "$d/summary.json" ] && [ -f "$d/per_task.tsv" ] && [ -d "$d/episodes" ] && [ ! -e "$d/call_logs.tar" ] || return 0
  ( cd "$d" || exit 1
    find episodes -mindepth 2 -maxdepth 2 \( -name blobs -o -name student_calls \) -type d | sort > .slim.list
    [ -s .slim.list ] || { rm -f .slim.list; exit 0; }
    n=$(xargs -a .slim.list -d '\n' find | wc -l)
    tar -cf call_logs.tar.tmp -T .slim.list && tar -df call_logs.tar.tmp > /dev/null 2>&1 && [ "$(tar -tf call_logs.tar.tmp | wc -l)" = "$n" ] || { echo "SLIM_FAIL $d"; rm -f call_logs.tar.tmp .slim.list; exit 1; }
    mv call_logs.tar.tmp call_logs.tar && sed "s#^#$d/#" .slim.list | xargs -d '\n' rm -rf && rm -f .slim.list && echo "$(date '+%T') SLIMMED $(basename "$d") entries=$n" ) }
slim_loop() { while :; do for e in "$1"/evaluations/*; do [ -d "$e" ] && slim "$e"; done; sleep 60; done; }
search() { # search BASIS_NAME OUT_NAME
  export PROJECT_ROOT=$T HPG_CONFIG=/work/xueqic/hq/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env CONFIG=$T/configs/base_e4b.yaml MODEL_PATH=/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 BASIS=$R/derived/$1 LIKE \
    TASKS=$Q/lists/t2_k20_train.json VERIFIED=/work/xueqic/hq/ICML27_data/verified/seeds_full_train_r2/subsets/k20 OUT=$R/search/$2 WORKERS=16 BATCH_INVARIANT=1 \
    VLLM_CACHE_ROOT=/work/xueqic/hq/tools/vllm-cache TRITON_CACHE_DIR=/work/xueqic/hq/tools/triton-cache VLLM_NO_USAGE_STATS=1 SED_HASH_CACHE_DIR=/work/xueqic/hq/ICML27_v0_runs/sed-hash-receipts
  mkdir -p "$R/search/$2"
  slim_loop "$R/search/$2" & local sl=$!
  bash "$T/scripts/wsp_search_job.sh"; local rc=$?
  kill "$sl" 2>/dev/null; wait "$sl" 2>/dev/null
  for e in "$R/search/$2"/evaluations/*; do [ -d "$e" ] && slim "$e"; done
  return $rc; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$Q/done/c5.basis.k20" ] && [ -f "$R/derived/c5-basis-k20-rand0/basis.json" ] || { echo "C5B_FAIL basis not ready"; exit 1; }
step "search (random basis)"; search c5-basis-k20-rand0 c5-k20-rand0 || { echo "C5B_FAIL search"; tail -n 3 "$R/search/c5-k20-rand0"/suite_logs/*.log 2>/dev/null | tail -n 12; exit 1; }
: > "$Q/done/c5.search.k20-rand0"; step C5B_DONE
