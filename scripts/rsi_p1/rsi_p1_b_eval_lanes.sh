#!/bin/bash
# P1 setting-B M2/M3 `evaluate` (prereg §13): 6 shards on ONE GPU in 3 lanes (<=3 processes per card, as in A).
# Lane 0 runs shard 0 (creates plan.json) then 3; lanes 1/2 start after plan.json exists and run 1,4 / 2,5.
# rsi_p1_stage.sh always exits 0 (its last command is echo), so success = its "exit 0" log line. Args: GPU
set -u
gpu=${1:?GPU}
P=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs
L=/home/xueqi/hq/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[ -e "$P/B/eval" ] && { echo "B/eval exists, refusing" >&2; exit 2; }
run() {
  bash /home/xueqi/hq/rsi_p1_stage.sh "$gpu" evaluate --pairs "$P/B/data/pairs.json" --conditions "$P/B/conditions" \
    --output "$P/B/eval" --shard "$1/6" > "$L/p1_B_eval_s$1.log" 2>&1 < /dev/null
  grep -a -q '^== p1 stage \[evaluate\] exit 0 ' "$L/p1_B_eval_s$1.log"
}
lane() {
  for s in "$@"; do
    run "$s" || { echo "== shard $s FAILED $(date -Is)"; return 1; }
    echo "== shard $s done $(date -Is)"
  done
}
echo "== B eval lanes start $(date -Is) GPU $gpu"
lane 0 3 & a=$!
until [ -e "$P/B/eval/plan.json" ]; do
  kill -0 "$a" 2>/dev/null || { echo "== shard 0 ended before plan.json"; wait "$a"; exit 1; }
  sleep 10
done
sleep 60
lane 1 4 & b=$!
lane 2 5 & c=$!
wait "$a"; ra=$?; wait "$b"; rb=$?; wait "$c"; rc=$?
echo "== B eval lanes done $(date -Is) rc=$ra/$rb/$rc"
