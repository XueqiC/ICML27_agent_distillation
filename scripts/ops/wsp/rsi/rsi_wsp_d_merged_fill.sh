#!/bin/bash
# rsi_wsp_d_merged_fill.sh GPU WAIT_PID — runs the D2 random-direction controls that rsi_wsp_d_merged_all.sh skipped.
# Cause (my bug, found 11-16 11:52 CST): in that script the helper settle() used the loop variable `i` without `local`,
# so inside `for i in 0 1 2 3 4; do cond d1 B2_$i ...; cond d2 B2_$i ...; done` the second call saw i=1 every time:
# d1-rand0..4 run as planned, but of the D2 controls only d2-rand1 ran (names and conditions stayed consistent:
# d2-randN always means conditions_d2/B2_N). This script waits for that queue to end and runs d2-rand0, 2, 3, 4.
# Each: export (tree d7b221c), 20 training tasks (t2), dev140. Stops (exit 3) if the GPU is in use before a step.
set -uo pipefail
gpu=${1:?GPU}; waitpid=${2:?pid to wait for (0 = none)}
D=/home/xueqi/hq/projects/ICML27_wsp_d; O=/home/xueqi/hq/projects/ICML27_wsp/eval
B=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs/B/data/pairs.json
busy() { local u; u=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -ge 2000 ]; }
settle() { local k; for k in $(seq 1 36); do busy || return 0; sleep 5; done; return 0; }
echo "== d merged fill on GPU $gpu start $(date -Is) pid $$, waiting for pid $waitpid"
if [ "$waitpid" != 0 ]; then while kill -0 "$waitpid" 2>/dev/null; do sleep 15; done; echo "$(date -Is) pid $waitpid gone"; settle; fi
one() { local exp=$1 name=$2 which=$3 run=$4; local out="$O/$name.merged.$which.$run"
  [ -e "$out/eval/summary.json" ] && { echo "SKIP_DONE $name $which $run"; return 0; }
  [ ! -e "$out" ] || { echo "PARTIAL_EXISTS $out (move it away to rerun)"; return 0; }
  [ -f "$exp/surgery_manifest.json" ] || { echo "NO_EXPORT $exp for $name"; return 0; }
  busy && { echo "$(date -Is) GPU_BUSY before $name $which: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_merged_eval.sh "$gpu" "$exp" "$which" "$out" > "$D/logs/merged_${name}_${which}_${run}.log" 2>&1 < /dev/null \
    && echo "$(date -Is) EVAL_OK $name $which $run" || echo "$(date -Is) EVAL_FAIL $name $which $run"
  settle; }
cond() { local tag=$1 c=$2 name=$3; local exp="$D/export_$tag/$c" cj="$D/conditions_$tag/$c.json"
  if [ ! -f "$exp/surgery_manifest.json" ]; then
    [ ! -e "$exp" ] || { echo "PARTIAL_EXPORT $exp (move it away)"; return 0; }
    [ -s "$cj" ] || { echo "NO_CONDITION $cj"; return 0; }
    busy && { echo "$(date -Is) GPU_BUSY before export $tag $c: stopping"; exit 3; }
    bash /home/xueqi/hq/rsi_wsp_d_stage.sh d7b221c "$gpu" export --pairs "$B" --condition "$cj" --output "$exp" > "$D/logs/export_merged_${tag}_$c.log" 2>&1 < /dev/null
    [ -f "$exp/surgery_manifest.json" ] && echo "$(date -Is) EXPORT_OK $tag $c" || { echo "$(date -Is) EXPORT_FAIL $tag $c"; settle; return 0; }
    settle
  fi
  one "$exp" "$name" t2 run1
  one "$exp" "$name" dev140 run1; }
for n in 0 2 3 4; do cond d2 "B2_$n" "d2-rand$n"; done
# the repeat dev140 runs of the first queue, in case it stopped before them (skipped when already done)
one "$D/export_d1/B1" d1-teacher dev140 run2
one "$D/export_d2/B1" d2-teacher dev140 run2
one "$D/export_d1/B0" d-raw dev140 run2
echo "== d merged fill done $(date -Is)"
