#!/bin/bash
# rsi_wsp_d_merged_all.sh GPU WAIT_PID — all merged-checkpoint evaluations of the D group on ONE rsi GPU, in priority order.
# Replaces rsi_wsp_d1_merged_queue.sh + rsi_wsp_d2_chain.sh (same steps, different order, plus the unedited reference):
#   0. wait for WAIT_PID (an evaluation already running on the GPU) to exit;
#   1. B0 = unedited student through the SAME merged path (name d-raw): the reference for every other number here;
#   2. D1 teacher edit (name d1-teacher; skipped where already done);
#   3. D2: freeze control conditions (controls of tree 9d897cc, python child terminated once B2_4.json exists: B3 / B5 are
#      not D-group controls, prereg section 16), then the D2 teacher edit (d2-teacher);
#   4. five random-direction controls of each session, interleaved (d1-rand0, d2-rand0, d1-rand1, ...);
#   5. second dev140 runs: d1-teacher, d2-teacher, d-raw.
# Each condition: export (surgery tree d7b221c), 20 training tasks (t2), dev140. Stops (exit 3) if the GPU is in use when
# a step is about to start. Start with nohup, record the pid.
set -uo pipefail
gpu=${1:?GPU}; waitpid=${2:?pid to wait for (0 = none)}
D=/home/xueqi/hq/projects/ICML27_wsp_d; O=/home/xueqi/hq/projects/ICML27_wsp/eval
B=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs/B/data/pairs.json
mkdir -p "$D/export_d1" "$D/export_d2" "$D/logs"
busy() { u=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -ge 2000 ]; }
settle() { for i in $(seq 1 36); do busy || return 0; sleep 5; done; return 0; }
echo "== d merged all on GPU $gpu start $(date -Is) pid $$, waiting for pid $waitpid"
if [ "$waitpid" != 0 ]; then while kill -0 "$waitpid" 2>/dev/null; do sleep 15; done; echo "$(date -Is) pid $waitpid gone"; settle; fi
one() { exp=$1; name=$2; which=$3; run=$4; out="$O/$name.merged.$which.$run"
  [ -e "$out/eval/summary.json" ] && { echo "SKIP_DONE $name $which $run"; return 0; }
  [ ! -e "$out" ] || { echo "PARTIAL_EXISTS $out (move it away to rerun)"; return 0; }
  [ -f "$exp/surgery_manifest.json" ] || { echo "NO_EXPORT $exp for $name"; return 0; }
  busy && { echo "$(date -Is) GPU_BUSY before $name $which: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_merged_eval.sh "$gpu" "$exp" "$which" "$out" > "$D/logs/merged_${name}_${which}_${run}.log" 2>&1 < /dev/null \
    && echo "$(date -Is) EVAL_OK $name $which $run" || echo "$(date -Is) EVAL_FAIL $name $which $run"
  settle; }
cond() { tag=$1; c=$2; name=$3; exp="$D/export_$tag/$c"; cj="$D/conditions_$tag/$c.json"
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
freeze_d2() { C=$D/conditions_d2; SUB=$D/runs/d2_s1/submission.json
  [ -s "$C/B2_4.json" ] && return 0
  [ ! -e "$C" ] || { echo "PARTIAL_CONDITIONS $C (move it away)"; return 0; }
  [ -s "$SUB" ] || { echo "no D2 submission"; return 0; }
  busy && { echo "$(date -Is) GPU_BUSY before controls: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_stage.sh 9d897cc "$gpu" controls --pairs "$B" --submission "$SUB" --output "$C" > "$D/logs/controls_d2.log" 2>&1 < /dev/null &
  stage=$!
  for i in $(seq 1 120); do [ -s "$C/B2_4.json" ] && break; kill -0 "$stage" 2>/dev/null || break; sleep 5; done
  [ -s "$C/B2_4.json" ] && sleep 5
  child=$(pgrep -P "$stage" | head -n 1); [ -z "$child" ] || kill -TERM "$child" 2>/dev/null; wait "$stage" 2>/dev/null
  [ -s "$C/B2_4.json" ] && echo "$(date -Is) CONTROLS_FROZEN $(ls "$C" | tr '\n' ' ')" || echo "$(date -Is) CONTROLS_FAIL see logs/controls_d2.log"
  settle; }
cond d1 B0 d-raw
cond d1 B1 d1-teacher
freeze_d2
cond d2 B1 d2-teacher
for i in 0 1 2 3 4; do cond d1 "B2_$i" "d1-rand$i"; cond d2 "B2_$i" "d2-rand$i"; done
one "$D/export_d1/B1" d1-teacher dev140 run2
one "$D/export_d2/B1" d2-teacher dev140 run2
one "$D/export_d1/B0" d-raw dev140 run2
echo "== d merged all done $(date -Is)"
