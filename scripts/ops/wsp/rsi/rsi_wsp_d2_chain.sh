#!/bin/bash
# rsi_wsp_d2_chain.sh GPU WAIT_PID — D2 post-processing on ONE rsi GPU, started while the D1 merged queue (WAIT_PID) runs:
#   1. wait for WAIT_PID to exit;
#   2. freeze the D2 control conditions (surgery tree 9d897cc, `controls`): B0, B1 (teacher submission), B2_0..4 (five
#      random-direction controls of the same structure). The command would go on to B3 / B5 (fixed rule, auto-LoRA), which
#      are not D-group controls (prereg section 16): the python child is terminated once B2_4.json exists, as for D1;
#   3. export each condition as a merged bf16 checkpoint (tree d7b221c) and evaluate it on the 20 training tasks (t2) and
#      on dev140, teacher edit first; second dev140 run of the teacher edit last.
# Stops (exit 3) if the GPU is in use when a step is about to start. Start with nohup, record the pid.
set -uo pipefail
gpu=${1:?GPU}; waitpid=${2:?pid of the D1 merged queue}
D=/home/xueqi/hq/projects/ICML27_wsp_d; C=$D/conditions_d2; E=$D/export_d2; O=/home/xueqi/hq/projects/ICML27_wsp/eval
B=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs/B/data/pairs.json
SUB=$D/runs/d2_s1/submission.json
mkdir -p "$E" "$D/logs"
busy() { u=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -ge 2000 ]; }
echo "== d2 chain on GPU $gpu start $(date -Is) pid $$, waiting for pid $waitpid"
while kill -0 "$waitpid" 2>/dev/null; do sleep 30; done
echo "$(date -Is) pid $waitpid gone"
if [ ! -s "$C/B2_4.json" ]; then
  [ ! -e "$C" ] || { echo "PARTIAL_CONDITIONS $C (move it away)"; exit 4; }
  [ -s "$SUB" ] || { echo "no submission"; exit 4; }
  busy && { echo "$(date -Is) GPU_BUSY before controls: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_stage.sh 9d897cc "$gpu" controls --pairs "$B" --submission "$SUB" --output "$C" > "$D/logs/controls_d2.log" 2>&1 < /dev/null &
  stage=$!
  for i in $(seq 1 120); do
    [ -s "$C/B2_4.json" ] && break
    kill -0 "$stage" 2>/dev/null || break
    sleep 5
  done
  if [ -s "$C/B2_4.json" ]; then
    sleep 5
    child=$(pgrep -P "$stage" | head -n 1)
    [ -z "$child" ] || kill -TERM "$child" 2>/dev/null
    wait "$stage" 2>/dev/null
    echo "$(date -Is) CONTROLS_FROZEN $(ls "$C" | tr '\n' ' ')"
  else
    child=$(pgrep -P "$stage" | head -n 1); [ -z "$child" ] || kill -TERM "$child" 2>/dev/null; wait "$stage" 2>/dev/null
    echo "$(date -Is) CONTROLS_FAIL (no B2_4.json after 10 min or stage exited): see logs/controls_d2.log"; exit 4
  fi
  for i in $(seq 1 24); do busy || break; sleep 5; done
fi
one() { cond=$1; name=$2; which=$3; run=$4; out="$O/$name.merged.$which.$run"
  [ -e "$out/eval/summary.json" ] && { echo "SKIP_DONE $name $which $run"; return 0; }
  [ ! -e "$out" ] || { echo "PARTIAL_EXISTS $out (move it away to rerun)"; return 0; }
  busy && { echo "$(date -Is) GPU_BUSY before $name $which: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_merged_eval.sh "$gpu" "$E/$cond" "$which" "$out" > "$D/logs/merged_${name}_${which}_${run}.log" 2>&1 < /dev/null \
    && echo "$(date -Is) EVAL_OK $name $which $run" || echo "$(date -Is) EVAL_FAIL $name $which $run"; }
for pair in B1:d2-teacher B2_0:d2-rand0 B2_1:d2-rand1 B2_2:d2-rand2 B2_3:d2-rand3 B2_4:d2-rand4; do
  cond=${pair%%:*}; name=${pair##*:}
  if [ ! -f "$E/$cond/surgery_manifest.json" ]; then
    [ ! -e "$E/$cond" ] || { echo "PARTIAL_EXPORT $E/$cond (move it away)"; continue; }
    busy && { echo "$(date -Is) GPU_BUSY before export $cond: stopping"; exit 3; }
    bash /home/xueqi/hq/rsi_wsp_d_stage.sh d7b221c "$gpu" export --pairs "$B" --condition "$C/$cond.json" --output "$E/$cond" > "$D/logs/export_merged_d2_$cond.log" 2>&1 < /dev/null \
      && [ -f "$E/$cond/surgery_manifest.json" ] && echo "$(date -Is) EXPORT_OK $cond" || { echo "$(date -Is) EXPORT_FAIL $cond"; continue; }
    for i in $(seq 1 24); do busy || break; sleep 5; done
  fi
  one "$cond" "$name" t2 run1
  one "$cond" "$name" dev140 run1
done
one B1 d2-teacher dev140 run2
echo "== d2 chain done $(date -Is)"
