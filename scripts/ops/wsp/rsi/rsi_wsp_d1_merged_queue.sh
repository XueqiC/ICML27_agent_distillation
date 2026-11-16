#!/bin/bash
# rsi_wsp_d1_merged_queue.sh GPU — D1 evaluation through merged checkpoints on ONE rsi GPU (run after the D2 session):
# export (surgery tree d7b221c) each frozen condition, then training-task rollouts (t2) and dev140 for each.
# Order: teacher edit first, then the five random-direction controls; a second dev140 run of the teacher edit last.
# Stops (exit 3) if the GPU is in use when a step is about to start. Start with nohup, record the pid.
set -uo pipefail
gpu=${1:?GPU}
D=/home/xueqi/hq/projects/ICML27_wsp_d; E=$D/export_d1; O=/home/xueqi/hq/projects/ICML27_wsp/eval
B=/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a/reports/p1_runs/B/data/pairs.json
mkdir -p "$E" "$D/logs"
busy() { u=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -ge 2000 ]; }
one() { cond=$1; name=$2; which=$3; run=$4; out="$O/$name.merged.$which.$run"
  [ -e "$out/eval/summary.json" ] && { echo "SKIP_DONE $name $which $run"; return 0; }
  [ ! -e "$out" ] || { echo "PARTIAL_EXISTS $out (move it away to rerun)"; return 0; }
  busy && { echo "$(date -Is) GPU_BUSY before $name $which: stopping"; exit 3; }
  bash /home/xueqi/hq/rsi_wsp_d_merged_eval.sh "$gpu" "$E/$cond" "$which" "$out" > "$D/logs/merged_${name}_${which}_${run}.log" 2>&1 < /dev/null \
    && echo "$(date -Is) EVAL_OK $name $which $run" || echo "$(date -Is) EVAL_FAIL $name $which $run"; }
echo "== d1 merged queue on GPU $gpu start $(date -Is) pid $$"
for pair in B1:d1-teacher B2_0:d1-rand0 B2_1:d1-rand1 B2_2:d1-rand2 B2_3:d1-rand3 B2_4:d1-rand4; do
  cond=${pair%%:*}; name=${pair##*:}
  if [ ! -f "$E/$cond/surgery_manifest.json" ]; then
    [ ! -e "$E/$cond" ] || { echo "PARTIAL_EXPORT $E/$cond (move it away)"; continue; }
    busy && { echo "$(date -Is) GPU_BUSY before export $cond: stopping"; exit 3; }
    bash /home/xueqi/hq/rsi_wsp_d_stage.sh d7b221c "$gpu" export --pairs "$B" --condition "$D/conditions_d1/$cond.json" --output "$E/$cond" > "$D/logs/export_merged_$cond.log" 2>&1 < /dev/null \
      && echo "$(date -Is) EXPORT_OK $cond" || { echo "$(date -Is) EXPORT_FAIL $cond"; continue; }
  fi
  one "$cond" "$name" t2 run1
  one "$cond" "$name" dev140 run1
done
one B1 d1-teacher dev140 run2
echo "== d1 merged queue done $(date -Is)"
