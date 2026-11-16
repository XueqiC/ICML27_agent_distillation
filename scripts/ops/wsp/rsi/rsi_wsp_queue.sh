#!/bin/bash
# rsi_wsp_queue.sh GPU LISTFILE — runs the units of queue/lists/LISTFILE one after another on ONE rsi GPU.
# Stops at once (exit 3) if the GPU is in use by anyone when a unit is about to start: the card is shared, we yield.
# A unit that fails is logged and skipped; finished units get queue/done/<unit>. Start with nohup, record the pid.
set -uo pipefail
gpu=${1:?gpu}; list=${2:?list}
Q=/home/xueqi/hq/projects/ICML27_wsp/queue
mkdir -p "$Q/done" "$Q/logs"
exec 9>"$Q/.gpu$gpu.lock"; flock -n 9 || { echo "QUEUE_ALREADY_RUNNING gpu $gpu"; exit 2; }
echo "== queue $list on GPU $gpu start $(date -Is) pid $$"
while IFS= read -r unit <&7; do
  [ -n "$unit" ] || continue
  [ ! -e "$Q/done/$unit" ] || { echo "SKIP_DONE $unit"; continue; }
  bash "$Q/rsi_wsp_eval_unit.sh" "$unit" "$gpu" > "$Q/logs/$unit.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" -eq 0 ]; then : > "$Q/done/$unit"; echo "$(date -Is) UNIT_OK $unit"
  elif [ "$rc" -eq 3 ]; then echo "$(date -Is) GPU_BUSY before $unit: stopping the queue"; exit 3
  else echo "$(date -Is) UNIT_FAIL $unit rc=$rc"; fi
done 7< "$Q/lists/$list"
echo "== queue $list done $(date -Is)"
