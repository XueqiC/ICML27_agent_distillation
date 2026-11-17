#!/bin/bash
# wsp_driver.sh — runs as ONE small CPU job on LONI (single partition) so that the round-2 queue keeps moving when rai is
# down (rai crashed twice on 2026-11-16 / 17 and nothing was submitted for 15 h and 5.5 h). Every 10 minutes it runs the
# current pass script (which submits only what is ready, respects the job caps and the file quota) and appends the output
# to queue/logs/driver.log. Stops by itself after HOURS (default 20) or when queue/STOP_DRIVER exists.
Q=/work/xueqic/hq/ICML27_wsp/queue; HOURS=${HOURS:-20}; end=$((SECONDS + HOURS * 3600))
echo "$(date '+%F %T') driver start on $(hostname), until +${HOURS}h"
while [ $SECONDS -lt $end ] && [ ! -e "$Q/STOP_DRIVER" ]; do
  P=$(cat "$Q/CURRENT_PASS" 2>/dev/null); [ -f "$Q/$P" ] || P=tc_wsp_pass7_loni.sh
  { bash "$Q/$P"; echo; } >> "$Q/logs/driver.log" 2>&1
  echo "$(date '+%F %T') pass $P done: $(squeue -u $USER -h | wc -l) jobs in queue"
  sleep 600
done
echo "$(date '+%F %T') driver stop"
