#!/bin/bash
# d_c4.sh (LONI, ONE GPU) — prereg section 18, C4: fixed-rule additive directions (4 alphas, K20 selection, 3 random controls),
# then the merged-checkpoint exports of every condition. Resumable. Ends D_C4_DONE.
source /work/xueqic/hq/ICML27_wsp_d/scripts/loni_wspd_common.sh
if [ ! -f "$D/conditions_c4/selection.json" ]; then
  if [ -e "$D/conditions_c4" ]; then mkdir -p "$D/_failed"; mv "$D/conditions_c4" "$D/_failed/conditions_c4.$(date +%s)"; fi
  run additive --pairs "$P" --output "$D/conditions_c4" || { echo D_C4_FAIL additive; exit 1; }
fi
exports c4 B0 C4_a0.5 C4_a1 C4_a2 C4_a4 C4_rand0 C4_rand1 C4_rand2 || { echo D_C4_FAIL export; exit 1; }
: > /work/xueqic/hq/ICML27_wsp/queue/done/d.c4; step D_C4_DONE
