#!/bin/bash
# d_d4.sh (LONI, ONE GPU) — prereg section 18: teacher session variant d4 (gpt-6-luna; dossier with the multi-student table),
# the frozen controls (teacher edit B1, unedited B0, five same-structure random controls), then the merged-checkpoint
# exports. Resumable after the session (a finished submission is never repeated). Ends D_D4_DONE.
source /work/xueqic/hq/ICML27_wsp_d/scripts/loni_wspd_common.sh
session d4 d4 d4_s1 || { echo D_D4_FAIL session; exit 1; }
exports d4 B0 B1 B2_0 B2_1 B2_2 B2_3 B2_4 || { echo D_D4_FAIL export; exit 1; }
: > /work/xueqic/hq/ICML27_wsp/queue/done/d.d4; step D_D4_DONE
