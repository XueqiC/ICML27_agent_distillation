#!/bin/bash
# d_d3.sh (LONI, ONE GPU) — prereg section 18: teacher session variant d3 (gpt-6-luna; dossier with the multi-student table),
# the frozen controls (teacher edit B1, unedited B0, five same-structure random controls), then the merged-checkpoint
# exports. Resumable after the session (a finished submission is never repeated). Ends D_D3_DONE.
source /work/xueqic/hq/ICML27_wsp_d/scripts/loni_wspd_common.sh
session d3 d3 d3_s1 || { echo D_D3_FAIL session; exit 1; }
exports d3 B0 B1 B2_0 B2_1 B2_2 B2_3 B2_4 || { echo D_D3_FAIL export; exit 1; }
: > /work/xueqic/hq/ICML27_wsp/queue/done/d.d3; step D_D3_DONE
