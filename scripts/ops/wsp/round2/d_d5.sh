#!/bin/bash
# d_d5.sh (LONI, ONE GPU) — prereg section 19, W3: teacher session variant d5 (tree 82eae94): the teacher is GIVEN the
# direction pairs of the working rank-1 K20 delta (magnitudes withheld) and must compose closed-form edits from them;
# then the frozen controls (B0, B1, five same-structure random controls) and the merged-checkpoint exports. Diagnostic only.
# Resumable after the session. Ends D_D5_DONE.
export TREE=/work/xueqic/hq/ICML27_runs/repair_eval_82eae94 EXPECT=82eae94
source /work/xueqic/hq/ICML27_wsp_d/scripts/loni_wspd_common.sh
O=$D/inputs/oracle_r1.json
if [ ! -s "$O" ]; then
  step "oracle directions from the rank-1 working adapter"
  CUDA_VISIBLE_DEVICES= "$PY" tools/p1_oracle_directions.py --adapter $W/ICML27_wsp/cells/a1-k20-r1-lr2e-4/stages/train-sft/attempt-000/output/checkpoint --out "$O" || { echo D_D5_FAIL oracle; exit 1; }
fi
session d5 d5 d5_s1 --oracle-directions "$O" || { echo D_D5_FAIL session; exit 1; }
exports d5 B0 B1 B2_0 B2_1 B2_2 B2_3 B2_4 || { echo D_D5_FAIL export; exit 1; }
: > /work/xueqic/hq/ICML27_wsp/queue/done/d.d5; step D_D5_DONE
