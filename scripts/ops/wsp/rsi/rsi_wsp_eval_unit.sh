#!/bin/bash
# rsi_wsp_eval_unit.sh UNIT GPU — ONE adapter evaluation unit on rsi (no scheduler; same launcher as hpg / LONI).
# Reads queue/jobs/UNIT.env (KEY='value' lines). Refuses an existing output, a wrong tree, or a GPU someone is using.
set -uo pipefail
unit=${1:?unit}; gpu=${2:?gpu}
Q=/home/xueqi/hq/projects/ICML27_wsp/queue
envf="$Q/jobs/$unit.env"; [ -f "$envf" ] || { echo "NO_ENV $unit"; exit 2; }
set -a; . "$envf"; set +a
[ ! -e "$OUT" ] || { echo "OUT_EXISTS $OUT"; exit 2; }
[ "$(cat "$PROJECT_ROOT/GIT_COMMIT")" = "$EXPECT_COMMIT" ] || { echo "TREE_MISMATCH"; exit 2; }
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) || exit 2
[ "$used" -lt 2000 ] || { echo "GPU_BUSY $gpu ${used}MiB"; exit 3; }
export CUDA_VISIBLE_DEVICES=$gpu
echo "== unit $unit start $(date -Is) host=$(hostname) GPU $gpu $(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader) tree $EXPECT_COMMIT"
cd "$PROJECT_ROOT" || exit 2
bash scripts/hpg_eval_adapters.slurm
rc=$?
echo "== unit $unit exit $rc $(date -Is)"
exit $rc
