#!/bin/bash
# inode_pack_1116c.sh (LONI) — third packing batch: finished training-task rollout units (a, b, c1s2), the step-0.125
# dev140 run and b2-k20-top50. Same procedure as inode_rescue_1116.sh (tar, tar -d compare, entry count, then remove).
set -u
pack() { cd "$1" || return 1; [ -d episodes ] || { echo "SKIP $1 (no episodes dir)"; return 1; }
  [ -f summary.json ] || { echo "SKIP $1 (no summary: not finished)"; return 1; }
  [ ! -e episodes.tar ] || { echo "SKIP $1 (episodes.tar exists)"; return 1; }
  n=$(find episodes | wc -l); tar -cf episodes.tar episodes || { echo "TAR_FAIL $1"; return 1; }
  tar -df episodes.tar > /dev/null 2>&1 || { echo "COMPARE_FAIL $1"; return 1; }
  m=$(tar -tf episodes.tar | wc -l); [ "$n" = "$m" ] || { echo "COUNT_FAIL $1 $n $m"; return 1; }
  echo "$(date '+%T') PACKED $1 entries=$n"; }
(
pack "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64-opt.run1/c1-k20-r64-opt/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64-opt.run1/c1-k20-r64-opt/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64-opt.run1/c1-k20-r64-opt/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n031/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n031/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n031/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1/eval/episodes"
) &
(
pack "/work/xueqic/hq/ICML27_wsp/eval/b2-k20-top50.run1/b2-k20-top50/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/b2-k20-top50.run1/b2-k20-top50/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/b2-k20-top50.run1/b2-k20-top50/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n062/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n062/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n062/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64-n375/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64-n375/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r64-n375/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n375/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n375/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n375/eval/episodes"
) &
(
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/google/gemma-4-E4B-it/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/google/gemma-4-E4B-it/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/google/gemma-4-E4B-it/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n125/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n125/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n125/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n062/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n062/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n062/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n031/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n031/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n031/eval/episodes"
) &
(
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/old-k20-s0/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/old-k20-s0/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/old-k20-s0/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n188/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n188/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-a.run1/c1-k20-r64-n188/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n125/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n125/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1-b.run1/c1-k20-r1-n125/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n062/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n062/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/t2-k20-c1s2.run1/c1a-k20-r64-n062/eval/episodes"
) &
wait
lfs quota -h -u $USER /work | sed -n 3p
echo PACK_DONE
