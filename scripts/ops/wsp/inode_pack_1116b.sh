#!/bin/bash
# inode_pack_1116b.sh (LONI) — second packing batch: the four finished SFT-checkpoint evaluations. Same procedure as
# inode_rescue_1116.sh: tar eval/episodes -> eval/episodes.tar, verify (tar status, tar -d compare, entry count), then
# remove the episodes directory. Summaries stay in place.
set -u
pack() { cd "$1" || return 1; [ -d episodes ] || { echo "SKIP $1 (no episodes dir)"; return 1; }
  [ -f summary.json ] || { echo "SKIP $1 (no summary: not finished)"; return 1; }
  [ ! -e episodes.tar ] || { echo "SKIP $1 (episodes.tar exists)"; return 1; }
  n=$(find episodes | wc -l); tar -cf episodes.tar episodes || { echo "TAR_FAIL $1"; return 1; }
  tar -df episodes.tar > /dev/null 2>&1 || { echo "COMPARE_FAIL $1"; return 1; }
  m=$(tar -tf episodes.tar | wc -l); [ "$n" = "$m" ] || { echo "COUNT_FAIL $1 $n $m"; return 1; }
  echo "$(date '+%T') PACKED $1 entries=$n bytes=$(stat -c %s episodes.tar)"; }
( pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u160.run1/a0-k20-r64-u160/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u160.run1/a0-k20-r64-u160/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u160.run1/a0-k20-r64-u160/eval/episodes" ) &
wait
lfs quota -h -u $USER /work | sed -n 3p
echo PACK_DONE
