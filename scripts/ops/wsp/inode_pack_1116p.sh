#!/bin/bash
# inode_pack_1116p.sh (LONI) — sixteenth packing batch: second runs of K100 rank 16 / 4 / 1 and MLP-only. Same procedure as inode_rescue_1116.sh (tar, tar -d compare, entry count, then remove).
set -u
pack() { cd "$1" || return 1; [ -d episodes ] || { echo "SKIP $1 (no episodes dir)"; return 1; }
  [ -f summary.json ] || { echo "SKIP $1 (no summary: not finished)"; return 1; }
  [ ! -e episodes.tar ] || { echo "SKIP $1 (episodes.tar exists)"; return 1; }
  n=$(find episodes | wc -l); tar -cf episodes.tar episodes || { echo "TAR_FAIL $1"; return 1; }
  tar -df episodes.tar > /dev/null 2>&1 || { echo "COMPARE_FAIL $1"; return 1; }
  m=$(tar -tf episodes.tar | wc -l); [ "$n" = "$m" ] || { echo "COUNT_FAIL $1 $n $m"; return 1; }
  echo "$(date '+%T') PACKED $1 entries=$n"; }
( pack "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r16/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r16/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r16/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r4/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r4/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a1-k100-r16+r4.run2/a1-k100-r4/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r1.run2/a1-k100-r1/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a1-k100-r1.run2/a1-k100-r1/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a1-k100-r1.run2/a1-k100-r1/eval/episodes" ) &
( pack "/work/xueqic/hq/ICML27_wsp/eval/a2-k100-r16-mlp.run2/a2-k100-r16-mlp/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a2-k100-r16-mlp.run2/a2-k100-r16-mlp/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a2-k100-r16-mlp.run2/a2-k100-r16-mlp/eval/episodes" ) &
wait
lfs quota -h -u $USER /work | sed -n 3p
echo PACK_DONE
