#!/bin/bash
# inode_rescue_1116.sh (LONI, CPU) — /work file-count quota hit (3,998,496 of 4,000,000) on 2026-11-16 03:5x CST: each
# adapter evaluation writes 35-50 thousand small files. For every listed evaluation: pack eval/episodes into
# eval/episodes.tar next to it, verify (tar exit status, tar -d compare against the files, entry count), and only then
# remove the episodes directory. Nothing is lost: the same bytes stay in one file. Summaries and per_task.tsv stay
# where they are. Units killed by the quota are afterwards moved to _failed/quota_20261116/ so they can be rerun.
set -u
E=/work/xueqic/hq/ICML27_wsp/eval
pack() { cd "$1" || return 1; [ -d episodes ] || { echo "SKIP $1 (no episodes dir)"; return 1; }
  [ ! -e episodes.tar ] || { echo "SKIP $1 (episodes.tar exists)"; return 1; }
  n=$(find episodes | wc -l); tar -cf episodes.tar episodes || { echo "TAR_FAIL $1"; return 1; }
  tar -df episodes.tar > /dev/null 2>&1 || { echo "COMPARE_FAIL $1"; return 1; }
  m=$(tar -tf episodes.tar | wc -l); [ "$n" = "$m" ] || { echo "COUNT_FAIL $1 $n $m"; return 1; }
  echo "$(date '+%T') PACKED $1 entries=$n bytes=$(stat -c %s episodes.tar)"; }
(
pack "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/google/gemma-4-E4B-it/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/google/gemma-4-E4B-it/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/google/gemma-4-E4B-it/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r4/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r4/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r4/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/old-k20-s0/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/old-k20-s0/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/old-k20-s0/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1/a0-k20-r64-u10/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u26.run1/a0-k100-r64-u26/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u26.run1/a0-k100-r64-u26/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u26.run1/a0-k100-r64-u26/eval/episodes"
) &
(
pack "/work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/google/gemma-4-E4B-it/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/google/gemma-4-E4B-it/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/old-k20-s0.run1/google/gemma-4-E4B-it/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64.run1/c1-k20-r64/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64.run1/c1-k20-r64/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/c1-k20-r64.run1/c1-k20-r64/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/old-k100-s0/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/old-k100-s0/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/old-k100-s0.run1/old-k100-s0/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1/a0-k20-r64-u40/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1/a0-k20-r64-u80/eval/episodes"
) &
(
pack "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r1.run1/c1-k20-r1/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r1.run1/c1-k20-r1/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/c1-k20-r1.run1/c1-k20-r1/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64.run1/a0-k20-r64/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64.run1/a0-k20-r64/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64.run1/a0-k20-r64/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s2.run1/old-k100-s2/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/old-k100-s2.run1/old-k100-s2/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/old-k100-s2.run1/old-k100-s2/eval/episodes"
pack "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r16/eval" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r16/eval/episodes" && echo "REMOVED /work/xueqic/hq/ICML27_wsp/eval/c1-k20-r4+r16.run1/c1-k20-r16/eval/episodes"
) &
wait
mkdir -p /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u10.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k20-r64-u10.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u40.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k20-r64-u40.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u80.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k20-r64-u80.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u26.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u26.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k100-r64-u26.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u160.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k20-r64-u160.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k20-r64-u160.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u104.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u104.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k100-r64-u104.run1"
[ -e "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u208.run1" ] && mv -- "/work/xueqic/hq/ICML27_wsp/eval/a0-k100-r64-u208.run1" /work/xueqic/hq/ICML27_wsp/_failed/quota_20261116/ && echo "MOVED a0-k100-r64-u208.run1"
lfs quota -h -u $USER /work | sed -n 3p
echo RESCUE_DONE
