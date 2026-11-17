#!/bin/bash
# inode_pack_auto.sh (LONI, CPU job or login node) — packs every FINISHED evaluation folder of the WSP workspace that still
# has an unpacked episodes/ directory (tar, tar -d compare, entry count, then remove), to keep /work under the file-count
# quota. Training-task rollouts (unit name t2-*) get their progress report written first (tools/wsp_progress.py), because
# the progress score needs the episode files. The closed-loop search folders are not touched here. Idempotent. Ends PACK_DONE.
set -u
R=/work/xueqic/hq/ICML27_wsp; T=/work/xueqic/hq/ICML27_runs/wsp_27a83f4; PY=/work/xueqic/hq/envs/eval-46b8527/bin/python
D=/work/xueqic/hq/ICML27_data/verified/seeds_full_train_r2/subsets
pack() { cd "$1" || return 1; [ -d episodes ] && [ -f summary.json ] && [ ! -e episodes.tar ] || return 1
  local n m; n=$(find episodes | wc -l); tar -cf episodes.tar episodes || { echo "TAR_FAIL $1"; rm -f episodes.tar; return 1; }
  tar -df episodes.tar > /dev/null 2>&1 || { echo "COMPARE_FAIL $1"; return 1; }
  m=$(tar -tf episodes.tar | wc -l); [ "$n" = "$m" ] || { echo "COUNT_FAIL $1 $n $m"; return 1; }
  echo "$(date '+%T') PACKED $1 entries=$n"; }
find "$R/eval" -mindepth 3 -maxdepth 5 -type f -name summary.json -path '*/eval/summary.json' | sort > "$R/queue/logs/pack_auto.$$.list"
while IFS= read -r s <&7; do d=$(dirname "$s"); [ -d "$d/episodes" ] || continue
  case "$d" in /work/xueqic/hq/ICML27_wsp/eval/*) ;; *) echo "SKIP_PATH $d"; continue;; esac
  unit=${d#/work/xueqic/hq/ICML27_wsp/eval/}; unit=${unit%%/*}
  case "$unit" in t2-*)
    if [ ! -s "$d/progress.json" ]; then case "$unit" in *k100*) v=k100;; *) v=k20;; esac
      ( cd "$T" && SED_HASH_CACHE_DIR=/work/xueqic/hq/ICML27_v0_runs/sed-hash-receipts PYTHONPATH="$T/src:$T" CUDA_VISIBLE_DEVICES= "$PY" tools/wsp_progress.py --eval "$d" --verified "$D/$v" --out "$d/progress.json" > /dev/null 2> "$d/progress.err" ) \
        && echo "PROGRESS $unit $("$PY" -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j["successes"], j["episodes"], round(j["mean_progress"], 4))' "$d/progress.json")" \
        || { echo "PROGRESS_FAIL $unit (kept unpacked): $(tail -n 1 "$d/progress.err" | cut -c1-160)"; continue; }
    fi;; esac
  pack "$d" && rm -rf "/work/xueqic/hq/ICML27_wsp/eval/${d#/work/xueqic/hq/ICML27_wsp/eval/}/episodes" && echo "REMOVED $unit"
done 7< "$R/queue/logs/pack_auto.$$.list"
rm -f "$R/queue/logs/pack_auto.$$.list"
echo "files: $(lfs quota -u $USER /work 2>/dev/null | grep -E '^ +/work' | awk '{print $6}')"
echo "sbatch from here: $(sbatch --test-only -A loni_depedlab04 -p single -N1 -n1 -t 00:05:00 --wrap true 2>&1 | tail -n 1 | cut -c1-120)"
echo PACK_DONE
