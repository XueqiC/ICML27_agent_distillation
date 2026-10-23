#!/bin/bash
# Move old-phase LONI data to hpg /orange (user decision 10-22 "A" + "直接走RAI吧"): stream LONI -> rai -> hpg one dir at a
# time (throttled), verify (file count, total bytes, sha256 of the largest file per immediate subdirectory), then delete the
# LONI source. Resumable: completed dirs are recorded in $DONE. Never staged on rai.
set -uo pipefail
P=/home/xueqi/hq/projects/ICML27_agent_distillation
LIST=$P/reports/engineering/loni_cold_move_list_1022.txt
DONE=$P/logs/loni_cold_move.done; LOG=$P/logs/loni_cold_move.log; touch $DONE
SRC=/work/xueqic/hq; DST=/orange/yd24f.fsu/xc25.fsu/tc-cold/loni_20261022
RATE=${RATE:-4000000}
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
manifest() {  # host base dir -> "count bytes" + one "sha path" line per immediate subdir (largest file)
  ssh -o BatchMode=yes "$1" "cd '$2' && find '$3' -type f -printf '%s\t%p\n'" 2>/dev/null | python3 -c '
import sys, collections
rows = [l.rstrip("\n").split("\t", 1) for l in sys.stdin if "\t" in l]
print(len(rows), sum(int(s) for s, _ in rows))
best = {}
for s, p in rows:
    parts = p.split("/"); key = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
    # ties in size are broken by path: find order differs between LONI and /orange (10-22: tc-hotpotqa-scan5)
    if key not in best or (int(s), p) > (int(best[key][0]), best[key][1]): best[key] = (s, p)
for k in sorted(best): print(best[k][1])'
}
sha_of() { ssh -o BatchMode=yes "$1" "cd '$2' && sha256sum $(printf "'%s' " "${@:3}")" 2>/dev/null | awk '{print $1}'; }
wait_hpg() { until ssh -O check hpg 2>/dev/null; do sleep 300; done; }
order=$(ssh -o BatchMode=yes loni "cd $SRC && du -s --block-size=1M $(sed 's#^hq/##' $LIST | tr '\n' ' ') 2>/dev/null" | sort -n | awk '{print $2}')
for d in $order; do
  case "$d" in ""|/*|*..*|ICML27_*|tc-alignment|tc-alignment/.venv*|tc-alignment/envs) log "SKIP unsafe/kept $d"; continue ;; esac
  grep -qx "$d" $DONE && continue
  for attempt in 1 2 3; do
    wait_hpg
    log "copy $d attempt $attempt"
    ssh -o BatchMode=yes loni "cd $SRC && tar cf - '$d'" 2>>$LOG | python3 $P/scripts/ops/throttle_pipe.py $RATE | \
      ssh -o BatchMode=yes hpg "mkdir -p $DST && tar xf - -C $DST" 2>>$LOG
    st=("${PIPESTATUS[@]}")
    [ "${st[0]}" = 0 ] && [ "${st[1]}" = 0 ] && [ "${st[2]}" = 0 ] && break
    log "copy $d FAILED (status ${st[*]})"; sleep 120
  done
  ml=$(manifest loni $SRC "$d"); wait_hpg; mh=$(manifest hpg $DST "$d")
  if [ -z "$ml" ] || [ "$ml" != "$mh" ]; then log "VERIFY MISMATCH $d: local [$(echo "$ml" | head -1)] remote [$(echo "$mh" | head -1)]"; continue; fi
  mapfile -t big < <(echo "$ml" | tail -n +2)
  if [ ${#big[@]} -gt 0 ]; then
    hl=$(sha_of loni $SRC "${big[@]}"); hh=$(sha_of hpg $DST "${big[@]}")
    if [ -z "$hl" ] || [ "$hl" != "$hh" ]; then log "VERIFY SHA MISMATCH $d"; continue; fi
  fi
  log "VERIFIED $d $(echo "$ml" | head -1) (${#big[@]} sha checks)"
  ssh -o BatchMode=yes loni "rm -rf -- '$SRC/${d:?}'" && { echo "$d" >> $DONE; log "DELETED source $SRC/$d"; } || log "delete FAILED $d"
done
log "ALL DONE ($(wc -l < $DONE) dirs)"
