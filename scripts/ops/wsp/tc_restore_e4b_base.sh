#!/bin/bash
# tc_restore_e4b_base.sh — runs ON hpg. COPY (not move) the Gemma-4-E4B-it HF snapshot from cold storage back to the
# dept hf-cache, where every Paper-1 training / eval job expects it (MODEL_PATH). The /orange copy is left untouched.
# Holds the shared /orange move lock for the whole transfer (ctl rule for transfers over ~10 GB). Ends RESTORE_E4B_DONE.
set -u
O=/orange/yd24f.fsu/xc25.fsu
SRC=/orange/yd24f.fsu/xc25.fsu/wm-cold/hf-cache/hub/models--google--gemma-4-E4B-it
DST=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--google--gemma-4-E4B-it
LOCK=$O/.orange_move.lock
[ -d "$SRC/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2" ] || { echo "NO_SRC"; exit 2; }
[ ! -e "$DST" ] && [ ! -L "$DST" ] || { echo "DST_EXISTS"; exit 2; }
exec 9>>"$LOCK"
flock -n 9 || { echo "LOCK_BUSY"; exit 1; }
echo "$(date '+%F %T %Z') start host=$(hostname)"
rsync -a "$SRC/" "$DST/"; rc=$?
echo "$(date '+%F %T %Z') rsync rc=$rc"
[ "$rc" -eq 0 ] || { echo "RESTORE_FAILED"; exit 1; }
sc=$(find "$SRC" -type f | wc -l); dc=$(find "$DST" -type f | wc -l)
sl=$(find "$SRC" -type l | wc -l); dl=$(find "$DST" -type l | wc -l)
sb=$(find "$SRC" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%d", s}')
db=$(find "$DST" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%d", s}')
big=$(find "$SRC" -type f -printf '%s %P\n' | sort -rn | head -n 1 | cut -d' ' -f2-)
s1=$(sha256sum "$SRC/$big" | cut -d' ' -f1); s2=$(sha256sum "$DST/$big" | cut -d' ' -f1)
echo "files $sc $dc | links $sl $dl | bytes $sb $db | largest $big | sha $s1 $s2"
bad=$(find "$DST/snapshots" -xtype l | wc -l)
echo "dangling links in DST snapshots: $bad"
if [ "$sc" = "$dc" ] && [ "$sl" = "$dl" ] && [ "$sb" = "$db" ] && [ "$s1" = "$s2" ] && [ "$bad" = 0 ]; then
  echo "$(date '+%F %T %Z') RESTORE_E4B_DONE"
else
  echo "RESTORE_MISMATCH"; exit 1
fi
