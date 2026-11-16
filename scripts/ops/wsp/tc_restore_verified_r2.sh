#!/bin/bash
# tc_restore_verified_r2.sh ROWS.tsv — runs ON hpg. COPY the listed big files (sha256<TAB>size<TAB>relpath, rows taken
# from the 11-11/12 move manifests) from /orange cold storage back to their original /blue paths. Source files stay.
# Every file: size checked before, sha256 checked after against the manifest value. Holds the /orange move lock.
set -u
ROWS="${1:?rows.tsv}"
O=/orange/yd24f.fsu/xc25.fsu
SRCROOT=/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_agent_distillation
DSTROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
exec 9>>"$O/.orange_move.lock"
flock -n 9 || { echo "LOCK_BUSY"; exit 1; }
echo "$(date '+%F %T %Z') start host=$(hostname) rows=$(wc -l < "$ROWS")"
fail=0; n=0
while IFS=$'\t' read -r sha size rel; do
  case "$rel" in data/verified/seeds_full_train_r2/manifest.json|data/verified/seeds_full_train_r2/recovery/source_files.json|data/verified/seeds_full_train_r2/recovery/source/manifest.json|data/verified/seeds_full_train_r2/teacher_budget/reservations.sqlite3) ;; *) echo "REFUSE path $rel"; fail=1; continue;; esac
  src="$SRCROOT/$rel"; dst="$DSTROOT/$rel"
  [ -f "$src" ] && [ "$(stat -c %s "$src")" = "$size" ] || { echo "BAD_SRC $rel"; fail=1; continue; }
  [ -d "$(dirname "$dst")" ] || { echo "NO_DST_DIR $rel"; fail=1; continue; }
  if [ -e "$dst" ]; then echo "DST_EXISTS $rel"; fail=1; continue; fi
  cp -p -- "$src" "$dst.restoring" && got=$(sha256sum "$dst.restoring" | cut -d' ' -f1) || { echo "COPY_FAILED $rel"; fail=1; continue; }
  if [ "$got" = "$sha" ]; then mv -n -- "$dst.restoring" "$dst" && { echo "RESTORED $size $rel"; n=$((n+1)); }
  else echo "SHA_MISMATCH $rel $got"; fail=1; fi
done < "$ROWS"
echo "$(date '+%F %T %Z') restored=$n fail=$fail"
[ "$fail" = 0 ] && echo "RESTORE_VERIFIED_R2_DONE" || { echo "RESTORE_VERIFIED_R2_FAILED"; exit 1; }
