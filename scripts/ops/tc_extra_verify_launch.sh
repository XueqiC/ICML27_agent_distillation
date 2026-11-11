#!/bin/bash
# Launch tc_extra_verify.sh detached on hpg (read-only deep check of a finished orange_move copy), log on /orange.
# usage: bash scripts/ops/tc_extra_verify_launch.sh <name> SRC DST      (or: <name> --stage-only)
# Holds ctl's global move lock while it runs, so no orange_move starts during the check and it never runs beside one.
# Success of the launch = the "launched pid" line; the verdict is the log's last line (EXTRA_PASS / EXTRA_FAIL / BUSY).
# This script always prints the log path; do not gate on its exit status alone.
set -u
NAME="${1:?name}"; SRC="${2:?src or --stage-only}"; DST="${3:-}"
case "$NAME" in *[!A-Za-z0-9._-]*|"") echo "bad name"; exit 2;; esac
STAGE=/orange/yd24f.fsu/xc25.fsu/tc-cold/_ops
LOCK=/orange/yd24f.fsu/xc25.fsu/.orange_move.lock
LOCAL="$(cd "$(dirname "$0")" && pwd)/tc_extra_verify.sh"
LSUM=$(sha256sum "$LOCAL" | cut -d' ' -f1)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 hpg"
RSUM=$($SSH "mkdir -p $STAGE/logs && { [ -f $STAGE/tc_extra_verify.sh ] && sha256sum $STAGE/tc_extra_verify.sh | cut -d' ' -f1 || echo none; }" 2>/dev/null | grep -v '^Loading' | tail -n 1)
if [ "$RSUM" != "$LSUM" ]; then
  scp -q -o BatchMode=yes -o ConnectTimeout=20 "$LOCAL" "hpg:$STAGE/tc_extra_verify.sh" || { echo "stage failed"; exit 1; }
  RSUM=$($SSH "sha256sum $STAGE/tc_extra_verify.sh | cut -d' ' -f1" 2>/dev/null | grep -v '^Loading' | tail -n 1)
fi
[ "$RSUM" = "$LSUM" ] || { echo "staged sha mismatch: rai=$LSUM hpg=$RSUM"; exit 1; }
echo "staged ok ${LSUM:0:16}"
[ "$SRC" = "--stage-only" ] && exit 0
[ -n "$DST" ] || { echo "dst missing"; exit 2; }
case "$SRC$DST" in *..*|*\'*) echo "REFUSE: no '..' or quotes in paths"; exit 2;; esac
LOG="$STAGE/logs/$NAME.extra.$(date -u +%Y%m%dT%H%M%SZ).log"
$SSH "setsid nohup bash -c 'exec 9>$LOCK; flock -n 9 || { echo BUSY: another move holds $LOCK; exit 3; }; bash $STAGE/tc_extra_verify.sh \"$SRC\" \"$DST\"' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!" 2>/dev/null | grep -v '^Loading'
echo "log: $LOG"
