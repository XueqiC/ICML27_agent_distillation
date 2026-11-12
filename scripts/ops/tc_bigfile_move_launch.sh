#!/bin/bash
# Launch tc_bigfile_move.sh detached on hpg (survives ssh drops), log on /orange.
# usage (from the project dir on rai):
#   bash scripts/ops/tc_bigfile_move_launch.sh NAME ROOT DSTROOT LOCAL_LIST [--delete-source|--verify-only]
# Stages the script (sha256-checked) and the list (sha256-checked), then prints "launched pid N" and the log
# path. Its exit status is not a gate: require the "launched pid" line, then watch the log with
#   bash scripts/ops/tc_move_watch.sh <copy|verify|delete> <log> 300
set -u
NAME="${1:?name}"; ROOT="${2:?root}"; DSTROOT="${3:?dstroot}"; LLIST="${4:?local list}"; FLAG="${5:-}"
case "$NAME" in *[!A-Za-z0-9._-]*|"") echo "bad name"; exit 2;; esac
case "$FLAG" in ""|--delete-source|--verify-only) ;; *) echo "bad flag"; exit 2;; esac
[ -s "$LLIST" ] || { echo "list missing or empty: $LLIST"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"; LOCAL="$HERE/tc_bigfile_move.sh"
STAGE=/orange/yd24f.fsu/xc25.fsu/tc-cold/_ops
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 hpg"
STEP=copy; [ "$FLAG" = "--delete-source" ] && STEP=delete; [ "$FLAG" = "--verify-only" ] && STEP=verify
U=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$STAGE/logs/$NAME.big$STEP.$U.log"; RLIST="$STAGE/bigmove_lists/$NAME.$U.list"
LSUM=$(sha256sum "$LOCAL" | cut -d' ' -f1); QSUM=$(sha256sum "$LLIST" | cut -d' ' -f1)
$SSH "mkdir -p $STAGE/logs $STAGE/bigmove_lists" 2>/dev/null || { echo "ssh failed"; exit 1; }
scp -q -o BatchMode=yes "$LOCAL" "hpg:$STAGE/tc_bigfile_move.sh" || { echo "stage failed (script)"; exit 1; }
scp -q -o BatchMode=yes "$LLIST" "hpg:$RLIST" || { echo "stage failed (list)"; exit 1; }
R=$($SSH "sha256sum $STAGE/tc_bigfile_move.sh '$RLIST' | cut -d' ' -f1 | tr '\n' ' '" 2>/dev/null | grep -v '^Loading' | tail -n 1)
[ "$R" = "$LSUM $QSUM " ] || { echo "staged sha mismatch: want '$LSUM $QSUM ' got '$R'"; exit 1; }
echo "staged ok: script ${LSUM:0:16} list ${QSUM:0:16} ($(wc -l < "$LLIST") paths)"
$SSH "setsid nohup bash $STAGE/tc_bigfile_move.sh '$NAME' '$ROOT' '$DSTROOT' '$RLIST' $FLAG > '$LOG' 2>&1 < /dev/null & echo launched pid \$! on \$(hostname)" 2>/dev/null | grep -v '^Loading'
echo "log: $LOG"
