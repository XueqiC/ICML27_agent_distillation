#!/bin/bash
# Poll one detached hpg move step from rai and exit with ONE event line. Low frequency, one ssh per poll,
# and it stops at the FIRST ssh failure (never retries in a loop).
# usage: bash scripts/ops/tc_move_watch.sh <copy|verify|delete|extra> <log path on hpg> [interval_s=300] [--once]
#   copy/verify: done = a "PASS:" line followed by the script's "source kept" line
#   delete:      done = a "DELETED" line
#   extra:       done = an "EXTRA_PASS" line
# Events: DONE …, FAILED … (FAIL/REFUSE/BUSY/EXTRA_FAIL line), DIED (lock free, no verdict -> state unknown,
# run --verify-only before anything else), SSH_FAIL.
# Liveness = the move flock, NOT pgrep: hpg has several login nodes and a new ssh master can land on a
# different one than the node running the move (11-11: move on login10, ssh on login11, pgrep said 0).
# Lustre is mounted with `flock`, so the lock is cluster-wide: held = some move is alive somewhere.
# (If another session's move holds the lock after ours died without a verdict, this keeps saying RUNNING;
# the 2 h background cap is the check-in for that.)
set -u
STEP="${1:?step}"; LOG="${2:?log}"; IV="${3:-300}"; ONCE="${4:-}"
case "$LOG" in /orange/yd24f.fsu/xc25.fsu/*) ;; *) echo "bad log path"; exit 2;; esac
LOCK=/orange/yd24f.fsu/xc25.fsu/.orange_move.lock
while true; do
  out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 hpg "grep -aE '^(FAIL|REFUSE|BUSY|EXTRA_FAIL)|^[0-9:]{8} (PASS:|DELETED |EXTRA_PASS|EXTRA_FAIL|source kept)' '$LOG' | tail -n 4; echo '@@LAST'; tail -n 2 '$LOG' | cut -c1-300; ( exec 9<'$LOCK' || exit 0; if flock -n -s 9; then echo '@@RUN 0'; else echo '@@RUN 1'; fi )" 2>/dev/null); rc=$?
  now=$(TZ=America/Chicago date '+%H:%M %Z')
  if [ $rc -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^@@RUN '; then echo "$now SSH_FAIL rc=$rc"; exit 1; fi
  marks=$(printf '%s\n' "$out" | awk '/^@@LAST$/{exit} {print}')
  last=$(printf '%s\n' "$out" | awk '/^@@RUN /{exit} f{print} /^@@LAST$/{f=1}')
  run=$(printf '%s\n' "$out" | sed -n 's/^@@RUN //p')
  ev=""
  if printf '%s\n' "$marks" | grep -qE '^(FAIL|REFUSE|BUSY|EXTRA_FAIL)| EXTRA_FAIL'; then ev=FAILED
  elif [ "$STEP" = delete ] && printf '%s\n' "$marks" | grep -q ' DELETED '; then ev=DONE
  elif [ "$STEP" = extra ] && printf '%s\n' "$marks" | grep -q ' EXTRA_PASS'; then ev=DONE
  elif { [ "$STEP" = copy ] || [ "$STEP" = verify ]; } && printf '%s\n' "$marks" | grep -q ' PASS: ' && printf '%s\n' "$marks" | grep -q ' source kept'; then ev=DONE
  elif [ "$run" = 0 ]; then ev=DIED
  fi
  if [ -n "$ev" ] || [ "$ONCE" = "--once" ]; then
    echo "$now ${ev:-RUNNING} step=$STEP lock_held=$run"; printf '%s\n' "$marks" | sed 's/^/  mark: /'; printf '%s\n' "$last" | sed 's/^/  last: /'
    [ "$ev" = DONE ] && exit 0; [ -z "$ev" ] && exit 0; exit 1
  fi
  sleep "$IV"
done
