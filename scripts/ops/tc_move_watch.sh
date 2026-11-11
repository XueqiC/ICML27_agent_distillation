#!/bin/bash
# Poll one detached hpg move step from rai and exit with ONE event line. Low frequency, one ssh per poll,
# and it stops at the FIRST ssh failure (never retries in a loop).
# usage: bash scripts/ops/tc_move_watch.sh <copy|verify|delete|extra> <log path on hpg> [interval_s=300] [--once]
#   copy/verify: done = a "PASS:" line and no orange_move process left
#   delete:      done = a "DELETED" line
#   extra:       done = an "EXTRA_PASS" line
# Events: DONE …, FAILED … (FAIL/REFUSE/BUSY/EXTRA_FAIL line), DIED (process gone, no verdict -> state unknown,
# run --verify-only before anything else), SSH_FAIL.
set -u
STEP="${1:?step}"; LOG="${2:?log}"; IV="${3:-300}"; ONCE="${4:-}"
case "$LOG" in /orange/yd24f.fsu/xc25.fsu/*) ;; *) echo "bad log path"; exit 2;; esac
if [ "$STEP" = extra ]; then PROC='tc_extra_verify.s[h] '; else PROC='orange_move.s[h] '; fi
while true; do
  out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 hpg "grep -aE '^(FAIL|REFUSE|BUSY|EXTRA_FAIL)|^[0-9:]{8} (PASS:|DELETED |EXTRA_PASS|EXTRA_FAIL)' '$LOG' | tail -n 4; echo '@@LAST'; tail -n 2 '$LOG' | cut -c1-300; echo \"@@RUN \$(pgrep -u xc25.fsu -f '$PROC' | wc -l)\"" 2>/dev/null); rc=$?
  now=$(TZ=America/Chicago date '+%H:%M %Z')
  if [ $rc -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^@@RUN '; then echo "$now SSH_FAIL rc=$rc"; exit 1; fi
  marks=$(printf '%s\n' "$out" | sed -n '1,/^@@LAST$/p' | grep -v '^@@LAST$')
  last=$(printf '%s\n' "$out" | sed -n '/^@@LAST$/,/^@@RUN /p' | grep -v '^@@')
  run=$(printf '%s\n' "$out" | sed -n 's/^@@RUN //p')
  ev=""
  if printf '%s\n' "$marks" | grep -qE '^(FAIL|REFUSE|BUSY|EXTRA_FAIL)| EXTRA_FAIL'; then ev=FAILED
  elif [ "$STEP" = delete ] && printf '%s\n' "$marks" | grep -q ' DELETED '; then ev=DONE
  elif [ "$STEP" = extra ] && printf '%s\n' "$marks" | grep -q ' EXTRA_PASS'; then ev=DONE
  elif { [ "$STEP" = copy ] || [ "$STEP" = verify ]; } && printf '%s\n' "$marks" | grep -q ' PASS: ' && [ "$run" = 0 ]; then ev=DONE
  elif [ "$run" = 0 ]; then ev=DIED
  fi
  if [ -n "$ev" ] || [ "$ONCE" = "--once" ]; then
    echo "$now ${ev:-RUNNING} step=$STEP procs=$run"; printf '%s\n' "$marks" | sed 's/^/  mark: /'; printf '%s\n' "$last" | sed 's/^/  last: /'
    [ "$ev" = DONE ] && exit 0; [ -z "$ev" ] && exit 0; exit 1
  fi
  sleep "$IV"
done
