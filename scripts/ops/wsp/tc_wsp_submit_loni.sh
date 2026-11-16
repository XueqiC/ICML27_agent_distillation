#!/bin/bash
# tc_wsp_submit_loni.sh — runs ON a LONI login node. Throttled submitter for the LONI lane of the WSP queue.
# No chains: submits only what is ready now; before EVERY sbatch it counts this user's wsp-* jobs and stops at CAP.
#   tc_wsp_submit_loni.sh list LISTNAME CAP     GPU jobs of lists/LISTNAME (gpu2, through ICML27_loni_wrap.sh)
#   tc_wsp_submit_loni.sh prep CELLSFILE CAP    one CPU preparation wave (single partition)
#   tc_wsp_submit_loni.sh cpu  SCRIPT.sh CAP    one CPU job running derive/SCRIPT.sh (single partition)
set -u
W=/work/xueqic/hq; Q=$W/ICML27_wsp/queue; T=$W/ICML27_runs/wsp_946b2a9; WRAP=$W/ICML27_loni_wrap.sh; ACCT=loni_depedlab04
MODE="${1:?mode}"; WHAT="${2:?name}"; CAP="${3:?cap}"
case "$CAP" in [1-9]|1[0-2]) ;; *) echo "REFUSE cap=$CAP"; exit 2;; esac
mkdir -p "$Q/logs" "$Q/done"; touch "$Q/submitted.tsv"
exec 8>>"$Q/.submit.lock"; flock -n 8 || { echo "SUBMIT_BUSY"; exit 1; }
count_jobs() { local out; out=$(squeue -u "$USER" -h -o '%j') || return 1
  NJ=$(printf '%s\n' "$out" | grep -c '^wsp-'); NALL=$(printf '%s\n' "$out" | grep -c .); return 0; }
jobid() { printf '%s\n' "$1" | grep -E '^[0-9]+$' | tail -n 1; }   # sbatch on LONI prints SU / lua notes around the id
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$ACCT" "$(date '+%F %T')" >> "$Q/submitted.tsv"; }
if [ "$MODE" = prep ] || [ "$MODE" = cpu ]; then
  if [ "$MODE" = prep ]; then f="$Q/lists/$WHAT"; cmd="bash $Q/tc_wsp_prep_wave.slurm"; tag="prep-${WHAT%.cells}"; lim=04:00:00
  else f="$Q/derive/$WHAT"; cmd="bash $Q/derive/$WHAT"; tag="cpu-${WHAT%.sh}"; lim=01:30:00; fi
  [ -f "$f" ] || { echo "NO_FILE $f"; exit 2; }
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP wsp=$NJ"; exit 0; }
  o=$(cd "$T" && sbatch --parsable -A "$ACCT" -p single -N1 -n1 --cpus-per-task=16 -t "$lim" -J "wsp-$tag" -o "$Q/logs/slurm-%x-%j.out" \
      --export=ALL,WSP_QUEUE="$Q",PREP_LIST="$Q/lists/$WHAT",PROJECT_ROOT="$T" --wrap "$cmd" 2>&1); jid=$(jobid "$o")
  [ -n "$jid" ] || { echo "SBATCH_FAILED $MODE $WHAT: $(printf '%s' "$o" | tail -n 2 | tr '\n' ' ')"; exit 1; }
  record "$MODE:$WHAT" "$jid"; echo "SUBMITTED $MODE:$WHAT $jid (wsp jobs before: $NJ, all: $NALL)"; exit 0
fi
[ "$MODE" = list ] || { echo "REFUSE mode"; exit 2; }
[ -f "$Q/lists/$WHAT" ] || { echo "NO_LIST"; exit 2; }
nsub=0
while IFS= read -r name <&7; do
  [ -n "$name" ] || continue
  if cut -f1 "$Q/submitted.tsv" | grep -q -x -F -- "$name"; then continue; fi
  envf="$Q/jobs/$name.env"; resf="$Q/jobs/$name.res"
  [ -f "$envf" ] && [ -f "$resf" ] || { echo "NOFILES $name"; continue; }
  ok=1; GPUS=; CPUS=; TIME=; STAGE_SCRIPT=
  while IFS='=' read -r k v; do
    case "$k" in
      REQUIRE) [ -e "$v" ] || ok=0;; FORBID) [ ! -e "$v" ] || ok=0;;
      GPUS) GPUS=$v;; CPUS) CPUS=$v;; TIME) TIME=$v;; STAGE_SCRIPT) STAGE_SCRIPT=$v;;
    esac
  done < "$resf"
  [ "$ok" = 1 ] || continue
  [ -n "$GPUS" ] && [ -n "$CPUS" ] && [ -n "$TIME" ] && [ -f "$T/$STAGE_SCRIPT" ] || { echo "BAD_RES $name"; continue; }
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP wsp=$NJ (submitted now: $nsub)"; exit 0; }
  o=$( set -a; . "$envf"; STAGE_SCRIPT="$STAGE_SCRIPT"; set +a; cd "$T" && sbatch --parsable -A "$ACCT" -p gpu2 -N1 -n1 --gres=gpu:"$GPUS" \
       --cpus-per-task="$CPUS" -t "$TIME" -J "wsp-$name" -o "$Q/logs/slurm-$name-%j.out" --export=ALL "$WRAP" 2>&1 ); jid=$(jobid "$o")
  [ -n "$jid" ] || { echo "SBATCH_FAILED $name: $(printf '%s' "$o" | tail -n 2 | tr '\n' ' ')"; exit 1; }
  record "$name" "$jid"; echo "SUBMITTED $name $jid (wsp jobs before: $NJ, all: $NALL)"; nsub=$((nsub + 1))
done 7< "$Q/lists/$WHAT"
echo "SUBMIT_PASS_DONE submitted=$nsub"
