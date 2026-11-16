#!/bin/bash
# tc_wsp_submit.sh — runs ON an hpg login node. Throttled submitter for the WSP queue; no chains, no dependencies:
# it submits only jobs that are ready NOW, and before EVERY sbatch it counts ALL jobs of the user (every session's)
# and stops at CAP (hard rule: at most 10 jobs in total).
#   tc_wsp_submit.sh list LISTNAME CAP          submit ready jobs of lists/LISTNAME, in order
#   tc_wsp_submit.sh prep  CELLSFILE CAP ACCT   submit one CPU preparation wave for lists/CELLSFILE
#   tc_wsp_submit.sh cpu   SCRIPT.sh CAP ACCT   submit one CPU job running derive/SCRIPT.sh
# CPU jobs (prep, cpu) use the burst QOS <account>-b (CPU only), so they do not take the group's main CPU quota.
# Account for list jobs: the .res ACCOUNT if set, else the account where this user has fewer jobs right now.
set -u
Q=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue
T=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/wsp_946b2a9
MODE="${1:?mode}"; WHAT="${2:?name}"; CAP="${3:?cap}"
case "$CAP" in [1-9]|10) ;; *) echo "REFUSE cap=$CAP"; exit 2;; esac
mkdir -p "$Q/logs" "$Q/done"; touch "$Q/submitted.tsv"
exec 8>>"$Q/.submit.lock"; flock -n 8 || { echo "SUBMIT_BUSY"; exit 1; }
count_jobs() {  # prints the number of this user's jobs (all states, all accounts); fails loudly if squeue fails
  local out; out=$(squeue -u xc25.fsu -h -o '%i|%a') || return 1
  NJ=$(printf '%s\n' "$out" | grep -c .); NYD=$(printf '%s\n' "$out" | grep -c '|yd24f.fsu$'); NDEPT=$(printf '%s\n' "$out" | grep -c '|fsu-compsci-dept$')
  return 0   # grep -c exits 1 on a zero count; the counts themselves are what matters
}
if [ "$MODE" = prep ]; then
  ACCT="${4:?account}"; case "$ACCT" in yd24f.fsu|fsu-compsci-dept) ;; *) echo "REFUSE account"; exit 2;; esac
  [ -f "$Q/lists/$WHAT" ] || { echo "NO_CELLS_FILE"; exit 2; }
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP jobs=$NJ"; exit 0; }
  jid=$(cd "$T" && WSP_QUEUE="$Q" PREP_LIST="$Q/lists/$WHAT" PROJECT_ROOT="$T" sbatch --parsable --account="$ACCT" --qos="$ACCT-b" \
        --job-name="wsp-prep-${WHAT%.cells}" --output="$Q/logs/slurm-prep-${WHAT%.cells}-%j.out" --export=ALL "$Q/tc_wsp_prep_wave.slurm") || { echo "SBATCH_FAILED prep $WHAT"; exit 1; }
  printf 'prep:%s\t%s\t%s\t%s\n' "$WHAT" "$jid" "$ACCT" "$(date '+%F %T')" >> "$Q/submitted.tsv"
  echo "SUBMITTED prep:$WHAT $jid $ACCT (jobs before: $NJ)"; exit 0
fi
if [ "$MODE" = cpu ]; then   # tc_wsp_submit.sh cpu SCRIPT.sh CAP ACCT : one CPU job running derive/SCRIPT.sh
  ACCT="${4:?account}"; case "$ACCT" in yd24f.fsu|fsu-compsci-dept) ;; *) echo "REFUSE account"; exit 2;; esac
  [ -f "$Q/derive/$WHAT" ] && [ -f "$Q/derive/tc_wsp_cpu.slurm" ] || { echo "NO_CMD_FILE"; exit 2; }
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP jobs=$NJ"; exit 0; }
  jid=$(cd "$T" && CMD_FILE="$Q/derive/$WHAT" sbatch --parsable --account="$ACCT" --qos="$ACCT-b" --job-name="wsp-cpu-${WHAT%.sh}" \
        --output="$Q/logs/slurm-cpu-${WHAT%.sh}-%j.out" --export=ALL "$Q/derive/tc_wsp_cpu.slurm") || { echo "SBATCH_FAILED cpu $WHAT"; exit 1; }
  printf 'cpu:%s\t%s\t%s\t%s\n' "$WHAT" "$jid" "$ACCT" "$(date '+%F %T')" >> "$Q/submitted.tsv"
  echo "SUBMITTED cpu:$WHAT $jid $ACCT (jobs before: $NJ)"; exit 0
fi
if [ "$MODE" = gpu ]; then   # tc_wsp_submit.sh gpu SCRIPT.sh CAP ACCT TIME MEM : one 1-GPU job running derive/SCRIPT.sh
  ACCT="${4:?account}"; lim="${5:?time}"; mem="${6:?mem}"
  case "$ACCT" in yd24f.fsu|fsu-compsci-dept) ;; *) echo "REFUSE account"; exit 2;; esac
  [ -f "$Q/derive/$WHAT" ] && [ -f "$Q/derive/tc_wsp_gpu.slurm" ] || { echo "NO_CMD_FILE"; exit 2; }
  if cut -f1 "$Q/submitted.tsv" | grep -q -x -F -- "gpu:$WHAT"; then echo "ALREADY_SUBMITTED gpu:$WHAT"; exit 0; fi
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP jobs=$NJ"; exit 0; }
  jid=$(cd "$T" && CMD_FILE="$Q/derive/$WHAT" sbatch --parsable --account="$ACCT" --qos="$ACCT" --time="$lim" --mem="$mem" \
        --job-name="wsp-gpu-${WHAT%.sh}" --output="$Q/logs/slurm-gpu-${WHAT%.sh}-%j.out" --export=ALL "$Q/derive/tc_wsp_gpu.slurm") || { echo "SBATCH_FAILED gpu $WHAT"; exit 1; }
  printf 'gpu:%s\t%s\t%s\t%s\n' "$WHAT" "$jid" "$ACCT" "$(date '+%F %T')" >> "$Q/submitted.tsv"
  echo "SUBMITTED gpu:$WHAT $jid $ACCT (jobs before: $NJ)"; exit 0
fi
[ "$MODE" = list ] || { echo "REFUSE mode"; exit 2; }
[ -f "$Q/lists/$WHAT" ] || { echo "NO_LIST"; exit 2; }
nsub=0
while IFS= read -r name <&7; do
  [ -n "$name" ] || continue
  if cut -f1 "$Q/submitted.tsv" | grep -q -x -F -- "$name"; then continue; fi
  envf="$Q/jobs/$name.env"; resf="$Q/jobs/$name.res"
  [ -f "$envf" ] && [ -f "$resf" ] || { echo "NOFILES $name"; continue; }
  ok=1; ACCOUNT=; PARTITION=; GPUS=; CPUS=; MEM=; TIME=; SCRIPT=
  while IFS='=' read -r k v; do
    case "$k" in
      REQUIRE) [ -e "$v" ] || ok=0;;
      FORBID) [ ! -e "$v" ] || ok=0;;
      ACCOUNT) ACCOUNT=$v;; PARTITION) PARTITION=$v;; GPUS) GPUS=$v;; CPUS) CPUS=$v;; MEM) MEM=$v;; TIME) TIME=$v;; SCRIPT) SCRIPT=$v;;
    esac
  done < "$resf"
  [ "$ok" = 1 ] || continue
  [ -n "$PARTITION" ] && [ -n "$GPUS" ] && [ -n "$CPUS" ] && [ -n "$MEM" ] && [ -n "$TIME" ] && [ -f "$T/$SCRIPT" ] || { echo "BAD_RES $name"; continue; }
  count_jobs || { echo "SQUEUE_FAILED"; exit 1; }
  [ "$NJ" -lt "$CAP" ] || { echo "AT_CAP jobs=$NJ (submitted now: $nsub)"; exit 0; }
  acct=$ACCOUNT
  if [ -z "$acct" ]; then if [ "$NYD" -le "$NDEPT" ]; then acct=yd24f.fsu; else acct=fsu-compsci-dept; fi; fi
  jid=$( set -a; . "$envf"; set +a; cd "$T" && sbatch --parsable --account="$acct" --qos="$acct" --partition="$PARTITION" --nodes=1 \
         --gpus="$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --job-name="wsp-$name" \
         --output="$Q/logs/slurm-$name-%j.out" --export=ALL "$SCRIPT" ) || { echo "SBATCH_FAILED $name"; exit 1; }
  printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$acct" "$(date '+%F %T')" >> "$Q/submitted.tsv"
  echo "SUBMITTED $name $jid $acct (jobs before: $NJ)"; nsub=$((nsub + 1))
done 7< "$Q/lists/$WHAT"
echo "SUBMIT_PASS_DONE submitted=$nsub"
