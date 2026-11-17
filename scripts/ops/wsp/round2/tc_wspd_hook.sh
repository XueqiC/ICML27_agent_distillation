#!/bin/bash
# tc_wspd_hook.sh — runs ON hpg, called by tc_wsp_pass2.sh. D lane of round 2 (C4 / D3 / D4 on hpg):
# (1) submits one merged-checkpoint evaluation job per finished export (counts ALL account jobs first, cap 30);
# (2) prints the state of the family jobs and the collected numbers.
H=/blue/fsu-compsci-dept/xc25.fsu/hq; D=$H/ICML27_wsp_d; SC=$D/scripts; CAP=30
touch "$D/submitted_evals.tsv"
for fam in c4 d3 d4 w3; do
  for m in "$D"/export_$fam/*/surgery_manifest.json; do
    [ -f "$m" ] || continue
    cond=$(basename "$(dirname "$m")"); name="$fam-$cond"
    cut -f1 "$D/submitted_evals.tsv" | grep -q -x -F -- "$name" && continue
    n=$(squeue -u xc25.fsu -h | wc -l) || { echo "SQUEUE_FAILED"; exit 1; }
    [ "$n" -lt "$CAP" ] || { echo "AT_CAP jobs=$n"; break 2; }
    runs=1; lim=00:50:00
    case "$cond" in B0|B1) runs=2; lim=01:15:00;; esac
    if [ -f "$D/conditions_$fam/selection.json" ] && grep -q "\"chosen_alpha\": *${cond#C4_a}\b" "$D/conditions_$fam/selection.json" 2>/dev/null && [ "${cond#C4_a}" != "$cond" ]; then runs=2; lim=01:15:00; fi
    jid=$(MERGED_MODEL_PATH="$(dirname "$m")" NAME="$name" OUTROOT="$D/eval" DEV140_RUNS="$runs" sbatch --parsable --account=fsu-compsci-dept --qos=fsu-compsci-dept \
          --partition=hpg-rtx6000 --gpus=1 --cpus-per-task=4 --mem=32G --time="$lim" --job-name="wspd-ev-$name" \
          --output="$D/logs/slurm-ev-$name-%j.out" --export=ALL "$SC/hpg_wspd_merged_eval.slurm") || { echo "SBATCH_FAILED $name"; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$runs" "$(date '+%F %T')" >> "$D/submitted_evals.tsv"
    echo "SUBMITTED wspd-ev-$name $jid (dev140 runs: $runs; jobs before: $n)"
  done
done
echo "== D lane: family jobs"; sacct -u xc25.fsu -S 2026-11-17T12:00 -X -n -o JobID,JobName%22,State%12,Elapsed 2>/dev/null | grep -E 'wspd-(c4|d3|d4|w3)' | grep -v 'wspd-ev'
for fam in c4 d3 d4 w3; do
  f=$(ls -t "$D"/logs/slurm-$fam-*.out 2>/dev/null | head -1); [ -n "$f" ] && echo "-- $fam: $(grep -c '^== step .* start' "$f") steps started; last: $(grep -E '^== ' "$f" | tail -n 1 | cut -c1-150)"
done
echo "== D lane: results (training tasks of 20 | dev140)"
for s in "$D"/eval/*.merged.*/eval/summary.json; do [ -f "$s" ] || continue
  python3 - "$s" <<'PY'
import json, sys
p = sys.argv[1]; j = json.load(open(p)); n = j.get("episodes"); s = j.get("successes")
if s is None and j.get("success_rate") is not None and n: s = round(j["success_rate"] * n)
print("%-34s %s/%s err=%s" % (p.split("/")[-3], s, n, j.get("error_episodes")))
PY
done
