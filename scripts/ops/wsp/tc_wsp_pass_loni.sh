#!/bin/bash
# tc_wsp_pass.sh — runs ON LONI. One submission pass in priority order (cap 12 GPU-queue jobs, 6 CPU jobs),
# then prints the queue, markers and finished results. Safe to run at any time; submits only what is ready.
Q=/work/xueqic/hq/ICML27_wsp/queue; R=/work/xueqic/hq/ICML27_wsp
S=$Q/tc_wsp_submit_loni.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') LONI pass"
[ -f "$Q/done/prep.sc-k20" ] && [ ! -f "$Q/done/c1.k20" ] && bash $S gpucmd c1_k20.sh 12 01:15:00
bash $S list ev1.list 8    # leave slots for trainings and the c1 jobs
[ -f "$Q/done/prep.sc-k100" ] && [ ! -f "$Q/done/c1.k100" ] && bash $S gpucmd c1_k100.sh 12 03:00:00
bash $S list main.list 12
bash $S list ev2.list 12
echo "== queue"; squeue -u $USER -h -o '%i %j %P %T %M %S' | grep ' wsp-' | sort -k2 | cut -c1-105
echo "== markers: $(ls $Q/done | tr '\n' ' ')"
echo "== prep waves"; for f in $Q/logs/slurm-wsp-prep-*.out; do case "$f" in *-118450[456].out) continue;; esac; echo "$(basename $f): $(grep -c PREP_OK $f) ok, $(grep -c PREP_FAIL $f) fail, last: $(tail -n 1 $f | cut -c21-70)"; done
echo "== trained cells: $(ls -d $R/cells/*/result.json 2>/dev/null | sed -E 's#.*/cells/([^/]+)/result.json#\1#' | tr '\n' ' ')"
echo "== eval dirs: $(ls $R/eval 2>/dev/null | tr '\n' ' ')"
echo "== last states"; sacct -u $USER -S now-3hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep wsp- | grep -v -E 'PENDING|RUNNING' | tail -n 12
