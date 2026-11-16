#!/bin/bash
# tc_wsp_pass.sh — runs ON hpg. One submission pass in priority order (each step counts all account jobs, cap 10),
# then prints the queue, markers and finished results. Safe to run at any time; submits only what is ready.
Q=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue; R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp
S=$Q/tc_wsp_submit.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') hpg pass"
[ -f "$Q/done/prep.sc-k20" ] && [ ! -f "$Q/done/c1.k20" ] && bash $S gpu c1_k20.sh 10 fsu-compsci-dept 00:45:00 48G
[ -f "$Q/done/c1.k20" ] && [ ! -f "$Q/done/c1t2.k20" ] && ! squeue -u xc25.fsu -h -o %j | grep -q -x -F wsp-cpu-c1t2_k20 && bash $S cpu c1t2_k20.sh 10 fsu-compsci-dept
bash $S list ev_t2.list 10  # closed-loop step-size check on the 20 training tasks (prereg section 14)
bash $S list ev1.list 6    # evaluations (hpg-rtx6000) wait on the dept GPU cap: keep six queued, leave four slots for trainings
[ -f "$Q/done/prep.sc-k100" ] && [ ! -f "$Q/done/c1.k100" ] && bash $S gpu c1_k100.sh 10 fsu-compsci-dept 01:30:00 48G
[ -f "$Q/derive/derive_shared.sh" ] && [ ! -f "$Q/done/derive.shared" ] && ! cut -f1 "$Q/submitted.tsv" | grep -q -x -F "cpu:derive_shared.sh" && bash $S cpu derive_shared.sh 10 yd24f.fsu
bash $S list main.list 10   # trainings need B200; slots outside the reservation do appear, so keep them queued
bash $S list ev2.list 6
echo "== queue"; squeue -u xc25.fsu -h -o '%i %j %a %T %M %R' | sort -k2 | cut -c1-105
echo "== markers: $(ls $Q/done | tr '\n' ' ')"
echo "== prep waves"; for f in $Q/logs/slurm-prep-*.out; do case "$f" in *-5153565[345].out|*-5153638[789].out|*-51536738.out) continue;; esac; echo "$(basename $f): $(grep -c PREP_OK $f) ok, $(grep -c PREP_FAIL $f) fail, last: $(tail -n 1 $f | cut -c21-70)"; done
echo "== trained cells: $(ls -d $R/cells/*/result.json 2>/dev/null | sed -E 's#.*/cells/([^/]+)/result.json#\1#' | tr '\n' ' ')"
echo "== eval dirs: $(ls $R/eval 2>/dev/null | tr '\n' ' ')"
echo "== last states"; sacct -u xc25.fsu -S now-3hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep wsp- | grep -v -E 'PENDING|RUNNING' | tail -n 12
