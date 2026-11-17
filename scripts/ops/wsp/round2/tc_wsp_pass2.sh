#!/bin/bash
# tc_wsp_pass2.sh (v2, 2026-11-17) — runs ON hpg. Round 2 (prereg sections 18-19) submission pass, in priority order;
# every step counts ALL account jobs (cap 30). Safe to run at any time; submits only what is ready.
# v2: also runs every executable hook in queue/hooks/ (D lane evaluation submitter).
Q=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue; R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp
S=$Q/tc_wsp_submit2.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') hpg pass2"
for l in $(cat $Q/lists/round2.order 2>/dev/null); do [ -f "$Q/lists/$l" ] && bash $S list "$l" 30; done
for h in $Q/hooks/*.sh; do [ -f "$h" ] && bash "$h"; done
echo "== queue ($(squeue -u xc25.fsu -h | wc -l) jobs on the account)"; squeue -u xc25.fsu -h -o '%i %j %P %T %M %R' | sort -k2 | cut -c1-110
echo "== results"; python3 $Q/tc_wsp_collect.py $R 2>/dev/null | grep -E '^== |icl-|b4|rules|w[0-9]|c5|a3|n1-' | cut -c1-200
echo "== last states"; sacct -u xc25.fsu -S now-3hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep -E 'wsp' | grep -v -E 'PENDING|RUNNING' | tail -n 14
