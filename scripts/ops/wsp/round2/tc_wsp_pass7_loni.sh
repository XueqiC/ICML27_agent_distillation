#!/bin/bash
# tc_wsp_pass7_loni.sh — runs ON LONI (login node or the wsp-driver job). Round 2 (prereg sections 18-20) submission pass.
# New in v7: file-quota guard (/work holds at most 4,000,000 files for the whole account): packs finished evaluation folders
# automatically, stops submitting above 3,750,000 files, releases held jobs below 3,600,000. Submits only what is ready now.
Q=/work/xueqic/hq/ICML27_wsp/queue; R=/work/xueqic/hq/ICML27_wsp; S=$Q/tc_wsp_submit_loni.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') LONI pass7"
inq() { squeue -u $USER -h -o %j | grep -q -x -F "$1"; }
nsub() { cut -f1 $Q/submitted.tsv | grep -c -x -F "$1"; }
files=$(lfs quota -u $USER /work 2>/dev/null | grep -E '^ +/work' | awk '{print $6}' | tr -d '*'); files=${files:-0}
echo "== files: $files"
unpacked=$(find $R/eval -mindepth 3 -maxdepth 5 -type d -name episodes 2>/dev/null | while read -r e; do [ -f "$(dirname "$e")/summary.json" ] && echo x; done | wc -l)
if [ "$unpacked" -gt 0 ] && { [ "$files" -gt 3500000 ] || [ "$unpacked" -ge 6 ]; } && ! inq wsp-cpu-inode_pack_auto; then bash $S cpu inode_pack_auto.sh 12; fi
if [ "$files" -gt 3750000 ]; then echo "QUOTA_HOLD: no submissions above 3,750,000 files ($unpacked finished folders unpacked)"
else
  if [ "$files" -lt 3600000 ]; then for j in $(squeue -u $USER -h -t PD -o '%i %j %r' | awk '$2 ~ /^wsp-/ && $3 == "JobHeldUser" {print $1}'); do scontrol release $j && echo "RELEASED $j"; done; fi
  [ ! -f "$Q/done/c5.search.k20" ] && ! inq wsp-gpu-c5a2_k20 && [ "$(nsub gpucmd:c5a2_k20.sh)" -lt 5 ] && bash $S gpucmd c5a2_k20.sh 16 05:00:00
  [ ! -f "$Q/done/c5.search.k20-rand0" ] && ! inq wsp-gpu-c5b2_k20 && [ "$(nsub gpucmd:c5b2_k20.sh)" -lt 5 ] && bash $S gpucmd c5b2_k20.sh 16 05:00:00
  for l in n1_main.list n1_ev2.list n1_ev3.list b4_main.list b4_ev.list a3_main.list a3_ev.list c2b.list c2b2.list w_ev.list icl_loni.list r2d.list; do [ -f "$Q/lists/$l" ] && bash $S list $l 12; done
fi
for h in $Q/hooks/*.sh; do [ -f "$h" ] && bash "$h"; done
echo "== queue"; squeue -u $USER -h -o '%i %j %P %T %M %S %r' | grep ' wsp' | sort -k2 | cut -c1-118
echo "== markers: $(ls $Q/done | grep -E 'a3check|w1|w12|c5|d\.' | tr '\n' ' ')"
echo "== cells and evaluations"; python3 $Q/tc_wsp_collect.py $R 2>/dev/null | grep -E '^== |n1-|b4-|12b|a3-|rules|icl-|raw27|w1-|w2-|c5' | cut -c1-170
echo "== progress (training-task rollouts)"; for p in $R/eval/t2-*/*/eval/progress.json $R/eval/t2-*/google/*/eval/progress.json; do [ -f "$p" ] && python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print("%-58s solved %s/%s progress %.4f" % ("/".join(sys.argv[1].split("/")[-5:-2]), j["successes"], j["episodes"], j["mean_progress"]))' "$p"; done
echo "== search"; for s in $R/search/*/state.json; do [ -f "$s" ] && python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); c=j["current"]; b=j["best_so_far"]; print(sys.argv[1].split("/")[-2], "iter", j["iteration"], "step", j["step"], "current", c and (c["solved"], round(c["mean_progress"],4), c["coefficients"]), "best", b and (b["solved"], round(b["mean_progress"],4), b["coefficients"]))' "$s"; done
echo "== last states"; sacct -u $USER -S now-4hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep wsp | grep -v -E 'PENDING|RUNNING' | tail -n 18
