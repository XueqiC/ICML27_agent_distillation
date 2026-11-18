#!/bin/bash
# wsp_status.sh — runs ON LONI. One condensed status page of round 2: last driver pass (submissions, quota, queue, new
# result lines), teacher sessions (settled / failed tools, K20 rollout outcomes, submission), C4 selection, C5 ledgers,
# D-lane evaluation results. Read-only.
W=/work/xueqic/hq; Q=$W/ICML27_wsp/queue; D=$W/ICML27_wsp_d; R=$W/ICML27_wsp
echo "== $(TZ=America/Chicago date '+%F %T %Z') driver pids: $(pgrep -u $USER -f wsp_driver.sh | tr '\n' ' ') | $(tail -n 1 $Q/logs/driver.out)"
awk '/^== .* LONI pass7/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $Q/logs/driver.log | grep -v -E 'SUBMIT_PASS_DONE|^== (markers|cells|trained|prep|n1 cells)' | grep -E '^SUBMITTED|RELEASED|QUOTA_HOLD|AT_CAP|FAIL|== files|== LONI|^[0-9]+ wsp|/140|/100|/20 |solved|^-- |exports' | cut -c1-150
echo "== sessions"
for r in d3_s1 d4_s1 d5_s1; do f=$D/runs/$r/tools/tools.jsonl; [ -f $f ] || continue
python3 - $f $r "$D/runs/$r" <<'PY'
import json, sys, collections, os
c = collections.Counter(); k = []
for l in open(sys.argv[1]):
    try: j = json.loads(l)
    except Exception: continue
    if j.get("event") == "settled":
        c[j.get("tool")] += 1
        if j.get("tool") == "k20_rollouts":
            t = j["result"]["data"]["totals"]; k.append("%s/%.3f" % (t["edited"]["success"], t["edited"].get("mean_progress") or 0))
    elif j.get("event") == "failed": c["FAILED:" + str(j.get("tool"))] += 1
turns = sum(1 for _ in open(os.path.join(sys.argv[3], "agent.jsonl")))
print(sys.argv[2], "turns", turns, dict(c), "k20 solved/progress:", k, "SUBMISSION" if os.path.exists(os.path.join(sys.argv[3], "submission.json")) else "")
PY
done
echo "== c4: $(ls $D/conditions_c4 2>/dev/null | tr '\n' ' ')"; grep -E 'candidate_completed|"completed"|failed' $D/conditions_c4/additive.jsonl 2>/dev/null | cut -c1-200
echo "== c5"
python3 - $R <<'PY'
import json, sys
for n in ("c5-k20", "c5-k20-rand0"):
    try:
        s = json.load(open("%s/search/%s/state.json" % (sys.argv[1], n))); rows = [json.loads(l) for l in open("%s/search/%s/search.jsonl" % (sys.argv[1], n))]
        c, b = s["current"], s["best_so_far"]
        print(n, "iter", s["iteration"], "step", s["step"], "rows", len(rows), "current", c and (c["solved"], round(c["mean_progress"], 4), c["coefficients"]), "best", b and (b["solved"], round(b["mean_progress"], 4), b["coefficients"]))
    except Exception as e: print(n, "ERR", e)
PY
for f in $Q/logs/slurm-wsp-gpu-d_*.out $Q/logs/slurm-wsp-gpu-c5?2_k20-*.out; do x=$(grep -E '_(DONE|FAIL)' $f | tail -n 1); [ -n "$x" ] && echo "$(basename $f): $x"; done
