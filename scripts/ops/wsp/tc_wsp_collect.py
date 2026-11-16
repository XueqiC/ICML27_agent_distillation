#!/usr/bin/env python3
"""tc_wsp_collect.py R_DIR — read-only summary of the WSP runs under R_DIR (cells/, eval/): one line per trained cell
(updates, training seconds, mean loss of the first / last epoch) and one line per evaluated adapter."""
import glob, json, os, statistics as st, sys
R = sys.argv[1]
print("== trained cells")
for res in sorted(glob.glob(R + "/cells/*/result.json")):
    cell = res.split("/")[-2]
    outs = sorted(glob.glob(f"{R}/cells/{cell}/stages/train-sft/attempt-*/output"))
    if not outs:
        continue
    o = outs[-1]
    try:
        L = [json.loads(l) for l in open(o + "/updates.jsonl")]
        r = json.load(open(o + "/result.json"))
        ep = sorted({x["epoch"] for x in L})
        first = st.mean(x["loss"] for x in L if x["epoch"] == ep[0]); last = st.mean(x["loss"] for x in L if x["epoch"] == ep[-1])
        print("%-28s updates=%d train_s=%.0f loss_first_epoch=%.4f loss_last_epoch=%.5f attempts=%d" % (cell, len(L), r.get("wall_time_seconds", -1), first, last, len(outs)))
    except Exception as e:
        print("%-28s ERR %s" % (cell, e))
print("== evaluations (dev140, greedy): unit | adapter | successes/episodes | errors, parser rejections, unknown | serving check")
rows = []
for d in sorted(glob.glob(R + "/eval/*/*/eval") + glob.glob(R + "/eval/*/*/*/eval")):
    rel = d[len(R + "/eval/"):].split("/")
    unit, name = rel[0], ("RAW:" + "/".join(rel[1:-1]) if len(rel) == 4 else rel[1])
    p = os.path.join(d, "summary.json")
    if not os.path.isfile(p):
        print("%-34s %-26s (running or incomplete)" % (unit, name)); continue
    try:
        j = json.load(open(p))
        n = j.get("episodes"); s = j.get("successes"); 
        if s is None and j.get("success_rate") is not None and n: s = round(j["success_rate"] * n)
        chk = os.path.join(os.path.dirname(d), "check", "report.json")
        passed = json.load(open(chk))["aggregate"]["passed"] if os.path.isfile(chk) else None
        bt = j.get("successes_by_task_type") or j.get("success_by_task_type") or {}
        rows.append((name, unit, s, n))
        print("%-34s %-26s %s/%s  err=%s parse=%s unknown=%s overflow=%s check=%s %s" % (unit, name, s, n, j.get("error_episodes"), j.get("parser_rejections"),
              j.get("unknown_outcomes", j.get("unknown_episodes")), j.get("context_overflow_episodes"), passed, json.dumps(bt)[:120] if bt else ""))
    except Exception as e:
        print("%-34s %-26s ERR %s" % (unit, name, e))
