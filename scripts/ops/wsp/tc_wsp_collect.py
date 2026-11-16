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
print("== evaluations")
for d in sorted(glob.glob(R + "/eval/*/*/eval")):
    unit, name = d.split("/")[-3], d.split("/")[-2]
    files = sorted(os.listdir(d))
    summ = [f for f in files if f.endswith(".json")]
    line = "%-34s %-28s files=%s" % (unit, name, ",".join(files)[:80])
    for f in ("summary.json", "result.json", "results.json", "metrics.json"):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            try:
                j = json.load(open(p))
                keys = {k: j[k] for k in j if isinstance(j[k], (int, float, str)) and len(str(j[k])) < 40}
                line += " | %s: %s" % (f, json.dumps(keys)[:260])
            except Exception as e:
                line += " | %s ERR %s" % (f, e)
            break
    print(line)
