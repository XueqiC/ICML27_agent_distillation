#!/usr/bin/env python3
"""t2_behaviour.py EVAL_ROOT UNIT [UNIT ...] — closed-loop behaviour per arm from stored episodes (events.jsonl)."""
import glob, json, os, re, sys
root, units = sys.argv[1], sys.argv[2:]
print("%-34s %5s %6s %8s %7s %6s %7s %6s %6s %7s" % ("unit/arm", "succ", "steps", "ph_turn1", "ph_all", "chars", "repeat", "inadm", "idle", "claims"))
for unit in units:
    for E in sorted(glob.glob(f"{root}/{unit}/*/eval") + glob.glob(f"{root}/{unit}/*/*/eval")):
        eps = sorted(glob.glob(E + "/episodes/*"))
        if not eps or not os.path.exists(E + "/summary.json"):
            continue
        n = ph = ph1 = rep = inad = idle = chars = succ = claims = 0
        for ep in eps:
            ev = []
            for l in open(ep + "/events.jsonl"):
                try:
                    e = json.loads(l)
                except Exception:
                    continue
                if isinstance(e, dict) and e.get("action_parsed"):
                    ev.append(e)
            try:
                succ += bool(json.load(open(ep + "/episode.json")).get("success"))
            except Exception:
                pass
            seen = set()
            for i, e in enumerate(ev):
                r = e.get("action_raw") or ""; a = e["action_parsed"]; n += 1; chars += len(r)
                p = r.startswith("Thought: ...\n"); ph += p; ph1 += (p and i == 0)
                rep += a in seen; seen.add(a)
                inad += e.get("action_outcome") == "inadmissible"
                idle += a.split()[0] in ("look", "help", "inventory")
                claims += bool(re.search(r"task is (now )?(complete|done|finished)|have (already )?(completed|finished)", r, re.I))
        name = unit.split(".")[0] + "/" + E.split(unit + "/")[1].replace("/eval", "").replace("google/", "RAW:")[:22]
        print("%-34s %5d %6.1f %7.0f%% %6.0f%% %6.0f %6.0f%% %5.0f%% %5.0f%% %6.0f%%" % (name[:34], succ, n / len(eps), 100 * ph1 / len(eps), 100 * ph / n, chars / n, 100 * rep / n, 100 * inad / n, 100 * idle / n, 100 * claims / n))
