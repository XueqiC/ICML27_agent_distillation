"""Run on hpg: compact episodes (actions, observations, thoughts) of E4B grid cells + teacher for selected dev140 tasks."""
import csv, glob, json, sys
R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"
P = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/probe"
CONV = {"tsft": {10: "tsft-k10-e32-lr5e-5", 20: "tsft-k20-e32-lr5e-5", 30: "tsft-k30-e64-lr5e-5", 100: "tsft-k100-e64-lr1e-4"},
        "AD": {10: "adc-k10-e32-lr1e-4", 20: "adc-k20-e32-lr1e-4", 30: "adc-k30-e64-lr1e-4", 100: "adc-k100-e64-lr1e-4"},
        "SAD": {10: "sadc-k10-b1", 20: "sadc-k20-b1", 30: "sadc-k30-b1", 100: "sadc-k100-b05"},
        "SmartAD": {10: "smc-k10-e32-lr1e-4", 20: "smc-k20-e32-lr1e-4", 30: "smc-k30-e32-lr1e-4", 100: "smc-k100-e64-lr1e-4"}}
want = set(json.load(open(sys.argv[1])))
def tsvs():
    for m, row in CONV.items():
        for k, c in row.items():
            cell = f"e4b-{c}"
            if c.startswith("sadc-"):
                fs = sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))
                if fs: yield f"{m}|{k}", fs[-1]
                continue
            fs = glob.glob(f"{R}/eval/chain/{cell}/{cell}/eval/per_task.tsv")
            if fs: yield f"{m}|{k}", fs[0]
            for s in (1, 2):
                fs = glob.glob(f"{R}/eval/chain/{cell}-s{s}/{cell}-s{s}/eval/per_task.tsv")
                if fs: yield f"{m}|{k}|s{s}", fs[0]
    yield "teacher", f"{P}/teacher140/per_task.tsv"
    fs = glob.glob(f"{R}/eval/chain/e4b-tsft-k30-e64-lr5e-5/google/gemma-4-E4B-it/eval/per_task.tsv")
    if fs: yield "base", fs[0]
out = {}
for cell, p in tsvs():
    rows = list(csv.DictReader(open(p), delimiter="\t")); assert len(rows) == 140, p
    for r in rows:
        if r["task_id"] not in want: continue
        ep = json.load(open(r["episode_dir"] + "/episode.json"))
        steps = [{"a": s.get("action_parsed"), "raw": (s.get("action_raw") or "")[:300],
                  "o": ((s.get("visible") or {}).get("observation") or "")[:400]} for s in ep["steps"]]
        out.setdefault(r["task_id"], {})[cell] = {"success": r["success"] == "True",
            "init": ((ep.get("initial") or {}).get("visible") or {}).get("observation", ""), "steps": steps}
json.dump(out, sys.stdout)
