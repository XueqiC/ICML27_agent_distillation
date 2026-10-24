"""Among failed episodes that never took the target object: did the agent take a DIFFERENT object and carry out the task
with it (substitution = goal-grounding error) or never take anything relevant (search)?
subst_placed = moved/put a non-target object into/onto the task receptacle type; subst_took = took a non-target object
(but did not place it in the target receptacle type); nothing = took nothing. Usage: subst_diag.py cell... (cell 'teacher' ok)"""
import csv, glob, json, re, sys
from collections import Counter
R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"
P = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/probe"
def tsv(cell):
    if cell == "teacher": return f"{P}/teacher140/per_task.tsv"
    if cell == "base": return f"{R}/eval/seedsft-sweep-bi/k5s0/google/gemma-4-12B-it/eval/per_task.tsv"
    if cell.startswith("sadc-"):
        return sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))[-1]
    return glob.glob(f"{R}/eval/chain/{cell}/*/eval/per_task.tsv")[0]
out = {}
for cell in sys.argv[1:]:
    cnt, pairs = Counter(), Counter()
    for r in csv.DictReader(open(tsv(cell)), delimiter="\t"):
        if r["success"] == "True": continue
        typ, obj, _m, recep = r["task_id"].split("/")[1].split("-")[:4]; obj, recep = obj.lower(), recep.lower()
        acts = [(s.get("action_parsed") or "").strip().lower() for s in json.load(open(r["episode_dir"] + "/episode.json"))["steps"]]
        if any(re.match(rf"take {obj} \d+ from ", a) for a in acts): continue
        others = [m.group(1) for a in acts for m in [re.match(r"take (\w+) \d+ from ", a)] if m]
        placed = [m.group(1) for a in acts for m in [re.match(rf"(?:move|put) (\w+) \d+ (?:to|in|on|in/on) {recep} \d+", a)] if m and m.group(1) != obj]
        k = "subst_placed" if placed else "subst_took" if others else "nothing"
        cnt[k] += 1
        if others: pairs[f"{obj}->{others[0]}"] += 1
    out[cell] = {"counts": dict(cnt), "pairs": pairs.most_common(12)}
    print(cell, dict(cnt), file=sys.stderr)
json.dump(out, sys.stdout)
