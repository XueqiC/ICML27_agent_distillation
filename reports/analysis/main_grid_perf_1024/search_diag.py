"""Where does search fail? (run on hpg; reads episode.json via per_task.tsv, like failure_modes.py)

Object location of a task = every receptacle from which ANY successful episode of that task (any cell given, incl. teacher)
took the target object ("take <obj> N from <recep> M"); the same game file has the same initial layout, so this is where the
object is. closed(loc) = some successful episode opened <loc> before taking from it.
For every failed episode that never took the target ("none"):
  no_loc       = no successful episode of the task anywhere (location unknown)
  never_went   = never went to any object location
  went_closed  = went to a location that needs opening but never opened it
  went_open    = went there (and opened it if needed) but did not take the object
Also: first step at which the agent reached a location, distinct receptacles visited, and for successful episodes the number of
distinct receptacles visited before the first take (search length).
Usage: search_diag.py cell... > out.json   (cell 'teacher' = reports/probe/teacher140)
"""
import csv, glob, json, re, sys
from collections import Counter, defaultdict

R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"
P = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/probe"


def tsv(cell):
    if cell == "teacher":
        return f"{P}/teacher140/per_task.tsv"
    if cell == "base":
        return f"{R}/eval/seedsft-sweep-bi/k5s0/google/gemma-4-12B-it/eval/per_task.tsv"
    if cell.startswith("sadc-"):
        return sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))[-1]
    return glob.glob(f"{R}/eval/chain/{cell}/*/eval/per_task.tsv")[0]


def target(task_id):
    typ, obj = task_id.split("/")[1].split("-")[:2]
    return typ, obj.lower()


def acts_of(ep):
    return [(s.get("action_parsed") or "").strip().lower() for s in ep["steps"]]


eps = {}
for cell in sys.argv[1:]:
    rows = list(csv.DictReader(open(tsv(cell)), delimiter="\t"))
    assert len(rows) == 140, cell
    for r in rows:
        ep = json.load(open(r["episode_dir"] + "/episode.json"))
        eps[(cell, r["task_id"])] = (r["success"] == "True", acts_of(ep), ep)

loc = defaultdict(set); closed = defaultdict(set)
for (cell, tid), (ok, acts, _) in eps.items():
    if not ok:
        continue
    _, obj = target(tid)
    opened = set()
    for a in acts:
        if a.startswith("open "):
            opened.add(a[5:])
        m = re.match(rf"take {obj} \d+ from (.+)$", a)
        if m:
            loc[tid].add(m.group(1))
            if m.group(1) in opened:
                closed[tid].add(m.group(1))

out = {"locations": {t: sorted(v) for t, v in loc.items()}, "cells": {}}
for cell in sys.argv[1:]:
    res = {}
    for tid in sorted(t for c, t in eps if c == cell):
        ok, acts, ep = eps[(cell, tid)]
        _, obj = target(tid)
        took_at = next((i for i, a in enumerate(acts) if re.match(rf"take {obj} \d+ from ", a)), None)
        went = [a[6:] for a in acts if a.startswith("go to ")]
        opened = {a[5:] for a in acts if a.startswith("open ")}
        L = loc.get(tid, set())
        first_loc = next((i for i, a in enumerate(acts) if a.startswith("go to ") and a[6:] in L), None)
        e = {"success": ok, "took": took_at is not None, "steps": len(acts), "visited": len(set(went)),
             "first_loc_step": first_loc}
        if ok and took_at is not None:
            e["search_len"] = len({a[6:] for a in acts[:took_at] if a.startswith("go to ")})
        if not ok and took_at is None:
            if not L:
                e["why"] = "no_loc"
            elif not (set(went) & L):
                e["why"] = "never_went"
            elif all(l in closed[tid] and l not in opened for l in set(went) & L):
                e["why"] = "went_closed"
            else:
                e["why"] = "went_open"
        res[tid] = e
    out["cells"][cell] = res
    F = [e for e in res.values() if "why" in e]
    print(cell, len(F), "none-failures;", dict(Counter(e["why"] for e in F)), file=sys.stderr)
json.dump(out, sys.stdout)
