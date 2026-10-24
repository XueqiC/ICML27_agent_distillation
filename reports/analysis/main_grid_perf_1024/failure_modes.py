"""Per-episode subgoal progress and looping for every main-grid cell (run on hpg; reads episode.json via per_task.tsv).

Milestones (ALFWorld task id <type>-<Obj>-<Movable>-<Recep>-<scene>; actions lower-case):
  took    = "take <obj> N from ..."
  trans   = clean/heat/cool <obj> N with ... (clean/heat/cool tasks); "use <lamp> N" at any point (look, with took; <lamp> = the task receptacle); second
            distinct <obj> instance taken (two); simple has no such step (counts as reached once took)
  placed  = "move <obj> N to <recep> M" / "put <obj> N in/on <recep> M" (look: n/a)
Stage of an episode = furthest milestone; "skipped" = placed without the required intermediate step.
Loop features: max run of one identical action; fraction of steps whose action already occurred in the last 4 steps.
Search features: seen_at = first observation index (0 = initial) mentioning "<obj> N"; n_recep = receptacles reachable by
"go to" at the start; visited = distinct ones the agent went to; opens = open actions.
"""
import csv, glob, json, re, sys
from collections import Counter

R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"


def tsv(cell):
    if cell == "base":
        return f"{R}/eval/seedsft-sweep-bi/k5s0/google/gemma-4-12B-it/eval/per_task.tsv"
    if cell.startswith("sadc-"):
        return sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))[-1]
    return glob.glob(f"{R}/eval/chain/{cell}/*/eval/per_task.tsv")[0]


def parse_task(task_id):
    name = task_id.split("/")[1]
    typ, obj, _mov, recep = name.split("-")[:4]
    return typ, obj.lower(), recep.lower()


def episode(ep, typ, obj, recep):
    acts = [(s.get("action_parsed") or "").strip().lower() for s in ep["steps"]]
    took = trans = placed = False
    taken = set(); lamp = False
    for a in acts:
        m = re.match(rf"take ({obj} \d+) from ", a)
        if m:
            took = True; taken.add(m.group(1))
        if took and typ == "pick_clean_then_place_in_recep" and re.match(rf"clean {obj} \d+ with ", a): trans = True
        if took and typ == "pick_heat_then_place_in_recep" and re.match(rf"heat {obj} \d+ with ", a): trans = True
        if took and typ == "pick_cool_then_place_in_recep" and re.match(rf"cool {obj} \d+ with ", a): trans = True
        if typ == "look_at_obj_in_light" and re.match(rf"use {recep} \d+", a): lamp = True
        if re.match(rf"(move {obj} \d+ to {recep} \d+|put {obj} \d+ (in|on|in/on) {recep} \d+)", a): placed = True
    if typ == "pick_two_obj_and_place": trans = len(taken) >= 2
    if typ == "pick_and_place_simple": trans = took
    if typ == "look_at_obj_in_light": trans = took and lamp   # success needs the lamp on and the object in hand, either order
    run = best = 1
    for i in range(1, len(acts)):
        run = run + 1 if acts[i] == acts[i - 1] else 1; best = max(best, run)
    rep = sum(1 for i, a in enumerate(acts) if a in acts[max(0, i - 4):i]) / max(1, len(acts))
    if typ == "look_at_obj_in_light":
        stage = "trans" if trans else "took" if took else "none"
    else:   # skipped = target placed without the required clean/heat/cool or second object
        stage = "placed" if placed and trans else "skipped" if placed else "trans" if trans and typ != "pick_and_place_simple" else "took" if took else "none"
    obs = [str((ep.get("initial") or {}).get("visible", {}).get("observation", ""))] + \
          [str((s.get("visible") or {}).get("observation", "")) for s in ep["steps"]]
    seen_at = next((i for i, o in enumerate(obs) if re.search(rf"\b{obj} \d+", o.lower())), None)
    recs = {c[6:] for c in (ep.get("initial") or {}).get("visible", {}).get("admissible_commands", []) if c.startswith("go to ")}
    visited = {a[6:] for a in acts if a.startswith("go to ")}
    return {"steps": len(acts), "took": took, "trans": trans, "placed": placed, "stage": stage,
            "seen_at": seen_at, "n_recep": len(recs), "visited": len(visited & recs) if recs else len(visited),
            "opens": sum(1 for a in acts if a.startswith("open ")),
            "max_run": best if acts else 0, "rep4": rep,
            "inad": sum(1 for s in ep["steps"] if str(s.get("inadmissible")) == "True"),
            "first_take": next((i for i, a in enumerate(acts) if re.match(rf"take {obj} \d+ from ", a)), None)}


out = {}
for cell in sys.argv[1:]:
    rows = list(csv.DictReader(open(tsv(cell)), delimiter="\t"))
    assert len(rows) == 140, cell
    res = {}
    for r in rows:
        typ, obj, recep = parse_task(r["task_id"])
        ep = json.load(open(r["episode_dir"] + "/episode.json"))
        e = episode(ep, typ, obj, recep); e["success"] = r["success"] == "True"; e["type"] = typ
        res[r["task_id"]] = e
    out[cell] = res
    F = [e for e in res.values() if not e["success"]]
    print(cell, len(F), "failed;", dict(Counter(e["stage"] for e in F)), file=sys.stderr)
json.dump(out, sys.stdout)
