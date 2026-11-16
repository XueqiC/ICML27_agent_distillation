#!/usr/bin/env python3
"""build_d_dossier.py B0_K20_EXECUTIONS_GLOB VERIFIED_ROOT K20_SUBSET_DIR OUT.json   (CPU, read-only on inputs)

Dossier for the D teacher sessions (prereg section 4 group D, section 16): what the UNEDITED student
(google/gemma-4-E4B-it) does in free greedy rollouts from the initial state on its 20 training tasks, next to the
teacher's verified trajectory for the same task. Training tasks only; no dev / test task, no dev-derived number.

Inputs
  B0_K20_EXECUTIONS_GLOB  P1 setting-B evaluation, condition B0 (no edit), metric M2_k20: one JSON per task with
                          initial.visible, steps[*].{action_raw, action_parsed, action_outcome, visible}, success.
  VERIFIED_ROOT           data/verified/seeds_full_train_r2 (episode dirs are relative to it)
  K20_SUBSET_DIR          .../subsets/k20 (manifest.json lists the 20 verified teacher episodes)
Output: {"title": ..., "text": ...} as required by `p1_surgery.py diagnose --dossier`.
"""
import collections
import glob
import json
import os
import re
import sys

exec_glob, root, subset, out = sys.argv[1:5]
IDLE = ("look", "help", "inventory")


def squeeze(actions):
    res, prev, n = [], None, 0
    for a in list(actions) + [None]:
        if a == prev:
            n += 1
            continue
        if prev is not None:
            res.append(prev if n == 1 else f"{prev} x{n}")
        prev, n = a, 1
    return " > ".join(res)


def thought(raw, limit=170):
    first = (raw or "").split("\nAction:")[0].replace("\n", " ").strip()
    return first if len(first) <= limit else first[:limit - 1] + "…"


def stats(action_lists):
    n = idle = rep = 0
    for acts in action_lists:
        seen = set()
        for a in acts:
            n += 1
            idle += a.split()[0] in IDLE if a.split() else 0
            rep += a in seen
            seen.add(a)
    return n, idle, rep


teacher = {}
manifest = json.load(open(os.path.join(subset, "manifest.json")))
for v in manifest["verified"]:
    ep = os.path.join(root, v["episode_dir"])
    meta = json.load(open(os.path.join(ep, "episode.json")))
    acts = []
    for line in open(os.path.join(ep, "events.jsonl")):
        e = json.loads(line)
        if e.get("action_parsed"):
            acts.append(e["action_parsed"])
    assert meta["task_id"].startswith("train/") and meta["task_id"] not in teacher
    teacher[meta["task_id"]] = acts
assert len(teacher) == 20, len(teacher)

student = {}
for path in sorted(glob.glob(exec_glob)):
    r = json.load(open(path))
    assert r["task_id"] not in student and r["max_steps"] == 40 and r["student_checkpoint"] is None
    student[r["task_id"]] = r
assert set(student) == set(teacher), (len(student), sorted(set(teacher) - set(student))[:2])

rows, inadmissible, unparsed, endings = [], 0, 0, collections.Counter()
for i, tid in enumerate(sorted(teacher), 1):
    r, t = student[tid], teacher[tid]
    obs = (r["initial"].get("visible") or {}).get("observation", "")
    m = re.search(r"Your task is to: (.*)", obs)
    goal = m.group(1).strip().rstrip(".") if m else "?"
    steps = r["steps"]
    s = [x["action_parsed"] if x["action_parsed"] is not None else "<unparsed>" for x in steps]
    inadmissible += sum(x["action_outcome"] == "inadmissible" for x in steps)
    unparsed += sum(x["action_parsed"] is None for x in steps)
    k = next((j for j in range(min(len(s), len(t))) if s[j] != t[j]), None)
    if k is None and len(s) != len(t):
        k = min(len(s), len(t))
    if not r["success"]:
        tail = s[-6:]
        idle_tail = sum(a.split()[0] in IDLE for a in tail if a.split())
        claims = sum(bool(re.search(r"task is (now )?(complete|done|finished)|have (already )?(completed|finished)", x["action_raw"] or "", re.I)) for x in steps)
        kind = ("claims the task is complete and keeps acting" if claims >= 3 else
                "ends in look / help / inventory" if idle_tail >= 4 else
                "repeats the same few actions" if len(set(tail)) <= 2 else "still searching at the step limit")
        endings[kind] += 1
    lines = [f"[{i:02d}] {tid.split('/')[1]} | goal: {goal} | student: {'SOLVED' if r['success'] else 'FAILED'} in {len(s)} steps"
             f" ({r['termination_reason']}) | teacher: solved in {len(t)} steps",
             f"  T: {squeeze(t)}", f"  S: {squeeze(s)}"]
    if k is not None:
        ta = t[k] if k < len(t) else "<teacher already finished>"
        sa = s[k] if k < len(s) else "<student already stopped>"
        lines.append(f"  first difference at step {k + 1}: teacher \"{ta}\" / student \"{sa}\"")
        if k < len(steps):
            lines.append(f"  student reply there: {thought(steps[k]['action_raw'])}")
    if not r["success"] and steps:
        lines.append(f"  student's last reply: {thought(steps[-1]['action_raw'])}")
    rows.append("\n".join(lines))

sn, sidle, srep = stats([[x["action_parsed"] or "<unparsed>" for x in student[t]["steps"]] for t in teacher])
tn, tidle, trep = stats(teacher.values())
solved = sum(student[t]["success"] for t in teacher)
head = [
    "Evidence for this session: the UNEDITED student (google/gemma-4-E4B-it, no adapter) in free greedy closed-loop "
    "rollouts from the initial state (H=40) on the same 20 training tasks that k20_rollouts uses, next to your own "
    "verified trajectory for each task. These are training tasks only.",
    f"Student totals: solved {solved} of 20; {sn} actions in total, {100 * sidle / sn:.0f}% of them look / help / inventory, "
    f"{100 * srep / sn:.0f}% repeat an action already issued in the same episode, {100 * inadmissible / sn:.0f}% inadmissible, "
    f"{unparsed} unparsed replies.",
    f"Teacher totals on the same tasks: solved 20 of 20; {tn} actions, {100 * tidle / tn:.0f}% look / help / inventory, "
    f"{100 * trep / tn:.0f}% repeats.",
    "How the student's failed episodes end: " + "; ".join(f"{v} x {k}" for k, v in endings.most_common()) + ".",
    "Notation: T = teacher trajectory, S = student free run, actions in order separated by ' > ', 'xN' = the same action "
    "N times in a row. The step-level diagnosis pairs you receive separately are teacher-history states where the "
    "student's greedy action differs from yours; this dossier shows what happens when the student runs on its own.",
]
text = "\n".join(head) + "\n\n" + "\n".join(rows) + "\n"
assert "valid_seen/" not in text and "valid_unseen/" not in text and len(text) <= 120000, len(text)
json.dump({"title": "Raw student free-run evidence on the 20 training tasks", "text": text}, open(out, "x"), ensure_ascii=False, indent=1)
print("dossier characters:", len(text), "| student solved", solved, "of 20 | endings", dict(endings))
print(text[:1800])
