"""V1b analysis by the pre-registered rule (docs/direction/2026-10-27_V1b_joint_guidance_prereg_ZH.md §4-§5, revision 2).

Reads reports/runs/method_v1/v1b_r2/runs/records (run 1 in v1b_main is void and never read).
Primary: sampled success per prefix (valid samples), averaged over the 38 prefixes; greedy reported separately.
Void episodes: excluded from the per-prefix mean; bounds with void = fail / void = success reported.
"""
import collections
import glob
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
M = os.path.join(ROOT, "reports/runs/method_v1")
OUT = os.path.dirname(os.path.abspath(__file__))

spec = json.load(open(os.path.join(M, "v1b_main_spec.json")))
udev = {c["case_id"]: c for c in json.load(open(os.path.join(M, "udev_prefix_spec_v1a.json")))}
CASES = [c["case_id"] for c in spec]


def stratum(cid):
    u = udev.get(cid)
    if u is None:
        return "CG"
    kind = {"target_seen": "seen", "search_after_10": "search"}[u["decision_point"]]
    return f"{kind}-{u['stratum']}"


def short(cid):
    g = cid.split("/")[1] if cid.startswith("train/") else cid.split("/")[0]
    return g


recs = collections.defaultdict(list)  # (case, mode, decoding) -> records
for f in glob.glob(os.path.join(M, "v1b_r2/runs/records/*.json")):
    r = json.load(open(f))
    recs[(r["case_id"], r["mode"], r["decoding"])].append(r)

# V1a teacher takeover success per prefix (T arm, reused, not rerun)
t_succ = collections.defaultdict(list)
for d in ("v1a_udev", "v1a_cg7"):
    for f in glob.glob(os.path.join(M, d, "records/*.json")):
        r = json.load(open(f))
        if r["status"] == "completed":
            t_succ[r["case_id"]].append(bool(r["success"]))


def valid(rs):
    return [r for r in rs if r["status"] == "completed"]


def rate(case, mode, dec="sampled", key="success", void_as=None, filt=None):
    rs = recs[(case, mode, dec)]
    vals = []
    for r in rs:
        if r["status"] != "completed":
            if void_as is not None:
                vals.append(float(void_as))
            continue
        if filt is not None and not filt(r):
            continue
        m = r["metrics"]
        v = m.get(key) if key != "success" else r["success"]
        vals.append(1.0 if v else 0.0)
    return sum(vals) / len(vals) if vals else None


def per_prefix(mode, **kw):
    return {c: rate(c, mode, **kw) for c in CASES}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs)


def boot_ci(a, b, cases, n=10000, seed=0):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [rng.choice(cases) for _ in cases]
        diffs.append(mean([a[c] for c in s]) - mean([b[c] for c in s]))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def compare(a_mode, b_mode, cases=CASES, **kw):
    a, b = per_prefix(a_mode, **kw), per_prefix(b_mode, **kw)
    a = {c: a[c] for c in cases if a[c] is not None and b[c] is not None}
    b = {c: b[c] for c in a}
    cs = list(a)
    d = mean([a[c] for c in cs]) - mean([b[c] for c in cs])
    up = sum(a[c] > b[c] for c in cs)
    down = sum(a[c] < b[c] for c in cs)
    lo, hi = boot_ci(a, b, cs)
    return {"a": a_mode, "b": b_mode, "n": len(cs), "a_mean": mean([a[c] for c in cs]),
            "b_mean": mean([b[c] for c in cs]), "diff": d, "ci95": [lo, hi], "up": up, "down": down,
            "tie": len(cs) - up - down, "pass": d >= 0.15 and up - down >= 4}


res = {}
res["primary_G_vs_U"] = compare("G", "U")
res["void_bounds_G_vs_U"] = {"void_fail": compare("G", "U", void_as=0), "void_success": compare("G", "U", void_as=1)}
res["G_vs_G1"] = compare("G", "G1")
res["G1_vs_U"] = compare("G1", "U")

# teacher-step share (sampled + greedy separately), pooled over episodes and mean per episode
share = {}
for dec in ("sampled", "greedy"):
    for mode in ("G", "G1"):
        te = ex = 0
        per = []
        for c in CASES:
            for r in valid(recs[(c, mode, dec)]):
                m = r["metrics"]
                te += m["teacher_executed_steps"]
                ex += m["steps_used_after_branch"]
                per.append(m["teacher_step_share"])
        share[f"{mode}-{dec}"] = {"pooled": te / ex, "teacher_steps": te, "executed_steps": ex,
                                  "episode_mean": sum(per) / len(per),
                                  "episodes_gt_50pct": sum(p > 0.5 for p in per), "episodes": len(per)}
res["teacher_step_share"] = share
res["takeover_limit_ok"] = share["G-sampled"]["pooled"] <= 0.30
res["sensitivity_drop_gt50"] = compare("G", "U", filt=lambda r: r["metrics"]["teacher_step_share"] <= 0.5)

# greedy
res["greedy"] = {m: sum(bool(valid(recs[(c, m, "greedy")])[0]["success"]) for c in CASES if valid(recs[(c, m, "greedy")]))
                 for m in ("U", "G", "G1")}

# strata
strata = collections.defaultdict(list)
for c in CASES:
    strata[stratum(c)].append(c)
res["strata"] = {}
for s, cs in sorted(strata.items()):
    row = {"n": len(cs)}
    for mode in ("U", "G", "G1"):
        pp = per_prefix(mode)
        row[mode] = mean([pp[c] for c in cs])
    g, u = per_prefix("G"), per_prefix("U")
    row["G>U"] = sum(g[c] is not None and g[c] > u[c] for c in cs)
    row["G<U"] = sum(g[c] is not None and g[c] < u[c] for c in cs)
    row["T"] = mean([sum(t_succ[c]) / len(t_succ[c]) for c in cs])
    res["strata"][s] = row

# secondary: found / picked
sec = {}
for key in ("found", "picked", "transform_done"):
    sec[key] = {m: mean(list(per_prefix(m, key=key).values())) for m in ("U", "G", "G1")}
res["secondary"] = sec

# steps to find (sampled, valid) + not found
stf = {}
for mode in ("U", "G", "G1"):
    xs, nf = [], 0
    for c in CASES:
        for r in valid(recs[(c, mode, "sampled")]):
            m = r["metrics"]
            if m["found"]:
                xs.append(m["steps_to_find"])
            else:
                nf += 1
    xs.sort()
    stf[mode] = {"found": len(xs), "not_found": nf, "median": xs[len(xs) // 2] if xs else None}
res["steps_to_find"] = stf

# mechanism (sampled, valid)
mech = {}
u_first = {}
for c in CASES:
    for r in valid(recs[(c, "U", "sampled")]):
        u_first[(c, r["seed"])] = r["metrics"]["first_post_branch_action"]
for mode in ("U", "G", "G1"):
    agg = collections.Counter()
    n = 0
    changed = paired = 0
    for c in CASES:
        for r in valid(recs[(c, mode, "sampled")]):
            m = r["metrics"]
            n += 1
            for k in ("accepts", "guides", "demonstrates", "guided_steps_followed", "refused_actions",
                      "repeated_refused_actions", "inadmissible_count", "steps_used_after_branch",
                      "teacher_executed_steps", "review_calls", "revisits_of_cleared"):
                agg[k] += m.get(k) or 0
            if mode != "U" and (c, r["seed"]) in u_first:
                paired += 1
                changed += m["first_post_branch_action"] != u_first[(c, r["seed"])]
    row = dict(agg)
    row["episodes"] = n
    if mode != "U":
        row["first_step_changed_vs_U"] = changed / paired
        dec = agg["accepts"] + agg["guides"] + agg["demonstrates"]
        row["decision_dist"] = {k: agg[k] / dec for k in ("accepts", "guides", "demonstrates")} if dec else None
        row["follow_rate"] = agg["guided_steps_followed"] / agg["guides"] if agg["guides"] else None
    row["refused_per_episode"] = agg["refused_actions"] / n
    row["repeated_refused_per_episode"] = agg["repeated_refused_actions"] / n
    row["steps_per_episode"] = agg["steps_used_after_branch"] / n
    mech[mode] = row
res["mechanism"] = mech

# per prefix table
rows = []
for c in CASES:
    rows.append({"case": c, "short": short(c), "stratum": stratum(c),
                 "T_v1a": f"{sum(t_succ[c])}/{len(t_succ[c])}",
                 **{m: rate(c, m) for m in ("U", "G", "G1")},
                 **{f"{m}_greedy": (bool(valid(recs[(c, m, 'greedy')])[0]["success"]) if valid(recs[(c, m, "greedy")]) else None)
                    for m in ("U", "G", "G1")},
                 "G_valid": len(valid(recs[(c, "G", "sampled")]))})
res["per_prefix"] = rows
res["void"] = [{"case": r["case_id"], "mode": r["mode"], "seed": r["seed"], "reason": r["void_reason"]}
               for v in recs.values() for r in v if r["status"] != "completed"]

json.dump(res, open(os.path.join(OUT, "v1b_r2_analysis.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_prefix"}, indent=1))
