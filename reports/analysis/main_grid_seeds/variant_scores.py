"""dev140 results of the student-base variants (run on hpg): e4b-/e2b- converged-column cells + the base student.
Writes JSON to stdout: {"e4b|tsft|30": [solved task ids], "e4b|tsft|30|s1": [...], ..., "e4b|base": [...]}; missing\nevals are left out."""
import csv, glob, json

R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"
CONV = {"tsft": {10: "tsft-k10-e32-lr5e-5", 20: "tsft-k20-e32-lr5e-5", 30: "tsft-k30-e64-lr5e-5", 100: "tsft-k100-e64-lr1e-4"},
        "AD": {10: "adc-k10-e32-lr1e-4", 20: "adc-k20-e32-lr1e-4", 30: "adc-k30-e64-lr1e-4", 100: "adc-k100-e64-lr1e-4"},
        "SAD": {10: "sadc-k10-b1", 20: "sadc-k20-b1", 30: "sadc-k30-b1", 100: "sadc-k100-b05"},
        "SmartAD": {10: "smc-k10-e32-lr1e-4", 20: "smc-k20-e32-lr1e-4", 30: "smc-k30-e32-lr1e-4", 100: "smc-k100-e64-lr1e-4"}}
# 10-26: E4B K10/K20 (rule-A rungs; E2B has no K10/K20 cells, its globs simply find nothing). SAD K20's reported rung is
# decided by rule A once b05/b1 are trained; b1 is listed provisionally and only an evaluated cell can appear.
REPO = {"e4b": "google/gemma-4-E4B-it", "e2b": "google/gemma-4-E2B-it"}


def solved(p):
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    assert len(rows) == 140, p
    return sorted(r["task_id"] for r in rows if r["success"] == "True")


out = {}
for tag in REPO:
    for m, row in CONV.items():
        for k, c in row.items():
            cell = f"{tag}-{c}"
            if c.startswith("sadc-"):
                fs = sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-*-greedy/attempt-*/output/per_task.tsv"))
            else:
                fs = glob.glob(f"{R}/eval/chain/{cell}/{cell}/eval/per_task.tsv")
            if fs:
                out[f"{tag}|{m}|{k}"] = solved(fs[-1])
            for seed in (1, 2):   # training seeds 1, 2 (SAD is seed-independent: no seed cells)
                fs = glob.glob(f"{R}/eval/chain/{cell}-s{seed}/{cell}-s{seed}/eval/per_task.tsv")
                if fs and not c.startswith("sadc-"):
                    out[f"{tag}|{m}|{k}|s{seed}"] = solved(fs[0])
    fs = glob.glob(f"{R}/eval/chain/{tag}-tsft-k30-e64-lr5e-5/{REPO[tag]}/eval/per_task.tsv")
    if fs:
        out[f"{tag}|base"] = solved(fs[0])
json.dump(out, open("/dev/stdout", "w"))
