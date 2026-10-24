"""Per-seed dev140 results of the main grid (run on hpg). Seed s>0 of a LoRA cell = <seed-0 cell>-s<s>; SAD is
seed-independent (full-parameter, fixed curriculum, no active dropout), so it has one run. Writes JSON to stdout:
{"conv|tsft|500": {"0": [solved task ids], "1": [...], ...}, ...}; missing evals are left out."""
import csv, glob, json

R = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs"
KS = [30, 100, 300, 500]
CONV = {"tsft": {30: "tsft-k30-e64-lr5e-5", 100: "tsft-k100-e64-lr1e-4", 300: "tsft-k300-e64-lr1e-4", 500: "tsft-k500-e64-lr5e-5"},
        "AD": {30: "adc-k30-e64-lr1e-4", 100: "adc-k100-e64-lr1e-4", 300: "adc-k300-e64-lr5e-5", 500: "adc-k500-e64-lr5e-5"},
        "SAD": {30: "sadc-k30-b1", 100: "sadc-k100-b05", 300: "sadc-k300-b1", 500: "sadc-k500-b2"},
        "SmartAD": {30: "smc-k30-e32-lr1e-4", 100: "smc-k100-e64-lr1e-4", 300: "smc-k300-e32-lr1e-4", 500: "smc-k500-e32-lr5e-5"}}
PAPER = {"tsft": {30: "tsft-k30-e2", 100: "tsft-k100-e2-lr1e-4", 300: "tsft-k300-e2", 500: "tsft-k500-e2"},
         "AD": {30: "adc-k30-e2", 100: "adc-k100-e2", 300: "adc-k300-e2", 500: "adc-k500-e2-lr1e-4"},
         "SAD": {30: "sadc-k30-b1", 100: "sadc-k100-b1-lr1e-5", 300: "sadc-k300-b1", 500: "sadc-k500-b1"},
         "SmartAD": {k: f"smc-k{k}-e2-lr2e-4" for k in KS}}


def tsv(cell):
    if cell.startswith("sadc-"):
        fs = sorted(glob.glob(f"{R}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))
    else:
        fs = glob.glob(f"{R}/eval/chain/{cell}/*/eval/per_task.tsv")
    return fs[-1] if fs else None


out = {}
for col, table in (("conv", CONV), ("paper", PAPER)):
    for m, row in table.items():
        for k, c0 in row.items():
            runs = {}
            for s in ((0,) if m == "SAD" else (0, 1, 2)):
                p = tsv(c0 if s == 0 else f"{c0}-s{s}")
                if not p:
                    continue
                rows = list(csv.DictReader(open(p), delimiter="\t"))
                assert len(rows) == 140, p
                runs[str(s)] = sorted(r["task_id"] for r in rows if r["success"] == "True")
            out[f"{col}|{m}|{k}"] = runs
json.dump(out, open("/dev/stdout", "w"))
