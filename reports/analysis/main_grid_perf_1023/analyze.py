"""Main-grid baseline performance on dev140 (seed 0): per-cell success, CIs, paired tests, per-type, overlap, failures."""
import csv, glob, json, math, os, sys
from collections import Counter, defaultdict

D = os.path.dirname(os.path.abspath(__file__))
KS = [30, 100, 300, 500]
METHODS = ["tsft", "AD", "SAD", "SmartAD"]
NAME = {"tsft": "轨迹SFT", "AD": "AD", "SAD": "SAD", "SmartAD": "SmartAD"}
CONV = {"tsft": {30: "tsft-k30-e64-lr5e-5", 100: "tsft-k100-e64-lr1e-4", 300: "tsft-k300-e64-lr1e-4", 500: "tsft-k500-e64-lr5e-5"},
        "AD": {30: "adc-k30-e64-lr1e-4", 100: "adc-k100-e64-lr1e-4", 300: "adc-k300-e32-lr1e-4", 500: "adc-k500-e64-lr5e-5"},
        "SAD": {30: "sadc-k30-b1", 100: "sadc-k100-b05", 300: "sadc-k300-b1", 500: "sadc-k500-b2"},
        "SmartAD": {100: "smc-k100-e64-lr1e-4", 300: "smc-k300-e32-lr1e-4"}}
for k, c in [(30, "smc-k30-e32-lr1e-4"), (500, None)]:
    if c and glob.glob(f"{D}/eval/chain/{c}/*/eval/per_task.tsv"): CONV["SmartAD"][k] = c
PAPER = {"tsft": {k: f"tsft-k{k}-e2" for k in KS}, "AD": {k: f"adc-k{k}-e2" for k in KS},
         "SAD": {k: f"sadc-k{k}-b1" for k in KS}, "SmartAD": {k: f"smc-k{k}-e2-lr2e-4" for k in KS}}
COLLAPSE = {("tsft", 100), ("tsft", 300), ("AD", 300), ("SmartAD", 100), ("SmartAD", 300)}  # converged-column flags


def path(cell):
    if cell.startswith("sadc-"):
        att = "attempt-001" if cell == "sadc-k500-b2" else "attempt-000"
        return f"{D}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/{att}/output/per_task.tsv"
    return glob.glob(f"{D}/eval/chain/{cell}/*/eval/per_task.tsv")[0]


def load(p):
    rows = {r["task_id"]: r for r in csv.DictReader(open(p), delimiter="\t")}
    assert len(rows) == 140, p
    return rows


BASE = load(f"{D}/eval/seedsft-sweep-bi/k5s0/google/gemma-4-12B-it/eval/per_task.tsv")
TASKS = sorted(BASE)
TYPE = {t: BASE[t]["task_type"] for t in TASKS}
TYPES = sorted(set(TYPE.values()))
SHORT = {"look_at_obj_in_light": "look", "pick_and_place_simple": "simple", "pick_clean_then_place_in_recep": "clean",
         "pick_cool_then_place_in_recep": "cool", "pick_heat_then_place_in_recep": "heat", "pick_two_obj_and_place": "two"}
data = {}
for col, table in (("conv", CONV), ("paper", PAPER)):
    for m in METHODS:
        for k, c in table[m].items():
            data[col, m, k] = load(path(c))
succ = lambda rows: {t for t in TASKS if rows[t]["success"] == "True"}
base_s = succ(BASE)


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d * n, (c + h) / d * n


def mcnemar(a, b):
    """Exact two-sided McNemar on paired task outcomes; returns (only_a, only_b, p)."""
    x, y = len(a - b), len(b - a); n = x + y
    if n == 0: return x, y, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n * 2
    return x, y, min(1.0, p)


out = []
P = out.append
P("# 主表 baseline performance(dev140,seed 0)\n")
P("dev140 = 140 个 ALFWorld valid_seen 任务(137 个 task group),greedy 解码,batch-invariant vLLM,40 步上限;"
  "base(gemma-4-12B-it 不训练)= 62/140。所有数字是单 seed。\n")
P("## 1. 成功数与 95% Wilson 区间\n")
for col, title in (("conv", "收敛列"), ("paper", "论文 recipe 列")):
    P(f"### {title}\n\n| 方法 | " + " | ".join(f"K{k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
    for m in METHODS:
        cells = []
        for k in KS:
            if (col, m, k) not in data: cells.append("—"); continue
            x = len(succ(data[col, m, k])); lo, hi = wilson(x, 140)
            flag = "c" if col == "conv" and (m, k) in COLLAPSE else ""
            cells.append(f"{x}{flag} [{lo:.0f}–{hi:.0f}]")
        P(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    P("")
P("## 2. 配对检验(同一批 140 个任务;exact McNemar)\n")
P("每格 = 只有该方法成功的任务数 / 只有对照成功的任务数,p 值。\n")
for ref_name, ref in (("轨迹SFT(同 K 收敛列)", None), ("base", base_s)):
    P(f"### 对照:{ref_name}\n\n| 方法 | " + " | ".join(f"K{k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
    for m in METHODS:
        if ref is None and m == "tsft": continue
        cells = []
        for k in KS:
            if ("conv", m, k) not in data: cells.append("—"); continue
            r = ref if ref is not None else succ(data["conv", "tsft", k])
            a, b, p = mcnemar(succ(data["conv", m, k]), r)
            cells.append(f"+{a}/−{b}, p={p:.3f}")
        P(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    P("")
P("## 3. 按任务类型(收敛列,成功数 / 该类任务数)\n")
counts = Counter(TYPE.values())
P("| 方法 | K | " + " | ".join(f"{SHORT[t]} ({counts[t]})" for t in TYPES) + " | 合计 |\n|---|---|" + "---|" * (len(TYPES) + 1))
P("| base | — | " + " | ".join(str(sum(1 for x in base_s if TYPE[x] == t)) for t in TYPES) + f" | {len(base_s)} |")
for m in METHODS:
    for k in KS:
        if ("conv", m, k) not in data: continue
        s = succ(data["conv", m, k])
        P(f"| {NAME[m]} | {k} | " + " | ".join(str(sum(1 for x in s if TYPE[x] == t)) for t in TYPES) + f" | {len(s)} |")
P("")
P("## 4. 方法之间的重叠(收敛列,同一 K)\n")
P("| K | 方法数 | 全部成功 | 至少一个成功(oracle) | 全部失败 | 各方法独有成功 |\n|---|---|---|---|---|---|")
for k in KS:
    ms = [m for m in METHODS if ("conv", m, k) in data]
    S = {m: succ(data["conv", m, k]) for m in ms}
    allm = set.intersection(*S.values()); anym = set.union(*S.values())
    uniq = {m: len(S[m] - set.union(*(S[o] for o in ms if o != m))) for m in ms}
    P(f"| {k} | {len(ms)} | {len(allm)} | {len(anym)} | {140 - len(anym)} | " + ", ".join(f"{NAME[m]} {uniq[m]}" for m in ms) + " |")
P("")
never = [t for t in TASKS if not any(t in succ(v) for (col, m, k), v in data.items() if col == "conv")]
P(f"收敛列所有格都没解决的任务:{len(never)} 个,类型 " + ", ".join(f"{SHORT[a]} {b}" for a, b in Counter(TYPE[t] for t in never).most_common()) + "\n")
P("## 5. 失败方式与效率(收敛列)\n")
P("失败 episode 全部是用满 40 步(仅论文列 轨迹SFT K30 有 5 个 context overflow)。\n")
P("| 方法 | K | 成功 episode 平均步数 | 失败 episode 无效动作占比 | 全部 episode 无效动作占比 | parser 错误 | 平均输出 token/步 |\n|---|---|---|---|---|---|---|")
def fm(rows):
    S = [r for r in rows.values() if r["success"] == "True"]; F = [r for r in rows.values() if r["success"] != "True"]
    steps = lambda rs: sum(int(r["steps"]) for r in rs)
    inad = lambda rs: sum(int(r["inadmissible_actions"]) for r in rs)
    s_mean = steps(S) / len(S) if S else float("nan")
    f_inad = inad(F) / steps(F) if F and steps(F) else float("nan")
    tok = sum(int(r["completion_tokens"] or 0) for r in rows.values()) / steps(rows.values())
    return s_mean, f_inad, inad(rows.values()) / steps(rows.values()), sum(int(r["parser_errors"] or 0) for r in rows.values()), tok
s, f, a, pe, tok = fm(BASE)
P(f"| base | — | {s:.1f} | {f:.0%} | {a:.0%} | {pe} | {tok:.0f} |")
for m in METHODS:
    for k in KS:
        if ("conv", m, k) not in data: continue
        s, f, a, pe, tok = fm(data["conv", m, k])
        P(f"| {NAME[m]} | {k} | {s:.1f} | {f:.0%} | {a:.0%} | {pe} | {tok:.0f} |")
P("")
open(f"{D}/perf_report.md", "w").write("\n".join(out))
json.dump({f"{c}|{m}|{k}": sorted(succ(v)) for (c, m, k), v in data.items()}, open(f"{D}/solved_sets.json", "w"))
print("\n".join(out))
