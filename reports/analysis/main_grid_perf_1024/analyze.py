"""Main-grid baseline performance on dev140 (seed 0): per-cell success, CIs, paired tests, per-type, overlap, failures."""
import csv, glob, json, math, os, sys
from collections import Counter, defaultdict

D = os.path.dirname(os.path.abspath(__file__))
KS = [30, 100, 300, 500]
METHODS = ["tsft", "AD", "SAD", "SmartAD"]
NAME = {"tsft": "轨迹SFT", "AD": "AD", "SAD": "SAD", "SmartAD": "SmartAD"}
AD_K300 = os.environ.get("AD_K300", "")  # converged rung of the AD K300 5e-5 rerun ladder, e.g. adc-k300-e16-lr5e-5
CONV = {"tsft": {30: "tsft-k30-e64-lr5e-5", 100: "tsft-k100-e64-lr1e-4", 300: "tsft-k300-e64-lr1e-4", 500: "tsft-k500-e64-lr5e-5"},
        "AD": {30: "adc-k30-e64-lr1e-4", 100: "adc-k100-e64-lr1e-4", 300: AD_K300 or "adc-k300-e32-lr1e-4", 500: "adc-k500-e64-lr5e-5"},
        "SAD": {30: "sadc-k30-b1", 100: "sadc-k100-b05", 300: "sadc-k300-b1", 500: "sadc-k500-b2"},
        "SmartAD": {100: "smc-k100-e64-lr1e-4", 300: "smc-k300-e32-lr1e-4", 500: "smc-k500-e32-lr5e-5"}}
for k, c in [(30, "smc-k30-e32-lr1e-4")]:
    if glob.glob(f"{D}/eval/chain/{c}/*/eval/per_task.tsv"): CONV["SmartAD"][k] = c
PAPER = {"tsft": {k: f"tsft-k{k}-e2" for k in KS}, "AD": {k: f"adc-k{k}-e2" for k in KS},
         "SAD": {k: f"sadc-k{k}-b1" for k in KS}, "SmartAD": {k: f"smc-k{k}-e2-lr2e-4" for k in KS}}
# Cells at 0/140 after divergence, rerun with lr halved (registered fallback); the original number is kept for the footnote.
RERUN = {("paper", "tsft", 100): ("tsft-k100-e2-lr1e-4", 0), ("paper", "AD", 500): ("adc-k500-e2-lr1e-4", 0),
         ("paper", "SAD", 100): ("sadc-k100-b1-lr1e-5", 0)}
if AD_K300: RERUN["conv", "AD", 300] = (AD_K300, 4)
for (col, m, k), (c, _) in RERUN.items():
    (CONV if col == "conv" else PAPER)[m][k] = c
# converged-column collapse flags (an epoch mean > 2x the earlier minimum and > minimum + 0.1, recovered by the end)
COLLAPSE = {("tsft", 100), ("tsft", 300), ("AD", 100), ("SmartAD", 100), ("SmartAD", 300)} | (set() if AD_K300 else {("AD", 300)})
NA = {}
OVERRIDE = {("conv", "SmartAD", 30)}  # evaluated despite a failed serving check (user option B), marked ‡


def path(cell):
    if cell.startswith("sadc-"):
        fs = sorted(glob.glob(f"{D}/baselines/{cell}/stages/eval-sad-1of1-attempt-000-greedy/attempt-*/output/per_task.tsv"))
        return fs[-1] if fs else None
    fs = glob.glob(f"{D}/eval/chain/{cell}/*/eval/per_task.tsv")
    return fs[0] if fs else None


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
data, MISSING = {}, []
for col, table in (("conv", CONV), ("paper", PAPER)):
    for m in METHODS:
        for k, c in table[m].items():
            p = path(c)
            if p: data[col, m, k] = load(p)
            else: MISSING.append(c)
if MISSING: print("MISSING per_task.tsv:", MISSING, file=sys.stderr)
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
            if (col, m, k) not in data: cells.append(f"N/A({NA[col, m, k]})" if (col, m, k) in NA else "待出"); continue
            x = len(succ(data[col, m, k])); lo, hi = wilson(x, 140)
            flag = "c" if col == "conv" and (m, k) in COLLAPSE else ""
            flag += "†" if (col, m, k) in RERUN else ""
            flag += "‡" if (col, m, k) in OVERRIDE else ""
            cells.append(f"{x}{flag} [{lo:.0f}–{hi:.0f}]")
        P(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    P("")
P("c = 训练中途 loss 崩过一次又恢复(某个 epoch 均值 > 之前最小值的 2 倍且 > 最小值 + 0.1)。"
  "† = 原 lr 下发散到 0 分附近,按预注册回退规则 lr 减半重跑:" +
  ";".join(f"{NAME[m]} K{k} {'收敛列' if col == 'conv' else '论文列'} 原 {orig}/140"
           for (col, m, k), (_, orig) in sorted(RERUN.items())) + "。"
  "‡ = serving 检查未通过、经用户批准照常评测(bf16 数值敏感:|vLLM − HF fp32| 0.067 > 容差 0.061;"
  "vLLM 两次逐位相同,HF bf16 与 fp32 的误差 0.078 更大)。\n")
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
FM = json.load(open(f"{D}/fm_all.json")) if os.path.exists(f"{D}/fm_all.json") else {}
if FM:
    P("## 6. 失败的 episode 卡在哪一步(failure_modes.py,按动作序列判定子目标)\n")
    P("阶段 = 失败 episode 走到的最远子目标。没拿到 = 从没拿起目标物体;拿到 = 拿起了但没完成中间步骤;"
      "跳步 = 没做 clean/heat/cool(或只拿了一个物体)就放进目标容器;处理完 = 做完中间步骤但没放对;其他 = 步骤都做过仍失败(如两件物体只放对一件)。"
      "见过 = 没拿到的 episode 里,观察中出现过目标物体的个数。覆盖 = 去过的容器数 / 开局可去的容器数。"
      "重复 = 动作在前 4 步内出现过的比例。\n")
    P("| 列 | 方法 | K | 失败 | 没拿到(见过) | 拿到 | 跳步 | 处理完 | 其他 | 没拿到的覆盖 | 没拿到的重复 | 成功的覆盖 |\n"
      "|---|---|---|---|---|---|---|---|---|---|---|---|")
    def frow(col, m, k, cell):
        if cell not in FM: return
        E = list(FM[cell].values()); S = [e for e in E if e["success"]]; F = [e for e in E if not e["success"]]
        st = Counter(e["stage"] for e in F); N = [e for e in F if e["stage"] == "none"]
        cov = lambda X: sum(e["visited"] / e["n_recep"] for e in X if e["n_recep"]) / max(1, sum(1 for e in X if e["n_recep"]))
        rep = lambda X: sum(e["rep4"] for e in X) / max(1, len(X))
        P(f"| {col} | {m} | {k} | {len(F)} | {st['none']}({sum(e['seen_at'] is not None for e in N)}) | {st['took']} | "
          f"{st['skipped']} | {st['trans']} | {st['placed']} | {cov(N):.2f} | {rep(N):.2f} | {cov(S):.2f} |")
    frow("—", "base", "—", "base")
    for col, table in (("收敛", CONV), ("论文", PAPER)):
        for m in METHODS:
            for k in KS:
                if k in table[m]: frow(col, NAME[m], k, table[m][k])
    P("")
P("## 7. 论文 recipe 与收敛训练的配对比较(同方法同 K)\n")
P("| 方法 | " + " | ".join(f"K{k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
for m in METHODS:
    cells = []
    for k in KS:
        if ("conv", m, k) not in data or ("paper", m, k) not in data or CONV[m][k] == PAPER[m][k]:
            cells.append("同一模型" if CONV[m].get(k) == PAPER[m].get(k) else "—"); continue
        a, b, p = mcnemar(succ(data["conv", m, k]), succ(data["paper", m, k]))
        cells.append(f"{len(succ(data['conv', m, k]))} vs {len(succ(data['paper', m, k]))}: +{a}/−{b}, p={p:.3f}")
    P(f"| {NAME[m]} | " + " | ".join(cells) + " |")
P("")
open(f"{D}/perf_report.md", "w").write("\n".join(out))
json.dump({f"{c}|{m}|{k}": sorted(succ(v)) for (c, m, k), v in data.items()}, open(f"{D}/solved_sets.json", "w"))
print("\n".join(out))
