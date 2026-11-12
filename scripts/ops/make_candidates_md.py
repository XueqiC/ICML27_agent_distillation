#!/usr/bin/env python3
"""Write the deletion-candidate list (markdown, Chinese) from the census and the moved-file lists.
usage: make_candidates_md.py units.tsv MOVED_LIST_DIR OUT.md "<status line>" """
import sys, os, glob, collections

units, moved_dir, out, status = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
GiB = 2.0 ** 30
D = "/blue/fsu-compsci-dept/xc25.fsu/hq/"; Y = "/blue/yd24f.fsu/xc25.fsu/hq/"
OD = "/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/"; OY = "/orange/yd24f.fsu/xc25.fsu/"
KEY = {D + "ICML27_agent_distillation": "dept-ICML27_agent_distillation", D + "ICML27_v0_runs": "dept-ICML27_v0_runs",
       D + "ICML27_runs": "dept-ICML27_runs", D + "ICML27_p1_backup": "dept-ICML27_p1_backup", D + "tc-alignment": "dept-tc-alignment",
       D + "tc-alignment-vllm": "dept-tc-alignment-vllm", D + "envs/p1-venv": "dept-envs-p1-venv",
       Y + "ICML27_archive": "yd-ICML27_archive", Y + "tc-alignment": "yd-tc-alignment", Y + "_trash": "yd-_trash"}
moved = collections.defaultdict(set)
for p in glob.glob(os.path.join(moved_dir, "*.list")):
    moved[os.path.basename(p).split(".")[0]] |= set(l.rstrip("\n") for l in open(p))

U = []; B = []
for line in open(units, errors="replace"):
    f = line.rstrip("\n").split("\t")
    if f[0] == "U": U.append((KEY[f[1]], f[2], int(f[3]), int(f[4])))
    elif f[0] == "B": B.append((KEY[f[1]], f[3], int(f[2])))

def agg(key, prefix):
    """census files/bytes of units under key/prefix, and moved big files under it"""
    n = b = mn = mb = 0
    for k, rel, fn, fb in U:
        if k == key and rel != "(shallow)" and (rel == prefix or rel.startswith(prefix + "/") or prefix == ""):
            n += fn; b += fb
    for k, rel, s in B:
        if k == key and (rel.startswith(prefix + "/") or prefix == "") and rel in moved[k]:
            mn += 1; mb += s
    return n, b, mn, mb

def g(b): return f"{b / GiB:,.1f}"
rows_total = {}
L = []
w = L.append
w("# hpg 存储:可删候选清单(tc / Paper 1,2026-11-12)")
w("")
w("**你点头之前什么都不删。** 下面 A–E 五件事是相互独立的决定,可以分别回答。")
w("")
w("- 数字来自 11-11 夜里的只读盘点(`lfs find`,tc 的 10 个目录,共 15,063,843 个文件 / 2,235.9 GiB;其中 698 个 ≥64 MiB 的文件占 2,023.3 GiB)。字节数是盘点时的精确值。")
w("- 已搬到 /orange 的大文件(≥64 MiB)每个都两边算过 sha256,一致后才删的 /blue 源;每个目录有清单 `_tc_moved_big_files.<批次>.tsv`,按清单 `rsync` 即可搬回原路径。")
w(f"- {status}")
w("- 路径缩写:`DEPT` = `/blue/fsu-compsci-dept/xc25.fsu/hq`,`YD` = `/blue/yd24f.fsu/xc25.fsu/hq`,`O` = `/orange/yd24f.fsu/xc25.fsu`。")
w("")
w("@@SUMMARY@@")
w("")

# ---------------- 1. trash
w("## 1. `_trash` 内容(最大的一项)")
w("")
w("这些是项目进行中我自己判为作废、按规则移进 `_trash` 而没有删的运行目录。大文件现在在 /orange,**占的是 yd24f 组的 /orange 配额**(规则是那里不能长期堆放),小文件余量还在 /blue。任何报告过的数字都不依赖它们。")
w("")
w("| # | 内容 | 大文件现在的位置(/orange) | 大文件 | 字节 | /blue 上的小文件余量 | 为什么作废 | 删了会失去什么 |")
w("|---|---|---|---|---|---|---|---|")
T = [
 ("1.1", "dept-ICML27_agent_distillation", "_trash/tainted_or_failed_1019", OD + "ICML27_agent_distillation/_trash/tainted_or_failed_1019/", "DEPT/ICML27_agent_distillation/_trash/tainted_or_failed_1019/",
  "10-19 teacher 429 配额耗尽时被污染的 14 个 LEAP K5–K20 cell,加上失败的 wkm/gad/had k20s0、agentdistill/coa k20s0;之后都重跑过", "只剩排查价值的失败运行权重和日志"),
 ("1.2", "dept-ICML27_agent_distillation", "_trash/leap_60s_timeout_1019", OD + "ICML27_agent_distillation/_trash/leap_60s_timeout_1019/", "DEPT/ICML27_agent_distillation/_trash/leap_60s_timeout_1019/",
  "LEAP 旧运行(14 个目录),teacher 60 s 超时截断了纠正;10-19 改成上游默认超时后整套重跑", "旧协议下的 LEAP 权重,不对应任何报告数字"),
 ("1.3", "yd-_trash", "ICML27_agent_distillation/superseded_1007", OY + "tc-cold/blue-yd24f/_trash/ICML27_agent_distillation/superseded_1007/", "YD/_trash/ICML27_agent_distillation/superseded_1007/",
  "10-07 配额告急时移走的被取代运行:leap-privileged-k20s0-m2/m3/m4、leap-zero-k20s0-m3/m4、eto-k20s0-tp…tp4、ipr-k20s0", "被后续版本取代的中间权重"),
 ("1.4", "dept-ICML27_v0_runs", "_failed", OD + "ICML27_v0_runs/_failed/", "DEPT/ICML27_v0_runs/_failed/",
  "11-06 v0_k30_r1 超时那次的半成品(三个训练臂的 manifest 和一个 adapter)等", "失败运行的半成品"),
]
tsum = [0, 0, 0, 0]
for num, key, pre, opath, bpath, why, lost in T:
    n, b, mn, mb = agg(key, pre)
    if key == "dept-ICML27_v0_runs":  # nested _failed dirs as well
        mn = sum(1 for k, rel, s in B if k == key and "_failed" in rel.split("/") and rel in moved[k])
        mb = sum(s for k, rel, s in B if k == key and "_failed" in rel.split("/") and rel in moved[k])
        n = sum(fn for k, rel, fn, fb in U if k == key and "_failed" in rel.split("/")); b = sum(fb for k, rel, fn, fb in U if k == key and "_failed" in rel.split("/"))
    w(f"| {num} | `{pre}` | `{opath}` | {mn} 个 | {mb:,} B({g(mb)} GiB) | `{bpath}`:约 {max(n - mn, 0):,} 个文件 / {g(max(b - mb, 0))} GiB | {why} | {lost} |")
    for i, v in enumerate((n, b, mn, mb)): tsum[i] += v
# the remaining small _trash entries of ICML27_agent_distillation
key = "dept-ICML27_agent_distillation"
on = ob = omn = omb = 0
for k, rel, fn, fb in U:
    if k == key and rel.startswith("_trash/") and not rel.startswith("_trash/tainted_or_failed_1019") and not rel.startswith("_trash/leap_60s_timeout_1019"):
        on += fn; ob += fb
for k, rel, s in B:
    if k == key and rel.startswith("_trash/") and not rel.startswith("_trash/tainted_or_failed_1019/") and not rel.startswith("_trash/leap_60s_timeout_1019/") and rel in moved[k]:
        omn += 1; omb += s
w(f"| 1.5 | `_trash/` 下其余小项(smc_prep_4gpu_1023/b/c、eval_reset_error_1022、两个 smc-k30 eval 参考、derive_k50_test_1020、partial_tree*) | `{OD}ICML27_agent_distillation/_trash/<同名>/` | {omn} 个 | {omb:,} B({g(omb)} GiB) | `DEPT/ICML27_agent_distillation/_trash/`:约 {max(on - omn, 0):,} 个文件 / {g(max(ob - omb, 0))} GiB | 4-GPU 预处理改 2-GPU 后的旧预处理、$HOME 满导致的无效评测、测试产物、半截目录 | 无用的中间产物 |")
for i, v in enumerate((on, ob, omn, omb)): tsum[i] += v
cores = [(rel, s) for k, rel, s in B if k == "dept-ICML27_runs" and "/core." in rel]
cb = sum(s for _, s in cores)
w(f"| 1.6 | 两个 core dump(`pt_nccl_watchdg` 崩溃转储) | `{OD}ICML27_runs/def44a9/`、`{OD}ICML27_runs/c265b5d/` | {len(cores)} 个 | {cb:,} B({g(cb)} GiB) | 无 | 训练进程崩溃时系统自动写的内存转储,不是实验产物 | 什么都不失去 |")
w(f"| | **合计** | | **{tsum[2] + len(cores)} 个** | **{tsum[3] + cb:,} B({g(tsum[3] + cb)} GiB)** | 约 {max(tsum[0] - tsum[2], 0):,} 个文件 / {g(max(tsum[1] - tsum[3], 0))} GiB | | |")
w("")
w("**我的建议:1.1–1.6 全部删除**(/orange 上的大文件和 /blue 上的余量一起)。这是最大的一项,删掉后 /orange 立刻少约 " + g(tsum[3] + cb) + " GiB。")
w("")

# ---------------- 2. envs
w("## 2. 运行环境(都还在 /blue,没有搬)")
w("")
w("环境可以重建,但没有锁文件,重建出的包版本可能和当时不完全一样。分两组,建议不同。")
w("")
w("### 2a. 旧阶段(tc-alignment,9-24 已归档)的 benchmark 环境 —— 建议删")
w("")
w("| # | 路径 | 文件数 | 字节 | 是什么 | 删了会失去什么 |")
w("|---|---|---|---|---|---|")
E_OLD = [
 ("2a.1", "dept-tc-alignment", "envs/bfcl", "DEPT/tc-alignment/envs/bfcl/", "BFCL 的 venv、gorilla 仓库、生成/打分缓存"),
 ("2a.2", "dept-tc-alignment", "envs/bfcl-venv", "DEPT/tc-alignment/envs/bfcl-venv/", "BFCL 的另一个 venv"),
 ("2a.3", "dept-tc-alignment", "envs/tau2", "DEPT/tc-alignment/envs/tau2/", "tau2 仓库 + venv"),
 ("2a.4", "dept-tc-alignment", "envs/tau-bench", "DEPT/tc-alignment/envs/tau-bench/", "tau-bench 仓库 + venv"),
 ("2a.5", "dept-tc-alignment", "envs/appworld-data", "DEPT/tc-alignment/envs/appworld-data/", "AppWorld 数据和旧实验输出"),
 ("2a.6", "dept-tc-alignment", "envs/appworld-official", "DEPT/tc-alignment/envs/appworld-official/", "AppWorld 官方 venv"),
 ("2a.7", "dept-tc-alignment", "envs/appworld-repo", "DEPT/tc-alignment/envs/appworld-repo/", "AppWorld 仓库"),
 ("2a.8", "dept-tc-alignment", "envs/appworld-venv", "DEPT/tc-alignment/envs/appworld-venv/", "AppWorld venv"),
 ("2a.9", "dept-tc-alignment", "envs/appworld-data-downloaded", "DEPT/tc-alignment/envs/appworld-data-downloaded/", "AppWorld 下载的数据"),
 ("2a.10", "dept-tc-alignment", "envs/ds1000", "DEPT/tc-alignment/envs/ds1000/", "DS-1000 的 HF 缓存"),
 ("2a.11", "dept-tc-alignment", "envs/bird", "DEPT/tc-alignment/envs/bird/", "BIRD 的 dev.zip 和 HF 缓存"),
 ("2a.12", "dept-tc-alignment", "envs/.python", "DEPT/tc-alignment/envs/.python/", "uv 装的 CPython 3.12"),
 ("2a.13", "yd-tc-alignment", ".venv", "YD/tc-alignment/.venv/", "旧阶段主 venv(`DEPT/tc-alignment/.venv` 是指向它的软链接)"),
 ("2a.14", "yd-tc-alignment", "envs", "YD/tc-alignment/envs/", "uv 缓存、AppWorld venv 和数据的另一份"),
]
esum = [0, 0]
for num, key, pre, path, what in E_OLD:
    n, b, _, _ = agg(key, pre)
    if n == 0: continue
    w(f"| {num} | `{path}` | {n:,} | {b:,} B({g(b)} GiB) | {what} | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |")
    esum[0] += n; esum[1] += b
w(f"| | **合计** | **{esum[0]:,}** | **{esum[1]:,} B({g(esum[1])} GiB)** | | |")
w("")
w("### 2b. Paper 1 恢复实验要用的环境 —— 建议留")
w("")
w("| # | 路径 | 文件数 | 字节 | 是什么 | 删了会失去什么 |")
w("|---|---|---|---|---|---|")
E_P1 = [
 ("2b.1", "dept-tc-alignment", "envs/vllm", "DEPT/tc-alignment/envs/vllm/", "vLLM 服务环境(`envs/vllm-serve/.venv` 和 `tc-alignment-vllm/envs/vllm-serve` 都指向它);dev140 评测用"),
 ("2b.2", "dept-tc-alignment", "envs/alfworld", "DEPT/tc-alignment/envs/alfworld/", "ALFWorld venv + 数据集(`tc-alignment-vllm/envs/alfworld` 指向它);dev140 评测用"),
 ("2b.3", "dept-envs-p1-venv", "", "DEPT/envs/p1-venv/", "P1 训练 venv"),
 ("2b.4", "dept-ICML27_agent_distillation", ".venv", "DEPT/ICML27_agent_distillation/.venv/", "项目 venv"),
]
psum = [0, 0]
for num, key, pre, path, what in E_P1:
    n, b, _, _ = agg(key, pre)
    w(f"| {num} | `{path}` | {n:,} | {b:,} B({g(b)} GiB) | {what} | 重跑 K30 ×3 / G5 或 dev140 评测前要重建;vLLM 那套有版本钉子(transformers 5.14.1 等),重建要小心 |")
    psum[0] += n; psum[1] += b
w(f"| | **合计** | **{psum[0]:,}** | **{psum[1]:,} B({g(psum[1])} GiB)** | | |")
w("")
rows_total["env_old"] = esum; rows_total["env_p1"] = psum

# ---------------- 3. live remainder
w("## 3. 实验数据的小文件余量(在 /blue)—— 不建议删,有一项例外")
w("")
w("大文件搬走后,各运行目录里剩下的是几百万个小文件:逐 episode 的记录、teacher 采集结果、评测输出。它们是基线和方法数字的原始证据,字节不多,系里的配额也不按文件数卡,所以**建议原地保留**。")
w("")
w("| 目录 | 盘点时文件数 | 盘点时字节 | 已搬走的大文件 | 现在大约还在 /blue |")
w("|---|---|---|---|---|")
for key, name in [("dept-ICML27_agent_distillation", "DEPT/ICML27_agent_distillation"), ("dept-ICML27_v0_runs", "DEPT/ICML27_v0_runs"), ("dept-ICML27_runs", "DEPT/ICML27_runs"),
                  ("dept-ICML27_p1_backup", "DEPT/ICML27_p1_backup"), ("dept-tc-alignment-vllm", "DEPT/tc-alignment-vllm"), ("yd-ICML27_archive", "YD/ICML27_archive")]:
    n = sum(fn for k, rel, fn, fb in U if k == key); b = sum(fb for k, rel, fn, fb in U if k == key)
    mn = sum(1 for k, rel, s in B if k == key and rel in moved[k]); mb = sum(s for k, rel, s in B if k == key and rel in moved[k])
    w(f"| `{name}` | {n:,} | {g(b)} GiB | {mn} 个 / {g(mb)} GiB | 约 {n - mn:,} 个文件 / {g(b - mb)} GiB |")
w("")
w("其中 `DEPT/ICML27_agent_distillation` 的大头(盘点时):`reports/runs/eval` 195 万个文件、`reports/smartad-selection/chain_k500_s0_r2_e4b` 192 万、`reports/smartad-selection/chain_k500_s0_r2` 187 万、`data/verified/seeds_full_train_r2` 98 万(teacher 采集,花了钱,不可再生)、`reports/runs/smartad-chain` 89 万。")
w("")
# exception: 12B / E2B student selection + collection data nobody references
w("### 例外:12B / E2B student 的选择与采集记录 —— 建议删(属于 11-02 你已同意的类别,当时清单漏了)")
w("")
w("这些目录的 `identity.json` 写明 student 是 gemma-4-12B-it 或 gemma-4-E2B-it。我查了现存 48 个基线 cell 的 `models.json`、`training/identity.json`、`training/provenance.json`、`training/tasks.json`:**没有任何一个引用它们**。(同名的 12B 时期目录 `reports/runs/smartad-chain/chain_k500_s0_r2` 和 `reports/runs/adftp-chain/chain_k500_s0_r2` **被 E4B 的 SmartAD / AD cell 引用,不能删**,不在本表。)我只查了这 48 个 cell 的这四类 json,没有查 `reports/runs/eval`、`method_v1` 等其他目录;真要删之前我会再全量 grep 一遍。")
w("")
w("3.1 里还有 4 个 ≥64 MiB 的清单文件(0.8 GiB)我特意没搬,留在 /blue 等你决定。")
w("")
w("| # | 路径(都在 /blue) | 文件数 | 字节 | 是什么 | 删了会失去什么 |")
w("|---|---|---|---|---|---|")
X = [
 ("3.1", "reports/smartad-selection/chain_k500_s0_r2_e2b", "E2B student 的 SmartAD K500 打分与选择"),
 ("3.2", "reports/smartad-selection/chain_k500_s0_r2", "12B student 的 SmartAD K500 打分与选择"),
 ("3.3", "reports/smartad-selection/k100_s0", "12B student 的 SmartAD K100 选择(seed 0)"),
 ("3.4", "reports/smartad-selection/k100_s1", "同上(seed 1)"),
 ("3.5", "reports/smartad-selection/k100_s2", "同上(seed 2)"),
 ("3.6", "reports/runs/smartad-collections/k100_s0", "12B 时期的 SmartAD K100 采集(seed 0)"),
 ("3.7", "reports/runs/smartad-collections/k100_s1", "同上(seed 1)"),
 ("3.8", "reports/runs/smartad-collections/k100_s2", "同上(seed 2)"),
 ("3.9", "reports/adftp-collections/k100_s0", "12B 时期的 AD-FTP K100 采集(seed 0)"),
 ("3.10", "reports/adftp-collections/k100_s1", "同上(seed 1)"),
 ("3.11", "reports/adftp-collections/k100_s2", "同上(seed 2)"),
]
xs = [0, 0]
for num, pre, what in X:
    n, b, mn, mb = agg("dept-ICML27_agent_distillation", pre)
    note = f";其中 {mn} 个大文件({g(mb)} GiB)已在 /orange" if mn else ""
    w(f"| {num} | `DEPT/ICML27_agent_distillation/{pre}/` | {n:,} | {b:,} B({g(b)} GiB){note} | {what} | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |")
    xs[0] += n; xs[1] += b
w(f"| | **合计** | **{xs[0]:,}** | **{xs[1]:,} B({g(xs[1])} GiB)** | | |")
w("")

# ---------------- 4. live weights on /orange
w("## 4. 已搬到 /orange 的有效权重 —— 不是删除候选,只有 4.4 请你定")
w("")
w("| # | 位置(/orange) | 大文件 | 字节 | 是什么 | 建议 |")
w("|---|---|---|---|---|---|")
def live(key, pred):
    xs_ = [(rel, s) for k, rel, s in B if k == key and rel in moved[k] and pred(rel)]
    return len(xs_), sum(s for _, s in xs_)
n1, b1 = live("dept-ICML27_agent_distillation", lambda r: not r.startswith("_trash/"))
n2, b2 = live("dept-ICML27_v0_runs", lambda r: "_failed" not in r.split("/"))
n3, b3 = live("dept-ICML27_p1_backup", lambda r: True)
n4, b4 = live("yd-ICML27_archive", lambda r: True)
w(f"| 4.1 | `{OD}ICML27_agent_distillation/`(`reports/runs/baselines`、`reports/runs/eval`、`method_v1` 等) | {n1} 个 | {b1:,} B({g(b1)} GiB) | 主实验网格的基线权重、评测用的合并模型、V2 方法权重 | 留。重跑或复评前按清单搬回 |")
w(f"| 4.2 | `{OD}ICML27_v0_runs/`(不含 `_failed`) | {n2} 个 | {b2:,} B({g(b2)} GiB) | v0 K10/K20/K30 的 adapter、数据集、标注 | 留。K30 ×3 / G5 重跑要用 |")
w(f"| 4.3 | `{OD}ICML27_p1_backup/` | {n3} 个 | {b3:,} B({g(b3)} GiB) | P1 评测运行树的备份 | 留 |")
w(f"| 4.4 | `{OY}icml27-cold/blue-yd24f/ICML27_archive/baselines_evaluated_1007/` | {n4} 个 | {b4:,} B({g(b4)} GiB) | 10-07 评测过的 LEAP 早期阶梯(privileged K1/K2/K5/K10/K20-m5、zero K20-m5)的各阶段权重 | **候选,请你定。** 它们是 10-19 修超时之前的旧协议运行,11-07 的总结里没有用到 LEAP;评测结果的小文件还在 /blue(`YD/ICML27_archive`,约 52 万个文件 / 7.7 GiB)。我没有逐个核对是否有别处引用这些权重 |")
w("")

# ---------------- restore
w("## 恢复方法(任何已搬走的文件)")
w("")
w("```bash")
w("# 例:把 ICML27_v0_runs 的大文件搬回 /blue 原路径(每个批次一份清单)")
w("M=/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_v0_runs")
w("cat $M/_tc_moved_big_files.*.tsv | grep -v '^#' | cut -f3 > /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/restore.list")
w("rsync -a --files-from=/orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/restore.list $M/ /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_v0_runs/")
w("```")
w("")
w("清单第一列是 sha256,搬回后可以用 `sha256sum -c` 核对。各目录的 /orange 路径和批次记在 `ICML27_agent_distillation/PROJECT_STATE.md`(11-11/11-12 条目)。")
S = ["## 要你定的五件事(细节见下)", "",
     "| 决定 | 内容 | 量 | 现在在哪 | 我的建议 |", "|---|---|---|---|---|",
     f"| A | `_trash` 作废运行 + 两个 core dump(第 1 节) | 大文件 {tsum[2] + len(cores)} 个 / {g(tsum[3] + cb)} GiB,另有约 {max(tsum[0] - tsum[2], 0):,} 个小文件 / {g(max(tsum[1] - tsum[3], 0))} GiB | 大文件在 /orange,小文件在 /blue | **删**。最大的一项,而且占着 yd24f 组的 /orange 配额 |",
     f"| B | 旧阶段 benchmark 环境(2a) | {esum[0]:,} 个文件 / {g(esum[1])} GiB | /blue | **删**,旧项目已归档 |",
     f"| C | Paper 1 恢复实验用的环境(2b) | {psum[0]:,} 个文件 / {g(psum[1])} GiB | /blue | **留** |",
     f"| D | 12B / E2B student 的选择与采集记录(第 3 节例外) | {xs[0]:,} 个文件 / {g(xs[1])} GiB | /blue | **删**,属于 11-02 你已同意的类别;删前我再全量核对一次引用 |",
     f"| E | 10-07 的 LEAP 早期阶梯权重(4.4) | {n4} 个文件 / {g(b4)} GiB | /orange | 倾向删,但请你确认 |",
     "", "其余(有效权重 4.1–4.3、实验数据的小文件余量)不建议动。"]
i = L.index("@@SUMMARY@@"); L[i:i + 1] = S
open(out, "w").write("\n".join(L) + "\n")
print("candidates 12B/E2B GiB", g(xs[1]), "files", xs[0])
print("wrote", out, "lines", len(L))
print("trash big total GiB", g(tsum[3] + cb), "old env GiB", g(esum[1]), "p1 env GiB", g(psum[1]))
