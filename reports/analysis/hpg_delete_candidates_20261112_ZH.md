# hpg 存储:可删候选清单(tc / Paper 1,2026-11-12)

**你点头之前什么都不删。** 下面 A–E 五件事是相互独立的决定,可以分别回答。

- 数字来自 11-11 夜里的只读盘点(`lfs find`,tc 的 10 个目录,共 15,063,843 个文件 / 2,235.9 GiB;其中 698 个 ≥64 MiB 的文件占 2,023.3 GiB)。字节数是盘点时的精确值。
- 已搬到 /orange 的大文件(≥64 MiB)每个都两边算过 sha256,一致后才删的 /blue 源;每个目录有清单 `_tc_moved_big_files.<批次>.tsv`,按清单 `rsync` 即可搬回原路径。
- 状态(11-12 01:18 CST):共搬走 535 个大文件 / 2,128,798,001,335 B(1.936 TiB),全部校验通过,/blue 上的源已删。我们账号在 /blue 上的用量从 2.29 TiB 降到 363 GiB;系里项目配额 19.47T → 16.75T(上限 20T)。
- 路径缩写:`DEPT` = `/blue/fsu-compsci-dept/xc25.fsu/hq`,`YD` = `/blue/yd24f.fsu/xc25.fsu/hq`,`O` = `/orange/yd24f.fsu/xc25.fsu`。

## 2026-11-15 执行结果(用户 11-15 批准:"建议删除的都可以删")

| 项 | 结果 |
|---|---|
| A | **已删。** /orange:220 个大文件 / 1,376,669,062,403 B(1,282.1 GiB);/blue:2,277,316 项(文件 + 目录),27.5 GiB |
| B | **已删。** 13 个目录,462,214 项,29.1 GiB |
| C | 保留 |
| D | **已删**(删前做了全量引用扫描:7,257 个扫描单元,0 报错,0 个 symlink 指向 D,4,075 个文本匹配都不是在用的数据)。/blue:11 个目录,8,394,037 项,74.7 GiB(下文的 53.1 GiB 是文件内容大小,配额按实际占用);/orange:9 个文件 / 1,279,841,993 B |
| E | **已删**(用户 11-15 17:31 CST 批准)。删前全量引用扫描:7,238 个扫描单元,0 报错,0 个 symlink 指向这批权重,307 个文本匹配都不是对权重文件的引用。/orange:38 个文件 / 460,787,521,352 B(429.1 GiB)。/blue 上这 6 个 LEAP cell 的评测结果小文件保留 |

本清单的两处错误(执行时已按更正处理):
- **2a.13 `YD/tc-alignment/.venv` 不是旧环境**:`configs/hpg.env` 的 `TRAIN_PYTHON` 指向它,是 Paper 1 在 hpg 上训练用的 Python。**保留**,应归入 2b。
- **第 1 节的 222 个大文件里有 2 个不属于 `_trash`**:`v0_k20_r1/_failed/label-bytecap-96605d3/{label/rows.json,dataset/examples.jsonl}`(153,967,410 B)在有效运行 `v0_k20_r1` 里。**保留**;实际删除 220 个。

被删的 /orange 文件逐行记在各冷存目录和 /blue 指针目录的 `_DELETED_20261115.{A,D}.tsv` 里。**按 `_tc_moved_big_files.*.tsv` 搬回时必须跳过这些行。**

## 要你定的五件事(细节见下)

| 决定 | 内容 | 量 | 现在在哪 | 我的建议 |
|---|---|---|---|---|
| A | `_trash` 作废运行 + 两个 core dump(第 1 节) | 大文件 222 个 / 1,282.3 GiB,另有约 1,269,043 个小文件 / 19.1 GiB | 大文件在 /orange,小文件在 /blue | **删**。最大的一项,而且占着 yd24f 组的 /orange 配额 |
| B | 旧阶段 benchmark 环境(2a) | 424,508 个文件 / 35.4 GiB | /blue | **删**,旧项目已归档 |
| C | Paper 1 恢复实验用的环境(2b) | 158,300 个文件 / 21.9 GiB | /blue | **留** |
| D | 12B / E2B student 的选择与采集记录(第 3 节例外) | 5,094,649 个文件 / 53.1 GiB | /blue | **删**,属于 11-02 你已同意的类别;删前我再全量核对一次引用 |
| E | 10-07 的 LEAP 早期阶梯权重(4.4) | 38 个文件 / 429.1 GiB | /orange | 倾向删,但请你确认 |

其余(有效权重 4.1–4.3、实验数据的小文件余量)不建议动。

## 1. `_trash` 内容(最大的一项)

这些是项目进行中我自己判为作废、按规则移进 `_trash` 而没有删的运行目录。大文件现在在 /orange,**占的是 yd24f 组的 /orange 配额**(规则是那里不能长期堆放),小文件余量还在 /blue。任何报告过的数字都不依赖它们。

| # | 内容 | 大文件现在的位置(/orange) | 大文件 | 字节 | /blue 上的小文件余量 | 为什么作废 | 删了会失去什么 |
|---|---|---|---|---|---|---|---|
| 1.1 | `_trash/tainted_or_failed_1019` | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_agent_distillation/_trash/tainted_or_failed_1019/` | 52 个 | 630,551,345,008 B(587.2 GiB) | `DEPT/ICML27_agent_distillation/_trash/tainted_or_failed_1019/`:约 464,456 个文件 / 7.2 GiB | 10-19 teacher 429 配额耗尽时被污染的 14 个 LEAP K5–K20 cell,加上失败的 wkm/gad/had k20s0、agentdistill/coa k20s0;之后都重跑过 | 只剩排查价值的失败运行权重和日志 |
| 1.2 | `_trash/leap_60s_timeout_1019` | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_agent_distillation/_trash/leap_60s_timeout_1019/` | 36 个 | 436,535,546,544 B(406.6 GiB) | `DEPT/ICML27_agent_distillation/_trash/leap_60s_timeout_1019/`:约 381,208 个文件 / 6.1 GiB | LEAP 旧运行(14 个目录),teacher 60 s 超时截断了纠正;10-19 改成上游默认超时后整套重跑 | 旧协议下的 LEAP 权重,不对应任何报告数字 |
| 1.3 | `ICML27_agent_distillation/superseded_1007` | `/orange/yd24f.fsu/xc25.fsu/tc-cold/blue-yd24f/_trash/ICML27_agent_distillation/superseded_1007/` | 100 个 | 288,149,541,808 B(268.4 GiB) | `YD/_trash/ICML27_agent_distillation/superseded_1007/`:约 412,911 个文件 / 6.5 GiB | 10-07 配额告急时移走的被取代运行:leap-privileged-k20s0-m2/m3/m4、leap-zero-k20s0-m3/m4、eto-k20s0-tp…tp4、ipr-k20s0 | 被后续版本取代的中间权重 |
| 1.4 | `_failed` | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_v0_runs/_failed/` | 14 个 | 5,627,888,479 B(5.2 GiB) | `DEPT/ICML27_v0_runs/_failed/`:约 4,725 个文件 / 0.1 GiB | 11-06 v0_k30_r1 超时那次的半成品(三个训练臂的 manifest 和一个 adapter)等 | 失败运行的半成品 |
| 1.5 | `_trash/` 下其余小项(smc_prep_4gpu_1023/b/c、eval_reset_error_1022、两个 smc-k30 eval 参考、derive_k50_test_1020、partial_tree*) | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_agent_distillation/_trash/<同名>/` | 18 个 | 10,484,907,782 B(9.8 GiB) | `DEPT/ICML27_agent_distillation/_trash/`:约 5,743 个文件 / 0.0 GiB | 4-GPU 预处理改 2-GPU 后的旧预处理、$HOME 满导致的无效评测、测试产物、半截目录 | 无用的中间产物 |
| 1.6 | 两个 core dump(`pt_nccl_watchdg` 崩溃转储) | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_runs/def44a9/`、`/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_runs/c265b5d/` | 2 个 | 5,473,800,192 B(5.1 GiB) | 无 | 训练进程崩溃时系统自动写的内存转储,不是实验产物 | 什么都不失去 |
| | **合计** | | **222 个** | **1,376,823,029,813 B(1,282.3 GiB)** | 约 1,269,043 个文件 / 19.1 GiB | | |

**我的建议:1.1–1.6 全部删除**(/orange 上的大文件和 /blue 上的余量一起)。这是最大的一项,删掉后 /orange 立刻少约 1,282.3 GiB。

## 2. 运行环境(都还在 /blue,没有搬)

环境可以重建,但没有锁文件,重建出的包版本可能和当时不完全一样。分两组,建议不同。

### 2a. 旧阶段(tc-alignment,9-24 已归档)的 benchmark 环境 —— 建议删

| # | 路径 | 文件数 | 字节 | 是什么 | 删了会失去什么 |
|---|---|---|---|---|---|
| 2a.1 | `DEPT/tc-alignment/envs/bfcl/` | 87,563 | 11,984,401,550 B(11.2 GiB) | BFCL 的 venv、gorilla 仓库、生成/打分缓存 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.2 | `DEPT/tc-alignment/envs/bfcl-venv/` | 39,829 | 9,934,693,408 B(9.3 GiB) | BFCL 的另一个 venv | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.3 | `DEPT/tc-alignment/envs/tau2/` | 18,342 | 1,219,820,078 B(1.1 GiB) | tau2 仓库 + venv | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.4 | `DEPT/tc-alignment/envs/tau-bench/` | 12,089 | 417,459,140 B(0.4 GiB) | tau-bench 仓库 + venv | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.5 | `DEPT/tc-alignment/envs/appworld-data/` | 104,834 | 390,654,734 B(0.4 GiB) | AppWorld 数据和旧实验输出 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.6 | `DEPT/tc-alignment/envs/appworld-official/` | 16,719 | 266,828,036 B(0.2 GiB) | AppWorld 官方 venv | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.7 | `DEPT/tc-alignment/envs/appworld-repo/` | 17,231 | 263,659,373 B(0.2 GiB) | AppWorld 仓库 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.8 | `DEPT/tc-alignment/envs/appworld-venv/` | 15,480 | 173,659,783 B(0.2 GiB) | AppWorld venv | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.9 | `DEPT/tc-alignment/envs/appworld-data-downloaded/` | 14,676 | 34,163,667 B(0.0 GiB) | AppWorld 下载的数据 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.10 | `DEPT/tc-alignment/envs/ds1000/` | 10 | 3,443,613 B(0.0 GiB) | DS-1000 的 HF 缓存 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.11 | `DEPT/tc-alignment/envs/bird/` | 8 | 291,818 B(0.0 GiB) | BIRD 的 dev.zip 和 HF 缓存 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.12 | `DEPT/tc-alignment/envs/.python/` | 3,727 | 103,996,984 B(0.1 GiB) | uv 装的 CPython 3.12 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.13 | `YD/tc-alignment/.venv/` | 39,095 | 7,708,450,202 B(7.2 GiB) | 旧阶段主 venv(`DEPT/tc-alignment/.venv` 是指向它的软链接) | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| 2a.14 | `YD/tc-alignment/envs/` | 54,905 | 5,553,738,953 B(5.2 GiB) | uv 缓存、AppWorld venv 和数据的另一份 | 旧阶段(BFCL/AppWorld/tau2 等)实验的原装环境;要复现旧数字得用目录里的 setup 脚本重建,版本可能漂移 |
| | **合计** | **424,508** | **38,055,261,339 B(35.4 GiB)** | | |

### 2b. Paper 1 恢复实验要用的环境 —— 建议留

| # | 路径 | 文件数 | 字节 | 是什么 | 删了会失去什么 |
|---|---|---|---|---|---|
| 2b.1 | `DEPT/tc-alignment/envs/vllm/` | 69,889 | 8,264,334,663 B(7.7 GiB) | vLLM 服务环境(`envs/vllm-serve/.venv` 和 `tc-alignment-vllm/envs/vllm-serve` 都指向它);dev140 评测用 | 重跑 K30 ×3 / G5 或 dev140 评测前要重建;vLLM 那套有版本钉子(transformers 5.14.1 等),重建要小心 |
| 2b.2 | `DEPT/tc-alignment/envs/alfworld/` | 54,065 | 9,787,425,860 B(9.1 GiB) | ALFWorld venv + 数据集(`tc-alignment-vllm/envs/alfworld` 指向它);dev140 评测用 | 重跑 K30 ×3 / G5 或 dev140 评测前要重建;vLLM 那套有版本钉子(transformers 5.14.1 等),重建要小心 |
| 2b.3 | `DEPT/envs/p1-venv/` | 25,262 | 5,145,449,261 B(4.8 GiB) | P1 训练 venv | 重跑 K30 ×3 / G5 或 dev140 评测前要重建;vLLM 那套有版本钉子(transformers 5.14.1 等),重建要小心 |
| 2b.4 | `DEPT/ICML27_agent_distillation/.venv/` | 9,084 | 352,099,349 B(0.3 GiB) | 项目 venv | 重跑 K30 ×3 / G5 或 dev140 评测前要重建;vLLM 那套有版本钉子(transformers 5.14.1 等),重建要小心 |
| | **合计** | **158,300** | **23,549,309,133 B(21.9 GiB)** | | |

## 3. 实验数据的小文件余量(在 /blue)—— 不建议删,有一项例外

大文件搬走后,各运行目录里剩下的是几百万个小文件:逐 episode 的记录、teacher 采集结果、评测输出。它们是基线和方法数字的原始证据,字节不多,系里的配额也不按文件数卡,所以**建议原地保留**。

| 目录 | 盘点时文件数 | 盘点时字节 | 已搬走的大文件 | 现在大约还在 /blue |
|---|---|---|---|---|
| `DEPT/ICML27_agent_distillation` | 12,278,723 | 1,382.3 GiB | 298 个 / 1,242.3 GiB | 约 12,278,425 个文件 / 140.0 GiB |
| `DEPT/ICML27_v0_runs` | 415,278 | 45.9 GiB | 87 个 / 36.1 GiB | 约 415,191 个文件 / 9.8 GiB |
| `DEPT/ICML27_runs` | 445,823 | 14.3 GiB | 2 个 / 5.1 GiB | 约 445,821 个文件 / 9.2 GiB |
| `DEPT/ICML27_p1_backup` | 239,242 | 5.0 GiB | 10 个 / 1.6 GiB | 约 239,232 个文件 / 3.4 GiB |
| `DEPT/tc-alignment-vllm` | 173,822 | 18.7 GiB | 0 个 / 0.0 GiB | 约 173,822 个文件 / 18.7 GiB |
| `YD/ICML27_archive` | 520,989 | 436.9 GiB | 38 个 / 429.1 GiB | 约 520,951 个文件 / 7.7 GiB |

其中 `DEPT/ICML27_agent_distillation` 的大头(盘点时):`reports/runs/eval` 195 万个文件、`reports/smartad-selection/chain_k500_s0_r2_e4b` 192 万、`reports/smartad-selection/chain_k500_s0_r2` 187 万、`data/verified/seeds_full_train_r2` 98 万(teacher 采集,花了钱,不可再生)、`reports/runs/smartad-chain` 89 万。

### 例外:12B / E2B student 的选择与采集记录 —— 建议删(属于 11-02 你已同意的类别,当时清单漏了)

这些目录的 `identity.json` 写明 student 是 gemma-4-12B-it 或 gemma-4-E2B-it。我查了现存 48 个基线 cell 的 `models.json`、`training/identity.json`、`training/provenance.json`、`training/tasks.json`:**没有任何一个引用它们**。(同名的 12B 时期目录 `reports/runs/smartad-chain/chain_k500_s0_r2` 和 `reports/runs/adftp-chain/chain_k500_s0_r2` **被 E4B 的 SmartAD / AD cell 引用,不能删**,不在本表。)我只查了这 48 个 cell 的这四类 json,没有查 `reports/runs/eval`、`method_v1` 等其他目录;真要删之前我会再全量 grep 一遍。

3.1 里还有 4 个 ≥64 MiB 的清单文件(0.8 GiB)我特意没搬,留在 /blue 等你决定。

| # | 路径(都在 /blue) | 文件数 | 字节 | 是什么 | 删了会失去什么 |
|---|---|---|---|---|---|
| 3.1 | `DEPT/ICML27_agent_distillation/reports/smartad-selection/chain_k500_s0_r2_e2b/` | 1,872,562 | 20,964,006,202 B(19.5 GiB);其中 1 个大文件(0.1 GiB)已在 /orange | E2B student 的 SmartAD K500 打分与选择 | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.2 | `DEPT/ICML27_agent_distillation/reports/smartad-selection/chain_k500_s0_r2/` | 1,871,962 | 20,954,572,472 B(19.5 GiB);其中 5 个大文件(0.9 GiB)已在 /orange | 12B student 的 SmartAD K500 打分与选择 | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.3 | `DEPT/ICML27_agent_distillation/reports/smartad-selection/k100_s0/` | 182,502 | 2,114,739,398 B(2.0 GiB);其中 1 个大文件(0.1 GiB)已在 /orange | 12B student 的 SmartAD K100 选择(seed 0) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.4 | `DEPT/ICML27_agent_distillation/reports/smartad-selection/k100_s1/` | 206,766 | 2,350,557,950 B(2.2 GiB);其中 1 个大文件(0.1 GiB)已在 /orange | 同上(seed 1) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.5 | `DEPT/ICML27_agent_distillation/reports/smartad-selection/k100_s2/` | 195,636 | 2,315,486,424 B(2.2 GiB);其中 1 个大文件(0.1 GiB)已在 /orange | 同上(seed 2) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.6 | `DEPT/ICML27_agent_distillation/reports/runs/smartad-collections/k100_s0/` | 158,470 | 1,669,477,446 B(1.6 GiB) | 12B 时期的 SmartAD K100 采集(seed 0) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.7 | `DEPT/ICML27_agent_distillation/reports/runs/smartad-collections/k100_s1/` | 181,206 | 1,887,598,650 B(1.8 GiB) | 同上(seed 1) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.8 | `DEPT/ICML27_agent_distillation/reports/runs/smartad-collections/k100_s2/` | 171,411 | 1,864,194,753 B(1.7 GiB) | 同上(seed 2) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.9 | `DEPT/ICML27_agent_distillation/reports/adftp-collections/k100_s0/` | 75,008 | 860,026,174 B(0.8 GiB) | 12B 时期的 AD-FTP K100 采集(seed 0) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.10 | `DEPT/ICML27_agent_distillation/reports/adftp-collections/k100_s1/` | 88,443 | 1,022,661,097 B(1.0 GiB) | 同上(seed 1) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| 3.11 | `DEPT/ICML27_agent_distillation/reports/adftp-collections/k100_s2/` | 90,683 | 1,058,158,954 B(1.0 GiB) | 同上(seed 2) | 12B / E2B student 的中间记录(含 teacher 调用结果);这两个 student 已不在任何表里 |
| | **合计** | **5,094,649** | **57,061,479,520 B(53.1 GiB)** | | |

## 4. 已搬到 /orange 的有效权重 —— 不是删除候选,只有 4.4 请你定

| # | 位置(/orange) | 大文件 | 字节 | 是什么 | 建议 |
|---|---|---|---|---|---|
| 4.1 | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_agent_distillation/`(`reports/runs/baselines`、`reports/runs/eval`、`method_v1` 等) | 192 个 | 256,366,230,583 B(238.8 GiB) | 主实验网格的基线权重、评测用的合并模型、V2 方法权重 | 留。重跑或复评前按清单搬回 |
| 4.2 | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_v0_runs/`(不含 `_failed`) | 73 个 | 33,102,903,267 B(30.8 GiB) | v0 K10/K20/K30 的 adapter、数据集、标注 | 留。K30 ×3 / G5 重跑要用 |
| 4.3 | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_p1_backup/` | 10 个 | 1,718,316,320 B(1.6 GiB) | P1 评测运行树的备份 | 留 |
| 4.4 | `/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-yd24f/ICML27_archive/baselines_evaluated_1007/` | 38 个 | 460,787,521,352 B(429.1 GiB) | 10-07 评测过的 LEAP 早期阶梯(privileged K1/K2/K5/K10/K20-m5、zero K20-m5)的各阶段权重 | **候选,请你定。** 它们是 10-19 修超时之前的旧协议运行,11-07 的总结里没有用到 LEAP;评测结果的小文件还在 /blue(`YD/ICML27_archive`,约 52 万个文件 / 7.7 GiB)。我没有逐个核对是否有别处引用这些权重 |

## 恢复方法(任何已搬走的文件)

```bash
# 例:把 ICML27_v0_runs 的大文件搬回 /blue 原路径(每个批次一份清单)
M=/orange/yd24f.fsu/xc25.fsu/icml27-cold/blue-dept/ICML27_v0_runs
cat $M/_tc_moved_big_files.*.tsv | grep -v '^#' | cut -f3 > /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/restore.list
rsync -a --files-from=/orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/restore.list $M/ /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_v0_runs/
```

清单第一列是 sha256,搬回后可以用 `sha256sum -c` 核对。各目录的 /orange 路径和批次记在 `ICML27_agent_distillation/PROJECT_STATE.md`(11-11/11-12 条目)。
