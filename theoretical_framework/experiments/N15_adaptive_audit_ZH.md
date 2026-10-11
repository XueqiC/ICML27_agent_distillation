# N15 · 实际状态上的自适应 audit；保真与改进 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N15_adaptive_audit_ZH.md)

**结论.** N15 成立。在 5 个配置上（每个配置 100,000 次独立训练，C16），自适应 audit 的实测增益始终高于两条保证：$`-\delta`$（式 [N15(c)](../proofs/N15_adaptive_audit_ZH.md)）和 $`\mathbb E\sum_tB_t-2\delta`$（式 [N15(c)](../proofs/N15_adaptive_audit_ZH.md)）。被选中的漂移超过 $`\sum_tB_t`$ 的频率为 0.990–0.999，高于每个名义值 $`1-\delta`$（结论 a）。

带号恒等式 [N15(b)](../proofs/N15_adaptive_audit_ZH.md) 与保真界 [N15(a)](../proofs/N15_adaptive_audit_ZH.md) 在 50 个有理数实例上精确成立（C21，残差 0）。理论包的精确期望（`adaptive_gate_expectations`）结论一致。

两个见证说明定理的形式是必要的：

- 不用下置信界的选择，精确期望增益为 $`-78/15625<0`$；
- 以"所有 audit 都通过"为条件，会把一个零漂移过程的价值从 $`1/4`$ 变为 $`119/494`$。

每个候选需要 1,016 个配对副本的 Hoeffding 代价得到了确认（C16、T13）。E7 在受控网络中测量了配对方差，并*预测*在 empirical-Bernstein 半径下 559 个副本就够了。这个数字是预测，不是一个已完成的 certificate。

![自适应 audit 的增益与代价](../figures/N15_adaptive_audit.png)

[PDF](../figures/N15_adaptive_audit.pdf)

## C16 · 有限学习过程上的自适应 audit

**动机.** [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 给出三条结论：一条高概率的漂移陈述 (a)，以及两条无条件期望界 [N15(c)](../proofs/N15_adaptive_audit_ZH.md)。若有违反，就会表现为实测增益低于 $`-\delta`$ 或低于 $`\mathbb E\sum B-2\delta`$，或者漂移成立的频率低于 $`1-\delta`$。

**设置.**

- 一个小的有限学习过程：4 个状态，$`R\le6`$，每个状态 $`M\in\lbrace 2,3\rbrace`$ 个候选 kernel，真实优势已知。
- audit 从真实模型中抽取有界的配对对比：完整参照尾部加 Bernoulli 部署。
- 用下置信界 $`e_t=\sqrt{2\log(2M/\delta_t)/n_t}`$ 选择，之后做一次新鲜更新。
- 压力设置：$`n_t`$ 小，$`\delta_t\in[0.1,0.4]`$ 大，候选的均值略为负、方差高。
- 种子 `SeedSequence([20261010, 16])`。

**做法.** 每个配置运行 100,000 次独立的完整训练（audit 加更新），然后：

- 估计增益 $`L_E(\mathrm{adaptive})-L_E(0)`$ 及其标准误；
- 以 4 个 SE 的余量检验 [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 和 [N15(c)](../proofs/N15_adaptive_audit_ZH.md)；
- 记录 $`\sum_tA_t\ge\sum_tB_t`$ 成立的频率。

**结果**（`theory/independent_checks/results_independent_v014.json`，`claims[id=C16].details.configurations`）：

| $`R`$ | $`M`$ | $`n`$ | $`\delta`$ | 半径 | 增益（SE） | $`\mathbb E\sum B`$ | 结论 (a) 频率 | ≥ $`1-\delta`$ |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 2 | 8 | 0.4 | 0.759 | 0.00084 (0.00091) | 0.00090 | 0.99412 | ✓ |
| 2 | 2 | 12 | 0.4 | 0.707 | −0.00067 (0.00102) | 0.00088 | 0.99002 | ✓ |
| 3 | 3 | 20 | 0.2 | 0.671 | 0.00003 (0.00106) | 0.00054 | 0.99407 | ✓ |
| 6 | 3 | 64 | 0.1 | 0.429 | 0.00115 (0.00108) | 0.00026 | 0.99628 | ✓ |
| 2 | 3 | 1000 | 0.2 | 0.090 | **0.13638** (0.00097) | 0.05153 | 0.99897 | ✓ |

status 为 PASS，共 500,450 个实例、468 条断言。在 327 个不同实例上做对抗搜索，最佳余量（界 − 增益）为 $`-0.199`$，即界始终低于增益（`adversarial_best_bound_minus_gain`）。

**必要性见证**（`claims[id=C16].details`）：

| 见证 | 值 | 含义 |
|---|---|---|
| `point_estimate_exact_gain` | $`-78/15625`$ | 按 $`\widehat A`$ 选择而不减去 $`e_t`$，期望上可能亏损，所以需要下置信界 |
| `zero_drift_conditioning_witness` | 条件价值 $`119/494`$，无条件价值 $`1/4`$（状态 1 上 audit 失败概率 $`9/128`$） | 以 audit 成功为条件会使终端价值产生偏差，所以 [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 必须是无条件的 |
| `conditioning_probe_found_45_violation` | false（搜索了 5 个配置） | 条件版本有偏差，但无条件界没有被违反 |
| `replicas_required_per_candidate` | 1016 | $`M=4,\delta=0.05,e=0.1`$ 下的 Hoeffding 分配 |

**说明了什么.** 这些保证恰好以陈述的强度成立，不多也不少：

- 副本少、$`\delta`$ 大时，audit 几乎从不选择候选，增益落在 $`-\delta`$ 的底线之内；
- 副本足够多时（$`n=1000`$），它会选中有益的候选，增益 $`0.136`$，远高于 $`\mathbb E\sum B-2\delta`$。

## C21 · 带号恒等式，以及保真与改进

**动机.** [N15(b)](../proofs/N15_adaptive_audit_ZH.md) 是依赖历史的 provider 选择下的精确恒等式，[N15(a)](../proofs/N15_adaptive_audit_ZH.md) 把保真和改进区分开。两者都做精确检验。

**设置与做法.**

- 50 个状态相关 kernel 选择的有理有限过程，外加对 $`\mathrm{LHS}-\mathrm{RHS}`$ 的对抗搜索。
- 三个保真例子：好的参照（$`\gamma>0`$）、无效的参照（$`\gamma<0`$）和中性的参照（$`\gamma=0`$），各配一个保真度 $`\varepsilon=1/100`$ 的候选。

**结果**（`claims[id=C21]`）：PASS，834 条断言，最坏残差 0，`adversarial_best_lhs_minus_rhs` = 0。

| 初始 | 参照 | 候选 | $`\gamma`$ | $`\varepsilon`$ | 候选增益 | [N15(a)](../proofs/N15_adaptive_audit_ZH.md)：增益 ≥ $`\gamma-\varepsilon`$ |
|---:|---:|---:|---:|---:|---:|---|
| 1/2 | 4/5 | 79/100 | 3/10 | 1/100 | 29/100 | ✓（取等） |
| 1/2 | 2/5 | 39/100 | −1/10 | 1/100 | −11/100 | ✓（取等） |
| 1/2 | 1/2 | 49/100 | 0 | 1/100 | −1/100 | ✓（取等） |

**说明了什么.** 带号望远镜恒等式是精确的。对一个无效参照的高保真复制，并不能改进初始学生：其增益为 $`-0.11`$。保真只继承参照的实测增益。

## 包内检验 · 自适应把关的精确期望

**动机.** 这是一项独立的精确计算（不含 Monte Carlo），检验 [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 以及 $`-\delta`$ 项的必要性。

**结果**（`theory/v014_repro/results_v014_rerun.json`，`adaptive_gate_expectations`；每个候选 $`n=400`$，$`\delta=0.05`$，半径 $`0.136`$）：

| 量 | 值 |
|---|---:|
| 无条件期望增益 | 0.200 |
| 被选中 LCB 的 $`\mathbb E\sum`$ | 0.132 |
| [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 的下界 $`-\delta`$ | −0.050 |
| [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 的下界 $`\mathbb E\sum B-2\delta`$ | 0.032 |
| 失败概率 | $`1.5\times10^{-8}`$ |
| 不存在精确的不退化：误选概率 / 期望增益（$`n=20`$，$`\delta=0.9`$，候选 0.49 对参照 0.50） | 0.00458 / $`-4.6\times10^{-5}`$ |

**说明了什么.** 在精确期望下两条界都成立。一个略差的候选以很小的正概率通过乐观的把关，所以期望变化可以严格为负。这就是 [N15(c)](../proofs/N15_adaptive_audit_ZH.md) 带有 $`-\delta`$ 而不是 $`0`$ 的原因。

## T13 与 E7 · audit 的代价

**动机.** 定理给出的分配有效但昂贵，用到真实模型之前必须清楚它的代价。E7 在受控网络中测量配对对比的方差，看对方差敏感的半径能省多少。

**设置（E7）.**

- [C23](N11_adam_cold_start_ZH.md)（v0.14 受控网络实验，其中 Adam 偏离仿射对照） 网络，一轮预先声明的 audit，四个候选 provider 对一个共享参照。
- 一个参照副本在各候选之间复用，保留候选之间的协方差。
- 2,400 个独立的配对 pilot 副本，采用正确的逆 CDF 耦合。
- 方差被精确分解为块抽样、suffix 训练和部署三部分。
- 种子 `SeedSequence([150150, 107])`。

**结果**（`docs/direction/v015_revision/results_v015.json` 的 `checks.T13_*`；`theory/v015_checks/results_v015_independent.json` 的 `groups[id=E7].data`）：

| 量 | 值 |
|---|---|
| T13：充分的 Hoeffding 配对数（$`M=4,\delta=.05,e=.1`$） | 1016；在 1016 处半径为 0.09995；`is_necessary_lower_bound` = false |
| E7 的 Hoeffding 分配 | 每个候选 1016；共享参照时 5,080 个训练分支（不复用时 8,128） |
| E7 的 empirical-Bernstein **预测** | 每个候选 559，在 $`n=559`$ 处半径为 0.0999，所用的最大 pilot 方差为 0.129 |

E7 中配对对比的方差分解：

| 候选 | 均值 | 块 | suffix | 部署 | 合计 | 部署占比 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.084 | 0.0133 | 0.0041 | 0.0908 | 0.1082 | 84% |
| 1 | −0.042 | 0.0084 | 0.0020 | 0.0454 | 0.0559 | 81% |
| 2 | −0.143 | 0.0096 | 0.0060 | 0.1202 | 0.1358 | 89% |
| 3 | 0.057 | 0.0096 | 0.0027 | 0.0410 | 0.0533 | 77% |

**说明了什么.**

- Hoeffding 分配是充分的，与定理所述一致。
- 在受控网络中，配对方差大部分来自部署噪声，而不是训练更新。因此预测一个对方差敏感的固定 $`n`$ 界能少用约 45% 的副本。
- 这个预测明确标为预测。将来的 audit 必须在固定 $`n`$ 下自行计算半径，不能做可选停止。
- E7 还根据实测的集群耗时，记录了 12B 模型的情景推算（`data.forecast`）。这些是规划用的输入，不是测量结果。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C16,C21 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E7 --output theory/v015_checks/quick/n15.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T13
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`claims[id=C16]` 与 `claims[id=C21]`。
- `theory/v014_repro/results_v014_rerun.json`：`adaptive_gate_expectations`。
- `theory/v015_checks/results_v015_independent.json`：`groups[id=E7]`。
- `docs/direction/v015_revision/results_v015.json`：`checks.T13_*`。
- 图：`../figures/N15_adaptive_audit.pdf` 与 `.png`（规格见 `../figures/specs/N15.md`）。
