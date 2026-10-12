# §4 充分性的复杂度 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/04_complexity_of_sufficiency_ZH.md) · [English](04_complexity_of_sufficiency.md)

**结论。** 本节每个陈述都通过了检验。除 Check 4.13 外，所有检验都是精确有理数计算或精确 LP；Check 4.13 是一个受控小网络。

- **[Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)（充分性对偶与迁移）：PASS。**
  - 在 300 个随机实例上，span 判据与独立的 LP 搜索从未出现分歧。
  - 最坏分布对差异等于最优一致逼近误差的两倍，偏差不超过 $`2.2\times10^{-14}`$（Check 4.1）；在 21 个结构化子空间上不超过 $`1.6\times10^{-15}`$（Check 4.10）。
- **[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)（仿射学习器的矩阶）：PASS。**
  - 在 1,080 棵随机生成并经对抗变异的部署树上，次数界和系数界全部成立。
  - 对至多 5 次更新的仿射学习器，矩匹配的分布给出完全相同的学习价值。
  - 对 $`N=1,\dots,14`$，每一阶都是必要的；对 $`N=1,\dots,8`$，parity 的差距保持为 1。
  - 18 个配置中网格统计量个数全部等于 $`\binom{k+N}N-1`$，二元壁垒被取到（Check 4.2–4.5）。
- **[Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)（与轮数无关的界、平均化速率）：PASS。**
  - 没有对抗实例越过界，最接近的 gap/bound 为 $`55/57`$（Check 4.6）。
  - 在全部六个情形中，第一个不匹配的 cumulant 同时给出衰减的指数和系数（Check 4.7）。
  - replay 和确定性后代样本在 256 倍曝光内都让差距保持不变（Check 4.8）。
- **[Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)（精确阶不等于近似复杂度）：PASS。**
  - $`p^N`$ 的 Chebyshev 截断误差等于二项尾概率；要达到 $`10^{-3}`$ 的精度，$`N=4,8,16,32`$ 时只需 $`q=4,6,9,13`$。
  - parity 在次数低于 $`N`$ 时误差恰为 $`1/2`$（Check 4.9）。
  - 最高次数同为 3 时，结构化的族需要 3 个统计量，parity 需要 7 个（Check 4.10）。
- **[Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)（更新复合）：PASS。**
  - 12 条随机更新链的块次数全部等于复合上界，平方构造取到了它。
  - SGD 展开的余项从不超过其界的 $`0.275`$ 倍（Check 4.11）。
  - 几何累积界可以取等号；没有稳定性条件时，关于 $`R`$ 线性的理解不成立（Check 4.12）。
  - 在受控网络中，切线仿射 SGD 预测在 $`\eta R`$ 小的区域内落在预先设定的阈值内，在大步长下失效（Check 4.13）。

| Check | 检验对象 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.1](#c4-1) | [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(a)–(d) | `v014:C5` | PASS |
| [Check 4.2](#c4-2) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(a) | `v014:C6` | PASS |
| [Check 4.3](#c4-3) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(b) | `v014:C7` | PASS |
| [Check 4.4](#c4-4) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(c) | `v014:C8`、`pkg:constant_gap_parity` | PASS |
| [Check 4.5](#c4-5) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(d) | `v014:C9`、`pkg:grid_response_dimensions` | PASS |
| [Check 4.6](#c4-6) | [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(a) | `v014:C10` | PASS |
| [Check 4.7](#c4-7) | [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(b)–(d) | `v014:C11`、`pkg:scalar_exact_decay` | PASS |
| [Check 4.8](#c4-8) | [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(e) | `v014:C18` | PASS |
| [Check 4.9](#c4-9) | [Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)(a),(b) | `v014:C12`、`pkg:scalar_approximation` | PASS |
| [Check 4.10](#c4-10) | [Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)；[Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(a),(b)；[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(d) | `v014:C19` | PASS |
| [Check 4.11](#c4-11) | [Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(a)–(d) | `v014:C13` | PASS |
| [Check 4.12](#c4-12) | [Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(e) | `v015:V7` | PASS |
| [Check 4.13](#c4-13) | [Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(d),(e)，网络中的适用范围 | `v014:C23`（SGD 部分） | 适用范围图 |

代码 ID：`v014:` 指 `theory/independent_checks`（结果文件 `results_independent_v014.json`）；`v015:` 指 `theory/v015_checks`（结果文件 `results_v015_independent.json`）；`pkg:` 指修订包自带的验证脚本 `docs/direction/v014_handoff/verify_environment_information_v014.py`，重新运行的结果在 `theory/v014_repro/results_v014_rerun.json`。

<a id="c4-1"></a>

## Check 4.1 · 随机实例上的 span 判据、对偶与迁移（Theorem 4.2）

**结论。** 在 300 个随机有限实例、6,014 条断言上 PASS。span 判据是精确的；对偶在 $`2.2\times10^{-14}`$ 内是等式；针对迁移界的对抗搜索最多只到 $`1.1\times10^{-16}`$；(d) 对 $`m=2,\dots,8`$ 精确成立。

**为什么做。** [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2) 把"反馈律之间哪些差别有影响"变成线性代数问题。每个部分都有对应的证伪方式，本检验逐一独立搜索：

- (a)：一对特征匹配的分布分离了一个落在 span 内的 response，或者 span 判据漏掉了一个落在 span 外的 response；
- (b)：某个可行分布对的差异超过两倍一致距离 $`2\mathrm{dist}_\infty`$，或者没有任何分布对能取到它——(b) 是等式，不只是上界；
- (c)：某个分布对违反迁移不等式；
- (d)：事件族的上确界不等于 $`\gamma\,\mathrm{TV}`$，或者 response 维数不等于 $`m-1`$。

**设置。**

- 300 个实例：字母表大小 $`m\sim\mathrm U\lbrace 4,\dots,12\rbrace`$，特征数 $`d\sim\mathrm U\lbrace 1,\dots,m-2\rbrace`$。
- 特征表 $`\phi\in\mathbb Q^{d\times m}`$ 和三个 response $`\mathcal G\subset\mathbb Q^m`$ 都取随机有理数。
- 每个实例另检验 $`g=\phi_1`$，作为预期差异为 0 的 span 内对照。
- 迁移实例中 $`b,r`$ 随机，并重新缩放，保证 $`g=\tfrac12+b^\top\phi+r\in[0,1]^m`$。
- (d) 在 $`m=2,\dots,8`$ 上取随机有理分布 $`P,Q`$，并取 $`a=\tfrac15`$、$`\gamma=\tfrac35`$。
- 种子：`SeedSequence([20261010, 5])`。

**做法。**

1. *精确秩。* 用 SymPy 计算 $`[\mathbf 1;\phi]`$ 与 $`[\mathbf 1;\phi;g]`$ 的秩，判断 $`g\in H_\phi`$；同时算 $`d_0=\mathrm{rank}[\mathbf 1;\mathcal G]-1`$。
2. *原问题 LP。* 用 HiGHS 直接以 $`(P,Q)`$ 为变量，在特征匹配的概率向量对上求 $`\max(P-Q)g`$。
3. *逼近 LP。* 用另一个 LP 计算 $`\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$。
4. *(a)(b) 的断言。* 断言 width $`=2\cdot`$ error，并断言 width $`\le10^{-8}`$ 当且仅当 $`g\in H_\phi`$。
5. *迁移 LP。* 在所有分布对上最大化 $`(P-Q)g-B\lVert\Delta\phi\rVert_\infty-2\tau`$，最优值必须 $`\le0`$。
6. *(d)。* 在精确有理数下枚举全部 $`2^m`$ 个事件，比较 $`\max_A\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert`$ 与 $`\gamma\cdot\tfrac12\lVert P-Q\rVert_1`$，并检验 $`[\mathbf 1;\lbrace\mathbf 1_A\rbrace]`$ 的秩等于 $`m`$。

**结果。**

| 量 | 值 | 来源（JSON 键） |
|---|---:|---|
| 实例数 / 断言数 | 300 / 6,014 | 条目 `v014:C5`，`n_instances`, `.n_assertions` |
| 状态 | PASS | 条目 `v014:C5`，`status` |
| $`\max\lvert\text{primal width}-2\cdot\text{approximation error}\rvert`$ | $`2.23\times10^{-14}`$ | 条目 `v014:C5`，`details.max_duality_residual` |
| 迁移对抗得分的最大值（必须 $`\le0`$） | $`1.11\times10^{-16}`$ | 条目 `v014:C5`，`details.adversarial_transfer_best_lhs_minus_rhs` |
| 全部断言中的最大残差 | $`5.76\times10^{-14}`$ | 条目 `v014:C5`，`worst_residual` |
| (d)：事件上确界 $`=\gamma\,\mathrm{TV}`$，维数 $`=m-1`$ | $`m=2,\dots,8`$ 全部精确 | 断言 `all-event total variation`、`event response dimension` |

以上数值均来自 `theory/independent_checks/results_independent_v014.json`。

![Figure 4.1：最坏的匹配分布对差异等于两倍逼近误差](../figures/fig4_1_duality.png)

*Figure 4.1.* 一个精确的五消息例子：图中给出一个 response、它在特征 span 中的最优一致逼近，以及取到两倍逼近误差的特征匹配分布对。第二个面板把原问题 LP 的最优值对两倍逼近误差作图，包括 Check 4.10 的全部结构化子空间和本检验的随机实例；所有点都落在对角线上。[PDF](../figures/fig4_1_duality.pdf)

**为什么支持 Theorem 4.2。**

- span 判据与 LP 分离搜索从未出现分歧，所以 (a) 的判据就是精确的检验。
- 两个 LP 是独立求解的：一个在分布对上，一个在逼近函数上。它们的最优值吻合到求解器精度，所以 (b) 是等式。
- 即使对手专门针对 (c) 做优化，也无法让左边超过右边。
- 证据的范围：字母表大小至多 12；定理本身也是对有限字母表陈述的。

<a id="c4-2"></a>

## Check 4.2 · 部署价值是有界多项式（Theorem 4.4(a)）

**结论。** 在 1,080 棵部署树（5,400 条断言）上 PASS。对抗搜索找到的"系数范数 / $`3^N`$"最大为 $`2/3`$。

**为什么做。** 仿射理论建立在一件事上：$`J_E`$ 是 policy 坐标的多项式，次数至多为 $`N`$，且 $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$（[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(a)）。只要有一棵树的精确展开次数更高或范数更大，它就被证伪。

**设置。**

- 90 轮搜索，每轮 $`k\in\lbrace 1,2,3\rbrace`$ 个 context，每条路径 $`N\in\lbrace 1,\dots,5\rbrace`$ 个决策。
- context 可以在路径上重复，机会节点使用有理概率。
- 每轮评估 12 棵候选树：每第四棵是全新的随机树，其余是对当前最坏比值树做叶子、context 或机会概率的随机变异（共 1,080 棵）。
- 种子：`SeedSequence([20261010, 6])`。

**做法。**

1. 用 SymPy 对每棵树做符号展开，得到精确的有理多项式。
2. 断言总次数 $`\le N`$ 且 $`\sum\lvert c_\alpha\rvert\le3^N`$。
3. 在 3 个随机有理点上断言 $`J\in[0,1]`$。
4. 对 $`\sum\lvert c_\alpha\rvert/3^N`$ 做爬山搜索，寻找最坏情形。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 树数 / 断言数 | 1,080 / 5,400 | 条目 `v014:C6`，`n_instances`, `.n_assertions` |
| 最大 $`\sum\lvert c_\alpha\rvert/3^N`$ | $`2/3`$ | 条目 `v014:C6`，`details.exact_best_ratio` |
| 见证 | $`N=1`$，$`J=1-p_1`$ | 条目 `v014:C6`，`details.witness` |
| 状态 | PASS（最大残差 0） | 条目 `v014:C6`，`status` |

**为什么支持 Theorem 4.4(a)。**

- 次数界和系数界在每棵树上都成立，包括对抗变异出来的树。
- 在这些树上，$`3^N`$ 界成立，但偏保守。

<a id="c4-3"></a>

## Check 4.3 · 矩匹配既充分又必要（Theorem 4.4(b)）

**结论。** PASS（46 个实例、385 条断言，最大残差恰为 0）。在 24 个随机配置（更新次数最多 $`R=5`$）中，矩匹配的分布给出完全相同的学习价值。$`N`$ 阶以内的每个矩都能被相应的单项式任务察觉。

**为什么做。** [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(b) 有两个方向，各有对应的证伪方式：

- 存在一对矩匹配的分布，在某个任务或仿射学习器下学习价值不同（否定充分性）；
- 存在某个矩，它的失配没有任何任务能察觉（否定必要性）。

**设置与做法。**

- *充分性*，24 个配置（$`k\in\lbrace 1,2\rbrace`$，$`N\in\lbrace 1,2,3\rbrace`$，$`R\in\lbrace 1,\dots,5\rbrace`$）：
  1. 在网格 $`\lbrace 0,\tfrac1{N+1},\dots,1\rbrace ^k`$ 上取随机的正分布 $`P`$；
  2. 构造 $`Q=P+av`$，其中 $`v`$ 取在所有次数 $`\le N`$ 的矩约束（含总质量）的精确零空间中；
  3. 随机抽取凸权重、$`p_0`$ 和任务树；
  4. 用精确的块矩卷积计算两个学习价值，断言所有被匹配的矩以及两个价值都相等。
- *必要性*，对 $`(k,N)\in\lbrace (1,4),(2,3),(3,2)\rbrace`$ 以及每个满足 $`\lvert\alpha\rvert\le N`$ 的非零 $`\alpha`$：
  1. 符号验证"访问 $`j`$ 恰好 $`\alpha_j`$ 次、结果全为 1 才回报"的任务等于 $`p^\alpha`$；
  2. 构造只在第 $`\alpha`$ 个矩上不同的分布对；
  3. 断言一步学习器 $`p=Z`$ 能把这对分布区分开。
- *两 context 例子*：边缘相同，交叉矩分别为 $`\tfrac12`$ 和 $`0`$。
- 种子：`SeedSequence([20261010, 7])`。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 46 / 385 | 条目 `v014:C7`，`n_instances`, `.n_assertions` |
| 仿射充分性配置 | 24 个，$`R`$ 最大为 5 | 条目 `v014:C7`，`details.affine_configurations` |
| 最大残差 | 0（精确） | 条目 `v014:C7`，`worst_residual` |
| 状态 | PASS | 条目 `v014:C7`，`status` |

**为什么支持 Theorem 4.4(b)。**

- 匹配全部 $`N`$ 阶以内的联合矩后，学习价值精确相等：差为零，而不只是很小。
- 反过来，$`N`$ 阶以内的每个矩都能被某个任务察觉，所以任何一个矩都不能省。
- 只匹配边缘分布是不够的（两 context 例子）。

<a id="c4-4"></a>

## Check 4.4 · 每一阶都必要；常数差距见证（Theorem 4.4(c)）

**结论。** PASS（22 个实例、8,983 条断言，最大残差 0）。标量见证对 $`N=1,\dots,14`$ 精确满足 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$，且所有低阶矩相等。在 $`N=1,\dots,8`$ 上，parity 分布对所有次数低于 $`N`$ 的联合多重指标都匹配，差距保持为 1；独立实现给出相同结果。

**为什么做。** [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(c) 说明层级是严格的：

- 对每个 $`N`$，都存在所有低阶矩一致、但 $`N`$ 阶学习价值不同的分布；
- 在 parity 构造中，差距不随 $`N`$ 缩小。

**设置与做法。**

- *标量见证。* 对 $`N=1,\dots,14`$，用精确有理数构造 $`P_N,Q_N`$。断言总质量为 1、所有 $`\ell<N`$ 阶的矩相等，且 $`m_{P_N}(N)-m_{Q_N}(N)=(-1)^NN!/(2^{N-1}N^N)`$。
- *Parity 见证。* 对 $`N=1,\dots,8`$ 枚举 $`\lbrace 0,1\rbrace ^N`$：
  1. 对总次数 $`<N`$ 的所有多重指标（不只是坐标子集）断言联合矩相等；
  2. 断言 $`J_{\rm even}`$ 在两个分布下分别取 1 和 0；
  3. 在 4 个随机有理点 $`p`$ 上，断言闭式表达式等于枚举得到的偶 parity 概率。
- 种子：`SeedSequence([20261010, 8])`。
- *独立实现*（`pkg:constant_gap_parity`，理论作者编写）：对 $`N=1,\dots,8`$，检验全部 $`2^N-1`$ 个真坐标子集的子集矩是否相等——二元坐标是幂等的，所以这已覆盖所有次数 $`<N`$ 的原始矩——然后精确计算两个价值。

**结果。**

| $`N`$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $`\Delta_N`$（精确） | $`-1`$ | $`1/4`$ | $`-1/18`$ | $`3/256`$ | $`-3/1250`$ | $`5/10368`$ | $`315/16777216`$ | $`567/800000000`$ | $`\approx9.58\times10^{-10}`$ |

来源：条目 `v014:C8`，`details.scalar`；全部 14 个值都等于闭式表达式。

| 量 | 值 | 来源 |
|---|---:|---|
| 所有低阶多重指标都匹配的 parity 阶数 | $`N=1,\dots,8`$ | 条目 `v014:C8`，`details.parity_all_multiindices`（true） |
| 断言数 | 8,983 | 条目 `v014:C8`，`n_assertions` |
| 状态 | PASS（最大残差 0） | 条目 `v014:C8`，`status` |
| 独立实现的 parity 差距，$`N=1,\dots,8`$ | 恰为 1（参考值 1.0，备选值 0.0） | `results_v014_rerun.json → constant_gap_parity` |
| 匹配的真子集检验数 | $`1,3,7,15,31,63,127,255`$ | 同上 |

重新运行的 `theory/v014_repro/results_v014_rerun.json` 在这一条目上与包内自带的 `docs/direction/v014_handoff/results_v014.json` 完全一致。

**为什么支持 Theorem 4.4(c)。**

- 对某个标量任务而言，每一阶 $`N`$ 都是必要的，差距公式精确成立。
- 标量差距随 $`N`$ 指数缩小，parity 差距则在每一阶都保持为 1。所以"精确必要性"和"固定精度下的困难性"是两回事；[Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6) 与 Check 4.9–4.10 把二者分开。

<a id="c4-5"></a>

## Check 4.5 · 精确统计量个数与二元壁垒（Theorem 4.4(d)）

**结论。** PASS（22 个实例、903 条断言，最大 LP 残差 $`2.3\times10^{-12}`$）。18 个配置的网格秩全部等于 $`\binom{k+N}N`$；237 个非常数单项式中删去任意一个，都会留下可分离的分布对；32 个特征数少于 $`2^k-1`$ 的随机二元接口，最坏分布对差异全部恰为 1。

**为什么做。** [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(d) 把 [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(a) 的 span 判据落成具体数字：

- 在网格上恰好需要、也只需要 $`\binom{k+N}N-1`$ 个统计量；
- 在二元消息、任意回报的设定下，少于 $`2^k-1`$ 个统计量就会留下最坏情形误差 1。

**设置与做法。**

1. *网格秩。* 对所有 $`k\le3,N\le5`$ 以及 $`k=4,N\le3`$（共 18 个配置），计算单项式求值矩阵在网格上的精确整数秩，断言其等于 $`\binom{k+N}N`$。
2. *删去特征。* 对每个配置中的每个非常数单项式（共 237 个），删去它后求解最坏分布对 LP，断言存在特征匹配的分布对能区分被删的单项式。
3. *二元壁垒。* 对 $`k=1,\dots,4`$，各抽 8 个特征数 $`d<2^k-1`$ 的随机有理特征表。取 $`[\mathbf 1;\phi]`$ 的一个零空间向量，以其正支撑为事件，断言 LP 差异等于 1。$`k\le3`$ 时还枚举全部事件，断言最优值为 1。
4. *完整接口。* 用 $`2^k-1`$ 个一般位置的特征，对每个示性 response 和一个随机 response 断言差异为 0。
5. *独立实现*（`pkg:grid_response_dimensions`）：在六个配置下计算 $`\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ 上单项式矩阵的数值秩（容差 $`10^{-10}`$）。

种子：`SeedSequence([20261010, 9])`。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 网格秩 $`=\binom{k+N}N`$ | 18/18，例如 $`(k,N)=(3,5)`$：56；$`(4,3)`$：35 | 条目 `v014:C9`，`details.grid_ranks` |
| 删去后均可分离的非常数特征 | 237 | 条目 `v014:C9`，`details.dropped_features` |
| $`d<2^k-1`$ 的二元接口；最坏分布对差异 | 32 个；全部 $`=1`$（最大偏差 $`4.4\times10^{-16}`$） | 条目 `v014:C9`，`details.binary_barriers` |
| 完整 $`2^k-1`$ 接口的差异 | 0 | 断言 `full generic interface discrepancy` |
| 断言数 / 最大 LP 残差 | 903 / $`2.3\times10^{-12}`$ | 条目 `v014:C9`，`n_assertions`, `.worst_residual` |

| $`(k,N)`$ | $`(1,1)`$ | $`(1,4)`$ | $`(2,2)`$ | $`(2,3)`$ | $`(3,2)`$ | $`(3,3)`$ |
|---|---:|---:|---:|---:|---:|---:|
| 字母表大小 | 2 | 5 | 9 | 16 | 27 | 64 |
| 秩 $`=\binom{k+N}N`$ | 2 | 5 | 6 | 10 | 10 | 20 |
| 最少非常数统计量 | 1 | 4 | 5 | 9 | 9 | 19 |

第二个表来自独立实现：`results_v014_rerun.json → grid_response_dimensions`。

![Figure 4.2：每一阶矩都必要；精确统计量个数是二项式系数](../figures/fig4_2_moment_order.png)

*Figure 4.2.* 标量见证在每一阶都能区分，差距按指数变小并与闭式一致；parity 分布的差距保持为 1（Check 4.4）。网格上单项式求值矩阵的秩在全部 18 个配置中都等于 $`\binom{k+N}N`$；在二元消息、任意回报的设定下，特征数少于 $`2^k-1`$ 的接口都留下最坏情形误差 1（本检验）。[PDF](../figures/fig4_2_moment_order.pdf)

**为什么支持 Theorem 4.4(d)。**

- 二项式计数是精确的：每个非常数矩单独都是必要的，合起来又是充分的。
- 二元壁垒被一个显式的分离事件取到，而不只是一个上界。

<a id="c4-6"></a>

## Check 4.6 · 不随更新次数增长的差异界（Theorem 4.5(a)）

**结论。** 在 288 个对抗搜索得到的实例（582 条断言，最大残差 0）上 PASS。界 $`C_J\delta_N`$ 从未被越过，最紧的实例 gap/bound $`=55/57\approx0.965`$。在一个固定问题上，$`R`$ 从 1 增加到 50，差距和界都保持不变。

**为什么做。** [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(a) 对每一个凸仿射学习器给出学习价值差距的上界 $`C_J\delta_N\le N3^N\delta_N`$，并且关于更新次数 $`R`$ 一致。证伪方式有两种：找到超过这个界的实例，或找到一族最坏比值随 $`R`$ 增长的实例。

**设置。**

- 24 次搜索试验，$`k\in\lbrace 1,2\rbrace`$、$`N\in\lbrace 1,2,3\rbrace`$，支撑为 $`\lbrace 0,\tfrac12,1\rbrace ^k`$。
- $`R`$ 在 $`\lbrace 1,2,5,10,25,50\rbrace`$ 中循环。
- 随机分布 $`P,Q`$、随机凸权重 $`w_0,\dots,w_R`$、随机 $`p_0`$，以及随机任务树（其精确多项式给出 $`C_J`$）。
- 种子：`SeedSequence([20261010, 10])`。

**做法。**

1. 每次试验对 $`Q`$ 做 12 步有理爬山（每步在支撑点之间移动 $`\le\tfrac1{20}`$ 的质量），以最大化 gap/bound，共 288 个实例。
2. 用独立块矩卷积精确计算每个学习价值。
3. 断言 gap $`\le C_J\delta_N\le N3^N\delta_N`$。
4. *固定问题扫描。* 在 $`\lbrace 0,1\rbrace`$ 上取 $`P=(\tfrac14,\tfrac34)`$、$`Q=(\tfrac34,\tfrac14)`$，$`J=p^2`$，学习器为 $`R`$ 个块的均匀平均，$`R=1,\dots,50`$。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 288 / 582 | 条目 `v014:C10`，`n_instances`, `.n_assertions` |
| 找到的最大 gap/bound | $`55/57\approx0.9649`$，出现在 $`N=1`$、$`R=10`$ | 条目 `v014:C10`，`details.best_exact_ratio`, `.best_instance` |
| 状态 | PASS（最大残差 0） | 条目 `v014:C10`，`status` |

| 固定问题，$`R`$ | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| 精确差距 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| 界 $`C_J\delta_N`$ | 1 | 1 | 1 | 1 | 1 | 1 |

来源：条目 `v014:C10`，`details.fixed_problem_R_sweep`。

**为什么支持 Theorem 4.5(a)。**

- 对抗搜索从未越过这个界，而界与实际差距只差 4%，接近可达。
- 更新次数完全不出现在界里：扫描中差距和界都不随 $`R`$ 变化。

<a id="c4-7"></a>

## Check 4.7 · 第一个不匹配的 cumulant 决定速率（Theorem 4.5(b)–(d)）

**结论。** PASS（1,530 个实例、1,797 条断言）。精确的标量律在 $`N=2,\dots,8`$、$`n=1,\dots,30`$ 的全部 210 个格点上成立。在全部六个 cumulant 情形中，精确差距恰好从 $`n^{1-r}`$ 开始，系数与预测一致，没有更慢的项；缩并为零时首项消失；均值不同时，差距以 $`O(1/n)`$ 收敛到 $`J(\mu_P)-J(\mu_Q)=13/32`$。一个不使用 cumulant 的独立实现在 20 个情形中复现了精确律。

**为什么做。** [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(b) 同时预测了差距随独立块数 $`n`$ 衰减的指数 $`1-r`$ 和系数 $`\langle D^rJ(\mu),\Delta\kappa_r\rangle/r!`$。证伪方式包括：首项系数不对；出现 $`j<r-1`$ 的更慢项 $`n^{-j}`$；余项不是 $`O(n^{-r})`$；缩并为零时首项却不为零。(c) 是标量见证的精确律 $`\Delta_N/n^{N-1}`$；(d) 给出均值不同时的极限 $`J(\mu_P)-J(\mu_Q)`$。

**设置。**

- *(c) 的精确律。* Check 4.4 的标量见证分布，$`J=p^N`$，$`N=2,\dots,8`$，$`n=1,\dots,30`$。
- *cumulant 情形*，共六个，$`k\in\lbrace 1,2\rbrace`$、$`r\in\lbrace 2,3,4\rbrace`$：
  - $`k=1`$：$`r`$ 阶标量见证分布；
  - $`k=2`$：二元提升 $`(x,(x+b)/2)`$，$`b\in\lbrace 0,1\rbrace`$ 等概率；
  - 任务为 $`J=(\tfrac1k\sum_jp_j)^{r+1}`$。
- *缩并为零。* $`J=(p-\tfrac12)^{r+2}`$，$`r=2,3,4`$；它在 $`\mu=\tfrac12`$ 处的 $`r`$ 阶导数为零。
- *均值不同。* 在 $`\lbrace 0,1\rbrace`$ 上取 $`P=(\tfrac14,\tfrac34)`$、$`Q=(\tfrac34,\tfrac14)`$，$`J=p^3`$。

**做法。**

1. 用样本下标的多项式展开，把 $`\mathbb EJ(\bar p_n)`$ 精确地算成 $`1/n`$ 的多项式。
2. 由矩独立地计算 cumulant。
3. 断言低阶 cumulant 相等、$`n^{1-r}`$ 的系数等于预测的缩并、所有更慢项的系数为零。
4. 对 $`n=1,\dots,220`$，断言 $`n\cdot\lvert n^{r-1}\text{gap}-\text{coefficient}\rvert`$ 始终不超过高阶系数构成的包络。
5. *独立实现*（`pkg:scalar_exact_decay`）：完全不用 cumulant，把分布卷积 $`n`$ 次，直接计算 $`\mathbb E\bar p_n^N`$，$`N=2,\dots,6`$，$`n\in\lbrace 1,2,4,8\rbrace`$。

**结果。**

| 情形 $`(k,r)`$ | 预测系数 | 精确差距（$`1/n`$ 的多项式） |
|---|---:|---|
| (1, 2) | $`3/8`$ | $`\tfrac38\,n^{-1}`$ |
| (1, 3) | $`-1/9`$ | $`-\tfrac19\,n^{-2}`$ |
| (1, 4) | $`15/512`$ | $`\tfrac{15}{512}\,n^{-3}`$ |
| (2, 2) | $`27/128`$ | $`\tfrac{27}{128}\,n^{-1}`$ |
| (2, 3) | $`-3/64`$ | $`-\tfrac3{64}\,n^{-2}`$ |
| (2, 4) | $`1215/131072`$ | $`\tfrac{1215}{131072}\,n^{-3}`$ |
| 缩并为零，$`r=2`$ | 0 | $`\tfrac3{16}n^{-2}-\tfrac18n^{-3}`$ |
| 缩并为零，$`r=3`$ | 0 | $`-\tfrac5{108}n^{-3}+\tfrac5{162}n^{-4}`$ |
| 缩并为零，$`r=4`$ | 0 | $`\tfrac{45}{4096}n^{-4}-\tfrac{15}{2048}n^{-5}`$ |
| 均值不同 | 极限 $`13/32`$ | $`\tfrac{13}{32}+\tfrac9{32}n^{-1}-\tfrac3{16}n^{-2}`$ |

- 来源：条目 `v014:C11`，`details.cumulant_cases[*].predicted_coefficient`、`.exact_gap_polynomial_in_inverse_n`、`.gap`（缩并为零的情形），以及 条目 `v014:C11`，`details.unequal_means_gap`、`.unequal_means_limit`。
- (c) 的精确律在全部 $`7\times30=210`$ 个 $`(N,n)`$ 格点上成立；状态 PASS。
- 独立实现：全部 20 个枚举差距都精确等于 $`\Delta_N/n^{N-1}`$，例如 $`N=4`$ 时依次为 $`3/256,\ 3/2048,\ 3/16384,\ 3/131072`$（`results_v014_rerun.json → scalar_exact_decay`；每一行的 `enumerated_gap_exact` 与 `predicted_gap_exact` 都相同）。

**为什么支持 Theorem 4.5(b)–(d)。**

- 每个情形的差距都恰好从 $`n^{1-r}`$ 开始，系数与预测一致：指数和系数都由第一个不匹配的 cumulant 决定。
- 缩并为零时，衰减变得更快（$`n^{-r}`$），与展开的预测一致。
- 均值不同时，差距以 $`O(1/n)`$ 逼近极限 $`J(\mu_P)-J(\mu_Q)`$。

<a id="c4-8"></a>

## Check 4.8 · 独立证据、replay 与合成后代样本（Theorem 4.5(e)）

**结论。** PASS（45 个实例、140 条断言）。使用独立块时，差距精确地按 $`\text{gap}_1/n^{N-1}`$ 衰减（$`N=2,3,4`$ 的双对数斜率为 $`-1,-2,-3`$）。重放一个块，或用同一母块的确定性后代训练，差距在 256 倍曝光内保持不变。

**为什么做。** [Theorem 4.5](../proofs/04_complexity_of_sufficiency_ZH.md#s4-5)(b)–(c) 的速率是关于独立块的个数。(e) 预测，两种只增加曝光、不带来新证据的做法会让差距停留在单块时的值：把一个块重放 $`n`$ 次，以及用同一母块的 $`n`$ 个确定性后代训练。这把信息与曝光区分开。

**设置。**

- $`N=2,3,4`$ 的标量见证分布，以及 $`N=2,3`$ 的 parity 分布。
- 曝光量 $`n\in\lbrace 1,2,4,\dots,256\rbrace`$。
- 后代通道 $`T(Z)=(Z+\tfrac12)/2`$，一个固定的、保信息的映射。
- 种子：`SeedSequence([20261010, 18])`。

**做法。** 对每个族、$`N`$ 和 $`n`$，精确计算三种差距：

- 独立块：$`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)`$；
- replay：同一个块的 $`n`$ 份拷贝，$`\bar p_n=Z_1`$；
- 后代样本：$`n`$ 份 $`T(Z_1)`$，$`\bar p_n=T(Z_1)`$。

然后断言：独立块差距等于单块差距除以 $`n^{N-1}`$；replay 差距等于单块差距；后代差距等于单块差距除以 $`2^N`$（这个通道的收缩因子）。

**结果。**

| 族，$`N`$ | $`n=1`$ 时差距 | 独立块，$`n=256`$ | replay，$`n=256`$ | 后代，$`n=256`$ | 独立块的双对数斜率 |
|---|---:|---:|---:|---:|---:|
| 标量，2 | $`1/4`$ | $`1/1024`$ | $`1/4`$ | $`1/16`$ | $`-1`$ |
| 标量，3 | $`-1/18`$ | $`-1/1179648`$ | $`-1/18`$ | $`-1/144`$ | $`-2`$ |
| 标量，4 | $`3/256`$ | $`3/4294967296`$ | $`3/256`$ | $`3/4096`$ | $`-3`$ |
| parity，2 | $`1`$ | $`1/256`$ | $`1`$ | $`1/4`$ | $`-1`$ |
| parity，3 | $`1`$ | $`1/65536`$ | $`1`$ | $`1/8`$ | $`-2`$ |

来源：条目 `v014:C18`，`details.curves`；斜率由 $`n=1`$ 和 $`n=256`$ 两行计算。

![Figure 4.3：只有独立块能让差异衰减](../figures/fig4_3_rates.png)

*Figure 4.3.* 独立块让差距按 $`n^{1-N}`$ 缩小，而重放一个块或用它的后代训练，差距保持不变（本检验）。乘以 $`n^{r-1}`$ 之后，精确差距在全部六个情形中都收敛到第一个不匹配 cumulant 给出的系数；缩并为零时首项消失（Check 4.7）。与轮数无关的界在 $`R`$ 直到 50 时都成立，并且几乎取到（Check 4.6）。[PDF](../figures/fig4_3_rates.pdf)

**为什么支持 Theorem 4.5(e)。**

- 只有独立证据能让差异衰减，而且衰减指数恰好是预测的 $`1-N`$。
- replay 和后代样本把训练曝光放大了 256 倍，学习价值差距却不变；后代样本只通过通道把差距缩放一次，之后不再衰减。
- 适用范围：通道是所述的确定性映射。它是一个构造出来的对照，不是关于所有后处理方式的论断。

<a id="c4-9"></a>

## Check 4.9 · Chebyshev 恒等式、精确尾概率与 parity minimax（Proposition 4.6）

**结论。** PASS（136 个实例、484 条断言，最大残差 $`1.0\times10^{-17}`$）。对 $`N=1,\dots,16`$ 与每个 $`q<N`$，$`p^N`$ 的截断误差都等于二项尾概率，并且低于 $`2e^{-q^2/N}`$（尾概率与界之比最大为 0.430）。一个扩展到 $`N=32`$ 的独立实现发现，要达到 $`10^{-3}`$ 的精度，$`N=4,8,16,32`$ 时只需 $`q=4,6,9,13`$。parity 的最佳次数 $`<N`$ 逼近误差恰为 $`1/2`$。

**为什么做。** [Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)(a) 有三条可证伪的断言：Chebyshev 展开是恒等式；截断误差等于二项尾概率 $`\Pr\lbrace\lvert B-N\rvert>q\rbrace`$（$`B\sim\mathrm{Bin}(2N,\tfrac12)`$），而不只是以它为上界；这个尾概率至多为 $`2e^{-q^2/N}`$。(b) 断言 parity 的最佳低次逼近误差恰为 $`1/2`$，找到更小的值就能将其证伪。

**设置。**

- 标量任务：$`N=1,\dots,16`$，以及所有 $`q<N`$。
- parity：$`N=1,\dots,5`$，在全部 $`2^N`$ 个立方体顶点上；每个 $`N`$ 另取 10 个随机有理点检验 $`\lvert J_{\rm even}-\tfrac12\rvert\le\tfrac12`$。
- 种子：`SeedSequence([20261010, 12])`。
- 独立实现（`pkg:scalar_approximation`）：$`N\in\lbrace 4,8,16,32\rbrace`$，$`q=0,\dots,N`$，8,001 点网格。

**做法。**

1. 用 SymPy 符号验证 Chebyshev 恒等式：$`\mathrm{expand}(\text{RHS}-p^N)=0`$。
2. 对每个 $`(N,q)`$：
   - 在含 $`p=1`$ 的 3,001 点网格上计算 $`\max\lvert p^N-h_q\rvert`$，断言它等于系数尾和；
   - 用精确有理数断言系数尾和等于二项尾概率；
   - 断言尾概率 $`\le2e^{-q^2/N}`$。
3. parity：用所有次数 $`<N`$ 的非常数单项式求解 minimax LP $`\min_{h\in\mathcal P_{N-1}}\max_{\rm vertices}\lvert J_{\rm even}-h\rvert`$，断言最优值为 $`0.5`$。
4. 独立实现：在网格上把 $`\lvert x^N-h_q\rvert`$ 与系数尾和、与 $`\min(1,2e^{-q^2/N})`$ 比较。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 136 / 484 | 条目 `v014:C12`，`n_instances`, `.n_assertions` |
| 恒等式与尾概率检验的 $`N`$ 范围 | 1–16 | 条目 `v014:C12`，`details.N_range` |
| 最大的 尾概率 $`/\,2e^{-q^2/N}`$ | 0.430 | 条目 `v014:C12`，`details.exhaustive_tail_best_ratio` |
| parity 最佳次数 $`<N`$ 逼近误差 | 恰为 0.5（$`N=1,\dots,5`$） | 条目 `v014:C12`，`details.parity_minimax_error` |
| 最大残差 | $`1.0\times10^{-17}`$ | 条目 `v014:C12`，`worst_residual` |
| 状态 | PASS | 条目 `v014:C12`，`status` |

独立实现：全部 64 行都满足 网格误差 = 尾和（$`\max\lvert\text{grid}-\text{tail}\rvert=8.9\times10^{-16}`$），且尾和 $`\le`$ 界。部分结果如下：

| $`N`$ | $`q`$ | 截断误差 | $`2e^{-q^2/N}`$ | parity 最佳 $`q`$ 次误差 |
|---:|---:|---:|---:|---:|
| 16 | 4 | 0.1102 | 0.7358 | 0.5 |
| 16 | 8 | 0.00210 | 0.0366 | 0.5 |
| 16 | 12 | $`2.56\times10^{-6}`$ | $`2.47\times10^{-4}`$ | 0.5 |
| 32 | 8 | 0.0328 | 0.2707 | 0.5 |
| 32 | 16 | $`2.44\times10^{-5}`$ | $`6.71\times10^{-4}`$ | 0.5 |
| 32 | 20 | $`1.01\times10^{-7}`$ | $`7.45\times10^{-6}`$ | 0.5 |

来源：`results_v014_rerun.json → scalar_approximation`（字段 `Chebyshev_truncation_error`、`grid_check_error`、`Hoeffding_upper`、`parity_best_degree_q_error`）。截断误差 $`\le10^{-3}`$ 的最小 $`q`$：$`N=4,8,16,32`$ 时分别为 $`4,6,9,13`$。

**为什么支持 Proposition 4.6。**

- $`p^N`$ 的截断误差恰好是二项尾概率，按 $`e^{-q^2/N}`$ 衰减；在检验范围内，所给的界至少保守 2.3 倍。
- 达到实用精度所需的阶数按 $`\sqrt N`$ 增长：$`N`$ 翻倍，只多出几阶；而精确相等需要 $`N`$ 阶。
- parity 在任何更低的次数下都无法比平凡的常数逼近得更好，所以 $`\sqrt N`$ 的行为是任务本身的性质，不是次数的性质。

<a id="c4-10"></a>

## Check 4.10 · 最高次数相同，response 空间的复杂度不同（Proposition 4.6；Theorem 4.2 与 4.4(d)）

**结论。** PASS（21 个子空间、1,445 条断言）。三个 response 族的最高次数都是 3。标量族和结构化族用 3 个统计量就被精确刻画；parity 族在 7 个非常数方向全部到位之前，一致误差始终至少 $`0.0625`$（差异至少 $`0.125`$）。含常数的精确 span 维数为 4、4、8。在每个选出的子空间上，差异与两倍一致误差之差不超过 $`1.6\times10^{-15}`$。

**为什么做。** response 族的次数并不决定它的近似复杂度；起作用的是逼近宽度，而 [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(b) 把它与最坏分布对差异等同起来。本检验比较三个最高次数相同的族，预期结构化的族代价低、parity 代价高（[Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)）。它同时在结构化（而非随机）的 response 上再次检验 Theorem 4.2(b) 的对偶，并检验 Theorem 4.2(a) 与 [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(d) 对一个给定的族所预测的 span 维数。

**设置。**

- 网格 $`\lbrace 0,\tfrac14,\dots,1\rbrace ^3`$（125 条消息）；每个族的最高次数都是 3。
- *标量仿射幂* $`(a+bx_1)^3`$：5 个 response。
- *多元 parity 拉回*：$`J_{\rm even}`$ 复合 8 个仿射映射，再加上原始 parity，共 9 个 response。
- *结构化求和幂* $`(a+b\bar x)^3`$：5 个 response。
- 种子：`SeedSequence([20261010, 19])`。

**做法。**

1. 计算每个族（含常数）的精确有理秩。
2. 拟合之前固定一个有限字典：各 response 向量、全部多重线性单项式，以及标量或结构化的 Chebyshev 基。
3. 对每个预算 $`d=1,\dots,7`$，贪心选出一个子空间；在全网格上用 LP 计算一致误差 $`\max_g\mathrm{dist}_\infty(g,H)`$，并用独立的原问题 LP 计算该子空间的最坏分布对差异。
4. 断言 差异 $`=2\times`$ 误差。由于贪心找到的子空间未必最优，正的误差值是最优宽度的上界；零误差则是精确值。

**结果（一致误差；差异恰为其两倍）。**

| 特征预算 $`d`$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 标量仿射幂 | 0.0569 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| 结构化求和幂 | 0.0582 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| 多元 parity | 0.25 | 0.125 | 0.125 | 0.125 | 0.0625 | 0.0625 | **0** |

- 来源：条目 `v014:C19`，`details.families[*].curve[*].uniform_error`（以及 `.discrepancy`、`.optimality`）。
- 含常数的 span 维数：4（标量）、8（parity）、4（结构化）（`span_dimension_including_constant`）。
- 全部 21 个（族, $`d`$）子空间上的对偶：$`\max\lvert\text{discrepancy}-2\cdot\text{uniform error}\rvert=1.6\times10^{-15}`$。
- 同族的留出 response：所选的标量和结构化子空间在 $`d=1`$ 时误差至多 $`0.017`$，从 $`d=3`$ 起为 $`0`$（`heldout_uniform_error`）。

![Figure 4.4：精确次数相同，近似信息阶数却大不相同](../figures/fig4_4_approximation.png)

*Figure 4.4.* $`p^{16}`$ 的低阶 Chebyshev 截断已经很接近它，截断误差等于一个低于 $`2e^{-q^2/N}`$ 的二项尾概率；parity 在次数低于 $`N`$ 时误差保持 $`1/2`$（Check 4.9）。在同一网格、同一最高次数下，标量族和结构化族用 3 个特征就被精确刻画，parity 要 7 个（本检验；空心标记是由找到的最优子空间给出的上界）。[PDF](../figures/fig4_4_approximation.pdf)

**为什么支持这些陈述。**

- 最高次数相同时，标量族和结构化族用 3 个统计量就能精确刻画，parity 需要全部 $`2^3-1=7`$ 个；只给 3 个统计量时，parity 仍留下 $`0.25`$ 的最坏分布对差异。决定信息阶数的是任务结构，而不是次数或决策步数（[Proposition 4.6](../proofs/04_complexity_of_sufficiency_ZH.md#s4-6)）。
- 精确的 span 维数就是 span 判据预测的数值，并且远低于 [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)(d) 统计的所有任务上的最坏情形。
- [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(b) 的对偶在非随机的结构化 response 上同样成立，包括差异始终很大的 parity 族。

<a id="c4-11"></a>

## Check 4.11 · 次数复合、上界可达、阈值学习器与 SGD 展开（Theorem 4.7(a)–(d)）

**结论。** PASS（72 个实例、371 条断言，最大残差 0）。12 条随机多项式更新链的实际块次数都等于上界 $`D_i=Nb_i\prod_{t>i}a_t`$，按 $`D_i`$ 逐块匹配矩后学习价值精确相等。平方构造在次数 1、2、4、8 上取到这个上界。阈值学习器把两个均值相同的分布区分为 $`2/3`$ 对 $`0`$。SGD 展开的余项从不超过其 $`\eta^3`$ 界的 0.275 倍。

**为什么做。** [Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(a) 断言第 $`i`$ 块上的 response 次数不超过 $`Nb_i\prod_{t>i}a_t`$，并且匹配这么多阶矩就足够；次数算错，矩充分性就不成立。(b) 断言这个上界可以取到，所以次数算松也会被发现。(c) 断言一旦更新不连续，任何矩阶都不够。(d) 的局部 SGD 展开是 [§5](05_optimizer_state_ZH.md) 所用的 SGD 一阶描述，所以它的余项界必须成立。

**设置。**

- 12 条随机更新链：$`R\in\lbrace 1,2,3\rbrace`$，$`N\in\lbrace 1,2\rbrace`$，次数 $`a_i,b_i\in\lbrace 1,2\rbrace`$，$`p_i=\lambda_ip_{i-1}^{a_i}+(1-\lambda_i)Z_i^{b_i}`$，$`\lambda_i\in\lbrace \tfrac14,\tfrac12,\tfrac34\rbrace`$ 随机。
- 用 SymPy 做精确的符号回拉，块矩用精确有理数；第 $`i`$ 块取 $`D_i+1`$ 阶的标量见证分布。
- 另有 $`R=1,\dots,4`$ 的平方构造，以及 (c) 的阈值分布。
- (d)：$`\mathbb R^3`$ 上 15 个随机三次函数 $`V`$，配 6 个原子上的随机离散梯度分布；再用 logistic 函数 $`V(\theta)=\sigma(w^\top\theta)`$ 重复；步长 $`\eta\in\lbrace 0.01,0.1,0.5,1\rbrace`$。

**做法。**

- 展开 $`J(p_R)`$，读出每块的次数并与 $`D_i`$ 比较；在按 $`D_i`$ 匹配矩的条件下计算两个学习价值。
- 对 (d)，计算精确的差 $`(K^P-K^Q)V(\theta)`$ 与展开的前两项，把余项与 $`\tfrac{M_3\eta^3}6(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)`$ 比较：
  - 三次函数的 $`M_3`$ 取常数三阶导数张量的 Frobenius 范数，它是算子范数的上界；
  - logistic 函数的 $`M_3=\lVert w\rVert^3/8`$，因为全局有 $`\lvert\sigma'''\rvert\le1/8`$。

**结果。** 数据：`theory/independent_checks/results_independent_v014.json`，`v014:C13`。

| 量 | 值 |
|---|---|
| 状态、实例数、断言数、最大残差 | PASS，72，371，0.0 |
| 声明次数 vs 实际块次数（`details.degrees`，12 条链） | 每条链都相等，例如 $`a=(2,2,2)`$、$`b=(2,2,1)`$、$`N=2`$：$`D=(16,8,2)`$，实际 $`(16,8,2)`$ |
| 逐块匹配下学习价值相等 | 12 条链全部精确相等 |
| 平方构造（`details.amplification`） | $`R=1,2,3,4`$；次数 $`1,2,4,8`$；差值 $`-1`$、$`1/4`$、$`3/256`$、$`315/16777216`$ |
| 阈值学习器（`check_C13` 中的断言） | 均值都是 $`1/2`$；价值 $`2/3`$ 对 $`0`$ |
| logistic 余项/界，15 个实例取最大（`details.logistic`） | $`\eta=0.01`$：0.274；$`0.1`$：0.274；$`0.5`$：0.270；$`1`$：0.265 |
| 全部检验中最大的余项/界（`best_ratio`） | 0.2743 |

**为什么支持 Theorem 4.7(a)–(d)。**

- 次数上界在随机更新链上是精确的，平方构造也取到了它：数据源需要匹配的阶数随更新的复合而增长。
- 阈值学习器说明，一旦更新不连续，任何固定阶数的矩都不够。
- (d) 的展开在所有测试步长下都成立，而且留有余量。

<a id="c4-12"></a>

## Check 4.12 · 逐步误差的累积（Theorem 4.7(e)）

**结论。** PASS（717 个实例、1,435 条断言，最大残差 0）。几何级数界在每个实例上都成立，并且可以取等号。反例 $`\eta=\tfrac12`$、$`L=C=1`$、$`R=4`$ 给出 $`e_4=65/32`$，超过线性值 $`RC\eta^2=1`$。

**为什么做。** 单步 $`O(\eta^2)`$ 的误差本身推不出整个训练 $`O(R\eta^2)`$ 的误差。[Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(e) 给出一个几何级数界，并断言它是紧的，因此线性累积需要稳定性条件。这两点都要精确检验。

**设置。** 精确有理数递推 $`e_{t+1}=s\cdot((1+\eta L)e_t+C\eta^2)`$，松弛系数 $`s\in(0,1]`$；$`\eta\in(0,1]`$、$`L\in[0,1]`$、$`C\in(0,1]`$、$`R\le40`$ 随机取值；另外在 $`(\eta,L,C,R,s)`$ 上做固定种子的随机重启加爬山搜索。

**做法。** 断言 $`e_R\le C\eta^2\sum_{s<R}(1+\eta L)^s`$，并在 $`s=1`$ 时断言取等号；搜索该界与线性简化 $`RC\eta^2`$ 之比的最大值；计算反例 $`\eta=\tfrac12`$、$`L=C=1`$、$`R=4`$。

**结果。** 数据：`theory/v015_checks/results_v015_independent.json`，`v015:V7`。

| 量 | 值 |
|---|---|
| 状态、实例数、断言数、最大残差 | PASS，717，1,435，0.0 |
| 几何级数界取到（`data.geometric_adversarial.best.score`） | 1.0（例如 $`\eta=4/5`$、$`L=1/5`$、$`C=7/10`$、$`R=13`$） |
| 反例（`data.linear_failure`） | $`e_4=65/32`$，界 $`65/32`$，线性值 $`RC\eta^2=1`$ |
| 搜索找到的最大"界/线性值"（`best_ratio`） | $`3.78\times10^{30}`$（$`\eta=2`$、$`L=2`$、$`R=47`$） |

**为什么支持 Theorem 4.7(e)。**

- 几何级数界是精确的：处处成立，并且可以取到。
- 一旦 $`\eta LR`$ 不小，"关于 $`R`$ 线性"的理解就不成立；所以沿训练累积局部展开时，必须满足所述的稳定性条件或短视野条件。

<a id="c4-13"></a>

## Check 4.13 · 局部仿射 SGD 描述在受控网络中的适用范围（Theorem 4.7(d),(e)）

**结论。** 适用范围图，与定理一致。在 $`\eta\le0.1`$ 且至多 4 次更新、或 $`\eta=0.01`$ 且 40 次更新的全部 35 个 SGD 设置上，切线仿射 SGD 预测都与实际网络一致，最大偏差 $`2.5\times10^{-4}`$。到 $`\eta\ge1`$ 时，42 个设置中有 35 个偏离，偏差最大 0.53。

**为什么做。** [Theorem 4.7](../proofs/04_complexity_of_sufficiency_ZH.md#s4-7)(d),(e) 预言：对 SGD，response 每一步由 consumed gradient 的均值决定，误差为 $`O(\eta^2)`$，并按几何级数累积。在非线性网络中，固定初始切线的仿射 SGD 预测因此应当在 $`\eta R`$ 小时准确，在 $`\eta`$ 大或视野长时失效。这一检验是看描述在哪里适用，而不是检验一个界。

**设置。** 受控小网络，不是大模型，也不是外部 benchmark。

- 网络：one-hot context，4 个 tanh 单元，sigmoid 输出；二元 cross-entropy；float64。
- 训练：每个采样块做一次普通 SGD，$`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$，$`R\in\lbrace 1,4,40\rbrace`$ 次更新。
- 任务：[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1) 的两组数据源、$`N=2,3,4`$ 的矩分布、$`N=2,3`$ 的 parity 分布（共 7 个任务）。
- 估计方式：能穷举的用完整有限历史枚举，否则每个数据源 2,400 次独立训练。
- 比较对象：实际策略在初始点处的切线仿射 SGD response。
- 一致性阈值在运行前写死在代码里：$`10^{-4}+0.1\lvert\text{prediction}\rvert+4\,\mathrm{SE}`$。

**做法。** 对每个设置，计算两个数据源之间实际的最终期望回报差和切线仿射预测，并按阈值记录是否一致。

**结果。** 数据：`theory/independent_checks/results_independent_v014.json → controlled_neural.details.configurations` 中的 SGD 行；分组汇总见 `details.observed_affine_comparison_cohorts`。

| $`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$ |
|---|---|---|---|
| 0.01 | 7/7 一致（最大偏差 $`1.1\times10^{-6}`$） | 7/7（$`4.1\times10^{-6}`$） | 7/7（$`2.4\times10^{-5}`$） |
| 0.1 | 7/7（$`1.1\times10^{-4}`$） | 7/7（$`2.5\times10^{-4}`$） | 5/7（$`1.9\times10^{-3}`$） |
| 1 | 3/7（0.013） | 0/7（0.053） | 3/7（0.084） |
| 3 | 0/7（0.52） | 0/7（0.53） | 1/7（0.51） |

- "SGD、步长 $`\le0.1`$、至多四次更新"这一组：28/28 一致，最大绝对偏差 $`2.52\times10^{-4}`$。
- 加性单步 null（同样的样本和权重，只改变分组方式）的参数残差恰为 $`0.0`$（`details.additive_null_parameter_residual`）。
- 同一实验中 Adam 的部分在 [§5](05_optimizer_state_ZH.md) 中分析。

![Figure 4.5：局部 SGD 展开的适用范围，以及复合结构的精确检验](../figures/fig4_5_regime.png)

*Figure 4.5.* 切线仿射 SGD 预测在 $`\eta\le0.1`$、短视野下与实际的受控网络一致，在 $`\eta\ge1`$ 时失效（本检验）。SGD 展开的三阶余项始终低于其界（Check 4.11）。逐步误差按几何级数累积，没有稳定性条件时，关于 $`R`$ 线性的理解不成立（Check 4.12）。[PDF](../figures/fig4_5_regime.pdf)

**为什么支持 Theorem 4.7(d),(e)。**

- SGD 的一阶仿射描述恰好在展开和稳定性条件所指的范围内准确：步长小、更新少，或者步长很小、更新很多。
- 偏差随 $`\eta`$ 和视野增长，与曲率项和累积项一致。
- 证据的范围：一个小网络结构、七个任务；它定位了适用范围的边界，并不认证任何真实模型。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C5,C6,C7,C8,C9,C10,C11,C12,C13,C18,C19,C23 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V7 \
  --output theory/v015_checks/quick/v7.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

`--only` 保留每条 claim 自己的随机流（`SeedSequence([20261010, claim 编号])`），所以子集运行能复现完整运行的数字。

## 产物

- 结果：
  - `theory/independent_checks/results_independent_v014.json`（`v014:C5`–`v014:C13`、`v014:C18`、`v014:C19`；`controlled_neural`）
  - `theory/v015_checks/results_v015_independent.json`（条目 `v015:V7`）
  - `theory/v014_repro/results_v014_rerun.json`（`constant_gap_parity`、`grid_response_dimensions`、`scalar_exact_decay`、`scalar_approximation`）
- 代码：`theory/independent_checks/check_v014_independent.py`（`check_C5` … `check_C13`、`check_C18`、`check_C19`、`check_C23`），`theory/v015_checks/mathematics.py`（`check_V7`），`docs/direction/v014_handoff/verify_environment_information_v014.py`
- 规格说明：`theory/independent_checks/SPEC.md`
- 已有图：`theory/independent_checks/figs/C18_evidence_exposure.png`、`theory/independent_checks/figs/C19_response_complexity.png`、`theory/independent_checks/figs/C23_controlled_neural.png`
- 图规格：`theoretical_framework/figures/specs/fig4_1.md` … `fig4_5.md`
