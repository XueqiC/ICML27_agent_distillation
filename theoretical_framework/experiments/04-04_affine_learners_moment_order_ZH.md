# Theorem 4.4 · 仿射 learner：矩阶充分且必要

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-04_affine_learners_moment_order_ZH.md) · [证明](../proofs/04-04_affine_learners_moment_order_ZH.md) · [实验](04-04_affine_learners_moment_order_ZH.md) · [English](04-04_affine_learners_moment_order.md)

**本页检验什么.** Theorem 4.4 的每个部分都有自己的精确检验：部署价值的多项式形式与系数界 (a)，矩匹配与学习价值相等的等价性 (b)，每一阶的必要性与常数差距见证 (c)，精确统计量个数与二元壁垒 (d)。另一个检验把结构化族的 span 维数与 (d) 的最坏情形作比较。

- **结论：PASS.** 在 1,080 棵随机生成并经对抗变异的部署树上，次数界和系数界全部成立（[Check 4.2](#c4-2)）。
- 对至多 5 次更新的仿射 learner，矩匹配的分布给出完全相同的学习价值，并且每个矩都能被它的单项式任务察觉（[Check 4.3](#c4-3)）。
- 对 $`N=1,\dots,14`$，每一阶都是必要的；对 $`N=1,\dots,8`$，parity 的差距保持为 1（[Check 4.4](#c4-4)）。
- 18 个配置中网格统计量个数全部等于 $`\binom{k+N}N-1`$，二元壁垒被取到（[Check 4.5](#c4-5)）。
- 全部计算都是精确有理数计算或精确 LP。

**本页用到的设定.**

- 学生 $`p\in[0,1]^k`$：在 context $`j`$ 以概率 $`p_j`$ 选择动作 1。部署是由机会、决策与终止节点构成的有限树，每条路径至多 $`N`$ 个决策；$`J_E(p)`$ 是它的期望奖励（[Definition 4.3](../statements/04-03_binary_context_student_affine_learner_ZH.md#s4-3)）。
- 经验块 $`Z_i\in[0,1]^k`$ IID 于来源分布 $`P`$ 或 $`Q`$。凸仿射 learner 用固定权重对 $`R`$ 个块取平均：$`p_R=w_0p_0+\sum_iw_iZ_i`$。一个来源的*学习价值*是 $`\mathbb EJ_E(p_R)`$。
- $`p^\alpha=\prod_jp_j^{\alpha_j}`$，$`\lvert\alpha\rvert=\sum_j\alpha_j`$，$`m_P(\alpha)=\mathbb E_PZ^\alpha`$ 是联合矩，$`\sum_\alpha\lvert c_\alpha\rvert`$ 是 $`J_E=\sum_\alpha c_\alpha p^\alpha`$ 的系数范数。
- *线性期望统计量*是对固定函数 $`\phi`$ 的 $`\mathbb E\phi(Z)`$；接口的最坏分布对差异，是统计量相等的两个分布之间最大的 response 差（[Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)）。
- 代码 ID：`v014:` 指 `theory/independent_checks`（结果文件 `results_independent_v014.json`）；`pkg:` 指修订包自带的验证脚本 `docs/direction/v014_handoff/verify_environment_information_v014.py`，重新运行的结果在 `theory/v014_repro/results_v014_rerun.json`。

**被检验的陈述（[Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）.**

- **(a)** $`J_E`$ 有唯一表示 $`J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，$`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$，且在 $`[0,1]^k`$ 上 $`J_E\in[0,1]`$。
- **(b)** 对每个 $`\lvert\alpha\rvert\le N`$ 都有 $`m_P(\alpha)=m_Q(\alpha)`$，当且仅当对每个至多 $`N`$ 次决策的部署和每个凸仿射 learner 都有 $`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$。
- **(c)** 每一阶都必要：$`\lbrace0,\tfrac1N,\dots,1\rbrace`$ 上的标量分布 $`P_N,Q_N`$（二项式权重 $`\binom Ni/2^{N-1}`$ 分别放在偶数点与奇数点上）在低于 $`N`$ 阶时一致，在 $`N`$ 阶相差 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$；$`\lbrace0,1\rbrace^N`$ 上的偶、奇 parity 分布在所有总次数低于 $`N`$ 的联合矩上一致，而 parity 价值 $`J_{\rm even}(p)=\tfrac12\big(1+\prod_j(1-2p_j)\big)`$ 在两者下的期望分别为 $`1`$ 与 $`0`$。
- **(d)** 在网格 $`\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上，充分的线性期望统计量最少需要 $`\binom{k+N}N-1`$ 个。在 $`\lbrace0,1\rbrace^k`$ 上、$`N\ge k`$ 且奖励任意时，每个少于 $`2^k-1`$ 个统计量的接口未分辨差距都恰为 $`1`$，而 $`2^k-1`$ 个仿射无关的特征给出 $`0`$。

<a id="c4-2"></a>

## Check 4.2 · 部署价值是有界多项式（Theorem 4.4(a)）

**结论.** 在 1,080 棵部署树（5,400 条断言）上 PASS。对抗搜索找到的"系数范数 / $`3^N`$"最大为 $`2/3`$。

**为什么做.** 仿射理论建立在一件事上：$`J_E`$ 是 policy 坐标的多项式，次数至多为 $`N`$，且 $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$（[Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(a)）。只要有一棵树的精确展开次数更高或范数更大，它就被证伪。

**设置.**

- 90 轮搜索，每轮 $`k\in\lbrace 1,2,3\rbrace`$ 个 context，每条路径 $`N\in\lbrace 1,\dots,5\rbrace`$ 个决策。
- context 可以在路径上重复，机会节点使用有理概率。
- 每轮评估 12 棵候选树：每第四棵是全新的随机树，其余是对当前最坏比值树做叶子、context 或机会概率的随机变异（共 1,080 棵）。
- 种子：`SeedSequence([20261010, 6])`。

**做法.**

1. 用 SymPy 对每棵树做符号展开，得到精确的有理多项式。
2. 断言总次数 $`\le N`$ 且 $`\sum\lvert c_\alpha\rvert\le3^N`$。
3. 在 3 个随机有理点上断言 $`J\in[0,1]`$。
4. 对 $`\sum\lvert c_\alpha\rvert/3^N`$ 做爬山搜索，寻找最坏情形。

**结果.**

| 量 | 值 | 来源 |
|---|---:|---|
| 树数 / 断言数 | 1,080 / 5,400 | 条目 `v014:C6`，`n_instances`, `.n_assertions` |
| 最大 $`\sum\lvert c_\alpha\rvert/3^N`$ | $`2/3`$ | 条目 `v014:C6`，`details.exact_best_ratio` |
| 见证 | $`N=1`$，$`J=1-p_1`$ | 条目 `v014:C6`，`details.witness` |
| 状态 | PASS（最大残差 0） | 条目 `v014:C6`，`status` |

**为什么支持 Theorem 4.4(a).**

- 次数界和系数界在每棵树上都成立，包括对抗变异出来的树。
- 在这些树上，$`3^N`$ 界成立，但偏保守。

<a id="c4-3"></a>

## Check 4.3 · 矩匹配既充分又必要（Theorem 4.4(b)）

**结论.** PASS（46 个实例、385 条断言，最大残差恰为 0）。在 24 个随机配置（更新次数最多 $`R=5`$）中，矩匹配的分布给出完全相同的学习价值。$`N`$ 阶以内的每个矩都能被相应的单项式任务察觉。

**为什么做.** [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(b) 有两个方向，各有对应的证伪方式：

- 存在一对矩匹配的分布，在某个任务或仿射 learner 下学习价值不同（否定充分性）；
- 存在某个矩，它的失配没有任何任务能察觉（否定必要性）。

**设置与做法.**

- *充分性*，24 个配置（$`k\in\lbrace 1,2\rbrace`$，$`N\in\lbrace 1,2,3\rbrace`$，$`R\in\lbrace 1,\dots,5\rbrace`$）：
  1. 在网格 $`\lbrace 0,\tfrac1{N+1},\dots,1\rbrace ^k`$ 上取随机的正分布 $`P`$；
  2. 构造 $`Q=P+av`$，其中 $`v`$ 取在所有次数 $`\le N`$ 的矩约束（含总质量）的精确零空间中；
  3. 随机抽取凸权重、$`p_0`$ 和任务树；
  4. 用精确的块矩卷积计算两个学习价值，断言所有被匹配的矩以及两个价值都相等。
- *必要性*，对 $`(k,N)\in\lbrace (1,4),(2,3),(3,2)\rbrace`$ 以及每个满足 $`\lvert\alpha\rvert\le N`$ 的非零 $`\alpha`$：
  1. 符号验证"访问 $`j`$ 恰好 $`\alpha_j`$ 次、结果全为 1 才回报"的任务等于 $`p^\alpha`$；
  2. 构造只在第 $`\alpha`$ 个矩上不同的分布对；
  3. 断言一步 learner $`p=Z`$ 能把这对分布区分开。
- *两 context 例子*：边缘相同，交叉矩分别为 $`\tfrac12`$ 和 $`0`$。
- 种子：`SeedSequence([20261010, 7])`。

**结果.**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 46 / 385 | 条目 `v014:C7`，`n_instances`, `.n_assertions` |
| 仿射充分性配置 | 24 个，$`R`$ 最大为 5 | 条目 `v014:C7`，`details.affine_configurations` |
| 最大残差 | 0（精确） | 条目 `v014:C7`，`worst_residual` |
| 状态 | PASS | 条目 `v014:C7`，`status` |

**为什么支持 Theorem 4.4(b).**

- 匹配全部 $`N`$ 阶以内的联合矩后，学习价值精确相等：差为零，而不只是很小。
- 反过来，$`N`$ 阶以内的每个矩都能被某个任务察觉，所以任何一个矩都不能省。
- 只匹配边缘分布是不够的（两 context 例子）。

<a id="c4-4"></a>

## Check 4.4 · 每一阶都必要；常数差距见证（Theorem 4.4(c)）

**结论.** PASS（22 个实例、8,983 条断言，最大残差 0）。标量见证对 $`N=1,\dots,14`$ 精确满足 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$，且所有低阶矩相等。在 $`N=1,\dots,8`$ 上，parity 分布对所有次数低于 $`N`$ 的联合多重指标都匹配，差距保持为 1；独立实现给出相同结果。

**为什么做.** [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(c) 说明层级是严格的：

- 对每个 $`N`$，都存在所有低阶矩一致、但 $`N`$ 阶学习价值不同的分布；
- 在 parity 构造中，差距不随 $`N`$ 缩小。

**设置与做法.**

- *标量见证.* 对 $`N=1,\dots,14`$，用精确有理数构造 $`P_N,Q_N`$。断言总质量为 1、所有 $`\ell<N`$ 阶的矩相等，且 $`m_{P_N}(N)-m_{Q_N}(N)=(-1)^NN!/(2^{N-1}N^N)`$。
- *Parity 见证.* 对 $`N=1,\dots,8`$ 枚举 $`\lbrace 0,1\rbrace ^N`$：
  1. 对总次数 $`<N`$ 的所有多重指标（不只是坐标子集）断言联合矩相等；
  2. 断言 $`J_{\rm even}`$ 在两个分布下分别取 1 和 0；
  3. 在 4 个随机有理点 $`p`$ 上，断言闭式表达式等于枚举得到的偶 parity 概率。
- 种子：`SeedSequence([20261010, 8])`。
- *独立实现*（`pkg:constant_gap_parity`，理论作者编写）：对 $`N=1,\dots,8`$，检验全部 $`2^N-1`$ 个真坐标子集的子集矩是否相等——二元坐标是幂等的，所以这已覆盖所有次数 $`<N`$ 的原始矩——然后精确计算两个价值。

**结果.**

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

**为什么支持 Theorem 4.4(c).**

- 对某个标量任务而言，每一阶 $`N`$ 都是必要的，差距公式精确成立。
- 标量差距随 $`N`$ 指数缩小，parity 差距则在每一阶都保持为 1。所以"精确必要性"和"固定精度下的困难性"是两回事；[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) 与 [Check 4.9](04-06_exact_order_not_approximate_complexity_ZH.md#c4-9)–[4.10](04-06_exact_order_not_approximate_complexity_ZH.md#c4-10) 把二者分开。

<a id="c4-5"></a>

## Check 4.5 · 精确统计量个数与二元壁垒（Theorem 4.4(d)）

**结论.** PASS（22 个实例、903 条断言，最大 LP 残差 $`2.3\times10^{-12}`$）。18 个配置的网格秩全部等于 $`\binom{k+N}N`$；237 个非常数单项式中删去任意一个，都会留下可分离的分布对；32 个特征数少于 $`2^k-1`$ 的随机二元接口，最坏分布对差异全部恰为 1。

**为什么做.** [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(d) 把 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(a) 的 span 判据落成具体数字：

- 在网格上恰好需要、也只需要 $`\binom{k+N}N-1`$ 个统计量；
- 在二元消息、任意回报的设定下，少于 $`2^k-1`$ 个统计量就会留下最坏情形误差 1。

**设置与做法.**

1. *网格秩.* 对所有 $`k\le3,N\le5`$ 以及 $`k=4,N\le3`$（共 18 个配置），计算单项式求值矩阵在网格上的精确整数秩，断言其等于 $`\binom{k+N}N`$。
2. *删去特征.* 对每个配置中的每个非常数单项式（共 237 个），删去它后求解最坏分布对 LP，断言存在特征匹配的分布对能区分被删的单项式。
3. *二元壁垒.* 对 $`k=1,\dots,4`$，各抽 8 个特征数 $`d<2^k-1`$ 的随机有理特征表。取 $`[\mathbf 1;\phi]`$ 的一个零空间向量，以其正支撑为事件，断言 LP 差异等于 1。$`k\le3`$ 时还枚举全部事件，断言最优值为 1。
4. *完整接口.* 用 $`2^k-1`$ 个一般位置的特征，对每个示性 response 和一个随机 response 断言差异为 0。
5. *独立实现*（`pkg:grid_response_dimensions`）：在六个配置下计算 $`\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ 上单项式矩阵的数值秩（容差 $`10^{-10}`$）。

种子：`SeedSequence([20261010, 9])`。

**结果.**

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

*Figure 4.2.* 标量见证在每一阶都能区分，差距按指数变小并与闭式一致；parity 分布的差距保持为 1（[Check 4.4](#c4-4)）。网格上单项式求值矩阵的秩在全部 18 个配置中都等于 $`\binom{k+N}N`$；在二元消息、任意回报的设定下，特征数少于 $`2^k-1`$ 的接口都留下最坏情形误差 1（本检验）。[PDF](../figures/fig4_2_moment_order.pdf)

**为什么支持 Theorem 4.4(d).**

- 二项式计数是精确的：每个非常数矩单独都是必要的，合起来又是充分的。
- 二元壁垒被一个显式的分离事件取到，而不只是一个上界。

### [Check 4.10](04-06_exact_order_not_approximate_complexity_ZH.md#c4-10) · 最高次数相同，response 空间的复杂度不同

- *对 Theorem 4.4 检验什么.* (d) 的最坏情形计数与结构化任务族的 span 维数相比如何。
- *设置.* 网格 $`\lbrace0,\tfrac14,\dots,1\rbrace^3`$（125 条消息）；三个最高次数为 3 的 response 族：标量仿射幂 $`(a+bx_1)^3`$、多元 parity 拉回、结构化求和幂 $`(a+b\bar x)^3`$；精确有理秩。
- *关键数字.* 含常数的精确 span 维数为 4（标量）、8（parity）、4（结构化）；标量族与结构化族用 3 个统计量就被精确刻画，parity 只有用全部 7 个非常数方向才行。
- *结论.* PASS（条目 `v014:C19`）。结构化的族远低于 (d) 统计的所有任务上的最坏情形。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C6,C7,C8,C9,C19 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

`--only` 保留每个 claim 自己的随机流（`SeedSequence([20261010, claim number])`），所以只跑子集也能复现完整运行的数字。

这些检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）、对树做精确符号展开并做精确有理数运算的 SymPy（[Meurer et al., 2017](#ref-meurer2017sympy)），以及 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）的 `linprog` 配合 HiGHS 求解器（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解线性规划。

## 产物

- 结果：`theory/independent_checks/results_independent_v014.json`（条目 `v014:C6`–`v014:C9`、`v014:C19`）；`theory/v014_repro/results_v014_rerun.json`（`constant_gap_parity`、`grid_response_dimensions`）。
- 代码：`theory/independent_checks/check_v014_independent.py`（`check_C6` … `check_C9`、`check_C19`）；`docs/direction/v014_handoff/verify_environment_information_v014.py`；规格说明 `theory/independent_checks/SPEC.md`。
- 图的规格：`theoretical_framework/figures/specs/fig4_2.md`。
