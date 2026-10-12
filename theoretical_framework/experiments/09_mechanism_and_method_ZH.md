# §9 机制与方法 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/09_mechanism_and_method_ZH.md) · [English](09_mechanism_and_method.md)

**结论.**

- **[Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)（耦合宽度）成立.** 在 320 个随机的 packet response 表上：可加表的宽度恰为零，不可加表的宽度为正；界 (b) 成立，中位松弛 0.231；受限配对与严格零假设两部分精确成立（最差残差 $`9.9\times10^{-16}`$）。一个独立实现给出相同结论。
- **严格零假设及其边界在训练过的网络中出现.** 在受控网络中，同一状态上的一步 SGD 对重组严格不变（参数差为 0）。用同样的标签、不同的分组做连续更新，最终回报的变化从 $`9.6\times10^{-6}`$ 到 $`0.75`$：SGD 下随步长和 horizon 增大，Adam 下符号相反。
- **按实测后果选择（[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)，Rule 2–3）.** 在受控网络的冻结留出面板上，测试过的五个静态 proxy 未能可靠地选对：在留出的学习器状态上，它们的判错率（wrong-sign rate）为 0.36–0.55，随机分数为 0.55；与 gold 对比的 Spearman 相关至多 0.10。实测的 4 步尾段在每一种状态、$`\epsilon`$、学习率与 warmth 的变化下，判错率不超过 3.1%，选择 regret 为零。在留出的阈值任务上，两种短尾段都比随机选得差，只有全尾段在每个划分上都精确。
- **warm-up 救不了静态 proxy（Rule 3）.** 学习率为 0 的矩 warm-up 压缩了候选之间的差别（随机选择的 regret 在 5 个划分中的 4 个下降），但五个静态 proxy 与 $`W\in\lbrace 5,20\rbrace`$ 的 10 种组合中，没有一种同时改进 regret、判错率和 Spearman。
- **范围.** 精确有限计算与一个受控小网络。本页不报告任何真实模型或 benchmark 结果；计划中的真实模型检验见 [Remark 9.4](../proofs/09_mechanism_and_method_ZH.md#s9-4)。同一面板上的一阶 margin 检验（code ID `v015:E12`）、排序器对比（`v016:C5`、`v016:C5b`）与配对替换证书（`v016:C6`、`v016:C6b`）随它们所检验的结论一起，分别放在 [§8](08_verification_ZH.md) 与 [§7](07_second_moment_channel_ZH.md) 的实验页。

| 检验 | 检验对象 | Code ID | 结果 |
|---|---|---|---|
| [Check 9.1](#c9-1) | [Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)(a)–(d) | `v014:C17` | PASS |
| [Check 9.2](#c9-2) | [Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)(a)、(b)、(d) | `pkg:coupling_and_additive_null` | PASS |
| [Check 9.3](#c9-3) | 训练网络中的 [Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)(d) | `v014:C23`（重组部分） | 零假设精确成立；连续更新有效应 |
| [Check 9.4](#c9-4) | [Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3) 的 Rule 2 与 Rule 5；[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2) 第 5 步 | `v015:E6` | 受控发现 |
| [Check 9.5](#c9-5) | [Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3) 的 Rule 3 | `v015:E9` | 受控发现 |

![Figure 9.1：耦合宽度——精确零假设与连续重组效应](../figures/fig9_1_coupling.png)

*Figure 9.1.* (A) 在受控网络中，把同样四个标签分组为 $`\lbrace(0,0),(1,1)\rbrace`$ 或 $`\lbrace(0,1),(0,1)\rbrace`$，会改变最终回报。同一状态上的一步，差距恰为零；SGD 下差距随步长和 horizon 增大，Adam 下符号相反、动态也不同。(B) 期望 packet response 在耦合多面体上的取值范围：可加表是一个点，不可加表的宽度为正。[PDF](../figures/fig9_1_coupling.pdf)

<a id="c9-1"></a>

## Check 9.1 · 耦合多面体、可加刻画与严格零假设（Proposition 9.1）

**结论.** PASS：320 个实例，2,567 条断言，最差残差 $`9.85\times10^{-16}`$（LP 可行性）。四个部分在每个实例上都成立。

**为什么做.** Proposition 9.1 有四个断言：宽度为零的"当且仅当"刻画、用可加残差给出的界 (b)、禁止配对时该刻画失效、以及单步零假设。任何一个若不成立，在某个随机实例上都会明显违反。

**设置.**

- $`\lvert\mathcal Y_1\rvert,\lvert\mathcal Y_2\rvert\in[2,5]`$，边缘分布严格为正。
- 随机可加表 $`H=a+b`$ 与随机不可加表。
- seed 为 `SeedSequence([20261010, 17])`。

**做法.**

1. 用两个线性规划计算宽度：$`\langle\Gamma,H\rangle`$ 在耦合多面体上的最大值与最小值。
2. 用第三个线性规划计算 $`\inf_{a,b}\lVert H-a-b\rVert_\infty`$，核对界 (b)。
3. 构造 (c) 部分的禁止配对例子。
4. 在有理数算术中精确核对零假设：线性模型在同一参数处走一步，同样的样本换一种分组；再做两步连续更新并重组。

**结果**（`theory/independent_checks/results_independent_v014.json`，条目 `v014:C17`）：

| 量 | 值 |
|---|---:|
| 状态 | PASS |
| 实例 / 断言 | 320 / 2,567 |
| 最差残差（LP 可行性） | $`9.85\times10^{-16}`$ |
| (b) 的松弛：中位数 / 最大值 | 0.231 / 0.712 |
| 可行宽度为零的禁止配对表 | $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$ |
| 一步重组的差别（线性模型，精确） | 0 |
| 两步连续重组的差别（线性模型，精确） | $`1/32`$ |

**为什么支持 Proposition 9.1.**

- 可加表的宽度恰为零，不可加表的宽度为正，与 (a) 一致。
- 界 (b) 成立但一般不紧；典型的松弛约为回报范围的四分之一，与证明中的严格性例子一致。
- 单步零假设在有理数算术中精确成立，所以任何测到的单步效应都只能是实现上的问题。
- 连续更新会产生真实的重组效应，即 (d) 证明中的 $`\eta^2H_{P_2}g_{P_1}`$ 项。

<a id="c9-2"></a>

## Check 9.2 · 独立实现（Proposition 9.1）

**结论.** PASS：可加表的宽度为 0，一个不可加的二次配对宽度为 0.25，冻结梯度零假设的残差为 $`2.8\times10^{-17}`$。

**为什么做.** 一个与 Check 9.1 分开编写的实现，可以排除两者共有的编码错误。

**设置.** 有限 packet 的线性规划与冻结梯度算术；不涉及神经网络 optimizer。

**结果**（`theory/v014_repro/results_v014_rerun.json`，`coupling_and_additive_null`）：

| 量 | 值 |
|---|---:|
| 可加表的宽度 | 0.0 |
| 二次配对：最小 / 最大 / 宽度 | 0.25 / 0.50 / 0.25 |
| 严格冻结梯度可加零假设，最大残差 | $`2.8\times10^{-17}`$ |

**为什么支持 Proposition 9.1.** 可加表宽度为零、不可加配对宽度为正，与 (a) 一致；(d) 的冻结梯度零假设在机器精度内精确成立。

<a id="c9-3"></a>

## Check 9.3 · 受控网络中的重组（Proposition 9.1(d)）

**结论.** 同一状态上的单步对重组严格不变（参数差 0.0）。连续更新之后，重组使最终回报在 SGD 下最多变化 0.749，在 Adam 下绝对值最多变化 0.073。

**为什么做.** 宽度是通过实际的学习后果定义的。一个训练过的网络，喂入同样的标签、不同的分组，在同一状态的一步上差别必须恰为零；连续更新之后则可能出现真实的差别。

**设置.**

- 网络：one-hot 上下文，4 个 tanh 单元，sigmoid 输出，float64。这是受控小网络，不是 Gemma，也不是外部 benchmark。
- 数据源 A 的 packet 为 $`\lbrace (0,0),(1,1)\rbrace`$，数据源 B 为 $`\lbrace (0,1),(0,1)\rbrace`$。两者含同样四个标签，边缘分布相同。
- optimizer：SGD，$`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$；Adam，$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$，$`\eta\in\lbrace 0.01,0.1\rbrace`$。
- 1、4 或 40 个循环，即 2、8 或 80 次更新。
- 精确枚举全部确定性历史。

**做法.** 计算两种分组之间最终期望回报的差。另外，在同一状态、相同权重下计算一步之后的参数差，并用中心差分核对手写梯度。

**结果**（`theory/independent_checks/results_independent_v014.json`，`controlled_neural.details`）：

- 同一状态单步：`additive_null_parameter_residual` = **0.0**；
- 手写梯度对中心差分：`gradient_max_residual` = $`8.7\times10^{-11}`$。

回报差 A − B（`regrouping`）：

| optimizer | $`\eta`$ | 2 次更新 | 8 次更新 | 80 次更新 |
|---|---:|---:|---:|---:|
| SGD | 0.01 | $`9.6\times10^{-6}`$ | $`3.8\times10^{-5}`$ | $`3.2\times10^{-4}`$ |
| SGD | 0.1 | $`9.2\times10^{-4}`$ | $`3.2\times10^{-3}`$ | $`7.9\times10^{-3}`$ |
| SGD | 1.0 | 0.040 | 0.062 | 0.076 |
| SGD | 3.0 | 0.658 | 0.747 | 0.749 |
| Adam | 0.01 | −0.0094 | −0.0136 | −0.0032 |
| Adam | 0.1 | −0.0729 | −0.0503 | −0.0029 |

**为什么支持 Proposition 9.1.**

- 效应恰好在 (d) 所说必须为零的地方为零：同一参数处的一步。
- 在实际的非线性学习器中，一旦更新是连续的，边缘分布相同的反馈怎样组织就会改变学到的学生。SGD 下效应随步长和 horizon 增大；Adam 下符号相反、动态也不同。
- 所以耦合宽度是学习后果层面一个真实的、依赖于 optimizer 的量。packet 的组织属于 [Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2) 要测量的 response（第 3 步），而不是数据本身的固定属性。

![Figure 9.2：哪些测量能选对反馈](../figures/fig9_2_selection.png)

*Figure 9.2.* 选择规则的受控学习器证据。(A) 冻结 proxy 在留出的学习器状态，以及 warmth、$`\epsilon`$、学习率变化下的判错率（`v015:E6`）：测试过的静态 proxy 与随机分数相当，实测的短尾段 response 接近零。(B) 选择 regret：短尾段在留出状态上几乎选得最优，在留出的阈值任务上却比随机差；只有全尾段处处精确。(C) 学习率为 0 的矩 warm-up 的效果（`v015:E9`）：候选之间的差距缩小（随机 regret 下降），静态 proxy 没有改进。受控小网络中的精确有限历史计算。[PDF](../figures/fig9_2_selection.pdf)

<a id="c9-4"></a>

## Check 9.4 · 用冻结 proxy 做留出选择（Remark 9.3，Rule 2 与 Rule 5）

**结论.** 受控发现。在留出的学习器状态上，测试过的五个静态 proxy 判错率为 0.359–0.547（随机 0.547），Spearman 至多 0.10。实测的 4 步尾段在每一种状态、$`\epsilon`$、学习率与 warmth 的变化下，判错率至多 0.031，regret 为零。在留出的阈值任务上，1 步与 4 步尾段的 regret 为 0.044 与 0.032，随机为 0.019。全尾段在每个划分上 regret 与判错率都为零。

**为什么做.** Rule 2 说候选必须按它实际的"更新加尾段"response 来评判。这个检验问：当学习器状态、任务、配方或 optimizer warmth 改变时，更便宜的统计量能否代替 response。每种改变都是单独的留出划分，所以只拟合了 pilot 的 proxy 无法通过。短尾段 proxy 是尾段被截断的一轮边界 response，即 [Definition 8.6](../proofs/08_verification_ZH.md#s8-6) 意义下的 surrogate；阈值划分检验这样的 surrogate 在新任务上能否保持排序，也就是 [Remark 8.11](../proofs/08_verification_ZH.md#s8-11) 的校准迁移问题。

**设置**（`theory/v015_checks/proxy_split_freeze.json`，SHA-256 `ac8f4aee…`，在任何留出测量之前写入）：

- **学习器.** `v014:C23` 的受控网络（one-hot 上下文，4 个 tanh 单元，sigmoid，平均交叉熵，float64）。
  - 状态：`fresh`、`warm2`、`warm5`、`warm9`（完整状态，含 Adam 矩）。
  - 配方 8 种：Adam，$`\eta\in\lbrace 0.01,0.1\rbrace`$；SGD，$`\eta\in\lbrace 0.1,1\rbrace`$；每种再配 $`\epsilon\in\lbrace 10^{-8},10^{-3}\rbrace`$。
- **候选.** 10 个双标签 packet 上的 9 个标签分布：`reference_even`，矩匹配的 `moment3_P`/`moment3_Q`，符号匹配的 `low_sign_A`/`B` 与 `high_sign_A`/`B`，`odd_parity` 与 `middle_mixture`。
- **gold.** 一轮候选边界之后接 7 次 IID 参考更新（共 8 次更新），对全部有限历史精确枚举，取与参考候选的对比值。
- **proxy**，只在 pilot 上用 ridge 回归拟合，并冻结在 `proxy_models_frozen.json`：
  - 静态：输入矩误差、平均梯度、只看符号、第一步的 $`\Delta\theta`$、带 $`(m,v)`$ 的 $`\Delta\theta`$；
  - 实测的短尾段 response：1 次更新后、4 次更新后；
  - 对照：随机分数、任务身份、训练步数。
- **pilot.** 状态 `fresh` 与 `warm2`，任务 `moment` 与 `parity`，配方 0 与 4；72 行。
- **留出划分.** 状态 `warm5`/`warm9`；任务 `threshold`；只变 $`\epsilon`$（配方 1、5）；只变学习率（2、6）；两者都变（3、7）；只用 fresh 拟合、在 warm 状态上评测，以及反过来。留出共 432 行；seed 为 `SeedSequence([150150, 106])`。
- **指标.**
  - *选择 regret*：最优候选的 gold 值减去 proxy 选中候选的 gold 值，对 cell 取平均。
  - *判错率*：gold 非零的候选中，预测对比值符号相反的比例。
  - *Spearman*：预测对比值与 gold 对比值的秩相关。

**做法.** 先把划分清单与冻结的 proxy 系数写入磁盘，再对每个留出划分只评测一次。任何留出结果都不改动 proxy、特征、ridge 惩罚或弃权阈值。

**结果**（`theory/v015_checks/results_v015_independent.json`，条目 `v015:E6`，`data.reports.<split>.metrics.<proxy>`）：

*判错率*（越低越好）：

|划分|输入矩|平均梯度|符号|Δθ|Δθ+m,v|短尾段 1|短尾段 4|随机|全尾段|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|0.359|0.547|0.406|0.516|0.516|0.047|0.016|0.547|0.000|
|`held_out_epsilon`|0.234|0.250|0.312|0.391|0.453|0.031|0.031|0.531|0.000|
|`held_out_lr`|0.438|0.344|0.297|0.312|0.344|0.391|0.031|0.469|0.000|
|`held_out_epsilon_lr`|0.438|0.344|0.297|0.312|0.344|0.391|0.031|0.562|0.000|
|`fresh_to_warm`|0.359|0.391|0.562|0.422|0.516|0.047|0.016|0.578|0.000|
|`warm_to_fresh`|0.125|0.125|0.188|0.250|0.438|0.062|0.031|0.469|0.000|
|`held_out_task`|0.250|0.250|0.083|0.250|0.083|0.000|0.042|0.458|0.000|

*选择 regret*（越低越好）：

|划分|输入矩|平均梯度|符号|Δθ|Δθ+m,v|短尾段 1|短尾段 4|随机|全尾段|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|0.0064|0.0043|0.0043|0.0043|0.0039|0.0000|0.0000|0.0056|0.0000|
|`held_out_epsilon`|0.0013|0.0013|0.0022|0.0013|0.0013|0.0000|0.0000|0.0067|0.0000|
|`held_out_lr`|0.0041|0.0041|0.0070|0.0041|0.0041|0.0007|0.0000|0.0105|0.0000|
|`held_out_epsilon_lr`|0.0040|0.0040|0.0070|0.0040|0.0040|0.0007|0.0000|0.0093|0.0000|
|`fresh_to_warm`|0.0064|0.0064|0.0054|0.0064|0.0064|0.0000|0.0000|0.0045|0.0000|
|`warm_to_fresh`|0.0000|0.0000|0.0012|0.0000|0.0000|0.0000|0.0000|0.0056|0.0000|
|`held_out_task`|0.0000|0.0000|0.0020|0.0000|0.0000|0.0439|0.0322|0.0186|0.0000|

*Spearman*（越高越好；— 表示预测为常数、无定义）：

|划分|输入矩|平均梯度|符号|Δθ|Δθ+m,v|短尾段 1|短尾段 4|随机|全尾段|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|-0.00|0.05|0.10|0.07|0.07|0.95|0.99|-0.04|1.00|
|`held_out_epsilon`|0.45|0.40|0.45|0.42|0.41|0.96|0.99|-0.07|1.00|
|`held_out_lr`|0.38|0.40|0.46|0.57|0.58|0.34|0.95|0.03|1.00|
|`held_out_epsilon_lr`|0.38|0.40|0.46|0.58|0.58|0.33|0.95|0.04|1.00|
|`fresh_to_warm`|-0.00|-0.07|-0.07|-0.06|-0.09|0.95|0.99|-0.07|1.00|
|`warm_to_fresh`|0.79|0.58|0.73|0.72|0.59|0.90|0.97|0.10|1.00|
|`held_out_task`|0.63|0.58|0.73|0.71|0.84|—|0.54|-0.05|1.00|

**为什么支持 Rule 2 与 Rule 5.**

1. *测试过的静态 proxy 不是 response.* 在留出的学习器状态上，以及从 fresh 状态迁到 warm 状态时，它们与随机分数相当：判错率 0.36–0.56，随机为 0.55–0.58；Spearman 至多 0.10。同一个更新统计量，在不同的完整状态上对应不同的学习后果，这正是 [Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1) 与 [§5](../proofs/05_optimizer_state_ZH.md) 的 optimizer state 结果所预言的。这是关于所测 proxy 的发现，不是对所有静态统计量的证明；被证明的不一致性只限于 [Proposition 7.3](../proofs/07_second_moment_channel_ZH.md#s7-3) 的两类。
2. *实测的 response 能选对.* 4 步尾段在每一种状态、$`\epsilon`$、学习率与 warmth 的变化下，判错率不超过 3.1%，regret 为 0。1 步尾段在学习率变化下变差（判错率 0.39），但仍几乎选得最优（regret 0.0007）。
3. *尾段长度是估计对象的一部分.* 在留出的 `threshold` 任务上，短尾段比随机选得差（regret 0.044 与 0.032，随机 0.019），因为它们的排序与 8 步全尾段不同。全尾段是唯一在每个划分上 regret 与判错率都为零的测量。用 [Remark 8.11](../proofs/08_verification_ZH.md#s8-11) 的话说：在一个任务族上校准好的 surrogate，到另一个任务族上可能失去排序，所以必须核对校准迁移。静态 proxy 在这个划分上碰巧达到零 regret，但判错率为 8%–25%，在留出状态上又与随机相当，所以它们不能跨划分迁移。
4. *量级.* regret 在回报单位下为 $`10^{-3}`$–$`10^{-2}`$，每个划分有 8 个 cell（任务划分与 warm-to-fresh 划分为 4 个）。cell bootstrap（`regret_bootstrap95`）描述的是这个有限面板内部的变化，不是总体置信区间。

<a id="c9-5"></a>

## Check 9.5 · 学习率为 0 的矩 warm-up 下的 proxy 迁移（Remark 9.3，Rule 3）

**结论.** 受控发现。warm-up 压缩了候选之间的差别：随机选择的 regret 在 5 个留出划分中的 4 个下降，$`W=5`$ 时平均下降 0.0079，$`W=20`$ 时平均下降 0.0100。它没有让静态 proxy 变得可预测：五个静态 proxy 与 $`W\in\lbrace 5,20\rbrace`$ 的 10 种组合中，没有一种同时改进 regret、判错率和 Spearman，它们的判错率上升 1–13 个百分点。

**为什么做.** 在 warm 的 optimizer state 上，候选必须匹配的一阶统计量会改变（[Proposition 5.6–5.8](../proofs/05_optimizer_state_ZH.md#s5-6)）。这个检验问：把矩预热之后，Check 9.4 中的静态统计量会不会变得可预测。

**设置**（`theory/v015_checks/proxy_split_freeze_E9.json`，SHA-256 `f895853d…`；模型 `proxy_models_frozen_E9.json`，SHA-256 `c96065f0…`；两者都在任何新的留出面板之前写入）：

- **臂.** `fresh`、`fresh_W5` 与 `fresh_W20`。warm 臂在 `reference_even` 的一个声明好的 IID 前缀上，以恰为 0 的学习率跑 $`W\in\lbrace 5,20\rbrace`$ 步 Adam（$`\beta=(0.9,0.999)`$）。参数不变，只有 $`m`$、$`v`$ 与步数计数器前进。
- **拟合.** Check 9.4 的候选菜单、特征、gold、ridge 拟合与指标都不变。只用原来冻结的 pilot 拟合，并核对系数完全相同。只变状态的对比用原来的 warm2 单独拟合。
- **划分.** 五个配对的 cold/warm 划分：留出状态、任务（`threshold`）、$`\epsilon`$、学习率、两者都变；留出共 918 行。随机对照在各臂之间配对。
- **seed.** `SeedSequence([150150, 109])`。

**做法.** 对每个划分、proxy 与 $`W`$，计算每个指标的 warm 减 cold 差，再按划分等权平均。改进指 regret 更低、判错率更低、Spearman 更高。

**结果**（`theory/v015_checks/results_v015_independent.json`，条目 `v015:E9`，`data.summary` 与 `cheap_proxy_results`；表中是 5 个划分上 warm − cold 的平均差，括号里是有改进的划分数）：

|proxy|W|Δ regret|Δ 判错率|Δ Spearman|三项都改进|
|---|---:|---|---|---|---|
|输入矩|5|+0.00000 (0/5)|+0.0375 (1/5)|−0.030 (0/5)|否|
|输入矩|20|+0.00021 (0/5)|+0.0625 (0/5)|−0.207 (0/5)|否|
|平均梯度|5|+0.00000 (0/5)|+0.0125 (1/5)|+0.042 (4/5)|否|
|平均梯度|20|+0.00021 (0/5)|+0.0500 (0/5)|−0.150 (0/5)|否|
|Δθ|5|+0.00000 (0/5)|+0.0750 (0/5)|−0.082 (2/5)|否|
|Δθ|20|+0.00021 (0/5)|+0.0375 (0/5)|−0.200 (0/5)|否|
|Δθ+m,v|5|+0.00000 (0/5)|+0.0500 (1/5)|−0.072 (2/5)|否|
|Δθ+m,v|20|+0.00021 (0/5)|+0.0750 (0/5)|−0.191 (0/5)|否|
|只看符号|5|−0.00062 (3/5)|+0.0625 (0/5)|−0.052 (0/5)|否|
|只看符号|20|−0.00090 (4/5)|+0.1250 (0/5)|−0.270 (0/5)|否|
|短尾段 1|5|−0.00625 (1/5)|−0.0187 (2/5)|+0.053 (4/5)|是|
|短尾段 1|20|−0.00662 (1/5)|+0.0063 (2/5)|−0.152 (2/5)|否|
|短尾段 4|5|−0.00626 (3/5)|+0.0000 (2/5)|−0.028 (2/5)|否|
|短尾段 4|20|−0.00662 (3/5)|+0.0000 (2/5)|+0.039 (5/5)|否|
|随机|5|−0.00785 (4/5)|−0.0125 (2/5)|−0.024 (2/5)|否|
|随机|20|−0.00996 (4/5)|−0.0875 (3/5)|+0.048 (3/5)|是|

各划分上随机选择的 regret，cold → $`W=5`$ / $`W=20`$（`comparisons[proxy=random]`）：

| 划分 | cold | W=5 | W=20 |
|---|---:|---:|---:|
| 留出状态 | 0.0067 | 0.0042 | 0.0034 |
| 留出任务（`threshold`） | 0.0430 | 0.0098 | 0.0039 |
| 留出 $`\epsilon`$ | 0.0076 | 0.0042 | 0.0035 |
| 留出学习率 | 0.0089 | 0.0086 | 0.0054 |
| 留出 $`\epsilon`$ + 学习率 | 0.0047 | 0.0049 | 0.0049 |

**为什么支持 Rule 3.**

1. *warm-up 压缩了候选之间的差别.* 均匀随机选择的 regret 衡量的是菜单上 gold 对比值的分散程度。它在 5 个划分中的 4 个下降，阈值任务上降得最多（0.043 降到 0.004）：warm-up 之后可选的差别变小了。这与 [Proposition 5.6](../proofs/05_optimizer_state_ZH.md#s5-6) 一致：在 warm 状态上，response 在所述残差以内只依赖于平均预条件梯度，所以主要在符号结构上不同的候选会彼此靠近。
2. *warm-up 没有让静态 proxy 变得可预测.* 除"只看符号"之外的四个静态 proxy，regret 在任何划分上都没有改进，判错率上升 1–8 个百分点，$`W=20`$ 时 Spearman 下降 0.15–0.21。"只看符号"的 regret 略有改进，但符号准确率下降 6–13 个百分点。没有一个静态 proxy 与 $`W`$ 的组合同时改进三项指标。
3. *对方法的含义.* warmth 决定的是哪个统计量携带信息（Rule 3）。这里用的静态 proxy 在两种状态下都不是那个统计量，所以仍然要测量 response，并且要在记录了 warmth 的状态上测量（[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2) 第 1 步与第 5 步）。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C17,C23 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # package check
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/s9.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：条目 `v014:C17`、`controlled_neural.details.regrouping`、`controlled_neural.details.additive_null_parameter_residual`。
- `theory/v014_repro/results_v014_rerun.json`：`coupling_and_additive_null`。
- `theory/v015_checks/results_v015_independent.json`：条目 `v015:E6`（`reports`、`negative_results`、`sign_pair_audit`）与 条目 `v015:E9`（`summary`、`cheap_proxy_results`、`comparisons`、`freeze_events`）。
- 冻结的契约文件：`theory/v015_checks/proxy_split_freeze.json`、`proxy_models_frozen.json`、`proxy_split_freeze_E9.json`、`proxy_models_frozen_E9.json`。
- 规范：`theory/v015_checks/SPEC.md`（`v015:E6`）与 `theory/v015_checks/README.md`（`v015:E9` 方法）。
- 图：`../figures/fig9_1_coupling.{pdf,png}` 与 `../figures/fig9_2_selection.{pdf,png}`（spec 为 `../figures/specs/fig9_1.md`、`fig9_2.md`）。
