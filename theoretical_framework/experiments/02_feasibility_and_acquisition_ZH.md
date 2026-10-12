# §2 可行性与信息获取 — 实验

[← 框架总览](../README_ZH.md) · [证明](../proofs/02_feasibility_and_acquisition_ZH.md) · [English](02_feasibility_and_acquisition.md)

**结论.**

- [Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)、[Proposition 2.3](../proofs/02_feasibility_and_acquisition_ZH.md#s2-3)、[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)：通过。在 500 个随机有限决策问题上用精确有理数运算检验（5,810 条断言）：分解的残差为 0；零信息判据在两个方向上都与直接计算一致；220 个实例上原始 LP 与对偶 LP 相等；任何 garbling，包括对抗选出的，都没有提高 Bayes 值。
- [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)、[Proposition 2.7](../proofs/02_feasibility_and_acquisition_ZH.md#s2-7)：通过。在 25 组 $`(\Delta,\delta)`$ 上（540 条断言，精确二项和与精确随机游走递推），没有任何策略低于下界 (2.6)。调好参数的 SPRT 在两组设定上恰好达到它（比值 1.000），所以一般而言它的常数不能改进。编译所需的标签数只依赖于 $`\delta_L`$。
- [Lemma 2.9](../proofs/02_feasibility_and_acquisition_ZH.md#s2-9)、[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)：通过。在全部 100 个穷举情形中，Bayes 最优风险等于 (2.7)，最优的按值自适应查询树等于按权重的贪心查询；其中 80 个情形的揭示集合含多个比特。把正部移到期望外面，在 100 个情形中有 36 个是错的。
- [Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11)：通过。预算不等式 (2.11) 在 297 个自适应查询–停止过程上都成立，并且能取等（比值 1.0）。
- replay（[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)）在全部 944 个精确格点上收益恰为 0；实际的 Adam 与 SGD 表格学习器从未低于 [Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10) 的信息下限。

| 检验 | 检验对象 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 2.1](#c2-1) | [Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)、[Propositions 2.3–2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-3) | `v014:C1` | 通过 |
| [Check 2.2](#c2-2) | [Proposition 2.3](../proofs/02_feasibility_and_acquisition_ZH.md#s2-3) | `pkg:robust_ambiguity_minimax` | 通过 |
| [Check 2.3](#c2-3) | [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)、[Proposition 2.7](../proofs/02_feasibility_and_acquisition_ZH.md#s2-7) | `v014:C2` | 通过 |
| [Check 2.4](#c2-4) | [Lemma 2.9](../proofs/02_feasibility_and_acquisition_ZH.md#s2-9)、[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10) | `v015:V5` | 通过 |
| [Check 2.5](#c2-5) | [Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11) | `v015:V6` | 通过 |
| [Check 2.6](#c2-6) | [Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)、[Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2) 的算法项 | `v015:E5` | 通过 |
| [Check 2.7](#c2-7) | [Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11) | `pkg:T09`、`pkg:T10`、`pkg:T11` | 通过 |

<a id="c2-1"></a>

## Check 2.1 · regret 分解、歧义判据、对偶与 data processing（Theorem 2.2，Propositions 2.3–2.4）

**结论.** 通过：500 个随机有限决策问题，5,810 条断言，没有违反。每个有理数恒等式的残差都恰为 0；最大的浮点残差（来自 LP 求解）为 $`1.33\times10^{-15}`$。

**为什么做.**

- 分解 (2.1) 是恒等式。$`B(D)`$、后验或学习器 kernel 的定义若有任何错误，都会表现为非零残差。
- 歧义判据 (2.2) 是"当且仅当"，任一方向的反例都能否定它。
- 对偶 (2.4) 与 data processing 不等式 (2.5) 若缺了某个假设，随机或对抗构造的实例就可能违反它们。

**设置.**

- 500 个随机实例：3–6 个环境，随机有理先验；$`\Pi`$ 中 2–5 个策略，$`\Pi_{\rm ref}`$ 再多两个策略。
- 值 $`J_E(\pi)\in[0,1]`$ 是分母为 12 的随机有理数。transcript 字母表大小 $`\lvert\mathcal D\rvert=4`$，信道 $`P(D\mid E)`$ 与学习器 kernel $`\lambda(\cdot\mid D)`$ 随机。
- 四分之一的实例强制带平局的共同最优（对所有 $`E`$ 有 $`J_E(\pi_1)=J_E(\pi_2)=1`$）；四分之一强制各环境最优策略不同。
- 精确运算（`fractions.Fraction`）。LP 用 HiGHS 求解，容差 $`10^{-8}`$。种子 `SeedSequence([20261010, 1])`。

**做法.** 对每个实例：

1. 计算 $`\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}`$；断言三者之和恰为总 regret，且各自 $`\ge0`$。
2. 判断是否对每个 $`D`$ 都存在共同的后验最优策略；断言这恰好在 $`\mathcal R_{\rm info}=0`$ 时成立。
3. 让 $`D`$ 经过 3 符号信道 garbling：一个随机的随机化信道；在前 30 个实例上还用全部 $`3^4=81`$ 个确定性信道（凸目标的极点）。断言 $`B(D')\le B(D)`$。
4. 用独立的 $`Z`$ 扩充 $`D`$，断言 $`B(D,Z)=B(D)`$ 精确成立。
5. 在 220 个实例上分别求解原始的"对混合策略取最小" LP (2.3) 与对偶的"对先验取最大" LP (2.4)；断言两者相等，且值为 $`0`$ 恰好在存在共同最优时成立。

**结果.**

| 量 | 值 | JSON 键（该检验的条目） |
|---|---|---|
| 实例数 / 断言数 | 500 / 5,810 | `n_instances`、`.n_assertions` |
| 最大残差（恒等式、LP 相等） | $`1.33\times10^{-15}`$ | `worst_residual` |
| 最大 LP 可行性残差 | $`1.33\times10^{-15}`$ | `worst_lp_feasibility_residual` |
| 分别求解原始/对偶 LP 的实例数 | 220 | `details.minimax_instances` |
| 所有 garbling 中最大的 $`B(D')-B(D)`$ | $`0.0`$ | `details.adversarial_garbling_best_lhs_minus_rhs` |
| 状态 | 通过 | `status` |

![Figure 2.1：regret 分解、garbling 与 minimax 对偶](../figures/fig2_1_decomposition.png)

*Figure 2.1.* (a) 一个三环境有限问题上四个学习器的精确 regret 分解：数据相同时，换学习器只改变算法项。(b) 让 transcript 经过二元对称信道 garbling，信息 regret 单调上升，从原始数据的 0.1217 升到无数据时的 0.1833。(c) Check 2.2 的两世界实例的稳健歧义值：对混合策略取最小与对先验取最大都等于 0.1。[PDF](../figures/fig2_1_decomposition.pdf)

**为什么支持这些陈述.**

- 在每个实例上分解都精确成立、每一项都非负，与 Theorem 2.2(a) 一致。
- 零信息判据在两个方向上都与直接计算一致，包括平局情形（Theorem 2.2(b)）。
- 没有任何 garbling，包括对抗选出的极点信道，提高了 Bayes 值（Proposition 2.4）；对偶 LP 总是达到原始值（Proposition 2.3）。
- 残差处于浮点 LP 求解的水平；每个有理数恒等式的残差都恰为零。

<a id="c2-2"></a>

## Check 2.2 · 典型的两世界歧义实例（Proposition 2.3）

**结论.** 通过：原始值与对偶值都是 0.1，minimax 混合策略与最不利先验都是 $`(0.5,0.5)`$。

**为什么做.** 一个可以手算的 Proposition 2.3 实例，最优混合有闭式。

**设置.** 两个环境偏好相反的策略，regret 矩阵 $`M=\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$。原始与对偶 LP 由 v0.14 包的验证脚本用 SciPy HiGHS 求解，种子 2026101014。

**结果.**

| 量 | 值 | JSON 键（该检验的条目） |
|---|---|---|
| 原始 $`r(C)`$ | 0.1 | `robust_ambiguity_minimax.primal_regret` |
| 对偶值 | 0.1 | `robust_ambiguity_minimax.dual_regret` |
| minimax 策略混合 | $`(0.5,0.5)`$ | `.policy_mixture` |
| 最不利先验 | $`(0.5,0.5)`$ | `.environment_mixture` |

**为什么支持这个陈述.**

- 两个世界无法区分时，每个学习器的输出分布在两个世界中相同，能达到的最好最坏期望 regret 是 $`r(C)=0.1>0`$，是押注任一策略时 regret 0.2 的一半（Proposition 2.3(a)）。
- 这个值是*混合平均*的 regret，正如 Proposition 2.4 之后的说明所指出的：单个抽样策略的 regret 是 0 或 0.2。

<a id="c2-3"></a>

## Check 2.3 · 两世界族中的查询复杂度与编译（Theorem 2.6，Proposition 2.7）

**结论.** 通过：25 组设定 $`\Delta\in\lbrace \tfrac18,\tfrac1{10},\tfrac1{16},\tfrac1{32},\tfrac1{64}\rbrace \times\delta\in\lbrace 0.01,0.05,0.1,0.25,0.4\rbrace`$，540 条断言，最大残差 0。没有任何精确计算的策略低于 (2.6)；SPRT 在 $`(\Delta,\delta)=(\tfrac18,\tfrac1{10})`$ 与 $`(\tfrac18,\tfrac14)`$ 上恰好落在界上。

**为什么做.**

- Theorem 2.6(b) 是对*所有*自适应策略的下界。一个在两个世界中误差都 $`\le\delta`$、期望查询数更少的策略就能否定它。
- Theorem 2.6(c) 与 Proposition 2.7 是充分性陈述。在所述样本量下精确误差超过 $`\delta`$（或 $`\delta_L`$）就能否定它们。
- 常数 64 依赖于 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$，这一点做了解析与数值核对。

**设置.**

- 精确的有理二项尾概率，用整数分子求和，不用正态近似。
- SPRT 的期望样本量与误差，通过对边界为 $`\pm h`$ 的吸收随机游走做精确三对角消元得到，$`h=\lceil\log\frac{1-\delta}{\delta}/\log\frac{1/2+2\Delta}{1/2-2\Delta}\rceil`$。
- 回报 $`J_v(p)`$ 与被动 minimax 用 SymPy 计算。
- KL 不等式在 $`(0,\tfrac14]`$ 上 20,001 个点的稠密网格上核对，另用有界标量优化器对抗地最大化比值。
- 种子 `SeedSequence([20261010, 2])`；检验是确定性的。

**做法.**

1. 找到最小的奇数 $`n`$，使阈值检验在两个世界中成功概率都 $`\ge1-\delta`$（平局算失败）；核对 $`n-1`$ 与 $`n-2`$ 失败。
2. 计算 SPRT 的精确期望停时与精确误差。
3. 断言两种停时都 $`\ge`$ 界 (2.6)，并且 (2.6) 的第一种形式 $`\ge`$ 其放松形式 $`\mathrm{kl}(1-\delta,\delta)/(64\Delta^2)`$。
4. 断言在 $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 处精确误差 $`\le\delta`$。
5. 编译：对 $`q\in\lbrace \tfrac18,\tfrac78\rbrace`$，计算 $`m=\lceil32\log(2/\delta_L)\rceil`$ 时 $`\lvert\widehat p-q\rvert>\tfrac18`$ 的精确概率；断言它 $`\le\delta_L`$，且每个落在容差内的结果 regret $`\le\Delta/2`$。

**结果（部分行；全部 25 行在 JSON 中）.**

| $`\Delta`$ | $`\delta`$ | 界 (2.6) | 最优固定 $`n`$ | SPRT $`\mathbb E N`$ | SPRT / 界 | $`n_{\rm query}`$ | $`n_{\rm query}`$ 处的精确误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1/8 | 0.01 | 8.198 | 19 | 9.918 | 1.210 | 37 | 5.1e-4 |
| 1/8 | 0.1 | 3.200 | 7 | 3.200 | **1.000** | 19 | 8.9e-3 |
| 1/8 | 0.25 | 1.000 | 1 | 1.000 | **1.000** | 12 | 5.4e-2 |
| 1/16 | 0.01 | 35.262 | 85 | 35.282 | 1.0005 | 148 | 1.3e-3 |
| 1/32 | 0.05 | 84.356 | 173 | 87.030 | 1.032 | 384 | 8.0e-3 |
| 1/64 | 0.01 | 575.661 | 1383 | 580.574 | 1.0085 | 2358 | 1.3e-3 |
| 1/64 | 0.1 | 224.703 | 421 | 233.224 | 1.038 | 1179 | 1.6e-2 |

| 汇总量 | 值 | JSON 键（该检验的条目） |
|---|---|---|
| 设定数 / 断言数 | 25 / 540 | `n_instances`、`.n_assertions` |
| SPRT / 界 的最小比值 | 1.0 | `details.minimum_sprt_ratio` |
| 固定 $`n`$ / 界 的最小比值 | 1.0（在 $`\Delta=\tfrac18,\delta=0.25`$） | `details.configurations[*].fixed_ratio` |
| $`(0,\tfrac14]`$ 上 $`\sup_u\mathrm{kl}(\tfrac12+u,\tfrac12-u)/u^2`$ 的解析值 | $`8\log3=8.789`$ | `details.analytic_kl_sup` |
| 优化器找到的最大比值 | 8.789（小于 16） | `details.optimizer_kl_ratio` |
| 编译所需 $`m`$（$`\delta_L=0.01,0.05,0.1,0.25,0.4`$） | 170、119、96、67、52 | `details.compilation[*].m` |
| 这些 $`m`$ 下的精确编译误差 | 4.2e-6、1.2e-4、2.4e-4、3.0e-3、3.9e-3 | `details.compilation[*].error` |
| 最大残差 | 0 | `worst_residual` |

![Figure 2.2：查询次数与下界之比，以及被动不可能](../figures/fig2_2_queries.png)

*Figure 2.2.* (a) $`\delta=0.05`$ 时精确查询次数随 $`1/\Delta^2`$ 的变化：SPRT 贴着 KL 下界 (2.6)，最小的成功固定样本检验与 Theorem 2.6(c) 的充分 $`n_{\rm query}`$ 在它上方，没有策略落到它下方。(b) 全部 25 组设定上各策略与界的比值；SPRT 在其中两组达到比值 1。(c) $`\Delta=1/8`$ 时，两个被动 regret $`2\Delta(1-m)`$ 与 $`2\Delta m`$ 随平均输出 $`m`$ 此消彼长，所以任何被动学习器的最坏情况至少为 $`\Delta`$。[PDF](../figures/fig2_2_queries.pdf)

**为什么支持这些陈述.**

- *被动不可能（Theorem 2.6(a)）.* 符号计算得到的回报恰为 $`\tfrac34+v\Delta(2p-1)`$，两个被动 regret 恰好在值 $`\Delta`$ 处相交。
- *下界（Theorem 2.6(b)）.* 没有精确计算的策略能打破这个界。SPRT 对这个问题接近最优，在两组设定上落在界上；在 $`\Delta=\tfrac1{64}`$、$`\delta\le0.25`$ 时与界相差不到 5%（比值 1.009–1.047）。所以这个界的常数也是对的，不只是阶对。$`\delta\le0.25`$ 时，除平凡的 $`n=1`$ 外，固定样本检验需要界的 1.6–2.4 倍，这对非序贯设计是预期之中的。
- *上界（Theorem 2.6(c)）.* 在 $`n_{\rm query}`$ 处，每组设定的精确误差都远低于 $`\delta`$：Hoeffding 是保守的。
- *编译（Proposition 2.7）.* 编译标签数为 $`O(\log(1/\delta_L))`$，与 $`\Delta`$ 无关；精确尾概率比 $`\delta_L`$ 低两到四个数量级。
- 两项代价，即 $`\Theta(\Delta^{-2}\log(1/\delta))`$ 次真实查询与 $`O(\log(1/\delta_L))`$ 个编译标签，因此分别得到了确认。

<a id="c2-4"></a>

## Check 2.4 · 隐藏比特的 coverage、硬查询预算与自适应性（Lemma 2.9，Theorem 2.10）

**结论.** 通过：348 个实例，26,214 条断言，最大残差 0。在全部 100 个穷举情形（$`d\le5`$、$`K\le4`$、$`n\le3`$）中，Bayes 最优风险等于 (2.7)/(2.9)，最优的按标签自适应查询决策树等于按权重的贪心规则。

**为什么做.**

- Theorem 2.10(a) 断言：无论同一条轨迹内的揭示事件如何相关，Bayes 风险都是 $`\tfrac12\sum_jw_j(1-q_j)^K`$。
- Theorem 2.10(b) 断言：没有按标签自适应的查询策略能胜过按权重贪心；并且正部必须留在期望里面。
- 若穷举得到的 Bayes 值或决策树值与公式不同，或出现 $`\mathbb E[(M-n)_+]<(\mathbb EM-n)_+`$ 的实例，就能否定这些陈述。

**设置.**

- 100 个情形：$`d=1,\dots,5`$（各 20 个）、$`K=0,\dots,4`$、$`n=0,\dots,3`$，随机有理权重 $`w`$，随机的位掩码揭示集合分布。
- 穷举全部 $`2^d`$ 个比特世界与每一种有序揭示历史；最大的情形有 2,321 条 transcript。按后验取 Bayes 动作。按标签的查询决策树精确最小化；最大的有 271 个查询节点。
- 均匀占据下 $`M_K`$ 的分布用两种独立方法计算（子集递推与容斥），$`d=2,\dots,5`$、$`K\le4`$。
- 对抗爬山搜索（8 次重启 × 20 步）寻找违反 Jensen 不等式或贪心最优性的实例。
- 种子 `SeedSequence([150150, 5])`；全程 `fractions.Fraction`。

**结果.**

| 量 | 值 | JSON 键（条目 `v015:V5`） |
|---|---|---|
| 实例数 / 断言数 / 最大残差 | 348 / 26,214 / 0 | `n_instances`、`n_assertions`、`worst_residual` |
| Bayes 风险等于 (2.7) 的情形 | 100 / 100 | `data.cases[*].Bayes` 对 `.A11` |
| 自适应最优等于贪心的情形 | 100 / 100 | `data.cases[*].adaptive` 对 `.greedy` |
| 揭示集合含多个坐标的情形 | 80 / 100 | `data.cases[*].reveal`（位掩码键） |
| $`\mathbb E[(M-n)_+]>(\mathbb EM-n)_+`$ 严格成立的情形 | 36 / 100 | `data.cases[*].E_positive` 对 `.positive_E` |
| 严格性的见证 | $`d=2,K=1,n=1`$：$`\mathbb E[(M-1)_+]=\tfrac12`$，而 $`(\mathbb EM-1)_+=0`$；风险 $`\tfrac5{56}`$ | `data.strict_Jensen` |
| 两种占据计算一致；均匀曲线 (2.10) 精确 | 所有 $`d\le5`$、$`K\le4`$、$`n\le3`$ | `data.uniform` |
| 对抗搜索找到的最大违反 | $`0`$ | `data.adversarial.best.score` |

**为什么支持这些陈述.**

- 信息下限与最优查询值恰好是 Theorem 2.10 的闭式，包括同一条轨迹内揭示事件相关的情形（Lemma 2.9 允许这种相关）。
- 按值自适应的查询从未带来收益，与 Lemma 2.9 的独立性论证一致。
- 把正部移到期望外面在超过三分之一的情形中是错的，所以 (2.9) 中的期望不能省。

<a id="c2-5"></a>

## Check 2.5 · 示范信息抵扣自适应查询预算（Proposition 2.11）

**结论.** 通过：297 个实例，2,969 条断言，最大残差 $`2.7\times10^{-15}`$。不等式 (2.11) 从未被违反，并且单个有信息的查询能使它取等（比值 1.0）。

**为什么做.** Proposition 2.11 必须对*每个*共同的、依赖历史的查询–停止规则成立。出现 $`\mathrm{kl}(1-\delta,\delta)>K\kappa_D+\kappa_X\mathbb E_vN`$ 的过程就能否定它；若比值处处远小于 1，说明界是松的。

**设置.**

- 两世界过程：$`K\in\lbrace 1,2\rbrace`$ 条对称的有信息示范，$`N_{\max}\le4`$，两个随机二元查询信道，两个世界共用的随机化、依赖历史的选择与停止规则。
- 精确的有理 transcript 概率。transcript KL 直接计算，并单独按自适应链式和计算；对数用 float64，容差 $`10^{-8}`$。
- 160 个随机过程，对抗爬山（8 次重启 × 16 步），以及两个构造情形：紧的情形（$`K=0`$，一次二元查询，信道 $`(\tfrac45,\tfrac15)`$，$`\delta=\tfrac15`$）与零信息情形（$`\kappa_D=\kappa_X=0`$）。
- 种子 `SeedSequence([150150, 6])`。

**结果.**

| 量 | 值 | JSON 键（条目 `v015:V6`） |
|---|---|---|
| 实例数 / 断言数 / 最大残差 | 297 / 2,969 / 2.7e-15 | `n_instances`、`n_assertions`、`worst_residual` |
| 满足前提 $`0<\delta<\tfrac12`$ 的随机过程 | 159 / 160 | `data.random_premise_count` |
| 随机过程中比值 $`\mathrm{kl}(1-\delta,\delta)/(K\kappa_D+\kappa_X\mathbb EN)`$ 的最大值 | 0.9973 | `data.worst_random.score` |
| 对抗搜索中的最大比值 | 0.9760 | `data.adversarial.best.score` |
| 构造的紧情形的比值 | **1.0**（KL $`=\mathrm{kl}(\tfrac45,\tfrac15)=0.8318`$，$`\mathbb EN=1`$） | `best_ratio`、`data.tight` |
| 零信息情形 | transcript KL $`=0`$；两个错误概率之和为 1 | `data.zero_information` |

**为什么支持这个陈述.**

- KL 链式恒等式（直接算的 transcript KL 等于累积的链式和）在自适应停止下精确成立，这是证明的核心。
- (2.11) 从未被违反，单个有信息的查询使它取等，所以它的常数不能改进。
- 示范与查询都不携带信息时，一致识别不可能，与 Proposition 2.11 的最后一句一致。

<a id="c2-6"></a>

## Check 2.6 · 示范数曲线、replay 与实际学习器（Theorem 2.10，Proposition 2.4，Theorem 2.2）

**结论.** 通过：944 个精确的 $`(K,\text{查询})`$ 格点与 5,664 行实际学习器结果，7,450 条断言，最大残差 $`5.6\times10^{-17}`$。每个格点上 replay 收益都为 0；预算相同时，定向查询胜过额外的被动示范；没有实际学习器低于信息下限。

**为什么做.** 这个检验把理论中分开的三件事分开看：

- 独立的新证据（更多轨迹，或定向查询）；
- 对已见证据的 replay，由 Proposition 2.4，它的信息价值为零；
- 信息是否可得与学习器实际用了多少之间的差，即 Theorem 2.2 的算法项。

**设置.**

- $`R_{\rm info}(K,\cdot)`$ 的精确曲线：$`m\in\lbrace 4,8,16\rbrace`$ 个上下文，均匀与非均匀的 $`q`$ 和 $`w`$，$`K=0,\dots,32`$，查询预算 $`0,1,2,4`$；用整数子集 Möbius 反演计算，风险为有理数。
- 预算相同的三种访问方式：*主动*（查询指向未见上下文）、*被动*（同样的预算用于额外的随机示范）、*replay*（重新展示已见标签）。
- 一个实际的表格学习器（每个上下文一个 logit），在已揭示的比特上用可分的求和交叉熵、Adam 或 SGD 训练 12 次更新；未见上下文的 logit 保持为 0，部署时按 Bernoulli 采样动作。
- 种子 `SeedSequence([150150, 105])`。

**结果.**

| 量 | 值 | JSON 键（条目 `v015:E5`） |
|---|---|---|
| 精确格点数 / 断言数 / 最大残差 | 944 / 7,450 / 5.6e-17 | `data.curves`、`n_assertions`、`worst_residual` |
| replay 收益 | 944 个格点上全为 $`0`$ | `data.curves[*].replay_gain` |
| $`m=8`$ 均匀，$`K=4`$：不查询 / 4 条被动示范 / 4 次主动查询 | $`0.2931`$ / $`0.1718`$ / $`0.0431`$（$`\tfrac{2401}{8192}`$、$`\tfrac{5764801}{33554432}`$、$`\tfrac{353}{8192}`$） | `data.curves[m=8,uniform,K=4]` |
| $`m=8`$ 均匀，$`K=8`$：不查询 / 4 次主动查询 | $`0.1718`$ / $`0.0013`$ | 同上 |
| 实际学习器行数 | 5,664（Adam、SGD × 主动、被动、replay） | `data.actual_learners` |
| $`\min(\text{学习器风险}-\text{信息下限})`$ | $`0.0`$（从未低于下限） | `data.actual_learners[*].risk`、`.information_floor` |
| 12 次更新后在*已揭示*比特上的平均错误率 | Adam 0.239，SGD 0.372 | `data.actual_learners[*].observed_bit_error` |

![Figure 2.3：信息下限随 K 的变化、实际学习器在下限之上，以及 KL 预算](../figures/fig2_3_floor.png)

*Figure 2.3.* (a) 隐藏比特的精确信息风险随 $`K`$ 的变化（$`m=8`$，均匀）：4 次定向查询使风险下降得远快于 4 条额外示范；4 次 replay 与"只有 $`K`$ 条示范"即 $`\tfrac12(7/8)^K`$ 重合。(b) 每个实际的 Adam 或 SGD 表格学习器都在信息下限之上或恰在下限上；超出的部分是它的算法误差。(c) Check 2.7 的包构造中，示范 KL 加查询 KL 总能覆盖 $`\mathrm{kl}(1-\delta,\delta)`$，与 (2.11) 一致：二元 KL 下界 ≤ 精确 transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$。[PDF](../figures/fig2_3_floor.pdf)

**为什么支持这些陈述.**

- 每条精确曲线都等于 (2.7) 与 (2.9)。
- replay 在每个格点上都不改变下限（Proposition 2.4）。
- 预算相同时，定向查询消除的风险远多于额外的被动示范：$`K=4`$、$`m=8`$ 时，4 次主动查询降到 $`0.043`$，被动示范只到 $`0.172`$。这正是 Theorem 2.10 中的 coverage 效应。
- 实际学习器始终在下限之上。超出的部分来自对已揭示比特拟合得不完美，即 Theorem 2.2 的算法项 $`\mathcal R_{\rm alg}`$；所以信息是否可得与信息是否被用上，可以分别看到。

<a id="c2-7"></a>

## Check 2.7 · 独立的包构造（Theorem 2.10，Proposition 2.11）

**结论.** 通过：理论包中三个独立构造都与 (2.7)、(2.9)、(2.10) 和 (2.11) 一致。

**为什么做.** 这些实现独立于 Checks 2.4–2.6，构造也不同。

**设置与结果.**

| 检验 | 构造 | 结果 | JSON 键（`docs/direction/v015_revision/results_v015.json`） |
|---|---|---|---|
| `pkg:T09` | $`d=3`$，$`w=(\tfrac12,\tfrac13,\tfrac16)`$，揭示集合 $`\varnothing,\lbrace 0,1\rbrace ,\lbrace 1,2\rbrace ,\lbrace 0,2\rbrace`$，概率 $`\tfrac1{10},\tfrac2{10},\tfrac3{10},\tfrac4{10}`$（$`q=(\tfrac35,\tfrac12,\tfrac7{10})`$）；完整后验枚举与决策树最小化，$`K\le4`$，$`n\le3`$ | 20/20 个情形等于公式。例如 $`K=0`$：$`\tfrac12,\tfrac14,\tfrac1{12},0`$；$`K=4`$：$`\tfrac{2099}{120000},\tfrac1{40000},\tfrac1{120000},0`$ | `checks.T09_exact_posterior_and_adaptive_query_optimum.exact_curves` |
| `pkg:T10` | 均匀 $`d=4`$，$`K\le6`$，$`n\le4`$ | 35/35 精确。$`K=3,n=2`$ 时正确风险为 $`\tfrac1{128}`$；把正部移到期望外面的公式给出 0 | `checks.T10_uniform_coverage_and_positive_part` |
| `pkg:T11` | 两世界，示范分布 $`(\tfrac3{10},\tfrac7{10})`$，两个查询信道，基于得分的停止，$`N_{\max}=4`$，$`K\le4`$ | 10/10 个方向行满足：二元 KL 下界 ≤ 精确 transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$，且 $`\mathbb EN`$ ≥ 界 (2.12)。下界/上界之比 0.37–0.56 | `checks.T11_adaptive_query_KL_enumeration.rows` |

**为什么支持这些陈述.** 又有三个构造（非均匀权重、多比特揭示集合、基于得分的停止）与 Theorem 2.10 和 Proposition 2.11 一致。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1,C2 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5,V6,E5 --output theory/v015_checks/quick/sec2.json
theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py --output theory/v014_repro/results_v014_rerun.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`v014:C1` 与 `v014:C2` 的条目（键 `details.configurations`、`details.compilation`、`details.analytic_kl_sup`、`details.optimizer_kl_ratio`、`details.minimum_sprt_ratio`）。
- `theory/independent_checks/SPEC.md`（`v014:C1`、`v014:C2` 两节）与 `theory/independent_checks/check_v014_independent.py`（`check_C1`、`check_C2`、`binomial_tail`、`walk_recursion`）。
- `theory/v014_repro/results_v014_rerun.json`：键 `robust_ambiguity_minimax`。
- `theory/v015_checks/results_v015_independent.json`：`v015:V5`、`v015:V6`、`v015:E5` 的条目；代码 `theory/v015_checks/mathematics.py`（`check_V5`、`check_V6`）与 `theory/v015_checks/experiments.py`（`check_E5`）。
- `docs/direction/v015_revision/results_v015.json`：`checks.T09_*`、`checks.T10_*`、`checks.T11_*`。
- 图：`figures/fig2_1_decomposition.{png,pdf}`、`figures/fig2_2_queries.{png,pdf}`、`figures/fig2_3_floor.{png,pdf}`；规格见 `figures/specs/fig2_1.md`、`fig2_2.md`、`fig2_3.md`。
