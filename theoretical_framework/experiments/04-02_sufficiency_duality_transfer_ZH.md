# Theorem 4.2 · 充分性对偶与迁移

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-02_sufficiency_duality_transfer_ZH.md) · [证明](../proofs/04-02_sufficiency_duality_transfer_ZH.md) · [实验](04-02_sufficiency_duality_transfer_ZH.md) · [English](04-02_sufficiency_duality_transfer.md)

**本页检验什么.** 在随机有限实例上用精确计算检验 Theorem 4.2 的全部四个部分：span 判据 (a)、对偶等式 (b)、迁移界 (c) 与不受限 learner 的情形 (d)。第二个检验在结构化的 response 族上再次检验对偶 (b) 与 span 计数 (a)。

- **结论：PASS.** 在 300 个随机实例上，span 判据与独立的 LP 搜索从未出现分歧；最坏分布对差异等于最优一致逼近误差的两倍，偏差不超过 $`2.2\times10^{-14}`$；针对迁移界的对手最多只超出 $`1.1\times10^{-16}`$；(d) 对 $`m=2,\dots,8`$ 精确成立（[Check 4.1](#c4-1)）。
- 在 21 个结构化子空间上，对偶在 $`1.6\times10^{-15}`$ 内成立（[Check 4.10](04-06_exact_order_not_approximate_complexity_ZH.md#c4-10)，摘要见下）。
- 全部计算都是精确有理数计算或精确 LP；不涉及训练过的网络。

**本页用到的设定.**

- $`\mathcal Y`$ 是大小为 $`m`$ 的有限消息字母表；$`\mathcal Y`$ 上的函数是 $`\mathbb R^m`$ 中的向量；分布 $`P,Q`$ 是概率向量；$`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$。
- $`\mathcal G`$ 是一组 response $`g:\mathcal Y\to[0,1]`$（最终任务回报作为消息的函数，[Definition 1.4](../README_ZH.md#s1-4)）。
- 特征接口传递 $`\phi_1,\dots,\phi_d`$ 的均值；$`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。*特征匹配的分布对*特征均值相等；$`\mathcal M_\phi`$ 是所有这样的分布对的集合。$`g`$ 的 *width*（最坏分布对差异）是 $`\max(P-Q)g`$ 在 $`\mathcal M_\phi`$ 上的值，未分辨差距 $`\mathfrak D_\phi(\mathcal G)`$ 是 $`g\in\mathcal G`$ 上最大的 width；*逼近误差*是 $`\mathrm{dist}_\infty(g,H_\phi)=\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$（[Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)）。
- $`d_0`$ 是使差距为零的最少特征个数；$`\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$。
- 代码 ID：`v014:` 指 `theory/independent_checks`（结果文件 `results_independent_v014.json`）。

**被检验的陈述（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)）.**

- **(a) 精确充分性.** 对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$，当且仅当 $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$。因此 $`d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$。
- **(b) 近似充分性.** $`\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H_\phi)`$；因此用 $`d`$ 个特征能达到的最小未分辨差距是 $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$，下确界取遍 $`H\ni\mathbf 1`$、$`\dim H\le d+1`$ 的子空间。
- **(c) 迁移.** 若每个目标 $`g=c+\sum_\ell b_\ell\phi_\ell+r`$，$`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则对所有 $`P,Q`$，$`\sup_g\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$。
- **(d) 不受限的 learner.** 若 learner 族能在任意事件 $`A\subseteq\mathcal Y`$ 上精确地部署两个 policy（价值 $`a`$ 与 $`a+\gamma`$，$`\gamma>0`$）中较好的一个，则 $`\sup_A\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q)`$，且 $`d_0(\mathcal G)=m-1`$。

<a id="c4-1"></a>

## Check 4.1 · 随机实例上的 span 判据、对偶与迁移（Theorem 4.2）

**结论.** 在 300 个随机有限实例、6,014 条断言上 PASS。span 判据是精确的；对偶在 $`2.2\times10^{-14}`$ 内是等式；针对迁移界的对抗搜索最多只到 $`1.1\times10^{-16}`$；(d) 对 $`m=2,\dots,8`$ 精确成立。

**为什么做.** [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 把"反馈律之间哪些差别有影响"变成线性代数问题。每个部分都有对应的证伪方式，本检验逐一独立搜索：

- (a)：一对特征匹配的分布分离了一个落在 span 内的 response，或者 span 判据漏掉了一个落在 span 外的 response；
- (b)：某个可行分布对的差异超过两倍一致距离 $`2\mathrm{dist}_\infty`$，或者没有任何分布对能取到它——(b) 是等式，不只是上界；
- (c)：某个分布对违反迁移不等式；
- (d)：事件族的上确界不等于 $`\gamma\,\mathrm{TV}`$，或者 response 维数不等于 $`m-1`$。

**设置.**

- 300 个实例：字母表大小 $`m\sim\mathrm U\lbrace 4,\dots,12\rbrace`$，特征数 $`d\sim\mathrm U\lbrace 1,\dots,m-2\rbrace`$。
- 特征表 $`\phi\in\mathbb Q^{d\times m}`$ 和三个 response $`\mathcal G\subset\mathbb Q^m`$ 都取随机有理数。
- 每个实例另检验 $`g=\phi_1`$，作为预期差异为 0 的 span 内对照。
- 迁移实例中 $`b,r`$ 随机，并重新缩放，保证 $`g=\tfrac12+b^\top\phi+r\in[0,1]^m`$。
- (d) 在 $`m=2,\dots,8`$ 上取随机有理分布 $`P,Q`$，并取 $`a=\tfrac15`$、$`\gamma=\tfrac35`$。
- 种子：`SeedSequence([20261010, 5])`。

**做法.**

1. *精确秩.* 用 SymPy 计算 $`[\mathbf 1;\phi]`$ 与 $`[\mathbf 1;\phi;g]`$ 的秩，判断 $`g\in H_\phi`$；同时算 $`d_0=\mathrm{rank}[\mathbf 1;\mathcal G]-1`$。
2. *原问题 LP.* 用 HiGHS 直接以 $`(P,Q)`$ 为变量，在特征匹配的概率向量对上求 $`\max(P-Q)g`$。
3. *逼近 LP.* 用另一个 LP 计算 $`\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$。
4. *(a)(b) 的断言.* 断言 width $`=2\cdot`$ error，并断言 width $`\le10^{-8}`$ 当且仅当 $`g\in H_\phi`$。
5. *迁移 LP.* 在所有分布对上最大化 $`(P-Q)g-B\lVert\Delta\phi\rVert_\infty-2\tau`$，最优值必须 $`\le0`$。
6. *(d).* 在精确有理数下枚举全部 $`2^m`$ 个事件，比较 $`\max_A\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert`$ 与 $`\gamma\cdot\tfrac12\lVert P-Q\rVert_1`$，并检验 $`[\mathbf 1;\lbrace\mathbf 1_A\rbrace]`$ 的秩等于 $`m`$。

**结果.**

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

*Figure 4.1.* 一个精确的五消息例子：图中给出一个 response、它在特征 span 中的最优一致逼近，以及取到两倍逼近误差的特征匹配分布对。第二个面板把原问题 LP 的最优值对两倍逼近误差作图，包括 [Check 4.10](04-06_exact_order_not_approximate_complexity_ZH.md#c4-10) 的全部结构化子空间和本检验的随机实例；所有点都落在对角线上。[PDF](../figures/fig4_1_duality.pdf)

**为什么支持 Theorem 4.2.**

- span 判据与 LP 分离搜索从未出现分歧，所以 (a) 的判据就是精确的检验。
- 两个 LP 是独立求解的：一个在分布对上，一个在逼近函数上。它们的最优值吻合到求解器精度，所以 (b) 是等式。
- 即使对手专门针对 (c) 做优化，也无法让左边超过右边。

**局限.** 字母表大小至多 12；定理本身也是对有限字母表陈述的。

### [Check 4.10](04-06_exact_order_not_approximate_complexity_ZH.md#c4-10) · 最高次数相同，response 空间的复杂度不同

- *对 Theorem 4.2 检验什么.* 在结构化、非随机的 response 族上检验对偶 (b)，以及 (a) 对一个给定的族所预测的 span 维数。
- *设置.* 网格 $`\lbrace0,\tfrac14,\dots,1\rbrace^3`$（125 条消息）；三个最高次数为 3 的族（标量仿射幂、多元 parity 拉回、结构化求和幂）；对每个特征预算 $`d=1,\dots,7`$ 贪心选出一个子空间，用两个独立的 LP 分别计算一致逼近误差与最坏分布对差异。
- *关键数字.* 在全部 21 个（族, $`d`$）子空间上，$`\max\lvert\text{discrepancy}-2\cdot\text{uniform error}\rvert=1.6\times10^{-15}`$。含常数的精确 span 维数为 4（标量）、8（parity）、4（结构化），正是 span 判据预测的数值。
- *结论.* PASS（21 个子空间、1,445 条断言；来源条目 `v014:C19`）。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C5,C19 \
  --output theory/independent_checks/subset.json
```

`--only` 保留每个 claim 自己的随机流（`SeedSequence([20261010, claim number])`），所以只跑子集也能复现完整运行的数字。

本检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）、做精确有理数运算的 SymPy（[Meurer et al., 2017](#ref-meurer2017sympy)），以及 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）的 `linprog` 配合 HiGHS 求解器（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解线性规划。

## 产物

- 结果：`theory/independent_checks/results_independent_v014.json`（条目 `v014:C5`、`v014:C19`）。
- 代码：`theory/independent_checks/check_v014_independent.py`（`check_C5`、`check_C19`）；规格说明 `theory/independent_checks/SPEC.md`。
- 图的规格：`theoretical_framework/figures/specs/fig4_1.md`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-meurer2017sympy"></a>

**Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, Anthony Scopatz (2017).** SymPy: Symbolic computing in Python. *PeerJ Computer Science* 3, e103. DOI [10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
