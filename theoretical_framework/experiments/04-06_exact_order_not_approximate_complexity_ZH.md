# Proposition 4.6 · 精确阶不等于近似复杂度

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-06_exact_order_not_approximate_complexity_ZH.md) · [证明](../proofs/04-06_exact_order_not_approximate_complexity_ZH.md) · [实验](04-06_exact_order_not_approximate_complexity_ZH.md) · [English](04-06_exact_order_not_approximate_complexity.md)

**本页检验什么.** 精确计算 Proposition 4.6 的 Chebyshev 恒等式 (4.7)、精确的二项尾截断误差以及 parity 的 minimax 值；然后比较三个最高次数相同的 response 族，说明决定所需统计量个数的是任务结构，而不是次数。

- **结论：PASS.** $`p^N`$ 的截断误差等于二项尾概率；要达到 $`10^{-3}`$ 的精度，$`N=4,8,16,32`$ 时只需 $`q=4,6,9,13`$；parity 在次数低于 $`N`$ 时误差恰为 $`1/2`$（[Check 4.9](#c4-9)）。
- 最高次数同为 3 时，结构化的族需要 3 个统计量，parity 需要 7 个（[Check 4.10](#c4-10)）。同一检验还再次检验了 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(b) 的对偶，以及 Theorem 4.2(a) 与 [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(d) 的 span 维数。
- 全部计算都是精确有理数、符号计算或精确 LP；不涉及训练过的网络。

**本页用到的设定.**

- 标量任务 $`J(p)=p^N`$，定义在 $`[0,1]`$ 上（一个 context 访问 $`N`$ 次，全部动作为 1 时成功）。parity 任务 $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$，定义在 $`[0,1]^N`$ 上（动作 1 的个数为偶数时回报 1）。
- $`T_j`$ 是第一类 Chebyshev 多项式，$`T_j(\cos\theta)=\cos(j\theta)`$（[Rivlin, 1990](#ref-rivlin1990chebyshev)）；$`h_q`$ 是 (4.7) 截断到 $`q`$ 阶的结果；$`\mathcal P_q`$ 是总次数不超过 $`q`$ 的多项式集合；$`B\sim\mathrm{Bin}(2N,\tfrac12)`$。
- 对有限消息网格上的 response 族 $`\mathcal G`$ 与包含常数的函数子空间 $`H`$，*一致误差*是 $`\max_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$，*最坏分布对差异*是在 $`H`$ 上一致的分布对之间最大的 $`(P-Q)g`$（[Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)）。特征预算 $`d`$ 指 $`\dim H=d+1`$。
- 代码 ID：`v014:` 指 `theory/independent_checks`（结果文件 `results_independent_v014.json`）；`pkg:` 指修订包验证脚本，重新运行的结果在 `theory/v014_repro/results_v014_rerun.json`。

**被检验的陈述（[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)）.** 对 $`p\in[0,1]`$，

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a)** 对 $`q<N`$，$`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$；矩匹配到 $`q`$ 阶的分布，以及带 IID 标量块的凸仿射 learner 的学习价值，在 $`p^N`$ 上至多相差 $`4e^{-q^2/N}`$。
- **(b)** 对每个 $`q<N`$，$`\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12`$，由 $`h\equiv\tfrac12`$ 取到。

<a id="c4-9"></a>

## Check 4.9 · Chebyshev 恒等式、精确尾概率与 parity minimax（Proposition 4.6）

**结论.** PASS（136 个实例、484 条断言，最大残差 $`1.0\times10^{-17}`$）。对 $`N=1,\dots,16`$ 与每个 $`q<N`$，$`p^N`$ 的截断误差都等于二项尾概率，并且低于 $`2e^{-q^2/N}`$（尾概率与界之比最大为 0.430）。一个扩展到 $`N=32`$ 的独立实现发现，要达到 $`10^{-3}`$ 的精度，$`N=4,8,16,32`$ 时只需 $`q=4,6,9,13`$。parity 的最佳次数 $`<N`$ 逼近误差恰为 $`1/2`$。

**为什么做.** [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)(a) 有三条可证伪的断言：Chebyshev 展开 (4.7) 是恒等式；截断误差等于二项尾概率 $`\Pr\lbrace\lvert B-N\rvert>q\rbrace`$，而不只是以它为上界；这个尾概率至多为 $`2e^{-q^2/N}`$。(b) 断言 parity 的最佳低次逼近误差恰为 $`1/2`$，找到更小的值就能将其证伪。

**设置.**

- 标量任务：$`N=1,\dots,16`$，以及所有 $`q<N`$。
- parity：$`N=1,\dots,5`$，在全部 $`2^N`$ 个立方体顶点上；每个 $`N`$ 另取 10 个随机有理点检验 $`\lvert J_{\rm even}-\tfrac12\rvert\le\tfrac12`$。
- 种子：`SeedSequence([20261010, 12])`。
- 独立实现（`pkg:scalar_approximation`）：$`N\in\lbrace 4,8,16,32\rbrace`$，$`q=0,\dots,N`$，8,001 点网格。

**做法.**

1. 用 SymPy 符号验证 Chebyshev 恒等式 (4.7)：$`\mathrm{expand}(\text{RHS}-p^N)=0`$。
2. 对每个 $`(N,q)`$：
   - 在含 $`p=1`$ 的 3,001 点网格上计算 $`\max\lvert p^N-h_q\rvert`$，断言它等于系数尾和；
   - 用精确有理数断言系数尾和等于二项尾概率；
   - 断言尾概率 $`\le2e^{-q^2/N}`$。
3. parity：用所有次数 $`<N`$ 的非常数单项式求解 minimax LP $`\min_{h\in\mathcal P_{N-1}}\max_{\rm vertices}\lvert J_{\rm even}-h\rvert`$，断言最优值为 $`0.5`$。
4. 独立实现：在网格上把 $`\lvert x^N-h_q\rvert`$ 与系数尾和、与 $`\min(1,2e^{-q^2/N})`$ 比较。

**结果.**

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

**为什么支持 Proposition 4.6.**

- $`p^N`$ 的截断误差恰好是二项尾概率，按 $`e^{-q^2/N}`$ 衰减；在检验范围内，所给的界至少保守 2.3 倍。
- 达到实用精度所需的阶数按 $`\sqrt N`$ 增长：$`N`$ 翻倍，只多出几阶；而精确相等需要 $`N`$ 阶。
- parity 在任何更低的次数下都无法比平凡的常数逼近得更好，所以 $`\sqrt N`$ 的行为是任务本身的性质，不是次数的性质。

<a id="c4-10"></a>

## Check 4.10 · 最高次数相同，response 空间的复杂度不同（Proposition 4.6；Theorem 4.2 与 4.4(d)）

**结论.** PASS（21 个子空间、1,445 条断言）。三个 response 族的最高次数都是 3。标量族和结构化族用 3 个统计量就被精确刻画；parity 族在 7 个非常数方向全部到位之前，一致误差始终至少 $`0.0625`$（差异至少 $`0.125`$）。含常数的精确 span 维数为 4、4、8。在每个选出的子空间上，差异与两倍一致误差之差不超过 $`1.6\times10^{-15}`$。

**为什么做.** response 族的次数并不决定它的近似复杂度；起作用的是逼近宽度，而 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(b) 把它与最坏分布对差异等同起来（最坏分布对差异恰为两倍一致误差）。本检验比较三个最高次数相同的族，预期结构化的族代价低、parity 代价高（[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)）。它同时再次检验：

- Theorem 4.2(b) 的对偶，在结构化（而非随机）的 response 上；
- Theorem 4.2(a) 对一个给定的族所预测的 span 维数（使差距为零的最少特征个数，等于常数与该族所张空间的维数减 1），以及它与 [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(d) 统计的所有任务上的最坏情形之间的关系。

**设置.**

- 网格 $`\lbrace 0,\tfrac14,\dots,1\rbrace ^3`$（125 条消息）；每个族的最高次数都是 3。
- *标量仿射幂* $`(a+bx_1)^3`$：5 个 response。
- *多元 parity 拉回*：$`J_{\rm even}`$ 复合 8 个仿射映射，再加上原始 parity，共 9 个 response。
- *结构化求和幂* $`(a+b\bar x)^3`$，$`\bar x`$ 为坐标均值：5 个 response。
- 种子：`SeedSequence([20261010, 19])`。

**做法.**

1. 计算每个族（含常数）的精确有理秩。
2. 拟合之前固定一个有限字典：各 response 向量、全部多重线性单项式，以及标量或结构化的 Chebyshev 基。
3. 对每个预算 $`d=1,\dots,7`$，贪心选出一个子空间；在全网格上用 LP 计算一致误差 $`\max_g\mathrm{dist}_\infty(g,H)`$，并用独立的原问题 LP 计算该子空间的最坏分布对差异。
4. 断言 差异 $`=2\times`$ 误差。由于贪心找到的子空间未必最优，正的误差值是最优宽度的上界；零误差则是精确值。

**结果（一致误差；差异恰为其两倍）.**

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

*Figure 4.4.* $`p^{16}`$ 的低阶 Chebyshev 截断已经很接近它，截断误差等于一个低于 $`2e^{-q^2/N}`$ 的二项尾概率；parity 在次数低于 $`N`$ 时误差保持 $`1/2`$（[Check 4.9](#c4-9)）。在同一网格、同一最高次数下，标量族和结构化族用 3 个特征就被精确刻画，parity 要 7 个（本检验；空心标记是由找到的最优子空间给出的上界）。[PDF](../figures/fig4_4_approximation.pdf)

**为什么支持这些陈述.**

- 最高次数相同时，标量族和结构化族用 3 个统计量就能精确刻画，parity 需要全部 $`2^3-1=7`$ 个；只给 3 个统计量时，parity 仍留下 $`0.25`$ 的最坏分布对差异。决定信息阶数的是任务结构，而不是次数或决策步数（[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)）。
- 精确的 span 维数就是 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(a) 的 span 判据预测的数值，并且远低于 [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)(d) 统计的所有任务上的最坏情形。
- Theorem 4.2(b) 的对偶在非随机的结构化 response 上同样成立，包括差异始终很大的 parity 族。

**局限.** 一个 125 条消息的网格、最高次数 3；正的误差来自贪心子空间，是最优宽度的上界。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C12,C19 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

`--only` 保留每个 claim 自己的随机流（`SeedSequence([20261010, claim number])`），所以只跑子集也能复现完整运行的数字。

这些检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）、做精确符号与有理数计算的 SymPy（[Meurer et al., 2017](#ref-meurer2017sympy)），以及 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）的 `linprog` 配合 HiGHS 求解器（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解 minimax 与逼近线性规划。

## 产物

- 结果：`theory/independent_checks/results_independent_v014.json`（条目 `v014:C12`、`v014:C19`）；`theory/v014_repro/results_v014_rerun.json`（`scalar_approximation`）。
- 代码：`theory/independent_checks/check_v014_independent.py`（`check_C12`、`check_C19`）；`docs/direction/v014_handoff/verify_environment_information_v014.py`；规格说明 `theory/independent_checks/SPEC.md`。
- 已有的图：`theory/independent_checks/figs/C19_response_complexity.png`。图的规格：`theoretical_framework/figures/specs/fig4_4.md`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-meurer2017sympy"></a>

**Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, Anthony Scopatz (2017).** SymPy: Symbolic computing in Python. *PeerJ Computer Science* 3, e103. DOI [10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103).

<a id="ref-rivlin1990chebyshev"></a>

**Theodore J. Rivlin (1990).** Chebyshev Polynomials: From Approximation Theory to Algebra and Number Theory. 2nd edition, Wiley, New York. ISBN 978-0-471-62896-5.

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
