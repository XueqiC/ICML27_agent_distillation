# Theorem 8.1 · 稳定性–近似的权衡

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-01_stability_approximation_tradeoff_ZH.md) · [证明](../proofs/08-01_stability_approximation_tradeoff_ZH.md) · [实验](../experiments/08-01_stability_approximation_tradeoff_ZH.md) · [English](08-01_stability_approximation_tradeoff.md)

**本页检验什么.**

- **[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)（稳定性–近似权衡）精确成立.** 在 220 个随机对抗实例上原始问题与对偶问题一致，最大对偶残差为 $`8.9\times10^{-16}`$。稳定性见证在全部 28 组设定上复现 $`W_\epsilon=\min\lbrace1,\epsilon/\alpha\rbrace`$。

**本页用到的设定与记号.**

设 $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。

- $`\mathcal Y`$ 上的函数是 $`\mathbb R^m`$ 中的向量，概率律是单纯形 $`\Delta(\mathcal Y)`$ 中的向量。对 $`h\in\mathbb R^m`$ 记 $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$。
- $`g\in\mathbb R^m`$ 是一个 response。$`\phi_1,\dots,\phi_d\in\mathbb R^m`$ 是固定的特征，$`\Phi\in\mathbb R^{m\times d}`$ 以 $`\phi_j`$ 为列，$`\Phi b=\sum_jb_j\phi_j`$。
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ 是容差，$`\epsilon_j\ge0`$。允许零容差，也允许特征线性相关。
- $`h`$ 的振幅（oscillation）为 $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$。

$`g`$ 的**容差差距**（tolerant discrepancy）定义为

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace\lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace .
```

它是特征均值之差不超过所声明容差的两个消息律之间，response 差距的最大值。一次有限 audit 把每个特征均值测到精度 $`\epsilon_j`$ 之后，面对的正是这种情形。

**被检验的陈述（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)）.**

**(a) 精确权衡.** 下面两个最小值都能取到，并且

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

特别地 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$；当 $`g\in[0,1]^m`$ 时它不超过 $`1`$。

- **(b) 精确情形.** $`\mathbf{\epsilon}=0`$ 时，$`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$，其中 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。这是 [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 的单个 response 版本。
- **(c) 稳定性反例.** 设 $`g\in\lbrace0,1\rbrace^m`$ 不是常数，$`d=1`$，$`\phi_1=\alpha g`$，$`\alpha>0`$。则 $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$。
- **(d) 有限测量的桥梁.** 固定一个同时成立的事件，在其上每个被 audit 的特征均值满足 $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ 与 $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$。令 $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$。则在该事件上 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$。若还有 $`\lVert\widehat g-g\rVert_\infty\le\eta`$，则 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。

<a id="c8-1"></a>

## Check 8.1 · 随机实例上的原始–对偶等式（Theorem 8.1）

**结论.** 在 220 个随机对抗实例上，在律对上直接计算的容差差异与两种对偶形式都相等。最大对偶残差为 $`8.9\times10^{-16}`$。

**动机.** [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1) 断言的是一个*等式*：在满足容差的律对中，最坏的 response 差异等于对特征系数的一个最小值。对偶写错，或漏掉零质量条件之类的约束，都会在某个随机实例上表现为严格的间隙。

**设置.**

- 随机有限实例：消息数 $`m\in[3,12]`$，特征数 $`d\in[1,m-1]`$。
- 随机 response $`g`$、随机特征 $`\phi`$、随机容差 $`\epsilon_j\ge0`$（包括恰好为零的情形）。
- 种子 `SeedSequence([20261010, 14])`。
- 用 HiGHS（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）独立求解三个线性规划：
  - 直接写在概率变量上的、对 $`(P,Q)`$ 的原始最大化；
  - 振幅对偶 $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$；
  - 中心化 sup 范数对偶 $`\min_{b,c}[2\lVert g-c-\Phi b\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert]`$。
- 另外断言：$`W\le\mathrm{osc}(g)`$；$`\mathbf{\epsilon}=0`$ 时退化为精确情形 Theorem 8.1(b)，即 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(b) 的单个 response 形式；以及 $`\mathbf{\epsilon}`$ 很大时 $`W\to\mathrm{osc}(g)`$。
- 原始 LP 在每个实例上都是全局最大化，因此搜索本身就是对抗的。

**结果**（`theory/independent_checks/results_independent_v014.json`，条目 `v014:C14`）：

| 量 | 值 |
|---|---:|
| status | PASS |
| `n_instances` | 220 |
| `n_assertions` | 2,420 |
| `worst_residual` | $`9.80\times10^{-13}`$ |
| `details.max_duality_residual` | $`8.88\times10^{-16}`$ |

**为什么支持该陈述.**

- 原始问题和两种对偶形式在每个实例上都一致，包括零容差和秩亏的特征。
- 所以精确对偶按所述成立，不需要内点条件或秩条件。

<a id="c8-2"></a>

## Check 8.2 · 缩放特征下的稳定性控制（Theorem 8.1(c)）

**结论.** 对特征 $`\phi=\alpha g`$，差异在全部 28 组设定上都等于 $`\min\lbrace1,\epsilon/\alpha\rbrace`$，最坏残差为 $`1.39\times10^{-17}`$。

**动机.** 稳定性见证 [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)(c) 是观察稳定性项最干净的方式。让特征等于 response 乘一个尺度，$`\phi=\alpha g`$。这时近似残差为零，但随着 $`\alpha\to0`$，证书必须变差：对 $`\phi`$ 的容差 $`\epsilon`$ 只能把 $`g`$ 约束在 $`\epsilon/\alpha`$ 以内。

**设置.**

- 二值 response $`g`$，特征 $`\phi=\alpha g`$。
- 尺度 $`\alpha\in\lbrace 1,\tfrac12,\tfrac1{10},\tfrac1{100}\rbrace`$，容差 $`\epsilon\in\lbrace 0,\tfrac1{1000},\tfrac1{100},\tfrac1{20},\tfrac15,1,2\rbrace`$，共 28 组设定。
- 独立求解概率变量上的原始 LP 与容差对偶 LP，并与以精确有理数保存的闭式比较。

**结果**（条目 `v014:C22`，`details.configurations`；28 行全部与公式一致）：

| $`\alpha`$ | $`\epsilon=10^{-3}`$ | $`\epsilon=10^{-2}`$ | $`\epsilon=1/20`$ | $`\epsilon=1/5`$ | $`\epsilon\ge1`$ |
|---|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.01 | 0.05 | 0.2 | 1 |
| 1/2 | 0.002 | 0.02 | 0.1 | 0.4 | 1 |
| 1/10 | 0.01 | 0.1 | 0.5 | 1 | 1 |
| 1/100 | 0.1 | 1 | 1 | 1 | 1 |

$`\epsilon=0`$ 时，对每个 $`\alpha`$ 差异都是 $`0`$。

**为什么支持该陈述.**

- 即使表示是精确的，证书也只有 $`\epsilon\lvert b\rvert`$ 那么好，其中 $`b=1/\alpha`$。
- $`\alpha=1/100`$ 时，1% 的容差就已经让 response 完全不确定（$`W=1`$）。
- 所以小的近似残差本身不构成证书，系数大小必须进入证书，正如 Theorem 8.1 所述。

<a id="c8-3"></a>

## Check 8.3 · 一个有闭式的容差对偶例子（Theorem 8.1）

**结论.** 在一个有闭式的例子上，原始值、对偶值与闭式在六个容差处都一致；另外 36 个随机 LP 上的最大原始–对偶残差为 $`8.9\times10^{-16}`$。

**动机.** 这是理论包里的一个独立实现，用在最优系数已知的例子上。

**设置.**

- $`\mathcal Y=\lbrace 0,\tfrac12,1\rbrace`$，$`\phi(y)=y`$，$`g=1\lbrace y=1\rbrace`$；闭式为 $`W_\epsilon=\min\lbrace 1,\tfrac12+\epsilon\rbrace`$。
- 在六个容差下求解原始与对偶 LP，记录见证律与最优系数。

**结果**（`theory/v014_repro/results_v014_rerun.json`，键 `noisy_feature_duality`）：

| 容差 | 原始 | 对偶 | 闭式 | 最优 $`b`$ | `uniform_residual` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | 0.50 | 1 | 0.25 |
| 0.05 | 0.55 | 0.55 | 0.55 | 1 | 0.25 |
| 0.25 | 0.75 | 0.75 | 0.75 | 1 | 0.25 |
| 0.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 0.8 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 1.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |

`uniform_residual` 是最优 $`b`$ 处的 $`\min_c\lVert g-c-b\phi\rVert_\infty`$。`random_LP_cases` = 36，`maximum_primal_dual_residual` = $`8.9\times10^{-16}`$。

**为什么支持该陈述.** 当测量代价 $`\epsilon`$ 超过近似收益时，最优解恰好从使用该特征（$`b=1`$）切换到忽略它（$`b=0`$）。Theorem 8.1 精确刻画的正是这个权衡。

![Figure 8.1：稳定性–近似权衡](../figures/fig8_1_stability.png)

*Figure 8.1.* 左：缩放特征 $`\phi=\alpha g`$ 的容差差异（[Check 8.2](../experiments/08-01_stability_approximation_tradeoff_ZH.md#c8-2)）；LP 值落在 $`\min\lbrace1,\epsilon/\alpha\rbrace`$ 上，所以一个精确但尺度很小的特征只能认证到 $`\epsilon/\alpha`$。右：[Check 8.3](../experiments/08-01_stability_approximation_tradeoff_ZH.md#c8-3) 的例子，原始值、对偶值与闭式重合，$`\epsilon\ge\tfrac12`$ 后最优解不再使用该特征。精确有限计算。[PDF](../figures/fig8_1_stability.pdf)

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C14,C22 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # noisy_feature_duality
```

这些脚本使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）；线性规划通过 SciPy 的 `linprog` 以 HiGHS 方法求解。

## 产出文件

- `theory/independent_checks/results_independent_v014.json`：条目 `v014:C14`、`v014:C22`；规范 `theory/independent_checks/SPEC.md`。
- `theory/v014_repro/results_v014_rerun.json`：`noisy_feature_duality`。
- 图：`../figures/fig8_1_stability`（`.pdf` 与 `.png`；规范在 `../figures/specs/`）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
