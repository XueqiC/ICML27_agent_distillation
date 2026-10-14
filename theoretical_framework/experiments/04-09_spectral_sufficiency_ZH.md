# Theorem 4.9 · 谱充分性与线性接口复杂度

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-09_spectral_sufficiency_ZH.md) · [证明](../proofs/04-09_spectral_sufficiency_ZH.md) · [实验](04-09_spectral_sufficiency_ZH.md) · [English](04-09_spectral_sufficiency.md)

**本页检验什么.**

- **(a)–(c) 加权谱.** 在一个随机的加权目录上，模式恒等式、Hilbert–Schmidt 界和最佳秩 3 残差都在舍入误差内成立（**通过**，[Check 4.14](#c4-14)）。
- **(f) 概率可行的下界.** 对三个任意特征，两个严格为正、特征均值相等的分布在测试上相差 $`0.0950`$，高于保证的 $`0.0804`$（**通过**，[Check 4.15](#c4-15)）。
- **(d)–(e) 平均值与有界性给不出什么.** 在试过的全部 12 个设定中，一般维数界要求的模式数都多于输入空间的维数；而一个光滑的 response 类在同样精度下只需 3 个模式。权重 $`10^{-6}`$ 的测试可以在平均误差 $`10^{-3}`$ 之下带着误差 1。8 个消息上的指示函数 response 的中心化秩是满的 7（**通过**，[Check 4.16](#c4-16)）。

**本页用到的设定.**

- 有限目录：$`p`$ 个测试，权重 $`w_i>0`$；$`m`$ 个消息，参考权重 $`q_j>0`$；两组权重各自和为 1；response 矩阵 $`G\in[0,1]^{p\times m}`$，$`G_{ij}=g_i(y_j)`$。
- 在正交归一输入坐标 $`x=\sqrt q\odot h`$ 下，[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8) 的 response operator 是矩阵 $`D_{\sqrt w}\,G\,D_{\sqrt q}\,\Pi_q`$，其中 $`\Pi_q=I-\sqrt q\sqrt q^\top`$；它的奇异值就是 $`\sigma_j`$。
- 模式差 $`\alpha_j=\langle v_j,h\rangle_\rho`$、秩 $`r`$ 线性接口与逐测试残差 $`e_r(a)`$ 同陈述页。

**被检验的陈述（[Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9)）.** (a) $`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\alpha_j^2`$；(b)–(c) 匹配前 $`r`$ 个模式后至多剩 $`B\sigma_{r+1}`$，这在秩 $`r`$ 线性接口中最优；(d) 逐测试的界要付 $`e_r(a)`$，稀有测试不受平均值控制；(e) 一般界 $`r+1\ge B^2/(4\varepsilon^2)`$ 在 $`J`$ 候选混合测度下没有压缩内容；(f) 在有限消息集上，$`r`$ 个特征总会留下两个统计量相等、response 差至少为 $`a\sigma_{r+1}`$ 的概率分布。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.14](#c4-14) | (a)、(b)、(c) | `v017:C2.spectrum` | 通过 |
| [Check 4.15](#c4-15) | (f) | `v017:C3.positive` | 通过 |
| [Check 4.16](#c4-16) | (d)、(e) | `v017:C4.generic`、`v017:C9.rare` | 通过 |

<a id="c4-14"></a>

## Check 4.14 · 加权谱、Hilbert–Schmidt 界与最佳截断（Theorem 4.9(a)–(c)）

**结论.** 所有恒等式都在舍入误差内成立；Hilbert–Schmidt 范数的平方为 $`0.044\le\tfrac14`$。

**为什么做.** 权重 $`\omega`$、$`\rho`$ 与中心化以特定方式进入算子。$`\sqrt w`$、$`\sqrt q`$ 或投影 $`\Pi_q`$ 的位置一旦放错，就会破坏模式恒等式、[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8) 中的密度差恒等式，或者最佳秩 $`r`$ 残差与 $`\sigma_{r+1}`$ 的相等关系。

**设置.**

- $`p=7`$ 个测试，$`m=11`$ 个消息，权重从 $`[0.3,1.3]`$ 均匀抽取后归一化，$`G=\sigma(N)`$，$`N`$ 为标准正态矩阵，$`\sigma`$ 为 logistic 函数。
- (a) 用中心化输入空间中的一个随机单位向量 $`x`$；(c) 用最佳秩 3 逼近在算子范数下的截断残差；密度差恒等式用两个从平坦 Dirichlet 分布抽取的分布 $`P,Q`$。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C2.spectrum`） |
|---|---|---|
| (4.12) 在随机单位输入上的误差 | $`2.8\times10^{-18}`$ | `spectral_identity_error` |
| $`\lvert\lVert\mathcal R-\mathcal R_3\rVert_{\rm op}-\sigma_4\rvert`$ | $`2.8\times10^{-17}`$ | `best_rank_residual_error` |
| $`\lVert\mathcal R\rVert_{\rm HS}^2`$（上界 $`\tfrac14`$） | $`0.04418`$ | `hilbert_schmidt_squared` |
| $`(\mathcal Rh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$ 的误差 | $`1.7\times10^{-17}`$ | `density_difference_error` |
| 奇异值 $`\sigma_1,\dots,\sigma_7`$ | $`0.1179,\ 0.1053,\ 0.0877,\ 0.0805,\ 0.0587,\ 0.0309,\ 0.0249`$ | `singular_values` |

**为什么支持这个陈述.** 由加权、中心化目录构造的算子精确满足 (a) 的模式恒等式，它的最佳秩 3 残差是 $`\sigma_4`$，与 (b)–(c) 一致；作用在密度差上时，它返回两个分布的加权得分差。

<a id="c4-15"></a>

## Check 4.15 · 三个特征分不开的两个正分布（Theorem 4.9(f)）

**结论.** 两个严格为正、三个任意特征均值相等的分布，在测试上相差 $`0.0950`$，高于保证的 $`a\sigma_4=0.0804`$。

**为什么做.** (c) 是在有符号球上证明的，球里有些差是任何一对概率分布都给不出的。(f) 断言：只要把扰动缩放到保持非负，真实的分布就能达到同样的下界。

**设置.** 用 [Check 4.14](#c4-14) 的目录结构以另一个种子重新生成；三个取标准正态值的特征；$`v`$ 是前四个输入模式张成空间中与三个特征正交的单位向量；$`a=1.8/\lVert v\rVert_\infty`$，严格落在可行范围 $`a\le2/\lVert v\rVert_\infty`$ 之内；$`P,Q`$ 由 (4.17) 给出。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C3.positive`） |
|---|---|---|
| 特征均值的最大差 | $`2.8\times10^{-16}`$ | `feature_match_error` |
| response 差 $`\lVert\mathcal R(p-q)\rVert_\omega`$ | $`0.0950`$ | `response_gap` |
| 保证值 $`a\sigma_4`$ | $`0.0804`$ | `guaranteed_gap` |
| $`P,Q`$ 中的最小概率 | $`0.0109`$ | `smallest_probability` |

**为什么支持这个陈述.** (f) 的构造给出两个真正的概率分布，任何只读这三个特征的接口都分不开它们，而它们的 response 差超过 $`a\sigma_{r+1}`$，与陈述一致。

<a id="c4-16"></a>

## Check 4.16 · 一般界、有结构的谱、稀有测试与满秩（Theorem 4.9(d)–(e)）

**结论.** 在全部 12 个设定中，一般界要求的模式数都多于输入空间的维数。一个光滑的 response 类用 3 个模式就把误差降到 $`0.01`$，而一般界要求 2,500 个。稀有测试可以在平均误差 $`10^{-3}`$ 之下带着误差 1。8 个消息上的指示函数 response 的中心化秩为 7。

**为什么做.**

- (e) 断言：仅由有界性得到的维数计数是有效的，但在混合测度下是空的。这个检验在能达到的最坏情形 $`B^2=2J`$ 下计算它。
- 与有结构的类对比，说明有用的压缩若存在，来自测得的谱。
- (d) 断言逐测试的保证不能从加权平均读出；满秩的类说明仅凭有界性与可测性推不出任何压缩。

**设置.**

- *一般界.* $`J\in\lbrace2,5,20,100\rbrace`$ 个落在不同消息上的点质量候选；$`\rho`$ 是它们的均匀混合；$`h`$ 是两个候选的密度差；$`\varepsilon\in\lbrace0.1,0.3,0.7\rbrace`$。把充分维数 $`B^2/(4\varepsilon^2)`$ 与可用输入维数 $`J-1`$ 对比。
- *有结构的类.* 30 个测试、网格 $`y\in[0,1]`$ 上的 40 个消息，$`g_k(y)=\sigma(3\sin(2\pi k/30+2y))`$，均匀权重。
- *稀有测试与满秩.* 一个权重 $`10^{-6}`$、误差为 1 的测试；8 个消息上的指示函数 response $`g_a=\mathbf 1\lbrace y=a\rbrace`$。

**结果.**

| 量 | 值 | JSON 键 |
|---|---|---|
| 达到的 $`B^2`$ | 每一行都是 $`2J`$ | `v017:C4.generic` `rows[].B_squared` |
| 充分维数 / 输入维数，$`J=20`$，$`\varepsilon=0.3`$ | $`111.1`$ / $`19`$ | `rows[].sufficient_dim`、`rows[].input_dim` |
| 充分维数 / 输入维数，$`J=100`$，$`\varepsilon=0.7`$ | $`102.0`$ / $`99`$ | 同上 |
| 一般界超过输入维数的行数 | 12 / 12 | `rows[].vacuous` |
| 有结构的类，$`\sigma_1,\dots,\sigma_6`$ | $`0.192,\ 0.0598,\ 0.0176,\ 0.00736,\ 0.00203,\ 0.00094`$ | `v017:C4.generic` `smooth_example_singular_values` |
| 有结构的类，数值秩（容差 $`10^{-10}`$） | 15 | `smooth_example_rank_tol_1e-10` |
| 加权 RMS 误差 / 稀有测试误差 | $`10^{-3}`$ / $`1`$ | `v017:C9.rare` `weighted_rms_error`、`rare_test_error` |
| 8 个指示函数 response 的中心化秩 | 7 | `v017:C9.rare` `centred_rank` |

**为什么支持这个陈述.**

- 一般维数从不低于本来就精确的统计量个数，与 (e) 一致。
- 在有结构的类中 $`\sigma_4=0.0074\le0.01<\sigma_3`$，所以取 $`B=1`$ 时，$`\varepsilon=0.01`$ 下的线性接口复杂度是 3，而 (4.16) 要求 $`r+1\ge2{,}500`$。谱衰减时，这个界有效但远不紧。
- 稀有测试与指示函数类正是 (d) 和定理的适用范围所警告的两种失效：平均值控制不了稀有测试，有界的 response 可能需要 $`m-1`$ 个模式。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C2.spectrum`、`v017:C3.positive`、`v017:C4.generic`、`v017:C9.rare`；`theory/v017_checks/check_v017.py` 中的函数 `c_spectrum`、`c_positive`、`c_generic`、`c_rare`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
