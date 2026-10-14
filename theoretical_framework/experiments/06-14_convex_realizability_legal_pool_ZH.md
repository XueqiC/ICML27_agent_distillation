# Proposition 6.14 · 有限合法消息池上的凸可实现性

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-14_convex_realizability_legal_pool_ZH.md) · [证明](../proofs/06-14_convex_realizability_legal_pool_ZH.md) · [实验](06-14_convex_realizability_legal_pool_ZH.md) · [English](06-14_convex_realizability_legal_pool.md)

**本页检验什么.** 一项检验 [Check 6.8](#c6-8)，在一个含 30 条消息、4 个特征的合成消息池上进行。

- 凸包内的目标被一个只用 5 条消息的顶点解精确匹配，正是 (b) 中的 $`r+1`$（**PASS**）。
- 稀疏解的密度远大于用来生成目标的稠密混合，与 (d) 的预言一致（**PASS**）。
- 凸包外的目标被判为不可行，与 (a) 的要求一致（**PASS**）。

这是合成计算；它说明不了真实目标落在真实消息池凸包内的频率。

**本页用到的设定与记号.**

- **合法消息池.** 有限个已核验的合法消息 $`\lbrace y_1,\dots,y_m\rbrace`$；合法混合为 $`q\in\Delta_m`$。
- **特征.** $`\phi(y)\in\mathbb R^r`$，$`\Phi=[\phi(y_1),\dots,\phi(y_m)]`$；$`q`$ 的特征均值为 $`\Phi q`$。
- **目标.** $`\mu_*\in\mathbb R^r`$。
- **参照测度.** $`\rho`$，$`\rho_i>0`$；$`dQ/d\rho=(q_i/\rho_i)_i`$，$`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$。量 $`\lVert dQ/d\rho-1\rVert_\rho`$ 是 $`q`$ 到 $`\rho`$ 的 $`\chi`$ 距离；$`\lVert dQ/d\rho\rVert_\rho^2=1+\lVert dQ/d\rho-1\rVert_\rho^2`$。

**被检验的陈述（[Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14)）.** 存在满足 $`\Phi q=\mu_*`$ 的合法混合，当且仅当 $`\mu_*\in\mathrm{conv}\lbrace\phi(y_i)\rbrace`$（a）；若存在，顶点解至多使用 $`r+1`$ 条消息（b）；在凸包之外，最好的特征误差是到凸包的正距离（c）；支撑为 $`S`$ 的混合满足 $`\lVert dQ/d\rho\rVert_\rho^2\ge1/\rho(S)`$（d）；密度上限使编译保持凸，并迫使 $`\rho(S)\ge1/\kappa`$（e）。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 6.8](#c6-8) | (a)、(b)、(d) | `v017:C10.realizable` | PASS |

<a id="c6-8"></a>

## Check 6.8 · 至多 r+1 条消息的精确编译、它的密度代价，以及凸包外的不可行（Proposition 6.14）

**结论.** PASS。顶点解只用 5 条消息就把目标的 4 个特征均值匹配到 $`8.3\times10^{-17}`$；它到参照测度的 $`\chi`$ 距离为 2.78，而生成目标的稠密混合只有 0.87；放在凸包外的目标被判为不可行。

**为什么做.** (a) 与 (b) 说的是：编译到已核验消息池是一个有稀疏解的线性可行性问题。(d) 是附在这个结论上的警告：稀疏解要在密度上付代价。如果求解器在顶点处返回一个用了超过 $`r+1`$ 条消息的可行点、为凸包外的目标给出可行点，或者给出密度很小的稀疏解，都与这个命题矛盾。

**设置.**

- 消息池有 $`m=30`$ 条消息，$`r=4`$ 个特征 $`\phi(y_i)`$，独立同分布地取自标准正态（固定种子）；参照测度均匀，$`\rho_i=1/30`$。
- 目标：$`\mu_*=\Phi w`$，$`w`$ 是一个随机的稠密混合（参数全为 1 的 Dirichlet），所以 $`\mu_*`$ 在凸包内。
- 编译：目标函数为零的线性规划 $`\lbrace q\ge0,\ \Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$，通过 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）调用 HiGHS 对偶单纯形法（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解；单纯形法返回的是顶点。
- 不可行情形：把目标移到消息池特征均值再在每个坐标上加 $`10\max_{i,k}\lvert\phi_k(y_i)\rvert`$，它一定在凸包之外。

**结果.**

| 量 | 取值 |
|---|---|
| 顶点解的支撑 | 5 |
| (b) 的上界 $`r+1`$ | 5 |
| 特征均值误差 $`\lVert\Phi q-\mu_*\rVert_\infty`$ | $`8.3\times10^{-17}`$ |
| 稠密混合 $`w`$ 的 $`\chi`$ 距离 | 0.875 |
| 稀疏解 $`q`$ 的 $`\chi`$ 距离 | 2.785 |
| 凸包外的目标 | 不可行（求解器状态 2） |

由 (d)，30 条消息中任意 5 条上的混合都有 $`\lVert dQ/d\rho\rVert_\rho^2\ge30/5=6`$，即 $`\chi`$ 距离至少为 $`\sqrt5\approx2.236`$。稀疏解的 2.785 满足这个下限，并且离参照测度的距离是特征均值相同的稠密混合的三倍多。

**为什么支持 Proposition 6.14.** 顶点解恰好达到 (b) 的支撑上界，不可行情形的表现符合 (a)，稀疏解的密度落在 (d) 的下限之上。与稠密混合的比较说明了 (d) 的实际含义：同一组特征均值，可以由密度相差三倍的分布来匹配，它们在 [Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9) 中的谱误差界也相应不同。

## 复现

在仓库根目录（只用 CPU，几秒钟）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验集用 NumPy（[Harris et al., 2020](#ref-harris2020array)），种子由 20261014 与检验名称确定（脚本中的 `rng_for`），并使用 SciPy 的 `linprog`。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C10.realizable`。
- `theory/v017_checks/check_v017.py`：函数 `c_realizable`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
