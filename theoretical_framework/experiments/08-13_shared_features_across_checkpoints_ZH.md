# Proposition 8.13 · 跨 checkpoint 共享的消息特征

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-13_shared_features_across_checkpoints_ZH.md) · [证明](../proofs/08-13_shared_features_across_checkpoints_ZH.md) · [实验](08-13_shared_features_across_checkpoints_ZH.md) · [English](08-13_shared_features_across_checkpoints.md)

**本页检验什么.**

- **(c)：反例.** 两个 checkpoint 对正交方向作出响应、输入随状态变化时，实际的 RMS response 为 1，共同输入的界 $`B\sigma_1(\mathcal S)`$ 给出 0.707，尾部之和的界给出 1（**PASS**：错误的界失效，正确的界以等号成立）。
- **(a)–(b)：尾部恒等式.** 在四个 checkpoint 算子的随机堆叠上，秩 3 共享接口按状态平均的 Hilbert–Schmidt 残差，与堆叠奇异值平方的尾部之和相差 $`5.3\times10^{-15}`$（**PASS**）。

**本页用到的设定与记号.**

- 共同消息空间上的一个参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$；测试与权重 $`\omega`$ 在每个 checkpoint 上含义相同；checkpoint 权重 $`\nu`$。
- $`\mathcal R_z`$ 是 checkpoint $`z`$ 处的 response operator；$`(\mathcal Sh)(z,a)=(\mathcal R_zh)(a)`$ 是堆叠算子，奇异值为 $`\sigma_j(\mathcal S)`$，$`\Pi_r`$ 投影到其前 $`r`$ 个右奇异向量上；$`h_z`$ 是 $`z`$ 处的反馈差，$`\lVert h_z\rVert_\rho\le B`$。

**被检验的陈述（[Proposition 8.13](../statements/08-13_shared_features_across_checkpoints_ZH.md#s8-13)）.**

```math
\mathbb E_{z\sim\nu}\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2 ,
```

$`\Pi_r`$ 在秩 $`r`$ 共享接口中使按状态平均的 Hilbert–Schmidt 残差最小；尾部之和不能换成 $`B^2\sigma_{r+1}(\mathcal S)^2`$。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 8.19](#c8-19) | (a)、(b)、(c) | `v017:C18.shared` | PASS |

<a id="c8-19"></a>

## Check 8.19 · 条件输入需要尾部之和（Proposition 8.13）

**结论.**

- 在两个 checkpoint 的构造中，真实的 RMS response 为 $`1.0`$。共同输入的算子界给出 $`0.7071`$，这个界**是错的**；Hilbert–Schmidt 尾部界给出 $`1.0`$，以等号成立。
- 在随机堆叠上，截断共享基的平均残差等于奇异值平方的尾部之和，误差只有舍入量级（$`5.3\times10^{-15}`$）。

**动机.** 一个诱人的捷径是：把所有 checkpoint 的 response 行堆起来，做一次奇异值分解，再把 $`B\sigma_{r+1}(\mathcal S)`$ 当作复用误差。只有当所有 checkpoint 共享同一个输入时这才正确。训练中参考反馈与替代反馈都随状态变化，每个 checkpoint 收到的是自己的输入。这个检验在最小的例子上展示捷径失效，并说明正确的量，即尾部之和，对共享的截断基是精确的。

**设置.**

- *反例.* 两个权重各为 $`\tfrac12`$ 的 checkpoint，在二维 $`\mathcal H`$ 的正交坐标下 response 行分别为 $`(1,0)`$ 与 $`(0,1)`$；第一个 checkpoint 的输入为 $`e_1`$，第二个为 $`e_2`$（$`B=1`$），$`r=0`$。堆叠算子为 $`I_2/\sqrt2`$。如证明所示，把分数按任意 $`c>0`$ 缩放后仍然有界，比较结论不变。
- *尾部恒等式.* 四个 checkpoint 算子，元素为独立标准正态的 $`5\times8`$ 矩阵，每个权重 $`\tfrac14`$（堆叠后除以 2），共享秩 $`r=3`$。
- 随机种子 `SeedSequence([20261014, tag])`；NumPy（[Harris et al., 2020](#ref-harris2020array)）。

**结果**（`theory/v017_checks/results_v017.json`，条目 `v017:C18.shared`）：

| 量 | 值 |
|---|---:|
| 实际 RMS response，$`r=0`$ | 1.0000 |
| 共同输入的界 $`B\sigma_1(\mathcal S)`$ | 0.7071 |
| 尾部之和的界 $`B(\sum_j\sigma_j(\mathcal S)^2)^{1/2}`$ | 1.0000 |
| 尾部恒等式误差，随机堆叠，$`r=3`$ | $`5.3\times10^{-15}`$ |

**为什么支持该陈述.**

- (c)：实际值超过了用第一个被丢弃奇异值给出的界，所以这个界对随状态变化的输入无效；尾部之和被取到。
- (a)–(b)：$`\Pi_r`$ 的平均 Hilbert–Schmidt 残差等于 $`\sum_{j>r}\sigma_j(\mathcal S)^2`$，正是控制条件输入误差、并被 $`\Pi_r`$ 最小化的那个量。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验程序使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C18.shared`；代码 `theory/v017_checks/check_v017.py`，函数 `c_shared`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
