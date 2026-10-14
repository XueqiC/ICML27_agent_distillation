# Proposition 4.13 · 有限时域 Gramian，以及控制理论中不能照搬的部分

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-13_finite_horizon_gramians_ZH.md) · [证明](../proofs/04-13_finite_horizon_gramians_ZH.md) · [实验](04-13_finite_horizon_gramians_ZH.md) · [English](04-13_finite_horizon_gramians.md)

**本页检验什么.** 在一个一步增益达到 $`1.28`$ 的时变线性系统上，有限时域 Hankel 奇异值与 $`W_c^{1/2}W_oW_c^{1/2}`$ 特征值的平方根之差为 $`1.0\times10^{-15}`$（**通过**）。没有用到稳定性。

**本页用到的设定.**

- 有限时域时变线性系统 $`\delta z_{t+1}=A_t\delta z_t+B_t\delta u_t`$、$`\delta o_t=C_t\delta z_t`$，在 $`\tau`$ 处分开；可达性矩阵 $`\mathscr C_\tau`$，可观性矩阵 $`\mathscr O_\tau`$，Hankel 算子 $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$，Gramian $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$ 与 $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$。

**被检验的陈述（[Proposition 4.13](../statements/04-13_finite_horizon_gramians_ZH.md#s4-13)）.** (a) 对每个 $`j`$，$`\sigma_j(H_\tau)^2=\lambda_j(W_c^{1/2}W_oW_c^{1/2})`$，不作稳定性或收缩假设。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.20](#c4-20) | (a) | `v017:C8.gramian` | 通过 |

<a id="c4-20"></a>

## Check 4.20 · 不依赖稳定性、由有限时域 Gramian 得到的 Hankel 奇异值（Proposition 4.13(a)）

**结论.** 在一个沿某方向扩张的系统上，恒等式在舍入误差内成立。

**为什么做.** 关于 Hankel 奇异值的经典陈述，通常针对稳定的时不变系统。(a) 断言该恒等式在有限时域上对任意时变矩阵成立；用一步增益大于 1 的系统检验，确认论证中没有暗含稳定性。

**设置.**

- 二维状态，时域 $`N=5`$，在 $`\tau=2`$ 处分开。
- $`A_t=\begin{pmatrix}1.15+0.03t&0.12\\0.04&0.75\end{pmatrix}`$，$`B_t=(1,\ 0.4(-1)^t)^\top`$，$`C_t=(1,\ 0.25+0.06t)`$。
- $`\mathscr C_\tau`$ 来自 $`t=0,1`$ 的输入；$`\mathscr O_\tau`$ 来自 $`t=2,\dots,5`$ 的输出；把 $`H_\tau`$ 的奇异值与 $`W_c^{1/2}W_oW_c^{1/2}`$ 的特征值比较，$`W_c^{1/2}`$ 由对称特征分解得到。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C8.gramian`） |
|---|---|---|
| 恒等式误差 | $`1.0\times10^{-15}`$ | `identity_error` |
| Hankel 奇异值 | $`4.746,\ 0.0371`$ | `hankel_singular_values` |
| 最大一步增益 $`\lVert A_t\rVert_2`$ | $`1.283`$ | `max_one_step_gain` |

**为什么支持这个陈述.** 系统不是收缩的，Hankel 奇异值仍恰好等于 (a) 中的 Gramian 量，所以有限时域恒等式不需要经典误差界所要求的稳定性。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C8.gramian`；`theory/v017_checks/check_v017.py` 中的函数 `c_gramian`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
