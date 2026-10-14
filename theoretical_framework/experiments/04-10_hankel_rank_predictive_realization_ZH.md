# Theorem 4.10 · Hankel 秩与线性 predictive 实现

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-10_hankel_rank_predictive_realization_ZH.md) · [证明](../proofs/04-10_hankel_rank_predictive_realization_ZH.md) · [实验](04-10_hankel_rank_predictive_realization_ZH.md) · [English](04-10_hankel_rank_predictive_realization.md)

**本页检验什么.**

- **(b) 由 Hankel 行空间实现.** 在一个有 4 个隐藏状态的随机受控过程上，长度至多为 3 的前缀与后缀构成的 Hankel 矩阵（含常数列）秩为 4。用长度至多为 2 的前缀拟合出的 4 维实现，预测长度为 4 与 5 的控制词（拟合范围之外）之后的每个测试，误差为 $`4.0\times10^{-15}`$（**通过**）。
- **(c) 条件化.** 在指示函数基下（此时 predictive 向量就是 belief 本身），归一化更新与 Bayes 公式精确相等（**通过**）。

**本页用到的设定.**

- 控制 $`b`$ 及其 Markov kernel $`K_b`$，控制词 $`w`$，$`K_w=K_{b_1}\cdots K_{b_\ell}`$，有界输出 $`F_j`$，初始分布 $`\mu_0`$。
- 带常数列的任务 Hankel 矩阵 $`H(p,(w,j))=\mu_0K_pK_wF_j`$；观察算子 $`T_{b,o}`$，$`\sum_oT_{b,o}=K_b`$。

**被检验的陈述（[Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization_ZH.md#s4-10)）.** (b) 若 $`\mathrm{rank}\,H=d`$，则存在 $`d`$ 维线性实现 $`q(pb)=A_bq(p)`$、$`H(p,(w,j))=c_j^\top A_wq(p)`$，且不存在更低维的。(c) 有观察时，$`q'=M_{b,o}q/(e_1^\top M_{b,o}q)`$ 是后验的 predictive state。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.17](#c4-17) | (b)、(c) | `v017:C5.realization` | 通过 |

<a id="c4-17"></a>

## Check 4.17 · 由 Hankel 行空间实现与条件化更新（Theorem 4.10(b)–(c)）

**结论.** Hankel 秩等于隐藏状态个数；行空间实现在舍入误差内预测没见过的更长控制词；条件化更新就是 Bayes 公式。

**为什么做.**

- (b) 断言的不只是低秩拟合：由列平移得到的转移映射，必须能预测没有用于拟合的控制词之后的期望。用长度至多为 2 的前缀拟合、去预测长度 4–5 的控制词，检验的正是这一点。
- (c) 是把开环预测变成在线更新的归一化；哪一行携带概率若弄错，会立刻暴露。

**设置.**

- 4 个隐藏状态、2 个控制、2 种观察、2 个输出。对每个控制，"观察与下一状态"的概率从 $`[0.1,1]`$ 均匀抽取后归一化，得到 $`T_{b,o}`$；$`K_b=\sum_oT_{b,o}`$。输出在 $`[0,1]`$ 上均匀抽取；$`\mu_0`$ 来自平坦 Dirichlet 分布。
- Hankel 矩阵取长度至多为 3 的全部前缀与后缀，含常数列。
- 实现：坐标取在行空间的一组正交归一基下；$`A_b`$ 由 $`\lvert p\rvert\le2`$ 的配对 $`(q(p),q(pb))`$ 最小二乘得到。
- 预测检验：对每个长度为 4 或 5 的控制词，从空前缀出发用拟合的 $`A_b`$ 传播 $`q`$，把预测的行与精确期望比较。
- 条件化：从 $`\mu_0`$ 出发，控制 1、观察 0，在指示函数基下进行。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C5.realization`） |
|---|---|---|
| 含常数列的 Hankel 秩 / 隐藏状态数 | 4 / 4 | `hankel_rank_with_constant`、`hidden_states` |
| 长度 4–5 控制词上的最大预测误差 | $`4.0\times10^{-15}`$ | `max_prediction_error_longer_words` |
| 条件化更新与 Bayes 后验之差 | $`0`$ | `conditioning_vs_bayes_error` |

**为什么支持这个陈述.**

- 只用 Hankel 行空间与短前缀构造的实现，重现了更长控制词之后的每个测试期望，与 (4.19) 一致：平移诱导的映射是精确的转移映射，不是拟合。
- 条件化更新重现了 Bayes 公式。这个检验用的是指示函数基，此时 (4.20) 退化为对 belief 的 Bayes 公式；它确认的是归一化与常数坐标的作用，不是在压缩基下估计 $`M_{b,o}`$。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C5.realization`；`theory/v017_checks/check_v017.py` 中的函数 `c_realization`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
