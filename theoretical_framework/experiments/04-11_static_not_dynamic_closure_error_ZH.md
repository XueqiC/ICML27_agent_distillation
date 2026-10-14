# Proposition 4.11 · 静态充分不等于动态充分；有限时域的闭合误差

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-11_static_not_dynamic_closure_error_ZH.md) · [证明](../proofs/04-11_static_not_dynamic_closure_error_ZH.md) · [实验](04-11_static_not_dynamic_closure_error_ZH.md) · [English](04-11_static_not_dynamic_closure_error.md)

**本页检验什么.**

- **(a) 反例.** $`(\mathbf 1,s)`$ 的最佳最小二乘闭合留下 $`0.5`$ 的残差；加入 $`b`$ 后空间闭合（秩 3）；当前价值相同的两个状态，一次更新后相差 1（**通过**）。
- **(b) 误差递推.** 在一个 6 状态的随机链上，用三个特征、五步，一步增益最大为 $`1.13`$（不是收缩），精确误差恒等式 (4.21) 在 $`2.5\times10^{-16}`$ 内成立（**通过**）。

**本页用到的设定.**

- 特征 $`\psi`$，kernel $`K_t`$，矩阵 $`A_t`$，残差 $`r_t=K_t\psi-A_t\psi`$，真实 predictive state $`q_t=\mathbb E\psi(Z_t)`$，模型预测 $`\hat q_{t+1}=A_t\hat q_t`$。

**被检验的陈述（[Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error_ZH.md#s4-11)）.** (a) 静态充分的特征集不一定闭合，当前价值相同时的未来差距可以是 1。(b) $`q_H-\hat q_H=A_{H-1:0}(q_0-\hat q_0)+\sum_tA_{H-1:t+1}\mathbb Er_t(Z_t)`$ 精确成立，无需收缩。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.18](#c4-18) | (a)、(b) | `v017:C6.closure` | 通过 |

<a id="c4-18"></a>

## Check 4.18 · 闭合反例与精确误差递推（Proposition 4.11）

**结论.** 反例的表现与陈述完全一致；误差递推在非收缩的模型上精确成立。

**为什么做.**

- (a) 是关于 $`(\mathbf 1,s)`$ 所有线性闭合的断言；最小二乘拟合是欧氏意义下最好的一个，$`0.5`$ 的残差确认不存在精确闭合。
- (b) 对任意拟合出的矩阵断言的是恒等式，而不是界。用非收缩的例子检验，证明中没有用到收缩。

**设置.**

- (a) 四个状态 $`(s,b)\in\lbrace0,1\rbrace^2`$，$`F=s`$，更新 $`(s,b)\mapsto(b,b)`$；把 $`K\psi`$ 对 $`\psi=(\mathbf 1,s)`$ 做最小二乘拟合；计算 $`(\mathbf 1,s,KF)`$ 的秩。
- (b) 6 个状态；5 个随机生成的 stochastic matrix $`K_t`$（每行来自平坦 Dirichlet 分布）；特征为常数加两列标准正态；$`A_t`$ 由最小二乘拟合，所以 $`r_t`$ 包含拟合误差；初始分布来自平坦 Dirichlet 分布；$`\hat q_0`$ 为 $`q_0`$ 加尺度 $`0.01`$ 的高斯噪声。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C6.closure`） |
|---|---|---|
| $`(\mathbf 1,s)`$ 的最大闭合残差 | $`0.5`$ | `closure_defect` |
| 加入 $`KF=b`$ 后的秩 | 3 | `closed_test_rank` |
| 当前价值相同时下一步的得分差 | $`1`$ | `future_gap_equal_current_value` |
| 恒等式 (4.21) 的误差 | $`2.5\times10^{-16}`$ | `recursion_identity_error` |
| 最大一步增益 $`\lVert A_t\rVert_2`$ | $`1.126`$ | `max_one_step_gain` |

**为什么支持这个陈述.** 这组特征精确预测今天的得分，却在明天的得分上错到最大，与 (a) 一致；递推用拟合出的、非收缩的映射精确重现了 predictive 误差，与 (b) 一致。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C6.closure`；`theory/v017_checks/check_v017.py` 中的函数 `c_closure`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
