# Proposition 6.15 · 线性高斯目标：精确风险、白化 response 与单次查询收益

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-15_linear_gaussian_target_ZH.md) · [证明](../proofs/06-15_linear_gaussian_target_ZH.md) · [实验](06-15_linear_gaussian_target_ZH.md) · [English](06-15_linear_gaussian_target.md)

**本页检验什么.** 一项检验 [Check 6.9](#c6-9)，在一个三维后果目标的合成线性高斯实验上进行。

- 迹风险等于广义特征值之和，误差 $`5.6\times10^{-17}`$（**PASS**）。
- 单次查询收益的闭式解等于两个迹之差，误差 $`9.0\times10^{-17}`$（**PASS**）。
- 广义最小二乘的 Monte Carlo 运行在 0.3% 以内重现了精确风险（容差 2%，**PASS**）。

**本页用到的设定与记号.**

- $`y=D\vartheta+\epsilon`$，$`\epsilon\sim\mathcal N(0,\Sigma)`$，$`\Sigma\succ0`$；目标 $`f=C\vartheta\in\mathbb R^k`$；损失 $`\lVert x\rVert_W^2=x^\top Wx`$。
- $`I=D^\top\Sigma^{-1}D`$，$`B=C^\top WC`$；广义特征值 $`Bv_j=\lambda_jIv_j`$；白化算子 $`\mathcal R_I=W^{1/2}CI^{-1/2}`$。
- 单位噪声的查询 $`a`$ 给 $`I`$ 加上 $`aa^\top`$，$`\Delta(a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$。

**被检验的陈述（[Proposition 6.15](../statements/06-15_linear_gaussian_target_ZH.md#s6-15)）.** 对 $`I\succ0`$，广义最小二乘具有常数风险 $`\mathrm{tr}(BI^{-1})=\sum_j\lambda_j`$，这是 minimax 风险（a）；$`\lambda_j`$ 是 $`\mathcal R_I`$ 的奇异值平方（c）；一次查询使 minimax 风险恰好下降 $`\Delta(a)`$（d）。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 6.9](#c6-9) | (a)、(c)、(d) | `v017:C11.gaussian` | PASS |

<a id="c6-9"></a>

## Check 6.9 · 精确的迹风险、特征值恒等式与单次查询收益（Proposition 6.15）

**结论.** PASS。两个恒等式都在舍入误差内成立，模拟的广义最小二乘达到了预言的风险。

**为什么做.** (a) 与 (d) 是可计算查询规则的基础：按 $`\Delta(a)`$ 给候选查询排序。迹公式、它的特征值的识别或 Sherman–Morrison 一步中的任何错误，都会让这个排序出错。Monte Carlo 运行把风险公式与一个真正运行的估计量对照，而不只是与另一个公式对照。

**设置.**

- $`d=4`$ 个参数，$`D\in\mathbb R^{9\times4}`$ 的元素独立同分布地取自标准正态（固定种子），$`\Sigma=I_9`$。
- 信息矩阵 $`I=D^\top D+0.4\,I_4`$；加上的这一项相当于先验信息（例如示范），并让 $`I`$ 条件良好。
- 目标 $`C\in\mathbb R^{3\times4}`$，元素取自标准正态，$`W=\mathrm{diag}(0.2,0.3,0.5)`$。
- 一个随机的查询向量 $`a\in\mathbb R^4`$。
- Monte Carlo：固定 $`\vartheta`$，独立抽取 200,000 次 $`y=D\vartheta+\mathcal N(0,I_9)`$，估计量 $`\hat\vartheta=(D^\top D)^{-1}D^\top y`$，与该估计量的精确风险 $`\mathrm{tr}(B(D^\top D)^{-1})`$ 比较。

**结果.**

| 量 | 取值 |
|---|---|
| $`\mathrm{tr}(BI^{-1})`$ | 0.40062 |
| $`\lvert\mathrm{tr}(BI^{-1})-\sum_j\lambda_j\rvert`$ | $`5.6\times10^{-17}`$ |
| 单次查询收益 $`\Delta(a)`$ | 0.05375 |
| $`\lvert\Delta(a)-[\mathrm{tr}(BI^{-1})-\mathrm{tr}(B(I+aa^\top)^{-1})]\rvert`$ | $`9.0\times10^{-17}`$ |
| 只用 $`D`$ 的最小二乘的精确风险 | 0.45909 |
| Monte Carlo 风险（200,000 次） | 0.45762 |

Monte Carlo 风险与精确风险相差 0.3%，在为 200,000 次抽样设定的 2% 容差之内。

**为什么支持 Proposition 6.15.** 由 $`I^{-1/2}BI^{-1/2}`$ 算出的特征值之和等于迹，这是 (a) 与 (c) 背后的恒等式；单次查询公式与对扩大后实验的直接重算一致，这是 (d)；模拟的估计量达到了迹风险，所以风险公式描述的是一个可以实际运行的估计量。minimax 性本身是一个证明，而不是数值陈述，这里不检验。

## 复现

在仓库根目录（只用 CPU，几秒钟）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验集用 NumPy（[Harris et al., 2020](#ref-harris2020array)），种子由 20261014 与检验名称确定（脚本中的 `rng_for`）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C11.gaussian`。
- `theory/v017_checks/check_v017.py`：函数 `c_gaussian`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
