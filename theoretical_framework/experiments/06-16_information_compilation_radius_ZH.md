# Proposition 6.16 · 信息—编译半径

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-16_information_compilation_radius_ZH.md) · [证明](../proofs/06-16_information_compilation_radius_ZH.md) · [实验](06-16_information_compilation_radius_ZH.md) · [English](06-16_information_compilation_radius.md)

**本页检验什么.** 一项检验 [Check 6.10](#c6-10)，由三个小的合成情形组成。

- 完全落在 $`\ker C`$ 中的不确定性不产生代价，而一个够不着的目标仍留下 0.1 的误差，即到合法集合的距离（**PASS**）。
- 两个相距 1 的被允许目标，合法集合同时包含二者时，半径为 0.5，即直径的一半（**PASS**）。
- 对盒子型的不确定集合，内层上确界在顶点取到：盒子中 200,000 个随机点都不超过顶点上的最大值（**PASS**）。

这些都是构造情形上的精确计算；还没有构造真实的 response 接口，因为真实模型的日志尚未记录逐条目的 response，也没有声明的目标。

**本页用到的设定与记号.**

- 测试权重 $`\omega_a>0`$；接口 $`g_a(y)=k_a+c_a^\top\phi(y)+e_a(y)`$，$`C\in\mathbb R^{p\times r}`$；$`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$。
- 目标特征均值的不确定集合 $`\mathcal M\subset\mathbb R^r`$；合法像 $`\mathcal K=\lbrace C\mu_Q\rbrace`$；$`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$，$`\mathfrak r=\min_{q\in\mathcal K}\Gamma(q)`$。

**被检验的陈述（[Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16)）.** $`\mathfrak r\ge\max\lbrace\tfrac12\mathrm{diam}_\omega(C\mathcal M),\sup_\mu\mathrm{dist}_\omega(C\mu,\mathcal K)\rbrace`$（b）；沿 $`\ker C`$ 的不确定性不改变半径，$`\mathcal K`$ 之外的单个目标留下它的距离（e）；对盒子 $`\mathcal M`$，内层上确界在顶点取到（f）。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 6.10](#c6-10) | (b)、(e)、(f) | `v017:C13.radius` | PASS |

<a id="c6-10"></a>

## Check 6.10 · 看不见的不确定性、够不着的目标与顶点最大值（Proposition 6.16）

**结论.** PASS。每个情形中算出的半径都等于 (b) 的下界所预言的值（这里两个下界都被取到），盒子情形确认了 (f) 中化为顶点的做法。

**为什么做.** 这个命题的实际用处是诊断：它把信息缺口（$`C\mathcal M`$ 直径的一半）与可实现性缺口（到 $`\mathcal K`$ 的距离）分开，并说明沿 $`\ker C`$ 的不确定性是免费的。前两个情形分别隔离出一种缺口；第三个情形检验 $`\mathcal M`$ 是盒子时所用的计算化简。

**设置.**

- *看不见的不确定性、够不着的目标.* $`r=2`$，一个测试，$`C=(1,0)`$。证据允许两个情景 $`\mu=(0.2,-100)`$ 与 $`\mu=(0.2,100)`$，它们只在 $`\ker C`$ 方向上不同，所以 $`C\mathcal M=\lbrace0.2\rbrace`$。合法 response 填满 $`\mathcal K=[0,0.1]`$，在 101 个网格点上搜索。
- *信息缺口.* 一个测试，$`C\mathcal M=\lbrace0,1\rbrace`$（两个情景），$`\mathcal K=[0,1]`$，101 个网格点。
- *盒子.* $`r=4`$，$`p=3`$，$`C`$ 的元素取自标准正态，$`\mathcal M`$ 是随机盒子 $`\prod_i[l_i,u_i]`$（$`l_i<0<u_i`$），球心 $`q`$ 随机。比较 16 个顶点上 $`\max_\mu\lVert C\mu-q\rVert`$ 与盒子中 200,000 个均匀随机点上的最大值。这一项使用单位测试权重。

**结果.**

| 情形 | 量 | 取值 | 预言 |
|---|---|---|---|
| 看不见的不确定性 | $`\mathrm{diam}(C\mathcal M)`$ | 0 | 0 |
| 够不着的目标 | 半径 $`\mathfrak r`$ | 0.1 | $`\mathrm{dist}(0.2,[0,0.1])=0.1`$ |
| 信息缺口 | 半径 $`\mathfrak r`$ | 0.5 | $`\tfrac12\mathrm{diam}=0.5`$ |
| 盒子 | 16 个顶点上的最大值 | 5.7366 | 每个点的上界 |
| 盒子 | 200,000 个随机点上的最大值 | 5.6436 | 不超过 5.7366 |

**为什么支持 Proposition 6.16.** 在第一个情形里，不确定性跨越 200 个单位，半径却完全由到合法集合的距离决定：关于第二个坐标的更多证据是浪费，只有更大的合法集合才有帮助，与 (e) 完全一致。在第二个情形里，合法集合已经尽可能丰富，半径仍是直径的一半，即 (b) 的信息下限。盒子情形说明 $`r`$ 较小时内层问题可以靠枚举顶点解决；它不检验一般情形的困难性，那是一个归约，不是计算。

## 复现

在仓库根目录（只用 CPU，几秒钟）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验集用 NumPy（[Harris et al., 2020](#ref-harris2020array)），种子由 20261014 与检验名称确定（脚本中的 `rng_for`）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C13.radius`。
- `theory/v017_checks/check_v017.py`：函数 `c_radius`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
