# Proposition 4.12 · response 经过下一步 predictive state 分解

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-12_response_predictive_state_factorization_ZH.md) · [证明](../proofs/04-12_response_predictive_state_factorization_ZH.md) · [实验](04-12_response_predictive_state_factorization_ZH.md) · [English](04-12_response_predictive_state_factorization.md)

**本页检验什么.**

- **(a)–(b) 分解与核相等.** 在一个有 9 个消息、3 维 predictive 基与 6 个测试的随机目录上，response operator 与 $`C\mathcal T_z`$ 之差为 $`1.4\times10^{-16}`$，两个核之差为 $`1.4\times10^{-15}`$（**通过**）。
- **(d) 带残差的版本.** 给每个测试值加上残差后，在一个随机单位输入上偏离 $`C\mathcal T_z`$ 的量为 $`0.0084`$，低于界 $`\epsilon_{\rm pred}=0.0447`$（**通过**）。

**本页用到的设定.**

- 有限消息，参考权重 $`\rho`$；测试的 response 为 $`g_a(y)=k_a+c_a^\top\Gamma(y)`$（(d) 中再加残差）；$`\mathcal T_zh=\mathbb E_\rho[\Gamma h]`$；response operator 取在正交归一输入坐标 $`x=\sqrt\rho\odot h`$ 下。

**被检验的陈述（[Proposition 4.12](../statements/04-12_response_predictive_state_factorization_ZH.md#s4-12)）.** (a) $`\mathcal R_z=C\mathcal T_z`$；(b) 若 $`C`$ 列满秩，则 $`\ker\mathcal R_z=\ker\mathcal T_z`$；(d) 有残差时，$`\lVert\mathcal R_zh-C\mathcal T_zh\rVert_\omega\le\epsilon_{\rm pred}\lVert h\rVert_\rho`$。

| Check | 检验内容 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 4.19](#c4-19) | (a)、(b)、(d) | `v017:C7.bridge` | 通过 |

<a id="c4-19"></a>

## Check 4.19 · 分解、核相等与残差界（Proposition 4.12）

**结论.** 分解与核相等在舍入误差内成立；带残差的版本留有充分余量。

**为什么做.**

- 分解依赖于中心化：因为 $`h`$ 的 $`\rho`$ 均值为零，常数 $`k_a`$ 必须消失。用含随机常数的 response 构造 $`\mathcal R_z`$，正是检验这一步。
- 核相等需要列满秩；$`p=6`$ 个测试、$`d=3`$ 个特征时 $`C`$ 列满秩，两个零空间从两个方向比较。

**设置.**

- $`m=9`$ 个消息，$`\rho`$ 来自平坦 Dirichlet 分布；$`\Gamma\in\mathbb R^{3\times9}`$、$`C\in\mathbb R^{6\times3}`$、$`k\in\mathbb R^6`$ 取标准正态；response 为 $`g=k\mathbf 1^\top+C\Gamma`$。
- $`\mathcal T_z`$ 与 $`\mathcal R_z`$ 在坐标 $`x=\sqrt\rho\odot h`$ 下写成矩阵，先在 $`\rho`$ 下中心化。
- 核的比较：$`\lVert\mathcal R_zN_T\rVert+\lVert\mathcal T_zN_R\rVert`$，$`N_T,N_R`$ 为两个零空间的正交归一基。
- 带残差的版本：尺度为 $`0.05`$ 的加性残差 $`e`$，均匀测试权重 $`\omega_a=1/6`$，中心化空间中的随机单位输入；$`\epsilon_{\rm pred}`$ 用未中心化的残差计算，即 (4.24) 中较大的那个界。

**结果.**

| 量 | 值 | JSON 键（条目 `v017:C7.bridge`） |
|---|---|---|
| $`\lVert\mathcal R_z-C\mathcal T_z\rVert`$（最大元素） | $`1.4\times10^{-16}`$ | `factorization_error` |
| 核相等的误差 | $`1.4\times10^{-15}`$ | `kernel_equality_error` |
| 带残差时 $`\lVert\mathcal R_zh-C\mathcal T_zh\rVert_\omega`$，$`\lVert h\rVert_\rho=1`$ | $`0.0084`$ | `residual_lhs` |
| 界 $`\epsilon_{\rm pred}`$ | $`0.0447`$ | `residual_bound` |

**为什么支持这个陈述.** 由测试 response 构造的 response operator 恰好是 $`C\mathcal T_z`$，常数被中心化去掉；$`C`$ 列满秩时两个零空间重合；残差算子不超过 $`\epsilon_{\rm pred}\lVert h\rVert_\rho`$。

## 复现

检验用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C7.bridge`；`theory/v017_checks/check_v017.py` 中的函数 `c_bridge`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
