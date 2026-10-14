# Proposition 8.12 · 有限目录的谱证书

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-12_finite_catalog_spectral_certificate_ZH.md) · [证明](../proofs/08-12_finite_catalog_spectral_certificate_ZH.md) · [实验](08-12_finite_catalog_spectral_certificate_ZH.md) · [English](08-12_finite_catalog_spectral_certificate.md)

**本页检验什么.**

- **(b)–(c)：无需间隙的残差.** 在一个逐格误差至多 0.004 的随机加权目录上，算子误差为 0.00115，估计的秩 3 接口的残差为 0.0666，低于保证值 $`\sigma_4+2\varepsilon=0.0746`$（**PASS**）。
- **(f) 与负担得起的预算下的半径.** 20 个候选、4 次复现的目录，若所有测试共用一个续训协议需要 80 次续训，若需要 10 个协议则需要 800 次。在 $`p=10`$、$`m=20`$、$`\delta=0.05`$ 时，每格 4、16、64、256 次复现对应的逐格 Hoeffding 半径为 1.06、0.53、0.26、0.13。全部超过 0.1，而 $`\sigma_1\le\tfrac12`$：证书成立，但在这个量级下是空的。

**本页用到的设定与记号.**

- 一个 checkpoint；$`p`$ 个测试，权重 $`w`$；$`m`$ 个冻结的候选，权重 $`q`$；格均值 $`G_{ij}\in[0,1]`$；每格 $`n`$ 次独立复现，平均为 $`\hat G`$；测试中有 $`p_{\rm train}`$ 个不同的续训协议。
- $`M=D_{\sqrt w}GD_{\sqrt q}\Pi_q`$，$`\Pi_q=I-\sqrt q\sqrt q^{\top}`$，奇异值 $`\sigma_k`$；$`\hat M`$、$`\hat\sigma_k`$ 以及到其前 $`r`$ 个右奇异向量上的投影 $`\hat\Pi_r`$ 都由 $`\hat G`$ 构造。
- $`\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$ 是逐格 Hoeffding 半径。

**被检验的陈述（[Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）.** 以至少 $`1-\delta`$ 的概率，所有格满足 $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$；在该事件上

```math
\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n,\qquad\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n,\qquad\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\sigma_{r+1}+2\varepsilon_n ,
```

逐测试的界因使用估计格而多出 $`2\varepsilon_n`$；完整目录需要 $`m\,n\,p_{\rm train}`$ 次续训。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 8.18](#c8-18) | (b)、(c)、(f) | `v017:C17.catalog` | PASS；在负担得起的 $`n`$ 下半径是空的 |

<a id="c8-18"></a>

## Check 8.18 · 无需间隙的截断残差与目录半径（Proposition 8.12）

**结论.**

- 估计矩阵的算子误差低于逐格界，估计接口的残差低于 $`\sigma_{r+1}+2\varepsilon`$，全程不用谱间隙：**PASS**。
- 每格 4 到 256 次复现时，半径 $`\varepsilon_n`$ 不小于 0.13。response operator 满足 $`\sigma_1\le\tfrac12`$，而真实模型上目前测到的候选差在 surrogate 上只有 $`10^{-3}`$ 量级（[Check 8.17](08-08_crn_paired_replacement_certificate_ZH.md#c8-17)）。所以在这些预算下，严格证书**成立但为空**。

**动机.**

- (b)–(c) 断言：由于权重归一，逐格误差以同样的常数控制整个谱和截断接口。把误差放在界的边缘上的随机实例，可以检验这些常数，以及不需要任何间隙条件这一点。
- (f) 与半径表在跑任何真实目录之前，就把严格目录的成本摆清楚。

**设置.**

- $`p=7`$ 个测试，$`m=11`$ 个候选，秩 $`r=3`$。权重 $`w_i`$、$`q_j`$ 从 $`[0.3,1.3]`$ 上均匀抽取并归一。$`G=\mathrm{sigmoid}(Z)`$，$`Z`$ 为标准正态。
- $`\hat G=\mathrm{clip}(G+U,0,1)`$，$`U_{ij}`$ 在 $`[-0.004,0.004]`$ 上均匀：这就是 $`\varepsilon=0.004`$ 时 (a) 的事件。
- 半径表：$`p=10`$、$`m=20`$、$`\delta=0.05`$ 时 $`n\in\lbrace4,16,64,256\rbrace`$ 的 $`\varepsilon_n`$；运行次数取 $`m=20`$、$`n=4`$、$`p_{\rm train}\in\lbrace1,10\rbrace`$。
- 随机种子 `SeedSequence([20261014, tag])`；NumPy（[Harris et al., 2020](#ref-harris2020array)）。

**结果**（`theory/v017_checks/results_v017.json`，条目 `v017:C17.catalog`）：

| 量 | 值 |
|---|---:|
| 逐格界 $`\varepsilon`$ | 0.004 |
| $`\lVert\hat M-M\rVert_{\rm op}`$ | 0.00115 |
| $`\lVert M(I-\hat\Pi_3)\rVert_{\rm op}`$ | 0.0666 |
| 保证值 $`\sigma_4+2\varepsilon`$ | 0.0746 |

| 每格复现数 $`n`$ | 4 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| $`\varepsilon_n`$（$`p=10`$、$`m=20`$、$`\delta=0.05`$） | 1.06 | 0.53 | 0.26 | 0.13 |

| 目录 | 续训次数 |
|---|---:|
| $`m=20`$、$`n=4`$、$`p_{\rm train}=1`$ | 80 |
| $`m=20`$、$`n=4`$、$`p_{\rm train}=10`$ | 800 |

**为什么支持该陈述.**

- 实测残差 0.0666 满足无需间隙的界 0.0746，算子误差 0.00115 满足逐格界，正如 (b)–(c) 所断言；$`\sigma_3`$ 与 $`\sigma_4`$ 之间的间隙既没有假设，也不需要。
- 半径表说明 (a) 的代价：它按 $`n^{-1/2}`$ 下降，每格 256 次复现时仍是 0.13。要用严格的目录证书分辨百分之几的差，所需复现数远超任何续训预算。这是基于取值范围的 Hoeffding 半径的性质，不证明不存在更便宜的证书。
- 运行次数说明，从同一次运行读出的指标是免费的：成本乘的是不同续训协议的个数，而不是测试个数。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验程序使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C17.catalog`；代码 `theory/v017_checks/check_v017.py`，函数 `c_catalog`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
