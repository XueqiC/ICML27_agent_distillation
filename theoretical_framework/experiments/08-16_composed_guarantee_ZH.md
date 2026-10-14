# Corollary 8.16 · 由复用、估计与编译组合而成的保证

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-16_composed_guarantee_ZH.md) · [证明](../proofs/08-16_composed_guarantee_ZH.md) · [实验](08-16_composed_guarantee_ZH.md) · [English](08-16_composed_guarantee.md)

**本页检验什么.**

- **(c)：单步的大小.** 20 个候选、每格 10 次复现（$`\varepsilon_n=0.578`$），实测评估噪声下 10 对漂移（$`\hat\Delta=0.718`$），$`B=2`$ 时绝对的单步下限 $`B(\epsilon_j+\Delta)`$ 为 2.59。还没有对各步求和，它就已超过整个差距的平凡界 1：这个保证在该量级下**成立但为空**。
- **有符号和对比逐项认证.** 对 $`R=20`$ 项、$`N=400`$ 次查询的有符号和估计量做 Monte Carlo，方差为 1.011，精确值 $`\sigma^2R^2/N=1`$（**PASS**）。在单位噪声、单位目标误差下，有符号和需要 400 次查询；把每一项认证到 $`\varepsilon/R`$ 再把绝对值相加需要 8,000 次。

**本页用到的设定与记号.**

- 参考（目标）与编译出的合法 feedback law $`P^0_t`$、$`P^S_t`$；近似价值 $`\widehat V_t`$；局部 response 差 $`d_t`$；Bellman 残差 $`r^0_t`$；终点误差 $`e_R`$；退出时刻 $`\tau^0`$、$`\tau^S`$（[Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value_ZH.md#s3-4)）。
- 在锚点为 $`j`$ 的被认证状态上：信息–编译半径 $`\mathfrak r_{t,z}`$ 与求解误差 $`\eta_{\rm opt}`$（[Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16)）；锚点第一个被丢弃的奇异值 $`\hat\sigma_{j,r_j+1}`$ 与估计误差 $`\epsilon_j`$（[Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）；漂移 $`\Delta_{t,z}`$（[Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)）；密度界 $`B_{t,z}`$；读出 $`\widehat V_{t+1}`$ 的测试的权重 $`\omega_{a_t}`$。

**被检验的陈述（[Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16)）.** 在每个被认证的状态上

```math
\lvert d_t(z)\rvert\le\beta_t(z)=\frac{\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}\big(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z}\big)}{\sqrt{\omega_{a_t}}},
```

且 $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit}`$；在实测量级下，$`B=2`$ 时单步下限为 2.59，高于平凡界。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 8.23](#c8-23) | (c) 与成本比较 | `v017:C22.composed` | PASS；成立但为空 |

<a id="c8-23"></a>

## Check 8.23 · 单步下限，以及有符号和对比逐项认证（Corollary 8.16）

**结论.**

- 单步下限 $`B(\epsilon_j+\Delta)=2\times(0.578+0.718)=2.59`$，$`\beta_t`$ 的其他项都非负。若初始状态被认证，组合界至少为 2.59；若不被认证，单是退出项就等于 1。差距至多为 1，所以无论哪种情况，这个界在该量级下都**为空**。下限是 $`1.296\,B`$：$`B=2`$ 是两候选的最坏情形，20 个候选的目录最坏可达 $`\sqrt{40}\approx6.3`$，离目标很近的编译律可能只有 $`B=0.5`$（下限 0.65）。即便如此，两个及以上被审计步之和也超过 1。
- 有符号和估计量的方差符合预测的 $`\sigma^2R^2/N`$（Monte Carlo 1.011，精确值 1.000）：**PASS**。

**动机.** 这个推论按构造成立；问题在于它能不能有信息量。它的锚点项与漂移项来自有界取值的集中不等式，大小与单步效应有多小无关。这个检验代入 [Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md#c8-18) 与 [Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md#c8-21) 的半径，并把认证方案与直接估计有符号和作比较。

**设置.**

- *下限.* 一个测试（$`\omega_{a_t}=1`$），$`B=2`$；$`\epsilon_j=\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$，$`p=1`$、$`m=20`$、$`n=10`$、$`\delta=0.05`$；$`\Delta=\hat\Delta`$ 取 [Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 的值，$`\bar W=(\sqrt{15}/300)^2`$，10 对，$`M=10`$，$`\delta=0.05`$。
- *有符号和.* $`R=20`$ 项 $`a_t`$，从标准正态各抽一次；$`N=400`$ 次查询均匀分配，每项 20 次，每次是 $`\sigma=1`$ 的带噪观测；估计量是各项均值之和；Monte Carlo 重复 20,000 次；NumPy（[Harris et al., 2020](#ref-harris2020array)）。
- *次数*（$`\sigma=\varepsilon=1`$）：有符号和为 $`\sigma^2R^2/\varepsilon^2`$；逐项认证到 $`\varepsilon/R`$，每项 $`\sigma^2R^2/\varepsilon^2`$ 次查询，再乘以 $`R`$ 项（省略 union bound 的对数因子）。

**结果**（`theory/v017_checks/results_v017.json`，条目 `v017:C22.composed`）：

| 量 | 值 |
|---|---:|
| $`\varepsilon_n`$（$`p=1`$、$`m=20`$、$`n=10`$） | 0.578 |
| $`\hat\Delta`$（10 对，实测噪声） | 0.718 |
| 单步下限 $`B(\epsilon_j+\Delta)`$ | 2.59 |
| 有符号和的方差，Monte Carlo | 1.011 |
| 有符号和的方差，精确值 $`\sigma^2R^2/N`$ | 1.000 |
| 单位误差下的查询次数，有符号和 | 400 |
| 单位误差下的查询次数，逐项绝对值认证 | 8,000 |

**为什么支持该陈述.**

- (c)：下限由两个半径精确算出；它超过平凡界，所以在这个量级下组合保证是一份误差预算，而不是有用的证书。对 $`R`$ 步求和，下限至多再乘 $`R`$。
- 这个比较说明浪费来自哪里：认证方案把 $`R`$ 个大小为 $`\varepsilon/R`$ 的绝对半径相加，比估计有符号和多花 $`R`$ 倍的查询。这是该方案的成本，不是问题的下界；它也把"估计一对协议的收益"与"认证每个局部 response"区分开来。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验程序使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C22.composed`；代码 `theory/v017_checks/check_v017.py`，函数 `c_composed`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
