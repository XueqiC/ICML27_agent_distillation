# Remark 8.5 · 范数与常数

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-05_norms_constants_ZH.md) · [实验](../experiments/08-05_norms_constants_ZH.md) · [English](08-05_norms_constants.md)

**本页检验什么.** 本结果由下面摘要中的检验一并检验。每条摘要说明该检验对本结果检验什么、设置与关键数字；完整描述在标题所链接的页面上。

**本页用到的设定与记号.**

沿用 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

**被检验的陈述（[Remark 8.5](../statements/08-05_norms_constants_ZH.md#s8-5)）.**

- [Proposition 8.4(a)](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) 的集中界在状态空间的 Hilbert 范数下成立，证明用到了内积（交叉项为零）。它不能以同样的常数推广到任意范数；例如在 $`\mathbb R^p`$ 的 sup 范数下，相应的界带有 $`\sqrt{\log p}`$ 因子。
- 界中不显含维数，但 $`b`$ 本身可以随维数增长。cold-start Adam 下每个坐标都移动约 $`\eta`$（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)），所以 $`\lVert\Delta\theta\rVert`$ 可以达到 $`\eta\sqrt p`$。
- 做排序决策只需要标量投影 $`\langle\nabla V(z),\Delta\rangle`$，它的波动可以远小于 $`\lVert\nabla V(z)\rVert\,b`$。[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 更进一步：它直接测量有限替换的效应，不需要 $`b`$ 的界。
- **代价对比.** 在一个状态上做完整尾段 audit，每个候选需要 [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 的副本数；(b) 每个候选只需 $`n_P+n_Q`$ 次梯度计算。(b) 单独做不到的，是给出 $`\Lambda`$ 与 $`M_2`$。

### [Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md#c8-13) · 平均增量的集中性（Proposition 8.4(a)、Remark 8.5）

- **它对 Remark 8.5 检验什么.** 在增量界 $`b`$ 固定时，Proposition 8.4(a) 的集中界是否随维数变差；以及“$`b`$ 本身可以随维数增长”这一点如何体现。
- **设置.** 维数 $`p\in\lbrace1,10,1000,100000\rbrace`$，样本数 $`n\in\lbrace8,32,128\rbrace`$，$`\delta\in\lbrace0.05,0.2\rbrace`$；三个 $`b=1`$ 的有界律（sign 类、裁剪高斯、稀疏尖峰）；每个单元格 20,000 次试验，精确抽样，不构造 $`p`$ 维向量。
- **关键数字.** 全部 144 个单元格的失败率都远低于 $`\delta`$；实测分位数与界之比的最大值在 $`p=1`$ 时为 0.615，在 $`p=100{,}000`$ 时为 0.574，$`b`$ 固定时没有任何量随 $`p`$ 增长。对未归一化的 sign 类增量（例如 cold-start Adam 的第一步，[Kingma & Ba, 2015](#ref-kingma2015adam)），$`b=\eta\sqrt p`$，一阶差也是同一尺度，所以相对精度仍由 $`1/\sqrt n`$ 决定。
- **结论.** PASS。在 Hilbert 范数下界中不显含维数，与该 Remark 一致；$`b`$ 的尺度是另一个问题。

## 复现

复现命令与结果文件见主检验所在的页面：[Proposition 8.4](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md)。该检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
