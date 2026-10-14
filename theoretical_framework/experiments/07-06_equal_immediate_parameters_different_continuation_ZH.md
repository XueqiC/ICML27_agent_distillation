# Proposition 7.6 · 即时参数相同并不决定后续

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [证明](../proofs/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [实验](07-06_equal_immediate_parameters_different_continuation_ZH.md) · [English](07-06_equal_immediate_parameters_different_continuation.md)

**本页检验什么.**

- **(c) 精确实例（[Check 7.8](#c7-8)）.** 通过。比值恒等式在有理数运算下精确成立（两个梯度都是 $`\tfrac{1960}{1249}`$）；浮点计算中两个第二步参数的差小于 $`10^{-12}`$，矩不同，第三步参数相差 $`-0.0712027`$。
- **(d) $`\epsilon`$ 为正（[Check 7.8](#c7-8)）.** 通过。$`\epsilon=10^{-8}`$ 时，根 $`g_B(\epsilon)\approx2.662025475`$ 让第二步参数重新相等（差小于 $`10^{-12}`$），第三步之差仍为 $`-0.0712027`$。
- **(e) warm 状态下的恒等式（[Check 7.9](#c7-9)）.** 通过。在 1,000 组随机状态与梯度上，$`v`$ 的相对变化与 $`(1-\beta_2)(g^2/v-1)`$ 的误差为 $`2.0\times10^{-16}`$；尽管 $`1-\beta_2=0.001`$，相对变化最大达到 $`0.119`$。动量的兑现份额是当步 $`0.1`$、十步之内 $`0.651`$。
- **真实模型上的对应.** 状态移植检验已预先注册，结果尚未得到；本页不报告任何真实模型结果。

**本页用到的设定与记号.**

- Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）的单个坐标，$`\beta_1=0.9`$、$`\beta_2=0.999`$，bias correction 为 $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$、$`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$，无 weight decay，学习率 $`\eta=1`$；除特别说明外 $`\epsilon=0`$。
- 步数计数为 $`s`$ 时，consumed gradient $`g`$ 给出 $`m'=\beta_1m+(1-\beta_1)g`$、$`v'=\beta_2v+(1-\beta_2)g^2`$ 与 $`\theta'=\theta-\eta q_{1,s}m'/(\sqrt{q_{2,s}v'}+\epsilon)`$。
- checkpoint $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$ 由 $`(0,0,0)`$ 用梯度 $`1`$ 走一步得到。第二步的两条消息的 consumed gradient 为 $`g_A=\tfrac12`$ 与 $`g_B=\tfrac{2103}{790}`$；第三步共用梯度 $`1`$。

**被检验的陈述（[Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）.**

- (c) 在 $`z_1`$ 处，$`g_A`$ 与 $`g_B`$ 满足 $`(0.9+g)^2/(0.999+g^2)=\tfrac{1960}{1249}`$；它们给出相同的第二步参数 $`\theta_2\approx-1.932179639`$ 和不同的矩，经过共同的第三步后 $`\theta_3^A-\theta_3^B\approx-0.0712027`$，sigmoid 得分之差 $`\approx-0.00364994`$。
- (d) 对小的 $`\epsilon>0`$，在 $`\tfrac{2103}{790}`$ 附近有一个梯度 $`g_B(\epsilon)`$ 让第二步参数重新相等，第三步之差仍不为零。
- (e) 在任何 $`v>0`$ 的状态下，

```math
\frac{v'-v}{v}=(1-\beta_2)\Big(\frac{g^2}{v}-1\Big),
```

并且在 $`v`$ 固定、尾段梯度不依赖 $`\theta`$ 时，前 $`K`$ 次更新承担梯度扰动总一阶位移中 $`1-\beta_1^{\,K}`$ 的份额。

| 检验 | 检验对象 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 7.8](#c7-8) | (c) 与 (d) | `v017:C14.memory` | 通过 |
| [Check 7.9](#c7-9) | (e) | `v017:C16.warm` | 通过 |

<a id="c7-8"></a>

## Check 7.8 · 即时参数相同，下一步不同（Proposition 7.6(c),(d)）

**结论.** 通过。两个梯度给出完全相同的第二步平方步长 $`\tfrac{1960}{1249}`$，浮点计算中第二步参数相同，矩不同，第三步参数相差 $`-0.0712027`$。$`\epsilon=10^{-8}`$ 时构造依然成立。

**为什么做.**

- (c) 断言的是一个精确的巧合，所以检验用精确的有理数运算来核对，不只看浮点数。
- (d) 断言这个巧合不是 $`\epsilon=0`$ 的产物。检验用求根找到扰动后的梯度，并确认下一步之差在所示位数上不变。

**设置.**

- 精确的有理数检验：用 Python 分数在 $`g=\tfrac12`$ 与 $`g=\tfrac{2103}{790}`$ 处计算 $`f(g)=(\tfrac9{10}+g)^2/(\tfrac{999}{1000}+g^2)`$；断言两者完全相等。
- 用 float64 浮点运行该实例的三步 Adam。
- $`\epsilon`$ 为正：三步都改用 $`\epsilon=10^{-8}`$ 重跑；$`g_B(\epsilon)`$ 是 $`\theta_2(g,\epsilon)-\theta_2(\tfrac12,\epsilon)`$ 在 $`[2,4]`$ 中的根，用 Brent 方法求得，容差 $`10^{-15}`$。
- 断言：比值精确相等；$`\epsilon=0`$ 与 $`\epsilon=10^{-8}`$ 时第二步参数之差都小于 $`10^{-12}`$；sigmoid 得分之差的绝对值大于 $`3\times10^{-3}`$；$`\epsilon=10^{-8}`$ 时第三步之差的绝对值大于 $`0.07`$。

**结果.**

| 量 | 值 |
|---|---|
| $`g_A`$ 与 $`g_B`$ 处的 $`(0.9+g)^2/(0.999+g^2)`$ | $`\tfrac{1960}{1249}`$ 与 $`\tfrac{1960}{1249}`$（精确） |
| checkpoint $`z_1=(\theta_1,m_1,v_1)`$ | $`(-1,\;0.1,\;0.001)`$ |
| 第二步参数 $`\theta_2`$，两边 | $`-1.932179639`$ |
| $`g_A`$ 之后的矩 $`(m_2,v_2)`$ | $`(0.14,\;0.001249)`$ |
| $`g_B`$ 之后的矩 $`(m_2,v_2)`$ | $`(0.356203,\;0.00808538)`$ |
| 第三步参数之差 $`\theta_3^A-\theta_3^B`$ | $`-0.0712027`$ |
| sigmoid 得分之差 | $`-0.00364994`$ |
| $`\epsilon=10^{-8}`$ 时的 $`g_B(\epsilon)`$ | $`2.662025475`$ |
| $`\epsilon=10^{-8}`$ 时第三步之差 | $`-0.0712027`$ |
| 步数计数为 2 时 $`v`$ 的 bias correction，$`1/(1-\beta_2^{\,2})`$ | $`500.25`$ |

**为什么支持该陈述.**

- 精确相等排除了舍入造成的巧合：两条消息留下相同的参数，这是代数上的事实。
- 两边的矩相差很大，$`m`$ 约差 $`2.5`$ 倍，$`v`$ 约差 $`6.5`$ 倍；用同一个损失做下一次更新，两个参数就已经相差 $`0.071`$。任何只记录消息之后参数的接口，都会认为这两条消息的后果相同。
- 最后一行说明这为什么是冷启动附近的构造：步数计数为 2 时，bias correction 因子约为 $`500`$，单个梯度就主导了 $`v`$。

**局限.** 单个坐标、三步。本检验核对的是构造本身，而不是真实网络中记忆效应的大小；后者是下面预先注册的移植检验要回答的。

<a id="c7-9"></a>

## Check 7.9 · warm 状态下记忆携带了什么（Proposition 7.6(e)）

**结论.** 通过。$`v`$ 的相对变化在舍入误差内服从 $`(1-\beta_2)(g^2/v-1)`$，样本中最大达到 $`0.119`$，而混合系数只有 $`0.001`$。动量在当步兑现一个扰动一阶位移的 $`0.1`$，十步之内兑现 $`0.651`$。

**为什么做.**

- 关于 $`v`$ 的恒等式纠正了一种常见的读法：把 $`1-\beta_2`$ 当作一次更新对 $`v`$ 的影响大小。它只是混合系数，影响的大小随 $`g^2/v`$ 变化。
- 动量份额说明，只读即时更新的接口能看到一条消息一阶效应的多少。

**设置.**

- 1,000 组随机的 $`(v,g)`$，$`v`$ 在 $`[0.1,2]`$ 上均匀，$`g`$ 服从标准差为 $`2`$ 的正态分布；把精确更新 $`v'=\beta_2v+(1-\beta_2)g^2`$ 与恒等式对比。
- 动量份额：$`\beta_1=0.9`$ 时 $`(1-\beta_1)\beta_1^{\,k}`$ 的部分和，以及前 20,000 项之和与 $`1`$ 的比较。

**结果.**

| 量 | 值 |
|---|---|
| 相对变化恒等式的最大误差 | $`2.0\times10^{-16}`$ |
| 样本中 $`v`$ 的最大相对变化 | $`0.119`$ |
| 消息自己这一步承担的一阶位移份额 | $`0.100`$ |
| 前十次更新承担的份额 | $`0.651`$ |

**为什么支持该陈述.** 恒等式是精确的，而样本表明一次更新可以让 $`v`$ 变化超过混合系数的一百倍。份额说明，在动量下，一个梯度一阶位移的十分之九是在消息自己这一步之后才兑现的，只读这一步的接口只看到十分之一；在 (e) 的前提下，其余部分是常数倍，所以排序只能经由曲率或 $`v`$ 改变。

**局限.** 动量份额假设 $`v`$ 保持不变、不做 bias correction、尾段梯度不依赖 $`\theta`$，与 (e) 的陈述一致。

## 真实模型上的对应（已预先注册，结果待出）

状态移植检验（[预注册文档](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md)）是本命题机制在真实模型上的检验。

- 在 ALFWorld（[Shridhar et al., 2021](#ref-shridhar2021alfworld)）蒸馏配方的一个 warm checkpoint 上，参照分支与备选反馈分支各跑相同次数的更新。然后按 2×2 设计重新组合它们的参数、动量与二阶矩：参照参数分别接上参照或备选的动量、参照或备选的二阶矩；另加一个重复的参照臂和自然的备选状态。
- 所有臂之后都跑相同的参照更新；主读数是留出的 surrogate 得分，判定阈值在任何训练之前就已写定。
- 结果尚未得到。运行完成后，无论正负，本页都会报告。

## 复现

在仓库根目录下运行（仅 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

本检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C14.memory` 与 `v017:C16.warm`。
- `theory/v017_checks/check_v017.py`：函数 `c_memory` 与 `c_warm`，以及共用的单坐标 Adam 步 `adam_step`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
