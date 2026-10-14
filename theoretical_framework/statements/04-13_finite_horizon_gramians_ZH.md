# Proposition 4.13 · 有限时域 Gramian，以及控制理论中不能照搬的部分

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-13_finite_horizon_gramians_ZH.md) · [证明](../proofs/04-13_finite_horizon_gramians_ZH.md) · [实验](../experiments/04-13_finite_horizon_gramians_ZH.md) · [English](04-13_finite_horizon_gramians.md)

**一段话.** 把训练看成受控系统，带来一个有用的想法，也招来几个错误的想法。有用的是：learner 状态的某个方向对声明的未来有影响，必须同时满足两点，feedback 能推动它、未来测试能看到它；在有限时域上，对线性化映射（先不计余项 $`\epsilon_t,\zeta_t`$），这恰好由两个 Gramian 度量，不需要任何稳定性假设；对实际训练，这个恒等式只在这些余项之内成立。一个在环境里很复杂、但合法 feedback 推不动的方向，或者推得动、但测试看不见的方向，都没有贡献；而 optimizer state 的一个小变化，若测试看得很清楚，就可能很重要。错误的是：照搬 balanced truncation 的先验误差界、线性时不变系统的稳定性，以及"一个 Hankel 块的低秩拟合就是因果模型"的假定。本页证明前者，并列出后者。

**本页用到的设定.**

*可选的局部线性化.* 在实际更新沿某条参考路径可微的地方，把状态、输入与输出的扰动写成

```math
\delta z_{t+1}=A_t\,\delta z_t+B_t\,\delta u_t+\epsilon_t,\qquad \delta o_t=C_t\,\delta z_t+\zeta_t ,
```

非线性余项 $`\epsilon_t,\zeta_t`$ 保留，不丢掉。$`\delta z`$ 是 complete learner state 的扰动（[Definition 1.3](../README_ZH.md#s1-3)），$`\delta u`$ 是 feedback 的扰动，$`\delta o`$ 是声明输出的扰动。

- *过去与未来.* 在 $`\tau`$ 处把时域分开。初始扰动为零、暂不计余项时，$`\delta z_\tau=\mathscr C_\tau\,\delta u_{\rm past}`$，$`\delta o_{\rm future}=\mathscr O_\tau\,\delta z_\tau`$，其中 $`\mathscr C_\tau=[\Phi(\tau,1)B_0,\dots,\Phi(\tau,\tau)B_{\tau-1}]`$，$`\mathscr O_\tau`$ 把 $`t=\tau,\dots,N`$ 的 $`C_t\Phi(t,\tau)`$ 叠起来，$`\Phi(t,j)=A_{t-1}\cdots A_j`$，$`\Phi(j,j)=I`$。*有限时域 Hankel 算子*为 $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$。
- 过去的输入带一个声明的能量范数，未来的输出带一个声明的加权范数；两者都通过平方根加权并入 $`B_t`$ 与 $`C_t`$，所以下面的范数都是欧氏范数。
- *Gramian.* $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$（可达性）与 $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$（可观性）；$`W_c^+`$ 是 Moore–Penrose 逆。

<a id="s4-13"></a>

**Proposition 4.13（有限时域 Gramian，以及控制理论中不能照搬的部分）.** 在上述设定下，对 $`A_t`$ 不作任何稳定性或收缩假设：

**(a) Hankel 奇异值.** 对每个 $`j`$，

```math
\sigma_j(H_\tau)^2=\lambda_j\big(W_c^{1/2}W_oW_c^{1/2}\big), \qquad(4.25)
```

两个序列都按降序排列，并用零补齐。

**(b) 状态方向的增益.** 对可达方向 $`x\in\mathrm{range}\,W_c`$，产生 $`\delta z_\tau=x`$ 所需的最小输入能量为 $`x^\top W_c^+x`$，而每个产生它的输入给出的未来输出能量都是 $`x^\top W_ox`$。因此

```math
\sigma_1(H_\tau)^2=\sup_{x\in\mathrm{range}\,W_c,\ x\ne0}\frac{x^\top W_o\,x}{x^\top W_c^+\,x}, \qquad(4.26)
```

并且 $`H_\tau=0`$ 当且仅当 $`W_oW_c=0`$，即每个可达方向都不可观。

**说了什么.**

- 在有限时域上，线性化训练映射（先不计余项 $`\epsilon_t,\zeta_t`$）的输入–输出增益精确地拆成两部分：feedback 把状态推向某个方向有多便宜，以及未来测试看这个方向有多清楚。单看任何一个都决定不了增益。对实际训练，这只在余项之内成立。
- 在环境里很复杂但不可达的方向，或者可达但不可见的方向，都没有增益。optimizer state 的某个方向（例如 Adam 的矩），对当前参数的直接影响很小，但若未来测试看得很清楚，仍可能带来很大的增益。
- 不需要稳定性；$`A_t`$ 的增益可以大于 1。

**直觉.** feedback 是输入，部署得分是输出，learner 的完整状态在两者之间。对蒸馏要紧的是能穿过去的部分：输入推得动，输出看得见。

**不能照搬的部分.**

- *balanced truncation 的误差界.* 按 Hankel 奇异值做 balancing 与截断可追溯到 [Moore (1981)](#ref-moore1981principal)，它经典的先验误差界对稳定的线性时不变系统及其实现成立。对非线性、非平稳的 Adam 训练，它不是保证，框架中也没有任何地方用到它。
- *低秩 Hankel 块不是因果模型.* 一个 Hankel 块的最佳低秩逼近，不一定具有任何有限维因果模型的平移结构。闭合必须另外检查（[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)）。
- *线性化是可选的.* 上面的局部模型是否描述真实训练，要通过测量 $`\epsilon_t`$ 与 $`\zeta_t`$ 来确认，否则分析只能限定为明确的局部分析。框架的基础恒等式，[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 与 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)，用的是精确 kernel，不依赖它。
- *换一种语言修不好失败的机制.* 平稳幅度定律（[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)）的真实模型检验没有支持它的预测；用谱的语言描述 response 并不能修复这次失败，究竟是哪个机制导致失败，需要单独的区分性实验。
- *把训练看成受控系统本身并不新.* 顺序 machine teaching 已经把教学写成最优控制问题 ([Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal))。

**在框架中的作用.**

- 它给出 [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12) 中分解的有限时域含义：那里 $`\mathcal T_z`$ 是单步的可达性映射，$`C`$ 是可观性映射；这里同样的拆分在多步上成立，并有精确的 Gramian。
- 它解释了为什么必须为完整的 optimizer state 建模（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)）：$`(m,v)`$ 的某些方向可能带着增益，而当前参数的变化显示不出来。

**推导要点.**

- (a) $`H^\top H=\mathscr C^\top W_o\mathscr C`$，而 $`XY`$ 与 $`YX`$ 有相同的非零特征值及重数；用两次即可。
- (b) $`\mathscr Cu=x`$ 的最小范数解是 $`\mathscr C^+x`$，其范数平方为 $`x^\top(\mathscr C\mathscr C^\top)^+x`$。输出只通过 $`x=\mathscr Cu`$ 依赖于 $`u`$。$`\ker\mathscr C`$ 中的输入不产生输出，所以 $`\lVert Hu\rVert^2/\lVert u\rVert^2`$ 的上确界在最小范数输入处达到。

**用到.**

- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)：单步的分解.

**被用于.** 后面没有结果依赖这个命题；它是 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 完整状态记账背后的解释。

## 相关工作

- 可达性与可观性 Gramian，以及由它们决定的 Hankel 奇异值，是 balanced realization 的基础 ([Moore, 1981](#ref-moore1981principal))；(a) 是它们在有限时域、时变情形下的形式。
- 顺序 machine teaching 把为已知 learner 选择训练样本写成最优控制 ([Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal))；另见 iterative machine teaching ([Liu et al., 2017](#ref-liu2017iterative))。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-lessard2019optimal"></a>

**Laurent Lessard, Xuezhou Zhang, Xiaojin Zhu (2019).** An optimal control approach to sequential machine teaching. *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 89:2495–2503. [proceedings.mlr.press/v89/lessard19a](https://proceedings.mlr.press/v89/lessard19a.html).

<a id="ref-liu2017iterative"></a>

**Weiyang Liu, Bo Dai, Ahmad Humayun, Charlene Tay, Chen Yu, Linda B. Smith, James M. Rehg, Le Song (2017).** Iterative machine teaching. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:2149–2158. [proceedings.mlr.press/v70/liu17b](https://proceedings.mlr.press/v70/liu17b.html).

<a id="ref-moore1981principal"></a>

**Bruce C. Moore (1981).** Principal component analysis in linear systems: Controllability, observability, and model reduction. *IEEE Transactions on Automatic Control* 26, no. 1, 17–32. DOI [10.1109/TAC.1981.1102568](https://doi.org/10.1109/TAC.1981.1102568).
