# Remark 8.5 · 范数与常数

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-05_norms_constants_ZH.md) · [实验](../experiments/08-05_norms_constants_ZH.md) · [English](08-05_norms_constants.md)

**一段话概括.** Proposition 8.4 的集中界不显含维数，这一点容易被读过头。这个 Remark 精确说明它在哪个范数下成立、增量界为什么仍可能随维数增长，以及为什么做排序决策并不需要完整的界。

**本页用到的设定与记号.**

沿用 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

<a id="s8-5"></a>

**Remark 8.5（范数与常数）.**

- [Proposition 8.4(a)](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) 的集中界在状态空间的 Hilbert 范数下成立，证明用到了内积（交叉项为零）。它不能以同样的常数推广到任意范数；例如在 $`\mathbb R^p`$ 的 sup 范数下，相应的界带有 $`\sqrt{\log p}`$ 因子。
- 界中不显含维数，但 $`b`$ 本身可以随维数增长。cold-start Adam 下每个坐标都移动约 $`\eta`$（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)），所以 $`\lVert\Delta\theta\rVert`$ 可以达到 $`\eta\sqrt p`$。
- 做排序决策只需要标量投影 $`\langle\nabla V(z),\Delta\rangle`$，它的波动可以远小于 $`\lVert\nabla V(z)\rVert\,b`$。[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 更进一步：它直接测量有限替换的效应，不需要 $`b`$ 的界。
- **代价对比.** 在一个状态上做完整尾段 audit，每个候选需要 [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 的副本数；(b) 每个候选只需 $`n_P+n_Q`$ 次梯度计算。(b) 单独做不到的，是给出 $`\Lambda`$ 与 $`M_2`$。

**说了什么.** Proposition 8.4 的界在状态空间的 Hilbert 范数下成立，换成任意范数常数就不同；增量界可以随维数增长（cold-start Adam 会让每个坐标都动）；排序只需要一个标量投影；而 Theorem 8.8 根本不需要增量界。

**直觉.** 不显含维数的意思是常数里没有维数，不是维数没有影响。

**在框架中的作用.** 它让一阶测量对自己的常数保持诚实，并引向 Theorem 8.8 的有限测量。

**用到.**

- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) — 被限定的集中界
- [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) — cold-start Adam 一步的大小

**被用于.**

- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) — 一阶证书
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) — 不需要增量界的测量

## Related work

- 这里被限定的 Hilbert 范数界由 Jensen 不等式与 McDiarmid 有界差分不等式（[McDiarmid, 1989](#ref-mcdiarmid1989method)）组合而来；内积只通过交叉项为零这一步进入。
- cold-start Adam 下 $`b`$ 随维度增长，反映的是 Adam 第一步的类 sign 行为（[Kingma & Ba, 2015](#ref-kingma2015adam)）；Adam 的 sign 视角见 [Balles & Hennig, 2018](#ref-balles2018dissecting)。
