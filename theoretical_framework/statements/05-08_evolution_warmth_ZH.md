# Proposition 5.8 · warmth 的演化

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-08_evolution_warmth_ZH.md) · [证明](../proofs/05-08_evolution_warmth_ZH.md) · [实验](../experiments/05-08_evolution_warmth_ZH.md) · [English](05-08_evolution_warmth.md)

**一段话.** Adam 的二阶矩是 $`\beta_2`$ 固定的指数滑动平均。这个命题计算典型梯度的 warmth 如何随二阶矩吸收的梯度评估次数演化。在平稳的 mean-field 情形下有精确规律和两侧界：warmth 只在 $`s(1-\beta_2)\ll1`$ 时按 $`1/(\beta_2s)`$ 下降，之后停在 $`(1-\beta_2)/\beta_2`$。命题还说明 bias correction 不改变 warmth，给出整个训练过程中由 warmth 控制的误差的累积界，并说明这个界的假设何时成立，包括学习率为零的 warm-up 之后。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

**Adam 与 AdamW.** Adam 带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

*Warm 状态的记号（[Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate_ZH.md#s5-4)）.*

考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

<a id="s5-8"></a>

**Proposition 5.8（warmth 的演化）.** 设二阶矩从 $`v_0=0`$ 开始，按 $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$ 逐坐标吸收梯度计算 $`G_1,G_2,\dots`$。记 $`\gamma=1-\beta_2`$。

**(a) 平稳二阶矩：精确的 mean-field warmth.** 设对所有 $`k`$ 有 $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$。定义 mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$。则

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ 关于 $`s`$ 递减，收敛到平台 $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$，不会趋于零。
- 早期它的行为像 $`1/(\beta_2s)`$：确切地说，$`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$，所以近似 $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ 的相对误差至多为 $`s(1-\beta_2)`$，只有在 $`s(1-\beta_2)\ll1`$ 时才准确。
- 若梯度恒为常数 $`G_k=g`$，上式对 warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 本身成立，不只对它的 mean-field 版本成立。

**(b) bias correction 不改变 warmth.** 用 bias-corrected 分母 $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ 与 $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$ 时，比值 $`d_i/a_i=\sqrt{1+\kappa_i}`$ 与不带 bias correction 时相同。所以 warmth 以及 Proposition 5.5(b) 的界都不受 bias correction 影响。

**(c) 沿训练过程的累积.** 经由 [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的 telescoping 恒等式，比较参考协议 $`0`$ 与替代协议 $`S`$ 在 $`W`$ 次 warm-up 计算之后的 $`R`$ 次真实更新 $`s=1,\dots,R`$。令 $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ 为这样的更新：在每个访问到的状态上，Proposition 5.6 的假设以下标为 $`s`$ 的常数成立，两个分布的平均 consumed gradient 相同，并且几乎必然地对某个常数 $`c>0`$ 有下式。

```math
\max_i\kappa_i\le c\Big(\frac1{W+s}+(1-\beta_2)\Big).
```

则

```math
\lvert L(0)-L(S)\rvert\le\big\lvert\lbrace s\le R:s\notin\mathcal S_{\rm warm}\rbrace\big\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
```

其中 $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$，$`\bar u_{i,s}`$ 是第 $`s`$ 次更新时 $`u_i`$ 的上界，$`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$。当 $`W\ge1`$ 时，

```math
\sum_{s=1}^{R}\min\Big\lbrace1,\;\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace\le c\log\Big(1+\frac RW\Big)+c(1-\beta_2)R .
```

- 所以由 warmth 控制的部分每次更新都有一个量级为 $`c(1-\beta_2)`$ 的下限，不随训练变长而消失。只有当 $`(1-\beta_2)R\lesssim\log(1+R/W)`$ 时对数项才占主导。
- $`W=0`$ 时，第一次真实更新从 $`v=0`$ 开始，不满足 (W)：它是 Proposition 5.3 的 cold 步，记为至多 $`1`$；由 Proposition 5.7，它的贡献可以是 $`\eta`$ 的一阶量。若 $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$，由 warmth 控制的和至多为 $`c\log R+c(1-\beta_2)(R-1)`$。

**(d) Warm-up，以及 (c) 的假设何时成立.** 长度为 $`W`$ 的*矩 warm-up*，用声明数据的梯度做 $`W`$ 次学习率恰为 $`0`$ 的 optimizer 步。它不改变 $`\theta`$，并按与真实步相同的递推推进 $`m`$、$`v`$ 与计数器；被 $`v`$ 吸收的每个梯度，无论来自 warm-up 还是真实更新，都算一次计算。所以第 $`s`$ 次真实更新面对的二阶矩已吸收 $`W+s-1`$ 次计算。若把实际的 $`v`$ 换成其均值（mean-field 近似），并且梯度满足 $`G_{k,i}^2\le\bar\gamma\,\sigma_i^2`$，则当 $`W+s\ge2`$ 时 (c) 的假设以 $`c=2\bar\gamma/\beta_2`$ 成立。把 $`v`$ 换成均值是一个近似；它需要指数滑动平均的集中性和平稳的梯度尺度。

**说了什么.** 梯度平稳时，吸收 $`s`$ 次计算后的 mean-field warmth 恰为 $`(1-\beta_2)/(\beta_2(1-\beta_2^s))`$。只有在 $`s(1-\beta_2)\ll1`$ 时它才按 $`1/(\beta_2s)`$ 下降，之后停在 $`(1-\beta_2)/\beta_2`$。沿整个训练累积，由 warmth 控制的误差至多 $`c\log(1+R/W)+c(1-\beta_2)R`$；bias correction 不改变 warmth；学习率为零的 warm-up 推进二阶矩的方式与真实步完全相同。

**直觉.** $`\beta_2`$ 固定就意味着平均窗口固定，所以每个新梯度都会按固定的相对幅度扰动分母。

**在框架中的作用.** 确定 warm 描述在训练中准确到什么程度，并说明 $`\beta_2`$ 固定时它永远不会变成精确的。

**推导要点.**

- 展开指数滑动平均并取期望；两侧界由 Bernoulli 不等式与 $`e^{-y}(1+y)\le1`$ 得到。
- 对 (c)，telescoping 恒等式中每个 warm 项用 Proposition 5.6 控制，再用 $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ 与 $`\sum_s1/(W+s)\le\log(1+R/W)`$。

**例示.** $`\beta_2=0.999`$ 时，前一百次计算里典型梯度的 warmth 约为 $`1/s`$，但它从不会降到 $`0.001`$ 以下太多：几千步之后，每个新梯度仍然把二阶矩扰动约千分之一，这个下限给累积误差加上一个关于更新次数线性的项。

**用到.**

- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) — warm Adam 关于 consumed gradient 一阶仿射
- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7) — cold 步在一阶就分开均值相同的反馈
- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — continuation response 刻画

**被用于.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器的一阶统计量
- [Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) — 在真实模型上检验 warm 状态的条件
