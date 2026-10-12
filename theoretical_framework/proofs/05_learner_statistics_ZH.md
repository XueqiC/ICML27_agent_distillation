# §5 学习器统计量 — 证明

[← Framework](../README_ZH.md) · [Proofs](05_optimizer_state_ZH.md) · [Experiments](../experiments/05_warm_adam_and_learners_ZH.md) · [English](05_learner_statistics.md)

本页包含 Proposition 5.9 至 Remark 5.11，沿用[原页面的设定](05_optimizer_state_ZH.md#设定)；展开式与 cold-start 证明见[原页面](05_optimizer_state_ZH.md)，warm-state 定义及证明见 [Warm Adam](05_warm_adam_ZH.md#s5-4)。

## 梯度裁剪

<a id="s5-9"></a>

**Proposition 5.9（部分激活的梯度裁剪）.** 令 $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$，$`c>0`$；约定 $`G=0`$ 时 $`G/\lVert G\rVert=0`$；令 $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ 为某个 $`G`$ 的分布下的激活率。则

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- 对两个分布，裁剪后均值与缩放后的平均方向之间的总偏差至多为 $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。例如两个分布的激活率都是 $`0.9`$ 时，用平均方向代替裁剪后均值，在差值上的误差至多为 $`0.2c`$。
- 只有在始终激活（$`p_{\rm act}=1`$）时，平均方向才恰好等于裁剪后均值。

*证明.* 第一式是按事件 $`\lbrace\lVert G\rVert\le c\rbrace`$ 拆分 $`\mathbb E\,\mathrm{clip}_c(G)`$。第二式：$`\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E[G/\lVert G\rVert]=\mathbb E\big[(G-cG/\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace\big]`$，在该事件上（包括 $`G=0`$）$`\lVert G-cG/\lVert G\rVert\rVert=\lvert\lVert G\rVert-c\rvert=c-\lVert G\rVert\le c`$。所以范数至多为 $`\mathbb E[(c-\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace]\le c\Pr(\lVert G\rVert\le c)=c(1-p_{\rm act})`$。两个分布的结论由三角不等式得到。$`\square`$

## 各学习器的统计量

**Optimizer.** 所有运算都按坐标进行。$`\Delta`$ 中与消息无关的项都会在 $`\mu^P-\mu^Q`$ 中抵消：weight decay、momentum 衰减，以及作用在旧状态上的 bias-correction 常数。

- *SGD：* $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta G`$。
- *Heavy-ball：* $`m^+=\beta m+G`$，$`\theta^+=\theta-\eta m^+`$。*Nesterov：* $`m^+=\beta m+G`$，$`\theta^+=\theta-\eta(G+\beta m^+)`$。
- *Adam 与 AdamW：* 见设定部分，warm Adam 记号见 Definition 5.4，并记 $`c_1=(1-\beta_1)/(1-\beta_1^{\,t})`$。
- *全局范数裁剪：* $`\mathrm{clip}_c(G)`$，在基础 optimizer 之前作用。
- *Lion 形式的 sign optimizer：* $`u=\mathrm{sign}(\beta_1m+(1-\beta_1)G)`$，$`\theta^+=\theta-\eta u`$，$`m^+=\beta_2m+(1-\beta_2)G`$。
- *LoRA：* 被适配的权重 $`W=W_0+\alpha BA`$，$`A\in\mathbb R^{r\times k}`$，$`B\in\mathbb R^{d\times r}`$，全权重梯度为 $`\Gamma=\nabla_W\ell`$。可训练参数为 $`(A,B)`$，标准初始化为 $`B_0=0`$，$`A_0`$ 随机。

对某状态处的两个分布 $`P,Q`$，记 $`\delta_1=\mathbb E_PG-\mathbb E_QG`$，$`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$。由 Proposition 5.1(c)，$`\mu^P-\mu^Q`$ 在 continuation 梯度张成的 $`U`$ 上的投影，正是两个 feedback law 在一阶上必须一致的量。

<a id="s5-10"></a>

**Theorem 5.10（各学习器的一阶统计量）.**

**(a) SGD.** $`\mu^P-\mu^Q=-\eta\,\delta_1`$。一阶统计量是平均 consumed gradient；二阶统计量是 $`\mathbb E\,GG^\top`$。

**(b) Heavy-ball 与 Nesterov momentum.** 增量关于 $`G`$ 仿射：heavy-ball 为 $`\Delta^\theta=-\eta(\beta m+G)`$，$`\Delta^m=(\beta-1)m+G`$；Nesterov 为 $`\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)`$，$`\Delta^m=(\beta-1)m+G`$。所以一阶项如下。统计量是平均 consumed gradient；momentum 改变的是它的权重，不是统计量本身。

```math
\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.
```

**(c) warm 状态下的 Adam 与 AdamW.** 精确的一阶统计量是 Adam 增量的均值（Proposition 5.5(a)）。在 $`(\theta,m,v)`$ 三块中，记 $`e(G)=\theta^+-\tilde\theta^+`$，$`\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace`$（Proposition 5.5(b)），

```math
\mu^P-\mu^Q=\Big(-\eta c_1\Phi\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

- 在 frozen-denominator 近似下，$`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定 $`\mu^P-\mu^Q`$；一般情形下只决定到 frozen-denominator 误差，每个坐标至多为 $`\eta\,(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]`$。相对于步长 $`\eta u_i`$，误差至多为 $`\kappa_i/2`$；由 Proposition 5.8，$`\beta_2`$ 固定时典型梯度的 warmth 不会沿训练趋于零，而是停在 $`(1-\beta_2)/\beta_2`$ 附近。
- 逐坐标的二阶矩通过 $`v`$ 在一阶上进入，权重为 $`(1-\beta_2)\nabla_vV`$。若 $`\delta_1=0`$，一阶项等于 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$，其中第二项的绝对值至多为 $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$。所以均值相同的分布可以通过二阶矩通道在一阶上不同，而且这个差别不随 warmth 变小。[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1) 用 adjoint 递推算出 $`\nabla_vV`$。

**(d) cold 状态下的 Adam.** 取 Proposition 5.3 中的 $`h_\epsilon`$，

```math
\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

参数块只看 $`\mathbb E\,h_\epsilon(G)`$，$`\epsilon\to0`$ 时就是符号分布。矩块保留 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$。若训练在这次更新后结束（$`R=1`$），continuation value 为 $`F_E(\theta)`$，一阶项为 $`-\eta\,\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle`$。

**(e) 梯度裁剪.** 对裁剪 SGD，精确地有 $`\mu^P-\mu^Q=-\eta\big(\mathbb E_P\mathrm{clip}_c(G)-\mathbb E_Q\mathrm{clip}_c(G)\big)`$，所以统计量是裁剪后均值。

- *反例.* 取 $`c=1`$，$`P(G=4e_1)=\tfrac14`$，$`P(G=-\tfrac43e_1)=\tfrac34`$，$`Q=\delta_0`$。此时 $`\delta_1=0`$，但 $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$，一阶项为 $`\tfrac\eta2\,\partial_{\theta_1}V`$。
- *平均方向.* 由 Proposition 5.9，每个分布下裁剪后均值与 $`c\,\mathbb E[G/\lVert G\rVert]`$ 的范数差至多为 $`c(1-p_{\rm act})`$。所以用缩放后的平均方向代替裁剪后均值，$`\mu^P_\theta-\mu^Q_\theta`$ 的变化在范数上至多为 $`\eta c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。
- *始终激活的裁剪.* 若在两个分布下都几乎必然有 $`\lVert G\rVert>c`$，统计量恰为 $`c\,\mathbb E[G/\lVert G\rVert]`$。若两个分布还诱导出相同的 $`G/\lVert G\rVert`$ 分布，增量分布就相同，两者在每一阶上都等价。
- *带裁剪的基础 optimizer.* 对 (a)–(d) 与 (f) 中的任一基础 optimizer，统计量就是对裁剪后梯度 $`\mathrm{clip}_c(G)`$ 取的相应统计量。

**(f) 带 momentum 的 sign optimizer（Lion 形式）.** 对 $`m_i\ne0`$，零计为半次翻转，

```math
\mathbb E\,\Delta^\theta_i=-\eta\,\mathrm{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \mathrm{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\mathrm{sign}(m_i)\rbrace ,
```

且 $`\Delta^m=(1-\beta_2)(G-m)`$。参数块的统计量是翻转概率向量 $`\pi`$。

- 若 momentum 占主导，即在两个分布下都几乎必然有 $`(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert`$，则 $`\pi_i=0`$，坐标 $`i`$ 的参数增量与消息无关。
- $`m`$ 块仍携带 $`(1-\beta_2)\delta_1`$。它的一阶效应 $`\langle\nabla_mV,(1-\beta_2)\delta_1\rangle`$ 在 continuation value 关于 $`m`$ 局部为常数处为零，也就是 $`m`$ 的小变化不会改变之后任何一步更新符号的地方。

**(g) 初始化时的 LoRA.** adapter 梯度为 $`G_A=\alpha B^\top\Gamma`$，$`G_B=\alpha\Gamma A^\top`$。在 $`B=B_0=0`$ 处：

- 对每条消息 $`G_A=0`$。在 SGD、从 $`m=0`$ 开始的 momentum 以及 cold Adam 下，第一次更新对每个分布都精确地不改变 $`A`$。
- $`B`$ 块的统计量对 SGD 与 momentum 为 $`\alpha\,\mathbb E[\Gamma]A_0^\top`$，对 cold Adam 为 $`\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)`$。
- 若两个分布的平均全权重梯度只相差满足 $`\Delta\Gamma A_0^\top=0`$ 的 $`\Delta\Gamma`$，它们在 SGD 与 momentum 下第 1 步一阶一致。在 cold Adam 下，若 sketch $`\Gamma A_0^\top`$ 的符号分布相同，$`\epsilon\to0`$ 时第一次更新相同。
- 第 1 步之后 $`B\ne0`$，$`G_A=\alpha B^\top\Gamma`$ 携带第二个 sketch，所以统计量的维数随训练增长。

**汇总.**

| 部分 | 学习器与状态 | feedback law 的一阶统计量 | 经由 |
|---|---|---|---|
| **(a)** | SGD | $`\mathbb E\,G`$ | $`\theta`$ |
| **(b)** | heavy-ball、Nesterov | $`\mathbb E\,G`$ | $`\theta`$ 与 $`m`$，权重为 $`-\eta\gamma\nabla_\theta V+\nabla_mV`$ |
| **(c)** | Adam/AdamW，warm | 精确地是 Adam 增量的均值；在 frozen-denominator 近似下为 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$，误差 $`O(\eta\,u\,\kappa)`$ | $`\theta`$（经 $`\Phi`$ 预条件）、$`m`$、$`v`$ |
| **(d)** | Adam/AdamW，cold | $`\mathbb E\,h_\epsilon(G)`$（$`\epsilon\to0`$ 时为符号分布）；$`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ | $`\theta`$；$`m`$ 与 $`v`$ |
| **(e)** | 裁剪 SGD | 精确地是 $`\mathbb E\,\mathrm{clip}_c(G)`$；平均单位方向准确到 $`c(1-p_{\rm act})`$，始终激活时精确 | $`\theta`$ |
| **(e)** | 带裁剪的基础 optimizer | 对 $`\mathrm{clip}_c(G)`$ 取基础 optimizer 的统计量 | 同基础 optimizer |
| **(f)** | Lion / 带 momentum 的 sign | 翻转概率 $`\pi`$；momentum 主导时 $`\theta`$ 中无依赖 | $`\theta`$；$`m`$ |
| **(g)** | LoRA，$`B=0`$，第 1 步 | rank-$`r`$ sketch $`\mathbb E[\Gamma]A_0^\top`$（SGD）或其符号分布（cold Adam）；$`A`$ 不动 | 只有 $`B`$ |

*证明.* 全程 $`\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD`$，其中 $`D(y)=\mathbb E_\xi\Delta(z,G(y,\xi))`$。$`\Delta`$ 中是状态 $`z`$ 的固定函数的项，在两个分布下期望相同，相互抵消。每种情形都从更新规则算出 $`\Delta`$，再取期望之差。

- *(a)* $`\Delta^\theta=-\eta\lambda_{\rm wd}\theta-\eta G`$。第一项抵消，由线性得 $`-\eta\delta_1`$。二阶统计量由 Proposition 5.1(a) 取 $`k=2`$ 得到：除去只依赖状态的项，$`\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top`$。
- *(b)* heavy-ball：$`\theta^+-\theta=-\eta(\beta m+G)`$，$`m^+-m=(\beta-1)m+G`$。Nesterov：$`\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)`$。依赖消息的部分在 $`\theta`$ 中是 $`-\eta\gamma G`$，在 $`m`$ 中是 $`G`$，所以 $`\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)`$；与 $`(\nabla_\theta V,\nabla_mV)`$ 配对即得。
- *(c)* 矩块：$`m^+-m=(1-\beta_1)(G-m)`$ 与 $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ 给出 $`(1-\beta_1)\delta_1`$ 与 $`(1-\beta_2)\delta_2`$。参数块：由 Definition 5.4，$`\tilde\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta c_1\Phi G`$，且 $`\theta^+=\tilde\theta^++e(G)`$。取期望之差得 $`\theta`$ 块；由 $`\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert`$ 与 Proposition 5.5(b) 得误差界。若 $`\delta_1=0`$，$`\theta`$ 块化为 $`\mathbb E_Pe-\mathbb E_Qe`$，$`m`$ 块为零，于是得到一阶项的所述形式与界。误差沿训练的大小是 Proposition 5.8(a)。
- *(d)* 在 $`m=v=0`$、$`t=1`$ 处，$`n=G`$，$`d=\lvert G\rvert`$，所以 $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta h_\epsilon(G)`$（Proposition 5.3(a)）。三个增量为 $`-\eta\lambda_{\rm wd}\theta-\eta h_\epsilon(G)`$、$`(1-\beta_1)G`$、$`(1-\beta_2)G^{\odot2}`$。$`\epsilon\to0`$ 时 $`h_\epsilon(g)_i\to\mathrm{sign}(g_i)`$，$`h_0(0)=0`$。若 $`R=1`$，continuation value 为 $`F_E(\theta)`$，故 $`\nabla_mV=\nabla_vV=0`$，只有参数块有贡献。
- *(e)* 对裁剪 SGD，除 weight decay 外 $`\Delta^\theta=-\eta\,\mathrm{clip}_c(G)`$。对反例，$`\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG`$，而 $`\mathrm{clip}_1(4e_1)=e_1`$、$`\mathrm{clip}_1(-\tfrac43e_1)=-e_1`$ 给出 $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$，于是 $`\mu^P-\mu^Q=\tfrac\eta2e_1`$。平均方向的界是对每个分布用 Proposition 5.9 再用三角不等式。若几乎必然 $`\lVert G\rVert>c`$，则 $`\mathrm{clip}_c(G)=cG/\lVert G\rVert`$；方向分布相同使增量 $`-\eta cG/\lVert G\rVert`$ 同分布，所以 Proposition 5.1(a) 的每一项以及精确 response 都相同。带裁剪的基础 optimizer 把基础更新作用在 $`\mathrm{clip}_c(G)`$ 上。
- *(f)* 记 $`c_i=\beta_1m_i+(1-\beta_1)G_i`$。对 $`m_i\ne0`$：不翻转时 $`\mathrm{sign}(c_i)=\mathrm{sign}(m_i)`$；严格翻转时 $`\mathrm{sign}(c_i)=-\mathrm{sign}(m_i)`$；$`c_i=0`$ 时 $`\mathrm{sign}(c_i)=0=\mathrm{sign}(m_i)(1-2\cdot\tfrac12)`$。所以 $`\mathbb E\,\mathrm{sign}(c_i)=\mathrm{sign}(m_i)(1-2\pi_i)`$。momentum 主导时 $`c_i`$ 几乎必然与 $`m_i`$ 同号，故 $`\pi_i=0`$。$`m`$ 块增量 $`(1-\beta_2)(G-m)`$ 给出 $`(1-\beta_2)\delta_1`$。之后的参数更新只通过之后各个 $`c`$ 的符号依赖 $`m`$；在 $`m`$ 的小扰动不改变这些符号的地方，continuation value 关于 $`m`$ 局部为常数，$`\nabla_mV=0`$。
- *(g)* 对 $`W=W_0+\alpha BA`$，链式法则给出 $`\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}`$ 与 $`\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}`$，即 $`G_B=\alpha\Gamma A^\top`$，$`G_A=\alpha B^\top\Gamma`$。在 $`B_0=0`$ 处 $`G_A`$ 恒为零：SGD 使 $`A`$ 移动 $`0`$；从 $`m=0`$ 开始的 momentum 给出 $`m^+=0`$；cold Adam 使 $`A`$ 移动 $`-\eta h_\epsilon(0)=0`$。$`B`$ 的增量在 SGD 下为 $`-\eta\alpha\Gamma A_0^\top`$，momentum 的第一步与之相同，cold Adam 下为 $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$；取期望即得统计量。若 $`\Delta\Gamma A_0^\top=0`$，$`B`$ 块均值相同，$`A`$ 块均值为零，所以第 1 步 $`\mu^P=\mu^Q`$；在 cold Adam 下，sketch 的符号分布相同，则 $`h_0(\alpha\Gamma A_0^\top)`$ 同分布。一步之后一般 $`B_1\ne0`$，$`G_A=\alpha B_1^\top\Gamma`$ 不再为零。$`\square`$

**Theorem 5.10 的推论.**

- *要看统计量，而不只是看数据.* 同一对 feedback law，可以对一个学习器一阶等价，对另一个学习器一阶不同。均值相同的两个标签分布，对 SGD 与 momentum 一阶等价；但对 cold Adam、sign optimizer 与裁剪 SGD 可以不同，因为它们的增量看的是符号分布或裁剪后均值；对 warm Adam 也可以不同，差别经由二阶矩通道。所以 teacher 数据必须在学生实际使用的 optimizer 下评判。
- *用 AdamW 加全局范数裁剪训练的 few-shot LoRA.* 初始化时适用 (g) 行：第一次更新只通过 rank-$`r`$ sketch $`\Gamma A_0^\top`$ 的符号分布看到反馈。一旦裁剪在部分 batch 上激活，(c)–(e) 行都对裁剪后的梯度取，而"只看平均方向"的简化只准确到 $`c(1-p_{\rm act})`$。
- *与 Proposition 5.3、5.6 的关系.* (d) 行统一了 Proposition 5.3(a)–(c) 与 (e)：参数块携带符号分布，矩块携带可以使第二步反向的幅度。(c) 行是 Proposition 5.6 的线性化形式：它把 Proposition 5.6 中的残差 $`L_v\kappa`$ 细化成显式的线性项 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$。

<a id="s5-11"></a>

**Remark 5.11（在真实模型上检验 warm 状态的条件）.**

- Proposition 5.6 的 warm 状态误差，是按敏感度加权的和 $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，再加上 $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$。关于 $`\kappa`$ 分位数的陈述（例如"大多数坐标的 warmth 很小"）对这两项都不构成验证：少数 $`L_iu_i\kappa_i`$ 很大的坐标就能主导这个和，而第二项取决于最大值。
- 所以在真实模型上应当报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ 及其最大的几项，以及 $`\max_i\kappa_i`$。敏感度 $`L_i`$ 是 continuation value 的性质，单看更新是观察不到的；任何替代量 $`\hat L_i`$（例如留出验证梯度的幅度）都是 proxy，必须作为 proxy 报告。
- 测得的更新映射非可加性（$`\Delta`$ 关于消息离仿射有多远）是 optimizer 映射的性质，不是任务 response 的性质。response 是否仿射必须在 response 本身上测，例如用 [Definition 8.6](08_certificate_ZH.md#s8-6) 的配对替换分数。
- 由 Proposition 5.8(a)，训练后期典型坐标 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$，而不是零。

## 相关工作

- Adam 带 bias correction 的第一步及其 sign descent 行为是已知的：Kingma & Ba（2015）；Balles & Hennig（2018）把 Adam 看成带方差自适应的 sign descent；Liu et al.（RAdam，ICLR 2020）讨论早期步的方差；signSGD（Bernstein et al., 2018）与 Lion（Chen et al., 2023）。decoupled weight decay 见 Loshchilov & Hutter（2019）。
- LESS（Xia et al., 2024）在 LoRA warm-up 之后用经 Adam 预条件的梯度特征做数据选择，对应 Theorem 5.10(c) 的参数块。Adam 版 In-Run Data Shapley（Ding et al., 2026）采用 fixed-state 近似，即只算当前一步的预条件更新。
- Proposition 5.1 与 stochastic modified equations（Li, Tai & E, 2019）、TracIn（Pruthi et al., 2020）、influence function（Koh & Liang, 2017）、metagradient（Engstrom et al., 2025）和 In-Run Data Shapley（Wang et al., 2025）属于同一类计算，见 Remark 5.2。
- 本节新增的是信息层面的陈述：给定 optimizer 使 feedback law 的哪个一阶统计量变得相关——在包括 $`m`$、$`v`$ 的完整状态上，残差作为 warmth 的函数显式给出——以及 $`\beta_2`$ 固定时 warmth 的精确长期行为。
