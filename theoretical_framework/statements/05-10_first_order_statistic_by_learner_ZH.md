# Theorem 5.10 · 各学习器的一阶统计量

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-10_first_order_statistic_by_learner_ZH.md) · [证明](../proofs/05-10_first_order_statistic_by_learner_ZH.md) · [实验](../experiments/05-10_first_order_statistic_by_learner_ZH.md) · [English](05-10_first_order_statistic_by_learner.md)

**一段话.** 这个定理对实际使用的 optimizer 算出 [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的一阶统计量，即两个反馈律要有相同一阶效应必须一致的那个梯度分布特征。SGD 与 momentum 需要梯度均值；warm Adam 需要 Adam 增量的精确均值，只有在冻结分母近似下才化为均值与二阶矩；cold Adam 的一步需要符号分布；裁剪需要裁剪后的均值；Lion 需要翻转概率；初始化时的 LoRA 只通过一个低秩 sketch 看到反馈。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

**Adam 与 AdamW.** Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

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
- 逐坐标的二阶矩通过 $`v`$ 在一阶上进入，权重为 $`(1-\beta_2)\nabla_vV`$。若 $`\delta_1=0`$，一阶项等于 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$，其中第二项的绝对值至多为 $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$。所以均值相同的分布可以通过二阶矩通道在一阶上不同，而且这个差别不随 warmth 变小。[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 用 adjoint 递推算出 $`\nabla_vV`$。

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

**说了什么.** Proposition 5.1 的统计量在常见学习器下的具体形式：SGD 与 momentum 是平均梯度；warm Adam 是 Adam 增量的精确均值，只在冻结分母近似下等于 $`(\mathbb EG,\mathbb EG^{\odot2})`$，二阶矩经由 $`v`$ 进入；cold Adam 的一步在参数上是符号分布；裁剪是裁剪后均值；Lion 是翻转概率；$`B=0`$ 的 LoRA 是 rank-$`r`$ sketch，第一步 $`A`$ 不动。

**直觉.** 平均梯度相同的两个反馈来源，在 cold Adam、裁剪，或经由二阶矩通道的 warm Adam 下，仍然可以在一阶上不同。

**在框架中的作用.** 告诉方法在每种训练配方下要匹配、要测量什么，并为 Theorem A 的 learner Jacobian 和 Theorem B 的状态通道提供输入。

**推导要点.**

对每条更新规则算出增量，去掉只依赖状态的项，取期望之差。(c) 行用到 Proposition 5.5；(e) 行用到 Proposition 5.9；(g) 行用到 $`B=0`$ 时 $`G_A=\alpha B^\top\Gamma=0`$。

**例示.** 两个平均梯度相同的标签分布，一个集中、一个分散，对 SGD 在一阶上可以互换。在裁剪下，分散的那个会被阈值削掉大梯度；在 warm Adam 下，它更大的二阶矩抬高 $`v`$，使之后每一步都变小。两种效应都是一阶的。

**用到.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) — optimizer 变换后的展开
- [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) — Cold-start Adam：符号商与 optimizer 记忆
- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5) — Adam 的精确一阶统计量与冻结分母误差
- [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) — 部分激活的梯度裁剪

**被用于.**

- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) — SGD 与 warm Adam 的学习器权重
- [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12) — 学习器之间的分离
- [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) — 状态通道分解
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — Consequence-Calibrated Experience Distillation，CED

## 相关工作

- 像 LESS（[Xia et al., 2024](#ref-xia2024less)）那样在 LoRA warm-up 之后用 Adam 预条件梯度特征做数据选择，对应 (c) 中的参数块。In-Run Data Shapley 的 Adam 版本（[Ding et al., 2026](#ref-ding2026inrun)）只用当前这一步的预条件更新。
- 各学习器都按其原始定义：Lion（[Chen et al., 2023](#ref-chen2023symbolic)）、全局范数裁剪（[Pascanu et al., 2013](#ref-pascanu2013difficulty)），以及采用 [Hu et al., 2022](#ref-hu2022lora) 的 $`B=0`$ 初始化的 LoRA。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-chen2023symbolic"></a>

**Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le (2023).** Symbolic discovery of optimization algorithms. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 49205–49233. arXiv [2302.06675](https://arxiv.org/abs/2302.06675).

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-hu2022lora"></a>

**Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022).** LoRA: Low-rank adaptation of large language models. *The Tenth International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9); arXiv [2106.09685](https://arxiv.org/abs/2106.09685).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
