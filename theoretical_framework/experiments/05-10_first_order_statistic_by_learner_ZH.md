# Theorem 5.10 · 各学习器的一阶统计量

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-10_first_order_statistic_by_learner_ZH.md) · [证明](../proofs/05-10_first_order_statistic_by_learner_ZH.md) · [实验](../experiments/05-10_first_order_statistic_by_learner_ZH.md) · [English](05-10_first_order_statistic_by_learner.md)

**本页检验什么.** [Check 5.16](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-16)–[5.18](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-18) 是 warm Adam、Lion 与 LoRA 的见证。[Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14)、[5.15](../experiments/05-09_partially_active_clipping_ZH.md#c5-15) 的完整描述在 Proposition 5.9 的页面上，覆盖裁剪。[Check 5.2](../experiments/05-01_optimizer_transformed_expansion_ZH.md#c5-2) 在 Proposition 5.1 的页面上，在受控网络中覆盖 cold Adam 一行。

**结论.**

- **Proposition 5.9 与 Theorem 5.10(e)（clipping）.**
  - 原始均值相同、clipped mean 不同的两个分布在一阶就分开，余项斜率为 2.0015。
  - 始终裁剪（$`p_{\rm act}=1`$）时，平均方向相同的分布一阶项恰为 0。
  - Proposition 5.9 中部分激活时的误差界只是两行的不等式，没有单独的数值检验。
- **Theorem 5.10（各学习器的一阶统计量）.** 五个见证全部通过，共 105 条断言，最差残差 $`8.0\times10^{-21}`$。
  - warm Adam：梯度均值相同时，一阶项完全由 $`v`$ 块承担，与预测相差小于 $`10^{-12}`$。
  - Lion：momentum 占主导时，参数块对消息完全不敏感。
  - LoRA：$`B=0`$ 时，零空间方向上的差别是二阶的，值域方向上的差别是一阶的。

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

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

**被检验的陈述（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）.**

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

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.2](../experiments/05-01_optimizer_transformed_expansion_ZH.md#c5-2) | [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)(b)、(d)；[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(d) | `v015:E10` | INFO（诊断）；所有光滑的精确单元格都在界内 |
| [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14) | [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e) | `v015:E11`（clipped mean） | PASS |
| [Check 5.15](../experiments/05-09_partially_active_clipping_ZH.md#c5-15) | $`p_{\rm act}=1`$ 时的 [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e) | `v015:E11`（始终裁剪） | PASS |
| [Check 5.16](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-16) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(c)、[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)(a) | `v015:E11`（warm Adam） | PASS |
| [Check 5.17](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-17) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(f) | `v015:E11`（Lion） | PASS |
| [Check 5.18](../experiments/05-10_first_order_statistic_by_learner_ZH.md#c5-18) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(g) | `v015:E11`（LoRA） | PASS |

<a id="c5-16"></a>

## Check 5.16 · Warm Adam：二阶矩经由 optimizer state 在一阶上进入（Theorem 5.10(c)、Proposition 5.5(a)）

**结论.**

- 梯度均值相同时，参数块与 $`m`$ 块的贡献恰为 0。$`v`$ 块贡献 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}`$，与预测相差小于 $`10^{-12}`$。
- 所以对 warm Adam，均值相同的两个分布在一阶上并不等价。

**动机.**

- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)(a) 说，Adam 的精确一阶统计量是完整状态上的平均 Adam 增量。它是梯度分布的非线性泛函，并通过 $`\Delta v=(1-\beta_2)(G\odot G-v)`$ 依赖 $`\mathbb E\,G\odot G`$。
- 因此，均值相同、二阶矩不同的两个分布，应当有非零的一阶项，由 $`v`$ 块承担，权重为 $`(1-\beta_2)\nabla_vV`$。
- 在这个见证里，参数块恰好为零，冻结分母误差也恰好为零，所以检验单独分离出了 [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(c) 中的 $`v`$ 通道。

**设置**（`v015:E11`，warm Adam）.

- 网络取 $`k=1`$ 个上下文、$`p=13`$ 个参数。
- warm 状态：$`m=0`$，每个坐标 $`v=0.02`$，步数计数 20。
  - 这是一个容许的 warm 状态，不是从 IID warm-up 中抽出来的；
  - 参数项为零依赖于标签分布的对称性。
- Adam，$`\beta=(0.9,0.999)`$，$`\eta=0.03`$。
- 标签：$`P`$ 在 $`\lbrace0,1\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace\tfrac14,\tfrac34\rbrace`$ 上均匀。两者均值都是 $`\tfrac12`$，二阶矩不同。
- continuation：之后用标签 $`\tfrac14`$ 更新两步，再部署 moment $`N=3`$。
- continuation value 对完整状态的梯度用中心差分计算。

**做法.**

1. 计算 $`\delta_1=\mathbb E_PG-\mathbb E_QG`$、$`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$，以及三个块的贡献 $`\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle`$、$`\langle\nabla_mV,\cdot\rangle`$、$`\langle\nabla_vV,\cdot\rangle`$。
2. 断言 $`v`$ 块等于 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$。
3. 与实测差距对比，并与 [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 中的 $`L_v\kappa`$ 项对比。

**结果**（`v015:E11`，`data.warm_Adam`）.

| 量 | 值 |
|---|---:|
| $`\max_i\lvert\delta_{1,i}\rvert`$（梯度均值相同） | 0 |
| $`\max_i\delta_{2,i}`$ | 0.1875 |
| 一阶各块 $`(\theta,\ m,\ v)`$ | $`(0,\ 0,\ 2.5043\times10^{-6})`$ |
| 预测的 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$ | $`2.5043\times10^{-6}`$（相差 $`<10^{-12}`$） |
| 实测差距 | $`8.753\times10^{-6}`$ |
| warm Adam 界对线性 $`v`$ 项给出的 $`L_v\max\kappa`$ | $`7.127\times10^{-6}`$（$`\max\kappa=0.0125`$） |

**为什么支持 Theorem 5.10(c) 与 Proposition 5.5(a).**

- 在 warm Adam 状态下，平均梯度相同的两个分布在一阶上被 $`v`$ 通道分开，系数与预测完全一致。
- $`\delta_1=0`$，所以 momentum 块为零。
- 参数块也为零，原因有二：
  - 它的均值项正比于 $`\delta_1`$，而 $`\delta_1=0`$；
  - 它的 warmth 误差，即精确参数增量与冻结分母参数增量之差（Proposition 5.5(b)），在这里恰好为零，因为标签分布对称且 $`m=0`$。
- 实测差距中的其余部分是 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的余项。
- Proposition 5.6 的残差界覆盖了线性的 $`v`$ 项；[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(c) 用显式的统计量 $`\mathbb E\,G^{\odot2}`$ 代替了这一残差。
- 证据的局限：这个见证只覆盖了冻结分母描述对参数块恰好精确的情形。一般情况下，精确的一阶对象是平均 Adam 增量；$`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ 只能把它确定到 Proposition 5.5(b) 的冻结分母误差为止。
- $`v`$ 通道在长尾段上如何起作用，见 [§7](../statements/07-01_state_channel_decomposition_ZH.md)。

<a id="c5-17"></a>

## Check 5.17 · Lion：momentum 占主导时参数块对消息不敏感（Theorem 5.10(f)）

**结论.** momentum 占主导时，两个差别很大的梯度分布在参数块上的对比恰为 0，没有符号翻转；$`m`$ 块恰好承载 $`0.01\,\delta_1`$。

**动机.** [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(f) 说，momentum 占主导的符号更新与消息无关，所以即使梯度分布差别很大，参数增量也相同；只有 $`m`$ 块记录了差别。

**设置**（`v015:E11`，Lion）.

- 网络配 Lion（[Chen et al., 2023](#ref-chen2023symbolic)；$`\beta_1=0.9`$，$`\beta_2=0.99`$，$`\eta=0.03`$）。
- 标签分别为 0 和 1 的两条消息，梯度相差 $`\delta_1`$，$`\max\lvert\delta_1\rvert=1`$。
- momentum 占主导的状态：每个坐标 $`m=0.2`$，使每条消息都满足 $`0.1\lvert G_i\rvert<0.9\cdot0.2`$。
- 对照：$`m=0.005`$。

**做法.**

- 计算逐坐标的翻转指示量，以及每个块上的增量之差。
- 断言占主导的情形下参数对比恰为零，对照情形下不为零。
- 断言 $`m`$ 块等于 $`(1-\beta_2)\delta_1=0.01\,\delta_1`$。

**结果**（`v015:E11`，`data.Lion`）.

| 状态 | 参数块对比 $`\max_i\lvert\cdot\rvert`$ | 翻转比例 | $`m`$ 块对比 $`\max_i\lvert\cdot\rvert`$ |
|---|---:|---:|---:|
| momentum 占主导，$`m=0.2`$ | 0（精确） | 0 | 0.0100 $`=0.01\max\lvert\delta_1\rvert`$ |
| 对照，$`m=0.005`$ | 0.0600 $`=2\eta`$ | 0.5 | 0.0100 |

**为什么支持 Theorem 5.10(f).**

- momentum 占主导时，两条消息的参数增量相同；它们在 $`\theta`$ 上的一阶统计量是翻转概率，这里为 0。
- 一旦某个符号翻转，对比就是完整的 $`2\eta`$ 一步。
- $`m`$ 块总是承载 $`(1-\beta_2)\delta_1`$。如果之后没有符号改变，continuation value 在 $`m`$ 方向上局部为常数，这一通道同样没有局部的一阶效应。

<a id="c5-18"></a>

## Check 5.18 · LoRA：第一步只看到一个低秩 sketch（Theorem 5.10(g)）

**结论.**

- $`B=0`$ 时，在 SGD、heavy-ball 与 cold Adam 下，对每个分布，第一步之后 $`A`$ 都逐位不变。
- 只在 $`A_0^\top`$ 零空间中不同的两个分布，一阶差距小于 $`10^{-20}`$，差距是二阶的（斜率 2.000）。
- 在值域中不同的两个分布，一阶项与实测差距在每个 $`\eta`$ 上都相差不到 0.3%；$`\eta=10^{-3}`$ 时相对误差不到 $`3\times10^{-5}`$。

**动机.** [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(g) 有两点断言：

- $`B=0`$ 时 adapter $`A`$ 在第一步不能移动，所以反馈只经由 $`\Gamma A_0^\top`$ 进入；
- 因此，平均全权重梯度的差别若落在 $`A_0^\top`$ 的零空间中，在一阶上看不见；落在值域中则看得见。

**设置**（`v015:E11`，LoRA）.

- 双线性 sigmoid 模型 $`W=W_0+\alpha BA`$，$`d=k=4`$，$`W_0=0.2`$，$`\alpha=1`$，$`r\in\lbrace1,2\rbrace`$，$`B_0=0`$，$`A_0`$ 随机。这正是 LoRA 的参数化与初始化（[Hu et al., 2022](#ref-hu2022lora)）。
- 消息是软标签的平均 cross-entropy 目标，精确实现指定的全权重梯度 $`\Gamma`$。
- 三组分布对：
  - *零空间*：平均梯度相差 $`\Delta\Gamma`$，且 $`\Delta\Gamma A_0^\top=0`$；
  - *值域*：差别形如 $`HA_0`$；
  - *符号*：sketch $`\Gamma A_0^\top`$ 的符号分布相同、幅度不同。
- optimizer：SGD、heavy-ball 与 cold Adam。逐位审计用 $`\eta=0.03`$，差距用完整的 $`\eta`$ 网格。

**做法.**

1. 断言对每个分布、每条消息、每个 optimizer，第一步之后 $`A`$ 都逐位不变。
2. 对零空间与值域两组分布对，计算第一步的精确差距与一阶项，并拟合零空间那组差距的斜率。
3. 对 cold Adam 下的符号那组，断言更新等于 $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$，并记录有限 $`\epsilon`$ 带来的差别。

**结果**（`v015:E11`，`data.LoRA`）.

| 量 | rank 1 | rank 2 |
|---|---:|---:|
| $`A`$ 不变的逐位检验（SGD、heavy-ball、cold Adam × 所有消息） | 21 / 21 | 21 / 21 |
| 零空间残差 $`\max\lvert\Delta\Gamma A_0^\top\rvert`$ | $`2.6\times10^{-18}`$ | $`4.8\times10^{-19}`$ |
| 零空间对：$`\eta=0.1`$ 时的一阶项 | $`-1.1\times10^{-20}`$ | $`2.5\times10^{-21}`$ |
| 零空间对：$`\eta=0.1`$ 时的差距；对 $`\eta`$ 的斜率 | $`-3.66\times10^{-9}`$；1.99994 | $`-4.83\times10^{-8}`$；2.0000017 |
| 值域对，$`\eta=10^{-3}`$：一阶项与差距 | $`6.06362\times10^{-7}`$ 与 $`6.06362\times10^{-7}`$ | $`1.86514\times10^{-7}`$ 与 $`1.86509\times10^{-7}`$ |
| 符号对，cold Adam：$`\epsilon=10^{-8}`$ 时的更新之差 | $`7.1\times10^{-8}`$ | $`7.8\times10^{-8}`$ |

**为什么支持 Theorem 5.10(g).**

- 在 LoRA 初始化处，第一步不能利用 rank-$`r`$ sketch $`\Gamma A_0^\top`$ 之外的任何反馈。
- 零空间中的差别是二阶的；值域中的差别是一阶的，系数与预测一致。
- 在 cold Adam 下，$`\epsilon\to0`$ 时只有 sketch 的符号分布起作用。$`\epsilon=10^{-8}`$ 时两次更新最多相差 $`7.8\times10^{-8}`$，正是 $`h_\epsilon`$ 的有限 $`\epsilon`$ 修正。

### [Check 5.2](../experiments/05-01_optimizer_transformed_expansion_ZH.md#c5-2) · 受控网络中、跨 optimizer 的整段训练表示（Proposition 5.1(b)、(d)）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.1 的页面上。对 Theorem 5.10，它检验的是：[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)(b)、(d)；[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(d)。

**结论.**

- 在整段训练上，两个反馈律之间的差距等于*完整状态增量*的一阶项，加上一个落在曲率界之内的余项。在 continuation value 光滑的每个精确单元格里都是如此，最大比值 0.9986。
- 在 `v014:C23` 的 cold Adam 单元格中，增量的一阶项解释了 98.4% 的差距；原始平均梯度项只有 $`-8.9\times10^{-11}`$。
- 余项的阶数符合 (d) 的矩层级。例外有两类：多轮 Adam，其二阶矩块使余项变成 $`\eta`$ 的一阶量；以及光滑性假设之外的单元格。

**设置概要.**

- *分布.* `v014:C23` 的标签分布：moment $`N=2,3,4`$ 与 parity $`N=2,3`$，它们在低于 $`N`$ 阶的标签矩上一致；另加在线性 logit 模型上实现的 clipped mean 见证。
- *11 种 optimizer.*
  - SGD；
  - heavy-ball 与 Nesterov（$`\beta=0.9`$）；
  - cold Adam（$`\beta=(0.9,0.999)`$，$`\epsilon=10^{-8}`$，带 bias correction）；
  - 先做 $`W=5`$ 和 $`W=20`$ 步 lr = 0 moment warm-up 之后的 warm Adam；
  - AdamW（$`\lambda=0.01`$）；
  - clipped SGD 与 clipped Adam；
  - signSGD；
  - Lion（$`\beta_1=0.9`$，$`\beta_2=0.99`$）。
- *裁剪阈值.*
  - 按任务选取，使被裁剪的概率质量最接近 50%。
  - 梯度范数相同时达不到 30%–70% 的区间：moment $`N=3`$、parity $`N=3`$ 为 25%。

### [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14) · Clipping：统计量是 clipped mean（Proposition 5.9、Theorem 5.10(e)）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.9 的页面上。对 Theorem 5.10，它检验的是：[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e)。

**结论.** 两个梯度分布的原始均值相同，clipped mean 相差 $`-\tfrac12`$。它们在一阶上就被分开：预测的差距 $`\tfrac\eta2\sigma'(0.4)`$ 与实测差距吻合，余项的 log–log 斜率为 2.0015。

**设置概要.**

- 线性 logit 的 sigmoid，$`\theta=0.4`$。
- $`P(G=4)=\tfrac14`$，$`P(G=-\tfrac43)=\tfrac34`$；$`Q=\delta_0`$。
- 在 $`c=1`$ 处裁剪；clipped SGD。

### [Check 5.15](../experiments/05-09_partially_active_clipping_ZH.md#c5-15) · 始终裁剪：只有方向起作用（完全激活时的 Proposition 5.9、Theorem 5.10(e)）

**这里检验什么.** 这项检验的完整描述在 Proposition 5.9 的页面上。对 Theorem 5.10，它检验的是：$`p_{\rm act}=1`$ 时的 [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)、[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)(e)。

**结论.** 每个梯度都被裁剪、且两个分布的平均单位方向相同时，一阶项恰为 0，差距是二阶的（log–log 斜率 1.99997）。

**设置概要.**

- 二维梯度，$`\lVert G\rVert\ge2>c=1`$。
- $`P`$ 在 $`\lbrace(\pm2,0)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace(0,\pm4)\rbrace`$ 上均匀。两者的平均单位方向都是 0，方向的二阶矩不同。
- 线性 logit 模型与 $`\eta`$ 网格同 [Check 5.14](../experiments/05-09_partially_active_clipping_ZH.md#c5-14)。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

这些检查使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。`v015:V2` 与 `v015:V8` 中的线性规划由 `scipy.optimize.linprog` 调用 HiGHS solver（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解。

## 产物

- `theory/v015_checks/results_v015_independent.json`：
  - `v015:V8`，以及 `v015:E10`（逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，cold Adam 单元格在 `data.C23_prediction`，设计在 `data.design`）；
  - `v015:V2`、`v015:V3`、`v015:V4`、`v015:E0`、`v015:E1`、`v015:E2`、`v015:E3`；
  - `v015:E8`：冻结分母、warm 仿射、cold 分离与 warmth 轨迹四个部分（最后一部分含 `per_step` 与 `fits`），以及 210 个 `data.gaps` 单元格；
  - `v015:E11`（`data.clipped_mean`、`data.always_active`、`data.warm_Adam`、`data.Lion`、`data.LoRA`）。
- `theory/v015_checks/c23_bundle.json`：精确分布、网络、optimizer 代码，以及每条历史的原始更新。
- `theory/v015_checks/raw/E8_trajectories.npz`：逐副本的 warmth 轨迹。它被 gitignore，由上面第一条命令重新生成；SHA-256 记录在 `v015:E8` warmth 轨迹部分的 `raw_sha256` 中。
- `theory/v016_checks/results_v016.json`：claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`。完整数组在 `theory/v016_checks/results_v016_raw.json.gz` 中，该文件被 gitignore，SHA-256 记录在 `raw_artifact`。
- `theory/independent_checks/results_independent_v014.json`：`controlled_neural`。
- `docs/direction/v015_revision/results_v015.json`：package checks `pkg:T03`–`pkg:T08`。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。
