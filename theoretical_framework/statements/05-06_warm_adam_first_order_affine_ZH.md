# Proposition 5.6 · warm Adam 关于 consumed gradient 一阶仿射

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-06_warm_adam_first_order_affine_ZH.md) · [证明](../proofs/05-06_warm_adam_first_order_affine_ZH.md) · [实验](../experiments/05-06_warm_adam_first_order_affine_ZH.md) · [English](05-06_warm_adam_first_order_affine.md)

**一段话.** 在 warm Adam 状态下（每个坐标的二阶矩都为正），单边界 response 是用消息 $`y`$ 做一步更新、再走完其余训练之后的期望最终得分。这个命题说明，response 是 consumed gradient 的仿射函数，外加一个显式余项；余项有三部分：冻结分母误差、二阶矩敏感项和曲率项。因此平均 consumed gradient 相同的两个反馈律，只相差这些余项。

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

<a id="s5-6"></a>

**Proposition 5.6（warm Adam 关于 consumed gradient 一阶仿射）.** 固定一轮 $`t\ge1`$、状态 $`z=(\theta,m,v,t-1)`$，以及有限消息集 $`Y`$，其 consumed gradient 为 $`G(y)`$。令 $`V:\mathcal Z\to[0,1]`$ 为更新之后的 continuation value（声明的参考尾段加部署，尾段随机性已积分），$`\psi(y)=V(z^+(y))`$ 为单边界 response，$`F(g)=V(\tilde z^+(g))`$。设下列条件成立，其中所有状态都由 $`Y`$ 中的消息从 $`z`$ 生成，计数器固定为 $`t`$：

- **(W)** 对每个坐标 $`v_i>0`$（warm 状态）；
- **(L)** 在 $`(m,v)=(m^+(y),v^+(y))`$ 处的点对 $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ 上，$`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$；
- **$`(\mathrm S_v)`$** 存在 $`L_v\ge0`$ 与上限 $`\bar k\in(0,\infty]`$，使得在 $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$ 处，对 $`k=\kappa(y)`$ 有 $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$（由于 $`V\in[0,1]`$，取 $`L_v\bar k\ge1`$ 时它总成立；有内容的是 $`L_v`$ 小的情形）；
- **(H)** $`F`$ 在 $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$ 的凸包 $`\mathcal C`$ 上二阶连续可微，且对所有 $`g\in\mathcal C`$、$`s\in\mathbb R^p`$ 有 $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$。

令 $`\psi_0=F(0)`$，$`b=\nabla F(0)`$，二者都与 $`y`$ 无关。则对每个 $`y\in Y`$，

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

并且对 $`Y`$ 上任意两个消息分布 $`P_0,P_1`$，

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- 特别地，平均 consumed gradient 相同的两个分布，response 之差至多为 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。
- 若 $`V`$ 关于 $`(\theta,m)`$ 可微，一阶统计量就是经过 preconditioner 的平均 consumed gradient，权重为

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

**说了什么.** warm 状态下，单边界 response 等于 consumed gradient 的一个仿射函数加显式残差：冻结分母项、二阶矩敏感度项与曲率项。平均 consumed gradient 相同的分布，response 之差至多为残差的期望。

**直觉.** 二阶矩积累起来以后，一步 Adam 就像一步带 preconditioner 的 momentum SGD。

**在框架中的作用.** 说明从训练的哪个阶段起，仿射与一阶理论适用于 Adam，并给出可计算的误差。

**推导要点.**

- 把 response 拆成三个差：真实与冻结的参数步（由 Proposition 5.5(b) 控制），真实与冻结的二阶矩（由 $`(\mathrm S_v)`$ 控制），以及 $`F`$ 的 Taylor 余项（由 (H) 控制）。

**用到.**

- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5) — Adam 的精确一阶统计量与冻结分母误差

**被用于.**

- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7) — cold 步在一阶就分开均值相同的反馈
- [Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) — warmth 的演化
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器的一阶统计量

## 相关工作

- LESS（[Xia et al., 2024](#ref-xia2024less)）用一段短 LoRA warm-up 训练得到的 optimizer 状态计算 Adam 预条件梯度，并把它作为训练样本作用的一阶 proxy。Proposition 5.6 说明这种对 warm Adam 步的仿射描述从训练的哪个阶段起成立，并给出显式残差。
