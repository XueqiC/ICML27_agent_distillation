# Proposition 5.6 · warm Adam 关于 consumed gradient 一阶仿射

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-06_warm_adam_first_order_affine_ZH.md) · [证明](../proofs/05-06_warm_adam_first_order_affine_ZH.md) · [实验](../experiments/05-06_warm_adam_first_order_affine_ZH.md) · [English](05-06_warm_adam_first_order_affine.md)

**本页证明什么.** 用三项分解证明逐点的余项界；两个消息律之间的界；以及仿射部分系数 $`b`$ 的公式。

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

**陈述（[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)）.** 固定一轮 $`t\ge1`$、状态 $`z=(\theta,m,v,t-1)`$，以及有限消息集 $`Y`$，其 consumed gradient 为 $`G(y)`$。令 $`V:\mathcal Z\to[0,1]`$ 为更新之后的 continuation value（声明的参考尾段加部署，尾段随机性已积分），$`\psi(y)=V(z^+(y))`$ 为单边界 response，$`F(g)=V(\tilde z^+(g))`$。设下列条件成立，其中所有状态都由 $`Y`$ 中的消息从 $`z`$ 生成，计数器固定为 $`t`$：

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

*证明.* 固定 $`y`$ 并在记号中略去。由于 $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$，

```math
\psi-\psi_0-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

- *(i)* 由 (L) 与 Proposition 5.5(b)，$`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$。
- *(ii)* 由 (W)，$`v^+=\beta_2v\odot(1+\kappa)`$，$`\kappa\ge0`$ 有限。在 $`(\tilde\theta^+,m^+)`$ 处用 $`(\mathrm S_v)`$，$`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$。
- *(iii)* 线段 $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ 落在 $`\mathcal C`$ 中。带积分余项的 Taylor 定理给出 $`F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau`$。由 (H)，被积函数的绝对值不超过 $`(1-\tau)H_G\lVert G\rVert_\infty^2`$，所以 $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$。

三个界相加即得逐点结论。对两个分布，写 $`\psi=\psi_0+b^\top G+e`$，$`\lvert e\rvert\le r`$。于是 $`\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e`$；常数 $`\psi_0`$ 抵消，因为两者都是概率测度；$`Y`$ 有限，所有期望都存在。三角不等式给出界，$`\psi\in[0,1]`$ 给出截断。对 $`b`$ 用链式法则：$`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)\Phi/(1-\beta_1^{\,t})`$，$`\partial m^+/\partial g=(1-\beta_1)I`$，$`\tilde z^+`$ 的 $`v`$ 分量与 $`g`$ 无关。$`\square`$
