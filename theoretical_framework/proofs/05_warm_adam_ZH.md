# §5 Warm Adam — 证明

[← Framework](../README_ZH.md) · [Proofs](05_optimizer_state_ZH.md) · [Experiments](../experiments/05_warm_adam_and_learners_ZH.md) · [English](05_warm_adam.md)

本页包含 Definition 5.4 至 Proposition 5.8，沿用[原页面的设定](05_optimizer_state_ZH.md#设定)；前面的展开式与 cold-start 证明见[原页面](05_optimizer_state_ZH.md)，clipping 及各学习器统计量见[后续页面](05_learner_statistics_ZH.md)。

## Warm Adam

<a id="s5-4"></a>

**Definition 5.4（warmth 与 frozen-denominator surrogate）.** 考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

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

<a id="s5-5"></a>

**Proposition 5.5（Adam 的精确一阶统计量与冻结分母误差）.** 在 Definition 5.4 的设定下：

- **(a) 精确统计量.** Adam 增量为 $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$，其中 $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$，$`\Delta m=(1-\beta_1)(G-m)`$，$`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$。Proposition 5.1 中的一阶统计量就是精确均值 $`\mu=\mathbb E\,\Delta(z,G)`$。它的参数块是 $`G`$ 的分布的非线性泛函：若 $`\beta_1<1`$，且对某个坐标 $`i`$ 有 $`v_i>0`$ 或 $`\epsilon>0`$，则存在两个有限支撑的分布，$`\mathbb E\,G`$ 相同、$`\mathbb E\,G^{\odot2}`$ 也相同，但参数块 $`\mu_\theta`$ 不同。
- **(b) 冻结分母误差.** 对每个坐标与每个 $`G`$，

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) 两种近似要分开.** frozen-denominator 平均增量 $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ 由 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定，且 $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$，$`m`$ 块与 $`v`$ 块则完全相同。所以"warm Adam 只需要 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"这一说法依赖两个不同的近似：
  - Proposition 5.1 的一阶近似，余项由增量的二阶矩控制，分块结构见 Remark 5.2；
  - frozen-denominator 近似，误差由 (b) 控制，随 warmth 变小。

*(a) 的证明.* 增量就是 AdamW 的定义。对第二个结论，固定坐标 $`i`$，令 $`G=g\,e_i`$（$`g`$ 为标量，其余坐标为零）。此时坐标 $`i`$ 的参数块为 $`\mathbb E f(g)`$，其中

```math
f(g)=-\eta\,\frac{\big(\beta_1m_i+(1-\beta_1)g\big)/(1-\beta_1^{\,t})}{\sqrt{\big(\beta_2v_i+(1-\beta_2)g^2\big)/(1-\beta_2^{\,t})}+\epsilon}.
```

- $`f`$ 在 $`\mathbb R`$ 上有界：$`v_i>0`$ 或 $`\epsilon>0`$ 时分母有正的下界；当 $`g\to\pm\infty`$ 时，$`f(g)`$ 趋于 $`\mp\eta(1-\beta_1)\sqrt{1-\beta_2^{\,t}}/\big((1-\beta_1^{\,t})\sqrt{1-\beta_2}\big)`$。因为 $`\beta_1<1`$，两个极限不同，所以 $`f`$ 不是常数。
- 在 $`\mathbb R`$ 上有界的不超过二次的多项式只能是常数，所以 $`f`$ 不是这样的多项式。于是存在四个点 $`g_1<g_2<g_3<g_4`$，$`f`$ 在其上与任何不超过二次的多项式都不一致（过图像上三点的二次多项式会错过某个第四点）。
- 线性映射 $`w\mapsto(\sum_kw_k,\sum_kw_kg_k,\sum_kw_kg_k^2)`$（从 $`\mathbb R^4`$ 到 $`\mathbb R^3`$）有非零核向量 $`w^*`$。它的行空间是 $`\mathrm{span}\lbrace1,g,g^2\rbrace`$ 在这四点上的限制，所以 $`\sum_kw^*_kf(g_k)\ne0`$，否则 $`f`$ 的限制就会落在这个张成空间里。
- 对小的 $`\tau>0`$，$`P=\tfrac14\mathbf 1+\tau w^*`$ 与 $`Q=\tfrac14\mathbf 1-\tau w^*`$ 是这四点上的概率向量（因为 $`\sum_kw^*_k=0`$）。它们的 $`\mathbb E\,g`$ 与 $`\mathbb E\,g^2`$ 相同，因此 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ 相同，而 $`\mathbb E_Pf-\mathbb E_Qf=2\tau\sum_kw^*_kf(g_k)\ne0`$。$`\square`$

*(b) 的证明.*

- *$`v_i>0`$ 的情形.* $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\big\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\big\rvert`$。通分得到下面的恒等式。由 $`\epsilon\ge0`$，最后一个因子不超过 $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$；它小于 $`1`$，又因 $`\sqrt{1+\kappa}-1\le\kappa/2`$ 与 $`\sqrt{1+\kappa}\ge1`$ 而不超过 $`\kappa_i/2`$。所以该因子不超过 $`\min\lbrace \kappa_i/2,1\rbrace`$。

```math
\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
```

- *$`v_i=0`$ 且 $`\epsilon>0`$ 的情形.* 此时 $`a_i=0`$，$`\kappa_i=\infty`$，$`u_i=\lvert n_i\rvert/\epsilon`$。两个倒数 $`1/(d_i+\epsilon)`$ 与 $`1/\epsilon`$ 都在 $`[0,1/\epsilon]`$ 中，差不超过 $`1/\epsilon`$，所以左边不超过 $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$。$`\square`$

*(c) 的证明.* 由 Definition 5.4，$`\tilde\theta^+-\theta=-\eta\lambda_{\rm wd}\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta\Phi(1-\beta_1)G/(1-\beta_1^{\,t})`$ 与 $`m^+-m=(1-\beta_1)(G-m)`$ 关于 $`G`$ 仿射，系数由状态固定；$`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ 关于 $`G^{\odot2}`$ 仿射。取期望后 $`\tilde\mu`$ 是 $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ 的仿射函数。$`\Delta`$ 与 $`\tilde\Delta`$ 的 $`m`$ 块、$`v`$ 块相同，$`\theta`$ 块相差 $`\theta^+-\tilde\theta^+`$，其期望逐坐标由 (b) 控制。$`\square`$

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

*证明.* 固定 $`y`$ 并在记号中略去。由于 $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$，

```math
\psi-\psi_0-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

- *(i)* 由 (L) 与 Proposition 5.5(b)，$`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$。
- *(ii)* 由 (W)，$`v^+=\beta_2v\odot(1+\kappa)`$，$`\kappa\ge0`$ 有限。在 $`(\tilde\theta^+,m^+)`$ 处用 $`(\mathrm S_v)`$，$`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$。
- *(iii)* 线段 $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ 落在 $`\mathcal C`$ 中。带积分余项的 Taylor 定理给出 $`F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau`$。由 (H)，被积函数的绝对值不超过 $`(1-\tau)H_G\lVert G\rVert_\infty^2`$，所以 $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$。

三个界相加即得逐点结论。对两个分布，写 $`\psi=\psi_0+b^\top G+e`$，$`\lvert e\rvert\le r`$。于是 $`\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e`$；常数 $`\psi_0`$ 抵消，因为两者都是概率测度；$`Y`$ 有限，所有期望都存在。三角不等式给出界，$`\psi\in[0,1]`$ 给出截断。对 $`b`$ 用链式法则：$`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)\Phi/(1-\beta_1^{\,t})`$，$`\partial m^+/\partial g=(1-\beta_1)I`$，$`\tilde z^+`$ 的 $`v`$ 分量与 $`g`$ 无关。$`\square`$

<a id="s5-7"></a>

**Proposition 5.7（cold 步在一阶就分开均值相同的反馈）.** 取 cold 状态（$`m=v=0`$，$`t=1`$）、理想符号映射 $`h_0`$（Proposition 5.3）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$，限制在 $`V\in[0,1]`$ 的邻域内。

- **(a) 均值相同.** 令 $`P_0(G=1)=\tfrac14`$，$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- **(b) 任意固定阶以下的矩都相同.** 对 $`N\ge1`$，令 $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数）与 $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数），$`0\le i\le N`$（即 [Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 中的分布），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。两个梯度分布的所有低于 $`N`$ 阶的矩都相同，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。
- 两种情形下的分离都是 $`\eta`$ 的一阶量。在 warm 状态、同一效用下，Proposition 5.6 把均值相同的分布之间的差距控制在 $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$ 以内。

*证明.*

- *(a)* 由 Proposition 5.3(a) 与 $`h_0=\mathrm{sign}`$，$`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$。在 $`P_0`$ 下 $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$，所以 $`\mathbb E\theta_1=\theta_0+\eta/2`$。在 $`P_1`$ 下 $`\mathrm{sign}\,0=0`$，所以 $`\mathbb E\theta_1=\theta_0`$。效用是仿射的，response 差为 $`\lvert w\rvert\eta/2`$。均值相同：$`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$。
- *(b)* 因为 $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$，$`P_N`$ 与 $`Q_N`$ 都是概率分布。对 $`\ell<N`$，$`\sum_i(-1)^i\binom Ni\,i^\ell=0`$，所以它们低于 $`N`$ 阶的矩相同，仿射像 $`G=q-Z`$ 的矩也相同。取 $`q\in(0,1/N)`$：在 $`P_N`$ 下，$`G>0`$ 当且仅当 $`i=0`$，概率为 $`2^{1-N}`$；否则 $`Z\ge2/N>q`$，$`G<0`$。所以 $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$。在 $`Q_N`$ 下总有 $`Z\ge1/N>q`$，所以 $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$。第一步的期望相差 $`\eta\,2^{2-N}`$，仿射效用给出差距 $`\lvert w\rvert\eta\,2^{2-N}`$。
- *warm 对照.* 对没有尾段的 $`V(\theta)=\tfrac12+w\theta`$，$`F`$ 关于 $`g`$ 仿射，故 $`H_G=0`$；$`V`$ 不依赖 $`(m,v)`$，故 $`L_v=0`$；且 $`L=\lvert w\rvert`$。均值相同时，Proposition 5.6 给出所示的界。$`\square`$

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

**(c) 沿训练过程的累积.** 经由 [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式，比较参考协议 $`0`$ 与替代协议 $`S`$ 在 $`W`$ 次 warm-up 计算之后的 $`R`$ 次真实更新 $`s=1,\dots,R`$。令 $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ 为这样的更新：在每个访问到的状态上，Proposition 5.6 的假设以下标为 $`s`$ 的常数成立，两个分布的平均 consumed gradient 相同，并且几乎必然地对某个常数 $`c>0`$ 有下式。

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

*(a) 的证明.* 展开递推，$`v_{s,i}=(1-\beta_2)\sum_{r=1}^s\beta_2^{\,s-r}G_{r,i}^2`$，所以 $`\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2)\sum_{j<s}\beta_2^{\,j}=\sigma_i^2(1-\beta_2^{\,s})`$。代入定义得 $`\bar\kappa_i(s)`$。关于界：

- 由 Bernoulli 不等式，$`1-(1-\gamma)^s\le s\gamma`$，所以 $`\bar\kappa\ge1/(\beta_2s)`$；又 $`1-(1-\gamma)^s\le1`$，所以 $`\bar\kappa\ge\gamma/\beta_2`$。
- 上界等价于 $`s\gamma\le(1+s\gamma)\big(1-(1-\gamma)^s\big)`$，即 $`(1-\gamma)^s(1+s\gamma)\le1`$。这由 $`(1-\gamma)^s\le e^{-s\gamma}`$ 以及对 $`y\ge0`$ 有 $`e^{-y}(1+y)\le1`$ 得到。
- 单调性是因为 $`1-\beta_2^{\,s}`$ 关于 $`s`$ 递增，极限是直接的。两侧界乘以 $`\beta_2s`$ 得 $`1\le\beta_2s\bar\kappa\le1+s\gamma`$。
- 梯度为常数时，$`v_{s,i}=g_i^2(1-\beta_2^{\,s})`$ 是确定的，$`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 就等于该公式。$`\square`$

*(b) 的证明.* $`d_i^2/a_i^2=v_i^+/(\beta_2v_i)=1+\kappa_i`$：因子 $`1/(1-\beta_2^{\,t})`$ 同时除在分子与分母上。$`\square`$

*(c) 的证明.* [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 给出

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

并且在每个访问到的状态上，$`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$，是两个取值于 $`[0,1]`$ 的期望之差。

- 对 $`s\notin\mathcal S_{\rm warm}`$，其绝对值以 $`1`$ 为界。
- 对 $`s\in\mathcal S_{\rm warm}`$，均值相同时 Proposition 5.6 把它控制在 $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$ 以内。用 $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$，以及无论 $`\kappa\le1`$ 还是 $`\kappa>1`$ 都成立的 $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$。由 $`\kappa_{\max}`$ 的假设，$`r_s`$ 的前两项至多为 $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)+c(1-\beta_2)\rbrace`$，曲率项至多为 $`\tfrac12\bar B_s`$。对两个分布求和得到因子 $`2`$。

对 $`s`$ 求和即得第一式。对第二式：当 $`a,b\ge0`$ 时 $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$，所以和至多为 $`\sum_{s=1}^R c/(W+s)+c(1-\beta_2)R`$；又 $`c/(W+s)\le\int_{s-1}^{s}c\,dz/(W+z)`$，所以 $`W\ge1`$ 时 $`\sum_{s=1}^Rc/(W+s)\le c\log(1+R/W)`$。$`W=0`$、$`s`$ 从 $`2`$ 开始时，同理 $`\sum_{s=2}^Rc/s\le c\log R`$。$`\square`$

*(d) 的证明.* 学习率为 $`0`$ 时参数更新为 $`\theta^+=(1-0\cdot\lambda_{\rm wd})\theta-0=\theta`$，而 $`m`$、$`v`$ 与计数器按与真实步相同的递推推进。所以经过 $`W`$ 次 warm-up 计算与 $`s-1`$ 次真实更新后，$`v`$ 已吸收 $`W+s-1`$ 次计算。在 mean-field 替换下，$`v_i=\sigma_i^2(1-\beta_2^{\,k})`$，$`k=W+s-1\ge1`$，所以由 (a)，$`\kappa_i=(G_i^2/\sigma_i^2)\,\bar\kappa_i(k)\le\bar\gamma\,(1/k+1-\beta_2)/\beta_2`$。当 $`W+s\ge2`$ 时 $`k\ge(W+s)/2`$，即 $`1/k\le2/(W+s)`$，因此 $`\kappa_i\le(2\bar\gamma/\beta_2)\big(1/(W+s)+(1-\beta_2)\big)`$。$`\square`$

**关于 Proposition 5.5–5.8 的讨论.**

- *切换.* 反馈来源必须匹配的统计量，随 optimizer state 而变。
  - cold Adam 状态下，要匹配 consumed gradient 的联合符号分布，以及保留在 $`m`$、$`v`$ 中的幅度（Proposition 5.3）。
  - warm Adam 状态下，精确统计量是 Adam 增量的均值；在 warmth 的一阶意义下，它是经过 preconditioner 的平均 consumed gradient，加上 $`v`$ 块中的 $`\mathbb E\,G^{\odot2}`$（Proposition 5.5–5.6）。
  - SGD 下，要匹配平均 consumed gradient，误差 $`O(\eta^2)`$（[Theorem 4.7](04_rates_and_composition_ZH.md#s4-7) 的局部 SGD 展开）。
- *warmth 的下限.* $`\beta_2`$ 固定时，典型梯度的 warmth 在此后的训练中一直停在 $`(1-\beta_2)/\beta_2`$ 附近（Proposition 5.8(a)）。所以 frozen-denominator 描述在每一步都只准确到这个相对阶，累积的非仿射贡献有一个关于 $`R`$ 线性的项（Proposition 5.8(c)）。
- *曲率项的阶.* 对光滑的尾段，$`F`$ 通过 $`\tilde\theta^+`$（被 $`\eta`$ 缩放）与 $`m^+`$（在之后的更新中被 $`\eta`$ 缩放）依赖 $`g`$。所以可以预期 $`H_G=O(\eta^2)`$。这是对尾段的预期，不是 (H) 的一部分。
- *warmth 消除不了的部分.* 曲率项 $`\tfrac12H_G\lVert G\rVert_\infty^2`$ 不依赖 $`\kappa`$，随 $`R`$ 线性累积。在长 horizon 上，均值相同的两个来源之间剩下的差别由曲率决定，而 SGD 同样有这一项。warm-up 只控制 cold 窗口。
- *消息与梯度.* Proposition 5.6 是关于 *consumed* gradient 的仿射性。当 $`G(y)`$ 本身关于 $`y`$ 非线性时，response 关于消息不是仿射的。例如序列级 teacher forcing（前缀会改变 Jacobian），以及处于激活状态的全局范数裁剪。
- *局部性.* Proposition 5.6 是单边界的结论，常数是 continuation value 的性质。要沿训练过程给出证书，还需要 [Theorem 8.2](08_verification_ZH.md#s8-2) 的 transport 与 coverage 项。
