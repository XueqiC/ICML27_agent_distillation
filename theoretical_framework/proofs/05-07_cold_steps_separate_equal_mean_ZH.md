# Proposition 5.7 · cold 步在一阶就分开均值相同的反馈

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-07_cold_steps_separate_equal_mean_ZH.md) · [证明](../proofs/05-07_cold_steps_separate_equal_mean_ZH.md) · [实验](../experiments/05-07_cold_steps_separate_equal_mean_ZH.md) · [English](05-07_cold_steps_separate_equal_mean.md)

**本页证明什么.** (a)、(b) 直接计算两个分布下 consumed gradient 符号的期望；warm 情形的对应结论由 Proposition 5.6 得到。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

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

**符号映射.** 对 $`\epsilon>0`$，$`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$。对 $`\epsilon=0`$，*理想符号更新*为 $`h_0(g)_i=\mathrm{sign}(g_i)`$，约定 $`\mathrm{sign}(0)=0`$。这个约定是定义的一部分；字面上的浮点 Adam 在 $`\epsilon=0`$ 时会算出 $`0/0`$。

**效用.** *即时参数效用*是 $`\theta_1`$ 的有界可测函数；*完整状态效用*是下一个完整状态 $`(\theta_1,m_1,v_1,1)`$ 的有界可测函数。consumed gradient 的分布 $`P,Q`$ 是 $`\mathbb R^p`$ 上的概率分布；在有限情形的陈述中，它们支撑在有限字母表 $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$ 上。记 $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$。

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

**陈述（[Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7)）.** 取 cold 状态（$`m=v=0`$，$`t=1`$）、理想符号映射 $`h_0`$（Proposition 5.3）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$，限制在 $`V\in[0,1]`$ 的邻域内。

- **(a) 均值相同.** 令 $`P_0(G=1)=\tfrac14`$，$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- **(b) 任意固定阶以下的矩都相同.** 对 $`N\ge1`$，令 $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数）与 $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数），$`0\le i\le N`$（即 [Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4) 中的分布），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。两个梯度分布的所有低于 $`N`$ 阶的矩都相同，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。
- 两种情形下的分离都是 $`\eta`$ 的一阶量。在 warm 状态、同一效用下，Proposition 5.6 把均值相同的分布之间的差距控制在 $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$ 以内。

*证明.*

- *(a)* 由 Proposition 5.3(a) 与 $`h_0=\mathrm{sign}`$，$`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$。在 $`P_0`$ 下 $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$，所以 $`\mathbb E\theta_1=\theta_0+\eta/2`$。在 $`P_1`$ 下 $`\mathrm{sign}\,0=0`$，所以 $`\mathbb E\theta_1=\theta_0`$。效用是仿射的，response 差为 $`\lvert w\rvert\eta/2`$。均值相同：$`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$。
- *(b)* 因为 $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$，$`P_N`$ 与 $`Q_N`$ 都是概率分布。对 $`\ell<N`$，$`\sum_i(-1)^i\binom Ni\,i^\ell=0`$，所以它们低于 $`N`$ 阶的矩相同，仿射像 $`G=q-Z`$ 的矩也相同。取 $`q\in(0,1/N)`$：在 $`P_N`$ 下，$`G>0`$ 当且仅当 $`i=0`$，概率为 $`2^{1-N}`$；否则 $`Z\ge2/N>q`$，$`G<0`$。所以 $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$。在 $`Q_N`$ 下总有 $`Z\ge1/N>q`$，所以 $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$。第一步的期望相差 $`\eta\,2^{2-N}`$，仿射效用给出差距 $`\lvert w\rvert\eta\,2^{2-N}`$。
- *warm 对照.* 对没有尾段的 $`V(\theta)=\tfrac12+w\theta`$，$`F`$ 关于 $`g`$ 仿射，故 $`H_G=0`$；$`V`$ 不依赖 $`(m,v)`$，故 $`L_v=0`$；且 $`L=\lvert w\rvert`$。均值相同时，Proposition 5.6 给出所示的界。$`\square`$
