# Proposition 5.7 · cold 步在一阶就分开均值相同的反馈

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-07_cold_steps_separate_equal_mean_ZH.md) · [证明](../proofs/05-07_cold_steps_separate_equal_mean_ZH.md) · [实验](../experiments/05-07_cold_steps_separate_equal_mean_ZH.md) · [English](05-07_cold_steps_separate_equal_mean.md)

**一段话.** 这个命题给出显式的反例：在 cold Adam 状态下，均值相同、甚至任意固定阶以下的矩都相同的反馈律，其 response 在学习率的一阶就不同。所以 cold 窗口是真正的一阶效应，而不是高阶修正。

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

<a id="s5-7"></a>

**Proposition 5.7（cold 步在一阶就分开均值相同的反馈）.** 取 cold 状态（$`m=v=0`$，$`t=1`$）、理想符号映射 $`h_0`$（Proposition 5.3）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$，限制在 $`V\in[0,1]`$ 的邻域内。

- **(a) 均值相同.** 令 $`P_0(G=1)=\tfrac14`$，$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- **(b) 任意固定阶以下的矩都相同.** 对 $`N\ge1`$，令 $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数）与 $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数），$`0\le i\le N`$（即 [Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4) 中的分布），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。两个梯度分布的所有低于 $`N`$ 阶的矩都相同，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。
- 两种情形下的分离都是 $`\eta`$ 的一阶量。在 warm 状态、同一效用下，Proposition 5.6 把均值相同的分布之间的差距控制在 $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$ 以内。

**说了什么.** cold 状态下，两个均值相同的梯度分布——甚至低于任意固定阶的矩全都相同——给出的 response 在学习率的一阶上就不同。

**直觉.** cold 步看的是相对于零的符号，而符号概率不是低阶矩的函数。

**在框架中的作用.** 证明 cold 窗口是真正的一阶效应，不是高阶修正。

**推导要点.**

cold 步是 $`-\eta\,\mathrm{sign}(G)`$，所以 response 关于 $`\mathbb E\,\mathrm{sign}(G)`$ 仿射；Theorem 4.4 的二项奇偶分布在 $`G>0`$ 的唯一那个点上放的质量不同。

**用到.**

- [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) — Cold-start Adam：符号商与 optimizer 记忆
- [Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4) — 仿射 learner：矩阶充分且必要

**被用于.**

- [Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) — warmth 的演化

## 相关工作

- Adam 最初几步行为特殊，这一点以另一种形式已为人知：[Liu et al., 2020](#ref-liu2020variance) 把学习率 warm-up 的必要性归因于训练早期自适应学习率的方差过大。Proposition 5.7 是信息层面的对应结论：在 cold 状态，低阶矩相同的分布在学习率的一阶上就能分开。
