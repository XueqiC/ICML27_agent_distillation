# Theorem 4.5 · 与轮数无关的界与平均化速率

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-05_round_independent_bound_averaging_rates_ZH.md) · [证明](04-05_round_independent_bound_averaging_rates_ZH.md) · [实验](../experiments/04-05_round_independent_bound_averaging_rates_ZH.md) · [English](04-05_round_independent_bound_averaging_rates.md)

**本页证明什么.** 对独立经验块取固定凸平均的 learner，两个来源的期望部署回报之差被它们的最大矩失配界住，与训练轮数无关；对 $`n`$ 个独立块取平均时，这个差按 $`n^{-(r-1)}`$ 衰减，$`r`$ 是第一个不匹配的 cumulant 的阶数；重放一个块不会让它衰减。证明用到仿射 learner 的块矩展开、乘积的 telescoping 界，以及矩–cumulant 公式。

**本页用到的设定**（[Definition 4.3](../statements/04-03_binary_context_student_affine_learner_ZH.md#s4-3)）.

- 学生 $`p\in[0,1]^k`$，在 context $`j`$ 选动作 1 的概率为 $`p_j`$；部署至多 $`N`$ 个决策；块 $`Z_i\in[0,1]^k`$ IID 于来源分布 $`P`$ 或 $`Q`$；凸仿射 learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，$`p_0`$ 固定，权重 $`w_i\ge0`$、$`\sum_{i=0}^Rw_i=1`$ 固定。
- $`p^\alpha=\prod_jp_j^{\alpha_j}`$，$`\lvert\alpha\rvert=\sum_j\alpha_j`$，$`m_P(\alpha)=\mathbb E_PZ^\alpha`$；$`e_j`$ 是第 $`j`$ 个单位多重指标。
- 由 [Theorem 4.4(a)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)（部署价值是有界的 $`N`$ 次多项式），部署价值为

```math
J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N. \qquad(4.3)
```

- *块矩展开*（[Theorem 4.4](04-04_affine_learners_moment_order_ZH.md) 证明中的 Claim 2）。对 $`\lvert\alpha\rvert=a`$，把 $`p_R^\alpha`$ 的坐标因子列为 $`j(1),\dots,j(a)`$，坐标 $`j`$ 出现 $`\alpha_j`$ 次；对来源映射 $`s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace`$，令 $`\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)}`$。分配律与块独立性给出

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)), \qquad(4.4)
```

- 其中系数 $`\prod_lw_{s(l)}`$ 非负、总和为 1，并且对每个 $`s`$ 至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$。
- 平均化 learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$，均值 $`\mu=\mathbb EZ`$，中心化的平均 $`W_n=\bar p_n-\mu`$。$`\kappa_s(P)`$ 是 $`Z\sim P`$ 的 $`s`$ 阶联合 cumulant 张量；$`D^sJ(\mu)`$ 是 $`J`$ 在 $`\mu`$ 处的 $`s`$ 阶导数张量，$`\langle D^sJ(\mu),T\rangle`$ 是它与张量 $`T`$ 的完全缩并。
- 标量见证（[Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）：偶数 $`i`$ 上 $`P_N(i/N)=\binom Ni/2^{N-1}`$，奇数 $`i`$ 上 $`Q_N(i/N)=\binom Ni/2^{N-1}`$；它们在低于 $`N`$ 阶时一致，在 $`N`$ 阶相差 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$。

**陈述（[Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates_ZH.md#s4-5)）.** 在 Definition 4.3 的设定下，取 Theorem 4.4(a) 中的 $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，记

```math
\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
```

- **(a) 与轮数无关的界.** 对每个凸仿射 learner，

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(4.5)
```

- 这个界与 $`R`$ 无关，并且可以截断在 1。
- **(b) 第一个不匹配的 cumulant.** 设 $`P,Q`$ 均值同为 $`\mu`$，且所有阶数 $`s<r`$ 的联合 cumulant 相等，其中 $`2\le r\le N`$。固定 $`P,Q,k,N,J`$，令 $`n\to\infty`$，

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(4.6)
```

- 若式中的缩并为零，差为 $`O(n^{-r})`$；速率为 $`\Theta(n^{1-r})`$ 当且仅当缩并非零。
- **(c) 标量见证的精确律.** 对 Theorem 4.4(c) 中的 $`P_N,Q_N`$ 与 $`J(p)=p^N`$，

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{对每个 }n\ge1 .
```

- **(d) 均值不同.** 若 $`\mu_P\ne\mu_Q`$，则 $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$。
- **(e) replay 与确定性派生不衰减.** 若 $`n`$ 个训练块为 $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$，其中 $`Z_1`$ 是单个父块，$`T`$ 是固定的确定性映射（replay 即 $`T=\mathrm{id}`$），则对每个 $`n`$ 都有 $`\bar p_n=T(Z_1)`$，学习价值之差等于 $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$，与 $`n`$ 无关。

**证明.** 用到三个断言。

*Claim 1（乘积的 telescoping）.* 对 $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$，$`\lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$。这是因为 $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$，括号内的乘积都在 $`[0,1]`$ 中。

*Claim 2（经验均值的 cumulant）.* 对 $`s\ge2`$，$`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$，且 $`\kappa_1(W_n)=0`$。$`s\ge2`$ 阶联合 cumulant 平移不变、多重线性，并且对独立求和项可加，因为独立向量的混合 cumulant 都为零。所以 $`\kappa_s\big(\tfrac1n\sum_iZ_i\big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z)`$；$`W_n`$ 的均值为零。

*Claim 3（矩–cumulant 公式）.* 对矩有限的随机向量 $`W`$，$`\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa(W_{j_b}:b\in B)`$，其中 $`\Pi(s)`$ 是 $`\lbrace1,\dots,s\rbrace`$ 的划分集合。这是 cumulant 的定义式反演（[Leonov & Shiryaev, 1959](#ref-leonov1959method)）。

*(a).* 固定 $`1\le\lvert\alpha\rvert=a\le N`$ 的 $`\alpha`$。把 (4.4) 写成 $`\mathbb Ep_R^\alpha=\sum_s\omega_s\pi_s\prod_{i=1}^Rm(\beta_i(s))`$，其中 $`\omega_s=\prod_lw_{s(l)}\ge0`$、$`\sum_s\omega_s=1`$、$`\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1]`$。

- 对固定的 $`s`$，至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$，每个都有 $`1\le\lvert\beta_i(s)\rvert\le N`$；由于 $`Z\in[0,1]^k`$，所有块矩都在 $`[0,1]`$ 中。
- 由 Claim 1，$`\big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N`$。
- 以 $`\omega_s`$ 为权求和，得 $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$。常数项 $`\alpha=0`$ 抵消，所以 $`\lvert\sum_\alpha c_\alpha(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha)\rvert\le C_J\delta_N`$。

最后，由 (4.3)，$`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$，并且两个学习价值都在 $`[0,1]`$ 中。由于对每个 $`R`$ 都有 $`\sum_s\omega_s=1`$，这个界关于 $`R`$ 一致。

*(b).* $`J`$ 是次数不超过 $`N`$ 的多项式，它在 $`\mu`$ 处的 Taylor 展开有限终止，所以

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle .
```

$`P`$ 与 $`Q`$ 有同一个 $`\mu`$，所以 $`D^sJ(\mu)`$ 相同。$`s=0`$ 的项抵消，$`s=1`$ 的项为零。对 $`s\ge2`$，对 $`W_n`$ 用 Claim 3。

- 含单点块的划分为零，因为 $`\kappa_1(W_n)=0`$。有 $`q`$ 个块、大小 $`b_1,\dots,b_q\ge2`$ 的划分，由 Claim 2 贡献 $`\prod_ln^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z)`$。
- 低于 $`r`$ 阶的 cumulant 相等，所以一个划分项在 $`P`$ 与 $`Q`$ 下可能不同，只有当某个块大小 $`b_l\ge r`$。此时 $`s=\sum_lb_l\ge r+2(q-1)`$，因此 $`q-s\le2-r-q\le1-r`$。
- $`q-s=1-r`$ 只在 $`q=1`$、$`s=r`$ 时成立：若 $`q=1`$ 且 $`s>r`$，则 $`q-s=1-s\le-r`$；若 $`q\ge2`$，则 $`q-s\le2-r-q\le-r`$。

所以阶数为 $`n^{1-r}`$ 的不同项只有 $`s=r`$ 处的单块划分，它贡献 $`\frac1{r!}\langle D^rJ(\mu),n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle`$。其余不同的项都带有 $`n^{-r}`$ 或更小的因子。这样的项只有有限多个（$`s\le N`$），并且由于 $`Z\in[0,1]^k`$ 的 cumulant 有界，每项系数有界；所以余项为 $`O(n^{-r})`$，常数只依赖于 $`P,Q,J`$。

*(c).* $`N=1`$ 时差为 $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$。设 $`N\ge2`$。

- $`P_N`$ 与 $`Q_N`$ 低于 $`N`$ 阶的原点矩相等，因而低于 $`N`$ 阶的 cumulant 也相等，因为 $`\kappa_s`$ 是不超过 $`s`$ 阶矩的多项式。
- 在 (b) 的论证中取 $`r=N`$，不同的划分必须含大小至少为 $`N`$ 的块，而 $`s\le\deg J=N`$。这样的划分只有 $`s=N`$ 处的单块，所以根本没有余项。
- $`D^N(p^N)=N!`$，且 $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$，因为 $`\kappa_N`$ 等于 $`m_N`$ 减去低阶矩的一个多项式，而低阶矩相等。所以差恰为 $`\Delta_Nn^{1-N}`$。

*(d).* 分别在 $`\mu_P`$ 与 $`\mu_Q`$ 处展开。$`s=0`$ 的项给出 $`J(\mu_P)-J(\mu_Q)`$。对 $`s\ge2`$，每个块大小至少为 2 的划分都有 $`q\le s/2`$，故 $`q-s\le-s/2\le-1`$，其余各项都是 $`O(n^{-1})`$。

*(e).* $`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$，所以对每个 $`n`$ 都有 $`\mathbb EJ(\bar p_n)=\mathbb EJ(T(Z_1))`$。$`\square`$

## 附注

- **为什么 (4.5) 中没有 $`R`$.** 凸权重使展开系数构成一个概率向量，与块数无关。带符号或无约束的权重会破坏这一点并可能放大差异；依赖数据的权重使分解失效。
- **总体量，而非单次运行.** 两个界讨论的都是训练过程的期望最终回报，不是某一个训练出的学生的分数。
- **依赖坐标.** $`C_J`$ 与 $`\delta_N`$ 是在概率坐标的单项式基下度量的。$`N`$ 大时 (4.5) 可能失去意义，所以 [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)（近似阶可以远低于精确阶）用逼近宽度代替次数。
- **什么控制速率.** (4.6) 中的变量是独立块的个数 $`n`$。replay 或一个块的确定性派生（(e)）增加了训练暴露量，但不增加 $`n`$，$`r`$ 阶差异停留在单块时的值。这是 [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10)（$`K`$ 条示范之后的信息下限）与 §9 方法所用的计数规则。
- **固定的分布.** $`O(n^{-r})`$ 的常数依赖于 $`(P,Q,J)`$；对 cumulant 增长的分布族，这个展开不是一致的。

## 相关工作

- 矩–cumulant 公式（[Leonov & Shiryaev, 1959](#ref-leonov1959method)）与 cumulant 对独立求和项的可加性是经典结果，在此注明，不作为新结果。
- 本页新增的是用它们在一个反馈来源的学习价值中把独立证据与暴露量分开。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-leonov1959method"></a>

**V. P. Leonov, A. N. Shiryaev (1959).** On a method of calculation of semi-invariants. *Theory of Probability and Its Applications* 4, no. 3, 319–329. DOI [10.1137/1104031](https://doi.org/10.1137/1104031).
