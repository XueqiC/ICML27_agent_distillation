# Theorem 4.5 · 与轮数无关的界与平均化速率

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-05_round_independent_bound_averaging_rates_ZH.md) · [证明](../proofs/04-05_round_independent_bound_averaging_rates_ZH.md) · [实验](../experiments/04-05_round_independent_bound_averaging_rates_ZH.md) · [English](04-05_round_independent_bound_averaging_rates.md)

**一段话.** [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) 说明两个经验来源要精确可互换必须共享哪些矩。这个定理把它定量化。若矩只差一点，学习结果能差多少？训练更久会不会更糟？当 learner 平均越来越多的独立块时，剩下的差以多快的速度消失？回答是：差距被一个矩失配界住，与训练轮数无关；它随*独立*块的个数衰减，速率由第一个不匹配的 cumulant 决定；而重放一个块，或用它的确定性改写训练，差距完全不衰减。

**本页用到的设定**（[Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3)）.

- 学生 $`p\in[0,1]^k`$（每个 context 选动作 1 的概率），部署至多 $`N`$ 个决策，经验块 $`Z_i\in[0,1]^k`$ IID 于来源分布 $`P`$ 或 $`Q`$，凸仿射 learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，权重固定、非负、和为 1。
- 由 [Theorem 4.4(a)](04-04_affine_learners_moment_order_ZH.md#s4-4)，部署价值是多项式 $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，$`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$。
- $`m_P(\alpha)=\mathbb E_PZ^\alpha`$ 是块的联合矩；$`\lvert\alpha\rvert=\sum_j\alpha_j`$。
- *平均化 learner* 为 $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$，$`\mu=\mathbb EZ`$，中心化的平均为 $`W_n=\bar p_n-\mu`$。
- Theorem 4.4(c) 的*标量见证* $`P_N,Q_N`$ 位于 $`\lbrace0,\tfrac1N,\dots,1\rbrace`$ 上：$`i`$ 为偶数时 $`P_N(i/N)=\binom Ni/2^{N-1}`$，$`i`$ 为奇数时 $`Q_N(i/N)=\binom Ni/2^{N-1}`$（其余为零）。它们在所有低于 $`N`$ 阶的矩上一致，在 $`N`$ 阶相差 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$。
- $`\kappa_s(P)`$ 是 $`Z\sim P`$ 的 $`s`$ 阶联合 cumulant 张量（多元 cumulant：$`\kappa_1`$ 是均值，$`\kappa_2`$ 是协方差，更高阶度量非 Gauss 结构）。$`D^sJ(\mu)`$ 是 $`J`$ 在 $`\mu`$ 处的 $`s`$ 阶导数张量，$`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$。

<a id="s4-5"></a>

**Theorem 4.5（与轮数无关的界与平均化速率）.** 在 Definition 4.3 的设定下，取 Theorem 4.4(a) 中的 $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，记

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
- **(c) 标量见证的精确律.** 对 [Theorem 4.4(c)](04-04_affine_learners_moment_order_ZH.md#s4-4) 中的 $`P_N,Q_N`$ 与 $`J(p)=p^N`$，

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{对每个 }n\ge1 .
```

- **(d) 均值不同.** 若 $`\mu_P\ne\mu_Q`$，则 $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$。
- **(e) replay 与确定性派生不衰减.** 若 $`n`$ 个训练块为 $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$，其中 $`Z_1`$ 是单个父块，$`T`$ 是固定的确定性映射（replay 即 $`T=\mathrm{id}`$），则对每个 $`n`$ 都有 $`\bar p_n=T(Z_1)`$，学习价值之差等于 $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$，与 $`n`$ 无关。

**说了什么.**

- (a) 小的矩失配只造成小的结果差，训练轮数不进入这个界。
- (b) 对 $`n`$ 个独立块取平均时，差按 $`n^{-(r-1)}`$ 衰减，$`r`$ 是两个来源第一个不一致的 cumulant 的阶数，首项系数是显式的。
- (c) 对 Theorem 4.4(c) 的见证，这个衰减律对每个 $`n`$ 都是精确的。
- (d) 若均值不同，差不会消失，而是趋于 $`J(\mu_P)-J(\mu_Q)`$。
- (e) 没有新证据的暴露（重放一个块，或用它的确定性改写训练）让单块时的差原封不动。

**直觉.** 只有独立的证据会缩小差距。对独立块取平均，使平均值的高阶 cumulant 按 $`n^{1-s}`$ 缩小；重复或确定性合成增加的是暴露量，不是信息。

**在框架中的作用.** 它把独立块的个数与 epoch 数分开。§2 对示范与查询的计数（[Theorem 2.10](02-10_few_shot_information_floor_ZH.md#s2-10)）与 §9 的方法（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)）都用这条计数规则。

**说明.**

- 为什么 (4.5) 中没有 $`R`$：凸权重使展开系数构成一个概率向量，与块数无关。带符号或无约束的权重会破坏这一点并可能放大差异；依赖数据的权重使分解失效。
- 两个界讨论的都是训练过程的期望最终回报，不是某一个训练出的学生的分数。
- $`C_J`$ 与 $`\delta_N`$ 是在概率坐标的单项式基下度量的。$`N`$ 大时 (4.5) 可能失去意义，所以 [Proposition 4.6](04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) 用逼近宽度代替次数。
- $`O(n^{-r})`$ 的常数依赖于 $`(P,Q,J)`$；对 cumulant 增长的分布族，这个展开不是一致的。

**推导要点.**

- (a) 在 $`\mathbb Ep_R^\alpha`$ 的块矩展开中，对每个 $`R`$ 系数都构成一个概率向量；至多 $`\lvert\alpha\rvert`$ 个 $`[0,1]`$ 中块矩之积的变化，不超过各块矩变化之和（乘积的 telescoping）。
- (b) 在公共均值处对多项式 $`J`$ 作 Taylor 展开，并用 cumulant 写出 $`\bar p_n`$ 的中心矩。一个有 $`q`$ 个块、总大小为 $`s`$ 的划分按 $`n^{q-s}`$ 缩放，阶数为 $`n^{1-r}`$ 的不同项只有大小为 $`r`$ 的单块。
- (c) 对 $`J=p^N`$ 不存在其他不同的划分。(d) 块大小都至少为 2 的划分是 $`O(n^{-1})`$。(e) 相同副本的平均就是副本本身。

**例示.** 两个数据源均值与方差相同、偏度不同（$`r=3`$），部署价值在均值处的三阶导数与偏度之差的缩并不为零：对 $`n`$ 个独立块取平均时，差距按 $`1/n^2`$ 缩小；而在同一个块上训练十个 epoch，单块时的差距原封不动。

**用到.**

- [Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3) — 模型。
- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 多项式形式 (a) 与见证 (c)。

**被用于.**

- [Theorem 2.10](02-10_few_shot_information_floor_ZH.md#s2-10) — 计数独立证据，而不是暴露量。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) — 方法的计数规则。

## Related work

- (b) 背后的联合 cumulant 与矩–cumulant 公式是经典的（[Leonov & Shiryaev, 1959](#ref-leonov1959method)）。速率来自它们在平均下的标度：$`n`$ 个 IID block 的平均值，其 $`s`$ 阶 cumulant 等于 $`n^{1-s}\kappa_s`$。
