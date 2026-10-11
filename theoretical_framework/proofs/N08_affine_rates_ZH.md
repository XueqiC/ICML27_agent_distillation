# N08 · 仿射理论中的速率 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N08_affine_rates_ZH.md) · [English](N08_affine_rates.md)

## 设定与假设

沿用 N07 的设定：学生 $`p\in[0,1]^k`$；部署价值 $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$（[N07(a)](N07_affine_moment_order_ZH.md)）；块 $`Z_i\in[0,1]^k`$ 独立同分布于固定的 $`P`$ 或 $`Q`$；凸仿射 learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，权重事先固定。

**定义。**

- 矩差异与 $`J`$ 的按次数加权的系数质量：

  ```math
  \delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
  ```

- 对平均 learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$，记 $`\mu=\mathbb E Z`$、$`W_n=\bar p_n-\mu`$。
- $`\kappa_s(P)`$ 是 $`Z\sim P`$ 的 $`s`$ 阶联合 cumulant 张量，分量为 $`\kappa(Z_{j_1},\dots,Z_{j_s})`$。
- $`D^sJ(\mu)`$ 是 $`s`$ 阶导数张量，$`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$。

在 (b) 中，$`P,Q,k,N,J`$ 固定，$`n\to\infty`$。

## 陈述

**(a)** **Theorem（与训练轮数无关的差异界）。** 对每个仿射 learner [N07](N07_affine_moment_order_ZH.md)，

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(29)
```

这个界与 $`R`$ 无关，并可截断在 1。

**(b)** **Theorem（第一个不匹配的 cumulant）。** 设 $`P,Q`$ 均值同为 $`\mu`$，且所有阶数 $`s<r`$ 的联合 cumulant 都相等，其中 $`2\le r\le N`$。则

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(30)
```

若式中的缩并为零，差为 $`O(n^{-r})`$。差距按 $`\Theta(n^{1-r})`$ 衰减，当且仅当缩并非零。

**(c)** **推论（标量见证的精确律）。** 对分布 [N07(c)](N07_affine_moment_order_ZH.md) 与 $`J(p)=p^N`$，

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{for every }n\ge1. \qquad(31)
```

**(b′)** **推论（均值不同）。** 若 $`\mu_P\ne\mu_Q`$，则 $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$。

**(c′)** **推论（replay 与后代样本不会衰减）。** 设 $`n`$ 个训练块为 $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$，即同一个母块 $`Z_1`$ 经固定确定性映射 $`T`$ 的结果（replay 对应 $`T`$ 为恒等映射）。则对所有 $`n`$ 都有 $`\bar p_n=T(Z_1)`$，学习价值之差等于 $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$，与 $`n`$ 无关。

## 证明

### 引理 A（乘积的伸缩估计）
对 $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$，$`\ \lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$。

*证明。* $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$，括号中的乘积都在 $`[0,1]`$ 内。$`\square`$

### (a) 的证明

固定 $`1\le\lvert\alpha\rvert=a\le N`$。对 $`p_R^\alpha`$ 用 [N07](N07_affine_moment_order_ZH.md) 的展开 [N07(b)](N07_affine_moment_order_ZH.md)：

```math
\mathbb E\,p_R^\alpha=\sum_s\omega_s\,\pi_s\prod_{i=1}^Rm(\beta_i(s)),\qquad\omega_s=\prod_lw_{s(l)}\ge0,\quad\sum_s\omega_s=1,\quad\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1].
```

对固定的分配 $`s`$，至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$，且这些块都有 $`1\le\lvert\beta_i(s)\rvert\le a\le N`$。由于 $`Z\in[0,1]^k`$，所有块矩都在 $`[0,1]`$ 内。由引理 A，

```math
\Big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\Big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N .
```

以权重 $`\omega_s`$ 对 $`s`$ 求和，得 $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$。常数项 $`\alpha=0`$ 相互抵消。于是

```math
\Big\lvert\sum_\alpha c_\alpha\big(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\big)\Big\rvert\le\sum_\alpha\lvert c_\alpha\rvert\lvert\alpha\rvert\,\delta_N=C_J\delta_N .
```

最后，由 [N07(a)](N07_affine_moment_order_ZH.md)，$`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$；两个价值都在 $`[0,1]`$ 内，所以差不超过 1。由于对每个 $`R`$ 都有 $`\sum_s\omega_s=1`$，这个界关于 $`R`$ 一致成立。$`\square`$

### 引理 B（经验均值的 cumulant）
对 $`s\ge2`$，$`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$；且 $`\kappa_1(W_n)=0`$。

*证明。* $`s\ge2`$ 阶联合 cumulant 在平移下不变，且是多重线性的。对独立的加项，cumulant 可加，因为独立向量的混合 cumulant 为零。所以

```math
\kappa_s\Big(\tfrac1n\sum_iZ_i\Big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z).
```

$`W_n`$ 的均值为零。$`\square`$

### 引理 C（矩–cumulant 公式）
对任何矩有限的随机向量 $`W`$，

```math
\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa\big(W_{j_b}:b\in B\big),
```

其中 $`\Pi(s)`$ 是 $`\lbrace 1,\dots,s\rbrace`$ 的划分集合。这就是 cumulant 的定义式反演（Leonov–Shiryaev）。$`\square`$

### (b) 的证明

*Taylor 展开。* $`J`$ 是次数 $`\le N`$ 的多项式，它在 $`\mu`$ 处的 Taylor 展开有限项终止：

```math
J(\mu+w)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),w^{\otimes s}\rangle .
```

因此

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle. \qquad(T)
```

$`P`$ 与 $`Q`$ 的 $`\mu`$ 相同，所以两者的 $`D^sJ(\mu)`$ 也相同。$`s=0`$ 项相消，$`s=1`$ 项为零。

*各划分的阶。* 对 $`s\ge2`$，把引理 C 用于 $`W_n`$。由于 $`\kappa_1(W_n)=0`$，含单点块的划分贡献为零。对于有 $`q`$ 个块、块大小 $`b_1,\dots,b_q\ge2`$ 的划分，引理 B 给出乘积

```math
\prod_{l=1}^q n^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z).
```

*哪些项可能不同。* 在 $`P`$ 和 $`Q`$ 下，阶数 $`<r`$ 的 cumulant 相同。只有当某个块的大小 $`b_l\ge r`$ 时，该划分项在两个分布下才可能不同。此时

```math
s=\sum_lb_l\ge r+2(q-1),\qquad\text{so}\qquad q-s\le2-r-q\le1-r.
```

$`q-s=1-r`$ 当且仅当 $`q=1`$ 且 $`s=r`$：

- 若 $`q=1`$ 且 $`s>r`$，则 $`q-s=1-s\le-r`$；
- 若 $`q\ge2`$，则 $`q-s\le2-r-q\le-r`$。

所以 $`n^{1-r}`$ 阶上唯一不同的项来自 $`s=r`$ 的单块划分，它贡献

```math
\frac1{r!}\langle D^rJ(\mu),\,n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle .
```

其余所有不同的项都带有 $`n^{-r}`$ 或更小的因子。这样的项只有有限多个（$`s\le N`$，划分有限），并且因为 $`Z\in[0,1]^k`$ 的 cumulant 有界，每个系数都有界。于是余项为 $`O(n^{-r})`$，常数只依赖于 $`P,Q,J`$。$`\square`$

### (c) 的证明

$`N=1`$ 时，(31) 就是 $`\mathbb E\bar p_n=\mu`$，差为 $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$。设 $`N\ge2`$。$`P_N,Q_N`$ 在阶数 $`<N`$ 的原始矩上一致；而 $`\kappa_s`$ 是阶数 $`\le s`$ 的矩的多项式，所以它们在所有阶数 $`<N`$ 的 cumulant 上也一致。取 $`r=N`$ 套用 (b) 的论证。因为 $`s\le\deg J=N`$，不同的划分必须含大小 $`\ge N`$ 的块，而这样的划分只有 $`s=N`$ 的单块划分，所以没有任何余项。又 $`D^N(p^N)=N!`$，且 $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$：$`\kappa_N`$ 等于 $`m_N`$ 减去一个低阶矩的多项式，而低阶矩相同。所以差恰为 $`\Delta_N\,n^{1-N}`$。$`\square`$

### (b′) 的证明

分别在 $`\mu_P`$ 与 $`\mu_Q`$ 处展开 (T)。$`s=0`$ 项给出 $`J(\mu_P)-J(\mu_Q)`$。对 $`s\ge2`$，每个块大小都 $`\ge2`$ 的划分满足 $`q\le s/2`$，因而 $`q-s\le-s/2\le-1`$。所以其余各项都是 $`O(n^{-1})`$。$`\square`$

### (c′) 的证明

$`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$，所以对每个 $`n`$，$`J(\bar p_n)`$ 的期望都是 $`\mathbb EJ(T(Z_1))`$。$`\square`$

## 附注

1. **为什么 (29) 中没有 $`R`$。** 凸权重使展开系数构成一个概率向量，与块数无关。带符号或无约束的权重会破坏这一点，可能放大差异；依赖数据的权重也会使分解失效。
2. **总体量，而非单次运行。** 两个定理讨论的都是训练过程的期望最终回报，而不是某一个训练出的学生的分数。
3. **对坐标的依赖。** $`C_J`$ 和 $`\delta_N`$ 是在概率坐标的单项式基下度量的。$`N`$ 较大时 (29) 可能失去意义，所以 N09 用逼近宽度代替次数。
4. **决定速率的是什么。** (30) 中的变量是*独立*块的个数 $`n`$。重放一个块，或用它的确定性后代，会增加训练曝光，但不会增加 $`n`$，$`r`$ 阶差异停留在单块时的值。
5. **固定的分布。** $`O(n^{-r})`$ 中的常数依赖于 $`(P,Q,J)`$。对 cumulant 不断增长的分布族，这个展开不是一致的。

## 依赖与来源

- **用到：** N07（[N07(a)](N07_affine_moment_order_ZH.md) 与块展开 [N07(b)](N07_affine_moment_order_ZH.md)）。
- **被用于：** 没有被直接使用。 是仿射分支的定量终点，它解释了 [N03](N03_few_shot_information_floor_ZH.md) 和 [N17](N17_ced_method_ZH.md) 用到的"replay 与独立证据"之分。

**出处：** [v0.14.1 Theorem 14](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-14-a-discrepancy-bound-independent-of-training-rounds); [v0.14.1 Theorem 15](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-15-the-first-unmatched-interaction-controls-averaging-decay) — Thms. 14–15; eqs. (29)–(31). v0.14.1 主框架 Theorems 14–15，式 (29)–(31)，§7，[Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md)；推论 15c 是 §13.1 中 replay/后代样本的陈述，也是 C18 规格中的内容。独立审阅：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A19–A21。cumulant 可加性与矩–cumulant 公式是经典工具。
