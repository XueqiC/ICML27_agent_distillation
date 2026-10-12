# §4 速率与复合 — 证明

[← Framework](../README_ZH.md) · [Proofs](04_complexity_of_sufficiency_ZH.md) · [Experiments](../experiments/04_complexity_of_sufficiency_ZH.md) · [English](04_rates_and_composition.md)

本页包含 Theorem 4.5 至 Theorem 4.7；所用的[特征充分性](04_complexity_of_sufficiency_ZH.md#特征充分性)、[仿射 learner 设定](04_complexity_of_sufficiency_ZH.md#s4-3)及前面的证明见[同节页面](04_complexity_of_sufficiency_ZH.md)。

<a id="s4-5"></a>

**Theorem 4.5（与轮数无关的界与平均化速率）.** 在 Definition 4.3 的设定下，取 [Theorem 4.4(a)](04_complexity_of_sufficiency_ZH.md#s4-4) 中的 $`J(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，记

```math
\delta_N(P,Q)=\max_{1\le\lvert\alpha\rvert\le N}\lvert m_P(\alpha)-m_Q(\alpha)\rvert,\qquad C_J=\sum_\alpha\lvert c_\alpha\rvert\,\lvert\alpha\rvert .
```

对平均化 learner $`\bar p_n=\frac1n\sum_{i=1}^nZ_i`$，记 $`\mu=\mathbb EZ`$、$`W_n=\bar p_n-\mu`$。令 $`\kappa_s(P)`$ 为 $`Z\sim P`$ 的 $`s`$ 阶联合 cumulant 张量，$`D^sJ(\mu)`$ 为 $`s`$ 阶导数张量，$`\langle D^sJ(\mu),T\rangle=\sum_{j_1,\dots,j_s}\partial_{j_1\cdots j_s}J(\mu)\,T_{j_1\cdots j_s}`$。

- **(a) 与轮数无关的界.** 对每个凸仿射 learner，

```math
\big\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\big\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q). \qquad(4.5)
```

  这个界与 $`R`$ 无关，并且可以截断在 1。
- **(b) 第一个不匹配的 cumulant.** 设 $`P,Q`$ 均值同为 $`\mu`$，且所有阶数 $`s<r`$ 的联合 cumulant 相等，其中 $`2\le r\le N`$。固定 $`P,Q,k,N,J`$，令 $`n\to\infty`$，

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}). \qquad(4.6)
```

  若式中的缩并为零，差为 $`O(n^{-r})`$；速率为 $`\Theta(n^{1-r})`$ 当且仅当缩并非零。
- **(c) 标量见证的精确律.** 对 [Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 中的 $`P_N,Q_N`$ 与 $`J(p)=p^N`$，

```math
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N=\frac{\Delta_N}{n^{N-1}}\quad\text{对每个 }n\ge1 .
```

- **(d) 均值不同.** 若 $`\mu_P\ne\mu_Q`$，则 $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=J(\mu_P)-J(\mu_Q)+O(n^{-1})`$。
- **(e) replay 与确定性派生不衰减.** 若 $`n`$ 个训练块为 $`Z^{(1)}=\dots=Z^{(n)}=T(Z_1)`$，其中 $`Z_1`$ 是单个父块，$`T`$ 是固定的确定性映射（replay 即 $`T=\mathrm{id}`$），则对每个 $`n`$ 都有 $`\bar p_n=T(Z_1)`$，学习价值之差等于 $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$，与 $`n`$ 无关。

*Proof.* 用到三个断言。

*Claim 6（乘积的 telescoping）.* 对 $`a_1,\dots,a_q,b_1,\dots,b_q\in[0,1]`$，$`\lvert\prod_ia_i-\prod_ib_i\rvert\le\sum_i\lvert a_i-b_i\rvert`$。这是因为 $`\prod_ia_i-\prod_ib_i=\sum_{l=1}^q\big(\prod_{i<l}b_i\big)(a_l-b_l)\big(\prod_{i>l}a_i\big)`$，括号内的乘积都在 $`[0,1]`$ 中。

*Claim 7（经验均值的 cumulant）.* 对 $`s\ge2`$，$`\kappa_s(W_n)=n^{1-s}\kappa_s(Z)`$，且 $`\kappa_1(W_n)=0`$。$`s\ge2`$ 阶联合 cumulant 平移不变、多重线性，并且对独立求和项可加，因为独立向量的混合 cumulant 都为零。所以 $`\kappa_s\big(\tfrac1n\sum_iZ_i\big)=n^{-s}\sum_{i=1}^n\kappa_s(Z_i)=n^{1-s}\kappa_s(Z)`$；$`W_n`$ 的均值为零。

*Claim 8（矩–cumulant 公式）.* 对矩有限的随机向量 $`W`$，$`\mathbb E[W_{j_1}\cdots W_{j_s}]=\sum_{\pi\in\Pi(s)}\prod_{B\in\pi}\kappa(W_{j_b}:b\in B)`$，其中 $`\Pi(s)`$ 是 $`\lbrace1,\dots,s\rbrace`$ 的划分集合。这是 cumulant 的定义式反演（Leonov–Shiryaev）。

*(a).* 固定 $`1\le\lvert\alpha\rvert=a\le N`$ 的 $`\alpha`$。把 (4.4) 写成 $`\mathbb Ep_R^\alpha=\sum_s\omega_s\pi_s\prod_{i=1}^Rm(\beta_i(s))`$，其中 $`\omega_s=\prod_lw_{s(l)}\ge0`$、$`\sum_s\omega_s=1`$、$`\pi_s=\prod_{l:s(l)=0}p_{0,j(l)}\in[0,1]`$。

- 对固定的 $`s`$，至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$，每个都有 $`1\le\lvert\beta_i(s)\rvert\le N`$；由于 $`Z\in[0,1]^k`$，所有块矩都在 $`[0,1]`$ 中。
- 由 Claim 6，$`\big\lvert\pi_s\prod_im_P(\beta_i(s))-\pi_s\prod_im_Q(\beta_i(s))\big\rvert\le\sum_{i:\beta_i(s)\ne0}\lvert m_P(\beta_i(s))-m_Q(\beta_i(s))\rvert\le a\,\delta_N`$。
- 以 $`\omega_s`$ 为权求和，得 $`\lvert\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha\rvert\le\lvert\alpha\rvert\,\delta_N`$。常数项 $`\alpha=0`$ 抵消，所以 $`\lvert\sum_\alpha c_\alpha(\mathbb E_Pp_R^\alpha-\mathbb E_Qp_R^\alpha)\rvert\le C_J\delta_N`$。

最后，由 (4.3)，$`C_J\le N\sum_\alpha\lvert c_\alpha\rvert\le N3^N`$，并且两个学习价值都在 $`[0,1]`$ 中。由于对每个 $`R`$ 都有 $`\sum_s\omega_s=1`$，这个界关于 $`R`$ 一致。

*(b).* $`J`$ 是次数不超过 $`N`$ 的多项式，它在 $`\mu`$ 处的 Taylor 展开有限终止，所以

```math
\mathbb EJ(\bar p_n)=\sum_{s=0}^N\frac1{s!}\langle D^sJ(\mu),\mathbb E\,W_n^{\otimes s}\rangle .
```

$`P`$ 与 $`Q`$ 有同一个 $`\mu`$，所以 $`D^sJ(\mu)`$ 相同。$`s=0`$ 的项抵消，$`s=1`$ 的项为零。对 $`s\ge2`$，对 $`W_n`$ 用 Claim 8。

- 含单点块的划分为零，因为 $`\kappa_1(W_n)=0`$。有 $`q`$ 个块、大小 $`b_1,\dots,b_q\ge2`$ 的划分，由 Claim 7 贡献 $`\prod_ln^{1-b_l}\kappa_{b_l}(Z)=n^{q-s}\prod_l\kappa_{b_l}(Z)`$。
- 低于 $`r`$ 阶的 cumulant 相等，所以一个划分项在 $`P`$ 与 $`Q`$ 下可能不同，只有当某个块大小 $`b_l\ge r`$。此时 $`s=\sum_lb_l\ge r+2(q-1)`$，因此 $`q-s\le2-r-q\le1-r`$。
- $`q-s=1-r`$ 只在 $`q=1`$、$`s=r`$ 时成立：若 $`q=1`$ 且 $`s>r`$，则 $`q-s=1-s\le-r`$；若 $`q\ge2`$，则 $`q-s\le2-r-q\le-r`$。

所以阶数为 $`n^{1-r}`$ 的不同项只有 $`s=r`$ 处的单块划分，它贡献 $`\frac1{r!}\langle D^rJ(\mu),n^{1-r}(\kappa_r(P)-\kappa_r(Q))\rangle`$。其余不同的项都带有 $`n^{-r}`$ 或更小的因子。这样的项只有有限多个（$`s\le N`$），并且由于 $`Z\in[0,1]^k`$ 的 cumulant 有界，每项系数有界；所以余项为 $`O(n^{-r})`$，常数只依赖于 $`P,Q,J`$。

*(c).* $`N=1`$ 时差为 $`m_{P_1}(1)-m_{Q_1}(1)=\Delta_1`$。设 $`N\ge2`$。

- $`P_N`$ 与 $`Q_N`$ 低于 $`N`$ 阶的原点矩相等，因而低于 $`N`$ 阶的 cumulant 也相等，因为 $`\kappa_s`$ 是不超过 $`s`$ 阶矩的多项式。
- 在 (b) 的论证中取 $`r=N`$，不同的划分必须含大小至少为 $`N`$ 的块，而 $`s\le\deg J=N`$。这样的划分只有 $`s=N`$ 处的单块，所以根本没有余项。
- $`D^N(p^N)=N!`$，且 $`\kappa_N(P_N)-\kappa_N(Q_N)=m_{P_N}(N)-m_{Q_N}(N)=\Delta_N`$，因为 $`\kappa_N`$ 等于 $`m_N`$ 减去低阶矩的一个多项式，而低阶矩相等。所以差恰为 $`\Delta_Nn^{1-N}`$。

*(d).* 分别在 $`\mu_P`$ 与 $`\mu_Q`$ 处展开。$`s=0`$ 的项给出 $`J(\mu_P)-J(\mu_Q)`$。对 $`s\ge2`$，每个块大小至少为 2 的划分都有 $`q\le s/2`$，故 $`q-s\le-s/2\le-1`$，其余各项都是 $`O(n^{-1})`$。

*(e).* $`\bar p_n=\frac1n\sum_{i=1}^nT(Z_1)=T(Z_1)`$，所以对每个 $`n`$ 都有 $`\mathbb EJ(\bar p_n)=\mathbb EJ(T(Z_1))`$。$`\square`$

*关于 Theorem 4.5 的附注.*

- **为什么 (4.5) 中没有 $`R`$.** 凸权重使展开系数构成一个概率向量，与块数无关。带符号或无约束的权重会破坏这一点并可能放大差异；依赖数据的权重使分解失效。
- **总体量，而非单次运行.** 两个界讨论的都是训练过程的期望最终回报，不是某一个训练出的学生的分数。
- **依赖坐标.** $`C_J`$ 与 $`\delta_N`$ 是在概率坐标的单项式基下度量的。$`N`$ 大时 (4.5) 可能失去意义，所以 [Proposition 4.6](#s4-6) 用逼近宽度代替次数。
- **什么控制速率.** (4.6) 中的变量是独立块的个数 $`n`$。replay 或一个块的确定性派生（(e)）增加了训练暴露量，但不增加 $`n`$，$`r`$ 阶差异停留在单块时的值。这是 [Theorem 2.10](02_feasibility_and_acquisition_ZH.md#s2-10) 与 §9 方法所用的计数规则。
- **固定的分布.** $`O(n^{-r})`$ 的常数依赖于 $`(P,Q,J)`$；对 cumulant 增长的分布族，这个展开不是一致的。

## 近似复杂度

<a id="s4-6"></a>

**Proposition 4.6（精确阶不等于近似复杂度）.** 令 $`T_j`$ 为第一类 Chebyshev 多项式（$`T_j(\cos\theta)=\cos(j\theta)`$，在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，$`T_j(1)=1`$），$`\mathcal P_q`$ 为总次数不超过 $`q`$ 的多项式。考虑 $`[0,1]`$ 上的标量任务 $`J(p)=p^N`$（同一 context 访问 $`N`$ 次，全部动作为 1 时成功），以及 [Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 中 $`[0,1]^N`$ 上的 parity 任务 $`J_{\rm even}`$。对 $`p\in[0,1]`$，

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a) 光滑的价值只需要低的近似阶.** 设 $`q<N`$，$`h_q\in\mathcal P_q`$ 为 (4.7) 截断到 $`q`$ 阶的结果，$`B\sim\mathrm{Bin}(2N,\tfrac12)`$。则：
  - $`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$；
  - 若 $`[0,1]`$ 上两个分布 $`P,Q`$ 的 $`\ell\le q`$ 阶矩都相等，则 $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$；
  - 对任何带独立同分布标量块的凸仿射 learner，学习价值 $`\mathbb E\,p_R^N`$ 满足同样的界。

  因此，匹配到 $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ 阶的矩就能保证差距不超过 $`\varepsilon`$，尽管要对所有分布精确相等需要 $`N`$ 阶（[Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4)）。
- **(b) Parity 在次数低于 $`N`$ 时不可压缩.** 对每个 $`q<N`$，

```math
\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12 ,
```

  且常数 $`h\equiv\tfrac12`$ 取到最小值。

*Proof.* *Claim 9（恒等式 (4.7)）.* 令 $`2p-1=\cos\theta`$，$`\theta\in[0,\pi]`$，则 $`p=\cos^2(\theta/2)`$，

```math
p^N=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}{2}\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.
```

对 $`j\ge1`$ 把 $`b=N\pm j`$ 合并，并用 $`\binom{2N}{N+j}=\binom{2N}{N-j}`$，得 $`p^N=4^{-N}\binom{2N}N+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta)`$，而 $`\cos(j\theta)=T_j(2p-1)`$。两边都是多项式且在 $`[0,1]`$ 上相等，所以 (4.7) 是多项式恒等式。

*(a)，第一条.* 记 $`a_0=4^{-N}\binom{2N}N`$，$`j\ge1`$ 时 $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$。则 $`p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1)`$。

- 所有 $`a_j\ge0`$ 且 $`\lvert T_j\rvert\le1`$，所以 $`\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j`$；在 $`p=1`$ 处每个 $`T_j(1)=1`$，等号成立。
- 把 $`b=N-j`$ 与 $`b=N+j`$ 配对，$`\sum_{j=q+1}^Na_j=\sum_{b:\lvert b-N\rvert>q}\binom{2N}b4^{-N}=\Pr\lbrace\lvert B-N\rvert>q\rbrace`$。
- $`B`$ 是 $`2N`$ 个独立 $`\lbrace0,1\rbrace`$ 变量之和，均值为 $`N`$，由 Hoeffding 不等式，$`\Pr\lbrace\lvert B-N\rvert\ge q\rbrace\le2\exp(-2q^2/(2N))=2e^{-q^2/N}`$。

*(a)，第二条.* 由于 $`h_q\in\mathcal P_q`$ 且两个分布的矩匹配到 $`q`$ 阶，$`\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)`$，所以 $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\lVert p^N-h_q\rVert_\infty\le4e^{-q^2/N}`$。

*(a)，第三条.* 对 $`\ell\le q`$，(4.4) 把 $`\mathbb Ep_R^\ell`$ 写成不超过 $`\ell`$ 阶块矩之积的固定组合。所以 $`p_R\in[0,1]`$ 在 $`P^R`$ 与 $`Q^R`$ 下的分布矩匹配到 $`q`$ 阶，第二条对它们适用。

*所需阶数.* $`4e^{-q^2/N}\le\varepsilon`$ 当且仅当 $`q\ge\sqrt{N\log(4/\varepsilon)}`$；$`q\ge N`$ 时由 [Theorem 4.4(b)](04_complexity_of_sufficiency_ZH.md#s4-4) 差距恰为 0。

*(b).* *上界.* $`J_{\rm even}`$ 是一个概率，所以 $`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$。

*下界.* 设 $`h\in\mathcal P_q`$，$`q<N`$；令 $`P,Q`$ 分别为 $`\lbrace0,1\rbrace^N`$ 中偶、奇 parity 顶点上的均匀分布。

- 在二元顶点上，$`a\ge1`$ 时 $`p_j^a=p_j`$，所以 $`h`$ 在 $`\lbrace0,1\rbrace^N`$ 上等于一个多重线性多项式，其单项式 $`\prod_{j\in S}p_j`$ 满足 $`\lvert S\rvert\le q<N`$。
- 两种分布下每个真坐标子集都是均匀的（[Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 的证明），所以 $`\mathbb E_Ph=\mathbb E_Qh`$。
- 在顶点上 $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$，所以 $`\mathbb E_PJ_{\rm even}=1`$、$`\mathbb E_QJ_{\rm even}=0`$。

因此

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace0,1\rbrace^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N},
```

即 $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$。等价地，由 [Theorem 4.2(b)](04_complexity_of_sufficiency_ZH.md#s4-2)，匹配对 $`(P,Q)`$ 证明了 $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$。$`\square`$

*关于 Proposition 4.6 的附注.*

- **对比.** $`p^N`$ 与 $`J_{\rm even}`$ 的精确次数都是 $`N`$，两者的精确充分性都需要直到 $`N`$ 阶的全部矩。但它们的近似需求完全不同：
  - $`p^N`$ 用 $`O(\sqrt{N\log(1/\varepsilon)})`$ 阶就能分辨到精度 $`\varepsilon`$；
  - $`J_{\rm even}`$ 用任何低于 $`N`$ 的阶都分辨不到 $`\tfrac12`$ 以内。
- 所以起作用的复杂度是 response 类的近似维数 $`d_\varepsilon`$（[Theorem 4.2(b)](04_complexity_of_sufficiency_ZH.md#s4-2)），而不是它的次数。
- **适用范围.** (a) 是对这个标量任务与凸仿射 learner 所需阶数的上界；不声称近似阶有匹配的下界。
- **稳定性.**
  - 精确的低阶匹配不是有限样本证书。若矩只是估计值，$`h_q`$ 在幂基下的系数会放大估计误差，而这些系数可以很大。
  - 有界的 Chebyshev 特征对单个标量消息避免了幂基放大，但经过多次仿射更新拉回之后，它们也需要自己的系数界。
  - 一般情形下相关的是 [Theorem 4.2(c)](04_complexity_of_sufficiency_ZH.md#s4-2) 中的 $`(B,\tau)`$，[Theorem 8.1](08_verification_ZH.md#s8-1) 把这一权衡精确化。
- **可实现性.** 两族见证都是可实现的有限反馈协议：reset 实验可以按指定分布发出一个块。这说明这样的环境存在，不说明某个 benchmark 允许它们。

## 阶数由 learner 决定

<a id="s4-7"></a>

**Theorem 4.7（更新复合放大信息需求）.** 设定分三部分。

- **多项式更新.** 固定 $`R\ge1`$ 与公共的 $`p_0\in[0,1]^k`$。相互独立的块 $`Z_1,\dots,Z_R`$ 取值于有限集 $`\mathcal Y\subset[0,1]^k`$；在参考数据源下块 $`Z_i`$ 的分布为 $`P_i`$，在备选数据源下为 $`Q_i`$。learner 为 $`p_i=U_i(p_{i-1},Z_i)`$，其中 $`U_i`$ 把 $`[0,1]^k\times\mathcal Y`$ 映入 $`[0,1]^k`$，$`U_i`$ 的每个坐标都是多项式，并满足
  - **(D1)** 关于 $`p_{i-1}`$ 的次数不超过 $`a_i\ge1`$（看作系数为 $`Z_i`$ 的多项式的、$`p_{i-1}`$ 的多项式）；
  - **(D2)** 关于 $`Z_i`$ 的次数不超过 $`b_i\ge1`$（看作系数为 $`p_{i-1}`$ 的多项式的、$`Z_i`$ 的多项式）。

  部署价值 $`J:[0,1]^k\to[0,1]`$ 是总次数不超过 $`N`$ 的多项式；Definition 4.3 中每个至多 $`N`$ 次决策的部署都满足这一点（[Theorem 4.4(a)](04_complexity_of_sufficiency_ZH.md#s4-4)）。
- **光滑局部设定.** $`\theta\in\mathbb R^p`$，$`V:\mathbb R^p\to\mathbb R`$ 是一个 continuation value（[Definition 1.4](../README_ZH.md#s1-4)）。consumed gradient 分布为 $`P`$ 的一步 SGD 作用为 $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$，记 $`\mu_P=\mathbb E_PG`$、$`M_P=\mathbb E_PGG^\top`$。
  - **(T3)** 对 $`P`$、$`Q`$ 支撑中的每个 $`g`$，$`V`$ 在线段 $`[\theta,\theta-\eta g]`$ 上为 $`C^3`$，并且在这些线段上 $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$。
- **误差递推.** 非负序列满足 $`e_0=0`$ 与 $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$，常数 $`\eta,L,C\ge0`$。

则：

- **(a) 次数的复合.** 在 (D1)–(D2) 下，把 $`J(p_R)`$ 看作 $`(Z_1,\dots,Z_R)`$ 的多项式，它关于第 $`i`$ 块的总次数不超过

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

  因此，若对每个 $`i`$ 与每个 $`\lvert\alpha\rvert\le D_i`$ 都有 $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$，则 $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$。若所有块服从同一个数据源分布，匹配到 $`\max_iD_i`$ 阶即可。
- **(b) 上界可达.** 对每个 $`R\ge1`$，都存在该类中的更新与一个 $`N=1`$ 的部署，使 response 恰为 $`Z_1^{2^{R-1}}`$。[Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 中 $`M=2^{R-1}`$ 阶的标量分布在所有低于 $`M`$ 阶的矩上一致，学习价值却不同，所以 (4.8) 一般不能改进。
- **(c) 阈值 learner.** 对 learner $`p=\mathbf 1\lbrace Z>1/2\rbrace`$ 与 $`J(p)=p`$（$`N=1`$），分布 $`P(Z=0)=\tfrac13`$、$`P(Z=\tfrac34)=\tfrac23`$ 与 $`Q(Z=\tfrac12)=1`$ 满足 $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$，但 $`\mathbb E_PJ(p)=\tfrac23`$、$`\mathbb E_QJ(p)=0`$。
- **(d) SGD 的局部展开.** 在 (T3) 下，

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big). \qquad(4.9)
```

- **(e) 误差累积.** 在误差递推下，

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

  当递推取等号时上式也取等号。$`L=0`$ 时线性界 $`e_R\le RC\eta^2`$ 成立；$`L>0`$ 时它一般不成立：取 $`\eta=\tfrac12`$、$`L=C=1`$，取等号的递推给出 $`e_4=\tfrac{65}{32}>1=RC\eta^2`$。

*Proof.* *(a).* 定义反向的 continuation 多项式 $`\Phi_R(p)=J(p)`$，以及 $`t=R,\dots,1`$ 时 $`\Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t(U_t(p,Z_t);Z_{t+1},\dots,Z_R)`$，于是 $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$。用反向归纳证明：$`\Phi_t`$ 关于 $`p`$ 的次数不超过 $`d_t:=N\prod_{s=t+1}^Ra_s`$（$`d_R=N`$），并且对每个 $`i>t`$，关于 $`Z_i`$ 的次数不超过 $`D_i`$。

- *起点 $`t=R`$.* $`\Phi_R=J`$ 关于 $`p`$ 的次数不超过 $`N`$，不含任何块。
- *归纳步.* 写 $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$，其中每个 $`c_\gamma`$ 对 $`i>t`$ 关于 $`Z_i`$ 的次数不超过 $`D_i`$。代入 $`p=U_t(p',Z_t)`$。单项式 $`p^\gamma`$ 变成 $`U_t`$ 的 $`\lvert\gamma\rvert`$ 个坐标多项式之积，每个因子由 (D1) 关于 $`p'`$ 的次数至多为 $`a_t`$，由 (D2) 关于 $`Z_t`$ 的次数至多为 $`b_t`$。乘积的次数相加，所以代入后的单项式关于 $`p'`$ 的次数至多为 $`a_td_t=d_{t-1}`$，关于 $`Z_t`$ 的次数至多为 $`b_td_t=Nb_t\prod_{s>t}a_s=D_t`$。系数 $`c_\gamma`$ 不含 $`p'`$ 与 $`Z_t`$，所以关于 $`Z_i`$（$`i>t`$）的次数不变。

在 $`t=0`$ 处 $`p=p_0`$ 是常数，所以 $`J(p_R)`$ 关于每个 $`Z_i`$ 的次数都不超过 $`D_i`$。展开 $`J(p_R)=\sum c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$，这是有限和，且 $`\lvert\alpha_i\rvert\le D_i`$。由独立性，$`\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}`$；在匹配假设下，每个因子在 $`P_i`$ 与 $`Q_i`$ 下都相同。若对所有 $`i`$ 有 $`P_i=P`$、$`Q_i=Q`$，匹配到 $`\max_iD_i`$ 阶就蕴含匹配到每个 $`D_i`$ 阶。

*(b).* 令 $`U_1(p,z)=z`$，$`i\ge2`$ 时 $`U_i(p,z)=p\odot p`$，于是 $`b_1=1`$，$`i\ge2`$ 时 $`a_i=2`$（其余次数以 1 为上界）。两个映射都保持 $`[0,1]^k`$ 不变。部署一次，$`J(p)=p_1`$（$`N=1`$）。由归纳，逐坐标 $`p_i=Z_1^{2^{i-1}}`$，所以 response 为 $`Z_{1,1}^{2^{R-1}}`$，而 (4.8) 给出 $`D_1=1\cdot1\cdot2^{R-1}`$。对 $`M=2^{R-1}`$ 的分布 $`P_M,Q_M`$，低于 $`M`$ 阶的矩相等，而 $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M\ne0`$。后面的块没有用到，这是 (D2) 允许的。

*(c).* $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$。在 $`P`$ 下，$`\mathbf 1\lbrace Z>1/2\rbrace=1`$ 当且仅当 $`Z=\tfrac34`$，其概率为 $`\tfrac23`$。在 $`Q`$ 下，$`Z=\tfrac12`$ 不大于 $`\tfrac12`$，所以价值为 0。

*(d).* 固定任一支撑中的 $`g`$，令 $`\varphi(\tau)=V(\theta-\tau\eta g)`$，$`\tau\in[0,1]`$。由 (T3)，$`\varphi\in C^3[0,1]`$，带积分余项的 Taylor 定理给出 $`\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\tfrac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau`$，其中

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

由 (T3)，$`\lvert\varphi'''\rvert\le M_3\eta^3\lVert g\rVert^3`$，又 $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$。所以 $`V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g)`$，$`\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3`$。分别取 $`\mathbb E_P`$、$`\mathbb E_Q`$ 再相减：$`V(\theta)`$ 抵消，一次项与二次项给出式中的两项，而 $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$ 由三角不等式满足所述的界。

*(e).* 对 $`t`$ 归纳得 $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$：$`t=0`$ 时是空和；并且 $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^t(1+\eta L)^s`$。每一步只用到递推，所以等号会传递下去。闭式是几何级数之和。对于反例，$`e_{t+1}=\tfrac32e_t+\tfrac14`$ 给出 $`e_1=\tfrac14`$、$`e_2=\tfrac58`$、$`e_3=\tfrac{19}{16}`$、$`e_4=\tfrac{65}{32}`$。$`\square`$

*关于 Theorem 4.7 的附注.*

- **信息阶数属于"learner 与部署的复合".** 只有一步凸仿射更新（$`a_i=b_i=1`$）时，(4.8) 回到 $`D_i=N`$，即 [Theorem 4.4(b)](04_complexity_of_sufficiency_ZH.md#s4-4)。后面每一步非线性更新都会把对所有更早块的需求成倍放大。
- **次数依赖坐标.** 可逆的重参数化可能改变更新的多项式形式。内在的对象仍是 response 类及其近似维数（[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2)）；次数只是在所选概率坐标下一个易于处理的证书。
- **非多项式 learner.** (c) 说明，一旦更新不连续，一般就没有哪一阶矩是充分的；即使只有一次决策，所需的也不止一阶矩。
- **(4.9) 的适用范围.** 这个展开是局部的、单步的：一阶统计量是 consumed gradient 的均值，二阶统计量是梯度二阶矩。对带状态的 optimizer，完整学习器状态的增量取代参数步（[Proposition 5.1](05_optimizer_state_ZH.md#s5-1)）；cold 与 warm Adam 分别见 [Proposition 5.3](05_optimizer_state_ZH.md#s5-3) 与 [Propositions 5.5–5.6](05_warm_adam_ZH.md#s5-5)。
- **累积需要稳定性.** (e) 对它的递推是紧的。关于 $`R`$ 线性的界需要 $`\eta LR=O(1)`$（因为 $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$），或一个压缩的误差映射。

## 相关工作

- LP 对偶、全变差的事件形式、有限差分、parity 独立性、矩–cumulant 公式（Leonov–Shiryaev）、Chebyshev 展开和 Hoeffding 不等式都是经典工具，在此注明，不作为新结果。
- 本节新增的是用它们计算：对给定的 learner 与部署类，一个 feedback law 必须携带多少信息——response 类的 span 维数或逼近宽度、凸仿射 learner 的矩阶数，以及复合更新对它的放大。
