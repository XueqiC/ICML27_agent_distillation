# N07 · Affine learner：矩阶数既必要又充分 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N07_affine_moment_order_ZH.md) · [English](N07_affine_moment_order.md)

## 设定与假设

**学生与部署。**

- 学生有 $`k`$ 个二元动作 context，参数为 $`p\in[0,1]^k`$。在 context 为 $`j`$ 的决策点，给定全部过往历史，它以条件概率 $`p_j`$ 选择动作 1。
- 部署过程是一棵有限树，含三类节点：
  - *机会节点*：转移概率与 $`p`$ 无关；
  - *决策节点*：各带一个 context 标签 $`j`$，标签可依赖历史，但不能直接依赖 $`p`$；
  - *终止节点*：奖励在 $`[0,1]`$ 中，且与 $`p`$ 无关。
- 每条根到叶的路径至多含 $`N`$ 个决策节点。同一 context 可在一条路径上重复出现。
- $`J_E(p)`$ 是期望终止奖励。

**经验与 learner。**

每个经验块为 $`Z\in[0,1]^k`$，块内各坐标可任意相关。

块 $`Z_1,\dots,Z_R`$ 独立同分布于 $`P`$（或 $`Q`$）。

learner 是凸仿射平均

```math
p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1,
```

其中 $`p_0\in[0,1]^k`$ 是公共且固定的，权重事先固定、与数据无关。于是 $`p_R\in[0,1]^k`$。

**记号。**

- 对多重指标 $`\alpha\in\mathbb N^k`$，记 $`p^\alpha=\prod_jp_j^{\alpha_j}`$、$`\lvert\alpha\rvert=\sum_j\alpha_j`$、$`m_P(\alpha)=\mathbb E_PZ^\alpha`$。
- 对多项式 $`f=\sum_\alpha c_\alpha p^\alpha`$，记 $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$。

## 陈述

**(a)** **Lemma（部署价值是多项式）。** $`J_E`$ 有唯一表示

```math
J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad \sum_\alpha\lvert c_\alpha\rvert\le3^N, \qquad(24)
```

且在 $`[0,1]^k`$ 上 $`J_E(p)\in[0,1]`$。

**(b)** **Theorem（联合信息阶数的充要条件）。** 以下两条等价：

1. 对所有 $`\lvert\alpha\rvert\le N`$，$`m_P(\alpha)=m_Q(\alpha)`$。
2. 对  中每个至多 $`N`$ 次决策的部署任务、每个仿射 learner ，都有 $`\ \mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$。

**(c)** **Theorem（严格层级；常数差距见证）。**

(i) *标量见证。* 设 $`N\ge1`$，$`z_i=i/N`$（$`i=0,\dots,N`$）。令 $`P_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数；奇数处为 $`0`$），$`Q_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数；偶数处为 $`0`$）。则

```math
m_{P_N}(\ell)=m_{Q_N}(\ell)\ \ (0\le\ell<N),\qquad \Delta_N:=m_{P_N}(N)-m_{Q_N}(N)=\frac{(-1)^NN!}{2^{N-1}N^N}.
```

任务 $`J(p)=p^N`$ 配合一步更新 learner $`p=Z`$，即可实现差距 $`\Delta_N`$。

(ii) *Parity 见证。* 取 $`k=N`$，$`P`$ 为 $`\lbrace 0,1\rbrace ^N`$ 中偶 parity 向量上的均匀分布，$`Q`$ 为奇 parity 向量上的均匀分布。两者所有总次数 $`<N`$ 的联合矩都相等。取

```math
J_{\rm even}(p)=\tfrac12\Big(1+\prod_{j=1}^N(1-2p_j)\Big),
```

一步 learner $`p=Z`$ 给出 $`\mathbb E_PJ_{\rm even}(Z)=1`$、$`\mathbb E_QJ_{\rm even}(Z)=0`$。

**(d)** **Theorem（精确统计量个数；二元壁垒）。**

(i) 在网格 $`\mathcal Y=\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ 上，要对 (b) 中所有任务和 learner 都充分，所需单个块线性期望统计量的最小个数是 $`\binom{k+N}{N}-1`$。

(ii) 在 $`\mathcal Y=\lbrace 0,1\rbrace ^k`$ 上，设 $`N\ge k`$，$`k`$ 维动作向量上的终止奖励任意。任何少于 $`2^k-1`$ 个统计量的特征接口，最坏分布对差异都恰好是 $`1`$。所以容忍任何 $`\varepsilon<1`$ 的差异都不会降低这个最坏情形维数。用 $`2^k-1`$ 个仿射独立的特征时差异为 $`0`$。

## 证明

### 引理 A（基本运算下的系数范数）

设 $`f,g`$ 是 $`p`$ 的多项式，$`j\in\lbrace 1,\dots,k\rbrace`$。

- $`\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}`$，因为 $`p^\beta\mapsto p^{\beta+e_j}`$ 把不同单项式映到不同单项式。
- $`\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}`$。
- 对权重 $`\pi_e\ge0`$、$`\sum_e\pi_e=1`$，$`\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}`$。

### (a) 的证明

对部署树的节点 $`u`$，令 $`V_u(p)`$ 为从 $`u`$ 出发的期望终止奖励，$`n(u)`$ 为从 $`u`$ 到叶的路径上决策节点数的最大值。对 $`u`$ 的高度归纳，证明

```math
\deg V_u\le n(u),\qquad\lVert V_u\rVert_{\rm coef}\le3^{n(u)}.
```

*叶节点：* $`V_u=r\in[0,1]`$ 为常数，范数 $`\le1=3^0`$。

*机会节点：* $`V_u=\sum_e\pi_eV_{u_e}`$，概率与 $`p`$ 无关。每个子节点都有 $`n(u_e)\le n(u)`$，所以次数 $`\le n(u)`$；由引理 A，范数 $`\le\max_e3^{n(u_e)}\le3^{n(u)}`$。

*context 为 $`j`$ 的决策节点：* 子节点为 $`u_1`$（动作 1）与 $`u_0`$（动作 0），

```math
V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}.
```

次数 $`\le1+\max(\deg V_{u_1},\deg V_{u_0})\le1+(n(u)-1)=n(u)`$；由引理 A，范数 $`\le3\cdot3^{n(u)-1}=3^{n(u)}`$。

根节点处 $`n\le N`$。$`V_u`$ 是 $`[0,1]`$ 中奖励的概率加权平均，所以 $`J_E\in[0,1]`$。

*唯一性：* 若两个多项式在 $`[0,1]^k`$ 上相等，它们的差在一个内部非空的集合上为零，于是在内点处所有偏导数为零，所有系数都为零。$`\square`$

### 引理 B（仿射 learner 的块矩展开）

对每个 $`\lvert\alpha\rvert=a`$ 的多重指标 $`\alpha`$，把 $`p_R^\alpha`$ 的坐标因子列为 $`j(1),\dots,j(a)`$，其中坐标 $`j`$ 出现 $`\alpha_j`$ 次。则

```math
p_R^\alpha=\sum_{s:\lbrace 1,\dots,a\rbrace \to\lbrace 0,\dots,R\rbrace }\ \Big(\prod_{l=1}^aw_{s(l)}\Big)\ \prod_{l:s(l)=0}p_{0,j(l)}\ \prod_{i=1}^R Z_i^{\beta_i(s)},
```

其中 $`\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)}`$。系数非负，总和为 $`\prod_l\sum_iw_i=1`$。此外：

$`\sum_i\lvert\beta_i(s)\rvert\le a`$；

至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$；

块独立同分布时，

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)). \qquad(B2)
```

*证明。* 对 $`a`$ 个线性形式 $`w_0p_{0,j(l)}+\sum_{i\ge1}w_iZ_{i,j(l)}`$ 的乘积用分配律展开，每个因子选定一个来源 $`s(l)`$。块间独立，于是每个块单项式乘积的期望可以分解。$`\square`$

### (b) 的证明

（1 ⇒ 2）由 (a)，$`J_E=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$，所以 $`\mathbb EJ_E(p_R)=\sum_\alpha c_\alpha\,\mathbb Ep_R^\alpha`$。在 (B2) 中出现的每个块矩都满足 $`\lvert\beta_i\rvert\le\lvert\alpha\rvert\le N`$，由 (1) 它们在 $`P`$ 和 $`Q`$ 下相等。因此两个学习价值相同。

（2 ⇒ 1）固定 $`1\le\lvert\alpha\rvert\le N`$ 的 $`\alpha`$（$`\alpha=0`$ 时两个矩都是 1，平凡成立）。考虑这样的部署：按固定顺序访问 context $`j`$ 恰好 $`\alpha_j`$ 次，当且仅当所有动作都是 1 时回报 1。它有 $`\lvert\alpha\rvert\le N`$ 个决策，价值为 $`J(p)=p^\alpha`$。取 $`R=1`$、$`w_0=0`$、$`w_1=1`$ 的 learner，即 $`p_1=Z_1`$。它的学习价值在 $`P`$ 下为 $`m_P(\alpha)`$，在 $`Q`$ 下为 $`m_Q(\alpha)`$，所以由 (2) 两者相等。$`\square`$

*例（边缘分布不够）。* 取 $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$、$`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$。两种分布下两个坐标都是 $`\mathrm{Bernoulli}(\tfrac12)`$，但 $`m_P(1,1)=\tfrac12`$、$`m_Q(1,1)=0`$。任务"两个 context 各访问一次，两个动作都为 1 时回报 1"会使训练出的学生相差 $`\tfrac12`$。

### (c) 的证明

**有限差分。** 令 $`(\mathsf Ef)(x)=f(x+1)`$、$`\Delta=\mathsf E-I`$。由可交换算子 $`\mathsf E`$、$`I`$ 的二项式定理，

```math
(\Delta^Nf)(0)=\sum_{i=0}^N(-1)^{N-i}\binom Ni f(i).
```

若 $`f`$ 是 $`d\ge1`$ 次多项式、首项系数为 $`c`$，则 $`\Delta f`$ 是 $`d-1`$ 次多项式、首项系数为 $`cd`$。所以 $`\deg f<N`$ 时 $`\Delta^Nf=0`$，且 $`\Delta^N x^N=N!`$。

**(i) 标量见证。**

*总质量：* $`N\ge1`$ 时，$`\sum_{i\text{ even}}\binom Ni=\sum_{i\text{ odd}}\binom Ni=2^{N-1}`$，因为两者之差为 $`(1-1)^N=0`$，之和为 $`2^N`$。所以 $`P_N,Q_N`$ 都是概率分布。

*矩：* 对 $`\ell\ge0`$，

```math
m_{P_N}(\ell)-m_{Q_N}(\ell)=\frac{1}{2^{N-1}N^\ell}\sum_{i=0}^N(-1)^i\binom Ni\,i^\ell=\frac{(-1)^N}{2^{N-1}N^\ell}\,\big(\Delta^Nx^\ell\big)(0).
```

$`\ell<N`$ 时为零，$`\ell=N`$ 时等于 $`(-1)^NN!/(2^{N-1}N^N)`$。

*实现：* 访问同一 context $`N`$ 次、仅当全部动作为 1 时回报 1 的部署，其价值为 $`J(p)=p^N`$。配合 $`p=Z`$，学习差距就是 $`\Delta_N`$。

**(ii) Parity 见证。**

*真子集上是均匀的。* 设 $`S\subsetneq\lbrace 1,\dots,N\rbrace`$，固定 $`x_S\in\lbrace 0,1\rbrace ^S`$。补集 $`S^c\ne\varnothing`$ 共有 $`2^{N-\lvert S\rvert}`$ 种补全，奇偶各占一半：翻转 $`S^c`$ 中某个固定坐标就是一个改变 parity 的双射。所以 $`P(Z_S=x_S)=Q(Z_S=x_S)=2^{-\lvert S\rvert}`$。

*矩。* 当 $`\lvert\alpha\rvert<N`$ 时，支撑 $`S=\lbrace j:\alpha_j>0\rbrace`$ 满足 $`\lvert S\rvert\le\lvert\alpha\rvert<N`$。在二元向量上，$`\alpha_j\ge1`$ 时 $`Z_j^{\alpha_j}=Z_j`$，所以 $`Z^\alpha=\prod_{j\in S}Z_j`$ 只依赖于 $`Z_S`$，它在 $`P`$、$`Q`$ 下的期望相同。

*价值。* 让部署对每个 context 访问一次，动作 $`a_j\sim\mathrm{Bernoulli}(p_j)`$ 相互独立，当 $`\sum_ja_j`$ 为偶数时回报 1。则

```math
\Pr(\text{even})=\tfrac12\big(1+\mathbb E(-1)^{\sum_ja_j}\big)=\tfrac12\Big(1+\prod_j(1-2p_j)\Big)=J_{\rm even}(p).
```

当 $`p=Z`$ 为二元向量时，动作确定地等于 $`Z`$，所以 $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\text{ even}\rbrace`$。由此在 $`P`$ 下价值为 $`1`$，在 $`Q`$ 下为 $`0`$。$`\square`$

### (d) 的证明

**引理 C（网格上的线性无关）。** 在 $`\mathcal Y=\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ 上，单项式集合 $`\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace`$ 作为 $`\mathcal Y`$ 上的函数线性无关；特别地，$`\lvert\alpha\rvert\le N`$ 的 $`\binom{k+N}N`$ 个单项式线性无关。

*证明。* 对 $`k`$ 归纳。

- $`k=1`$：次数 $`\le N`$ 的一元多项式若在 $`N+1`$ 个不同点为零，则恒为零。
- $`k>1`$：写 $`f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)`$，设 $`f`$ 在 $`\mathcal Y`$ 上为零。固定子网格中的 $`(p_2,\dots,p_k)`$，关于 $`p_1`$ 的函数次数 $`\le N`$、在 $`N+1`$ 个点为零，所以每个 $`f_a`$ 在 $`k-1`$ 维子网格上为零。由归纳假设，每个 $`f_a`$ 都是零多项式。

总次数单项式是其子集。它们的个数为 $`\binom{k+N}{N}`$（隔板法：$`\lvert\alpha\rvert\le N`$ 的 $`\alpha\in\mathbb N^k`$ 与 $`\lvert\alpha\rvert=N`$ 的 $`\alpha\in\mathbb N^{k+1}`$ 一一对应）。$`\square`$

**(i) 计数。**

- *上界：* 由 (b)，$`\binom{k+N}N-1`$ 个非常数矩 $`\lbrace m(\alpha):1\le\lvert\alpha\rvert\le N\rbrace`$ 已经充分。
- *下界：* 一步 learner $`p=Z`$ 配合 (b) 中的单项式任务，其 response 类 $`\mathcal G`$ 包含所有 $`\lvert\alpha\rvert\le N`$ 的单项式 $`p^\alpha`$（限制在 $`\mathcal Y`$ 上）。有限网格上的所有分布都允许。任何充分的接口都必须让"特征均值相等"推出这些 response 的值全部相等，所以由 [N06(a)](N06_sufficiency_duality_ZH.md)，$`\mathrm{span}\mathcal G\subseteq H_\phi`$。由引理 C，$`\dim\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\ge\binom{k+N}N`$，因此 $`d\ge\binom{k+N}N-1`$。

**(ii) 二元壁垒。**

*每张奖励表都是一个 response。* 取 learner $`p=Z`$，部署对 $`k\le N`$ 个 context 各访问一次。由于 $`Z\in\lbrace 0,1\rbrace ^k`$，动作向量就等于 $`Z`$，于是任何终止奖励 $`f:\lbrace 0,1\rbrace ^k\to[0,1]`$ 都能实现为 $`g=f`$。所以 $`\mathcal G=[0,1]^{\mathcal Y}`$。

*少于 $`2^k-1`$ 个特征时差异为 1。* 设 $`d<2^k-1`$，则 $`\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert`$，所以存在 $`v\in H_\phi^\perp\setminus\lbrace 0\rbrace`$。由 $`\mathbf 1\in H_\phi`$，$`v_\pm`$ 的质量相同，记为 $`s>0`$。令 $`P=v_+/s`$、$`Q=v_-/s`$：两者是支撑不相交的分布，且 $`(P-Q)\phi_j=v^\top\phi_j/s=0`$。取 $`f=\mathbf 1_{\mathrm{supp}v_+}\in\mathcal G`$，得

```math
(P-Q)f=1-0=1.
```

所有 response 都在 $`[0,1]`$ 中，差异不会超过 1，所以最坏分布对差异恰为 1，任何 $`\varepsilon<1`$ 都要求 $`d\ge2^k-1`$。

*用 $`2^k-1`$ 个特征时差异为 0。* 若 $`\lbrace \mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace`$ 线性无关，则 $`H_\phi=\mathbb R^{\mathcal Y}`$，特征均值相等就推出 $`P=Q`$，差异为 0。$`\square`$

## 附注

1. **全称量词。** (b) 的必要性是对*整个*任务/learner 族而言的。对固定的任务或固定的更新配方，所需的矩组合可能少得多。
2. **保留的是什么。** (b) 保留的是期望最终任务回报，不是训练所得参数的分布，也不是任务类中不出现的尾部事件。
3. **(24) 的松紧。** 界 $`3^N`$ 是保守的。对抗搜索找到的 $`\sum\lvert c_\alpha\rvert/3^N`$ 最大比值是 $`2/3`$（见实验 C6）。
4. **标量见证的大小。** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ 随 $`N`$ 指数衰减。它证明了每一阶的*精确*必要性，而不是固定容差下的困难性。Parity 见证在每一阶都给出常数差距，代价是需要 $`k=N`$ 个坐标。N09 展开这一对比。
5. **假设在哪里起作用。**
   - 分解 (B2) 需要独立同分布、与状态无关的块。依赖当前学生的 on-policy 反馈会破坏它。
   - 展开需要事先固定的非负凸权重；依赖数据的权重会破坏它。
   - 次数是在概率坐标 $`p`$ 中度量的。重参数化（例如 $`p=\sigma(\theta)`$）一般会破坏有限次数。
6. **(d)(ii) 有意取得很宽。** 任意终止奖励使每个函数都成为 response。结构化的任务族维数可以小得多（N09，C19）。

## 依赖与来源

- **用到：** N06（span 判据 [N06](N06_sufficiency_duality_ZH.md)–[N06(a)](N06_sufficiency_duality_ZH.md)，用于 (d)）。
- **被用于：** [N08](N08_affine_rates_ZH.md)（定量速率）、[N10](N10_update_composition_ZH.md)（多项式更新）。

**出处：** [v0.14.1 §7](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#7-an-explicit-closed-theory-deployment-dependence-composed-with-learning) — Lemma 10; Thms. 11–13; eqs. (23)–(28). v0.14.1 主框架 §7，Lemma 10、Theorems 11–13，式 (23)–(28)，[Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md)。独立审阅：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A16–A18、B3、C2。有限差分与 parity 独立性是经典工具。
