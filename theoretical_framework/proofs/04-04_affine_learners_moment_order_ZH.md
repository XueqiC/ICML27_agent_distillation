# Theorem 4.4 · 仿射 learner：矩阶充分且必要

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-04_affine_learners_moment_order_ZH.md) · [证明](04-04_affine_learners_moment_order_ZH.md) · [实验](../experiments/04-04_affine_learners_moment_order_ZH.md) · [English](04-04_affine_learners_moment_order.md)

**本页证明什么.** 对做二元决策的学生、至多 $`N`$ 次决策的部署，以及对独立经验块取固定凸平均的 learner，两个来源对每个这样的任务与 learner 给出相同的期望部署回报，当且仅当它们直到 $`N`$ 阶的联合矩相等；每一阶都必要；在网格上统计量的精确个数是一个二项式系数。证明依赖两个事实：部署价值是动作概率的 $`N`$ 次多项式；IID 块的仿射平均的多项式，其期望可以分解成块矩。

**本页用到的设定**（[Definition 4.3](../statements/04-03_binary_context_student_affine_learner_ZH.md#s4-3)）.

- 学生：$`k`$ 个二元动作 context，参数 $`p\in[0,1]^k`$；在 context 为 $`j`$ 的决策点，给定历史，它以条件概率 $`p_j`$ 选择动作 1。
- 部署：由机会节点（转移概率与 $`p`$ 无关）、决策节点（带 context 标签 $`j`$，可依赖历史但不直接依赖 $`p`$）与终止节点（奖励在 $`[0,1]`$ 中）构成的有限树；每条根到叶的路径至多 $`N`$ 个决策节点；$`J_E(p)`$ 是期望终止奖励。
- 经验：块 $`Z_1,\dots,Z_R\in[0,1]^k`$ IID 于 $`P`$（或 $`Q`$），块内坐标可任意相关。learner：$`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，$`p_0`$ 固定，权重 $`w_i\ge0`$、$`\sum_{i=0}^Rw_i=1`$ 固定。
- 记号：$`p^\alpha=\prod_jp_j^{\alpha_j}`$，$`\lvert\alpha\rvert=\sum_j\alpha_j`$，$`m_P(\alpha)=\mathbb E_PZ^\alpha`$；对 $`f=\sum_\alpha c_\alpha p^\alpha`$，$`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$；$`e_j`$ 是第 $`j`$ 个单位多重指标。
- 特征接口、$`H_\phi`$ 与未分辨差距如 [Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)；response 类 $`\mathcal G`$ 收集所考虑的任务与 learner 实现的函数 $`Z\mapsto J_E(p_R)`$。

**陈述（[Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）.** 在 Definition 4.3 的设定下：

- **(a) 部署价值是多项式.** $`J_E`$ 有唯一表示

```math
J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N, \qquad(4.3)
```

- 且在 $`[0,1]^k`$ 上 $`J_E(p)\in[0,1]`$。
- **(b) 联合阶数的充要条件.** 以下两条等价：
  - 对所有 $`\lvert\alpha\rvert\le N`$，$`m_P(\alpha)=m_Q(\alpha)`$；
  - 对每个至多 $`N`$ 次决策的部署和每个凸仿射 learner，$`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$。
- **(c) 每一阶都必要.**
  - *标量见证.* 设 $`N\ge1`$，$`z_i=i/N`$，$`i=0,\dots,N`$。令 $`P_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数；奇数处为 $`0`$），$`Q_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数；偶数处为 $`0`$）。则 $`0\le\ell<N`$ 时 $`m_{P_N}(\ell)=m_{Q_N}(\ell)`$，而

```math
\Delta_N:=m_{P_N}(N)-m_{Q_N}(N)=\frac{(-1)^NN!}{2^{N-1}N^N},
```

  - 并且任务 $`J(p)=p^N`$ 配合一步 learner $`p=Z`$ 实现了差距 $`\Delta_N`$。
  - *Parity 见证.* 取 $`k=N`$，$`P`$ 为 $`\lbrace0,1\rbrace^N`$ 中偶 parity 向量上的均匀分布，$`Q`$ 为奇 parity 向量上的均匀分布。两者所有总次数低于 $`N`$ 的联合矩都相等。取 $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$，一步 learner $`p=Z`$ 给出 $`\mathbb E_PJ_{\rm even}(Z)=1`$、$`\mathbb E_QJ_{\rm even}(Z)=0`$。
- **(d) 精确统计量个数与二元壁垒.**
  - 在网格 $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上，要对 (b) 中所有任务与 learner 都充分，单个块的线性期望统计量最少需要 $`\binom{k+N}{N}-1`$ 个。
  - 在 $`\mathcal Y=\lbrace0,1\rbrace^k`$ 上，设 $`N\ge k`$，$`k`$ 维动作向量上的终止奖励任意。任何少于 $`2^k-1`$ 个统计量的特征接口，未分辨差距都恰好是 $`1`$。所以容忍任何 $`\varepsilon<1`$ 的差距都不会降低这一最坏情形维数；$`2^k-1`$ 个仿射无关的特征给出差距 $`0`$。

**证明.** 用到三个断言。

*Claim 1（基本运算下的系数范数）.* 设 $`f,g`$ 是 $`p`$ 的多项式，$`j\in\lbrace1,\dots,k\rbrace`$。

- $`\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}`$，因为 $`p^\beta\mapsto p^{\beta+e_j}`$ 把不同单项式映到不同单项式。
- $`\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}`$。
- 对权重 $`\pi_e\ge0`$、$`\sum_e\pi_e=1`$，$`\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}`$。

*Claim 2（仿射 learner 的块矩展开）.* 设 $`\lvert\alpha\rvert=a`$，把 $`p_R^\alpha`$ 的坐标因子列为 $`j(1),\dots,j(a)`$，其中坐标 $`j`$ 出现 $`\alpha_j`$ 次。则

```math
p_R^\alpha=\sum_{s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace}\Big(\prod_{l=1}^aw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^RZ_i^{\beta_i(s)},\qquad\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)} .
```

- 系数 $`\prod_lw_{s(l)}`$ 非负，总和为 $`\prod_l\sum_iw_i=1`$。
- 对每个 $`s`$，$`\sum_i\lvert\beta_i(s)\rvert\le a`$，所以至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$。
- 块独立同分布时，

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)). \qquad(4.4)
```

展开式就是对 $`a`$ 个线性形式 $`w_0p_{0,j(l)}+\sum_{i\ge1}w_iZ_{i,j(l)}`$ 的乘积用分配律，每个因子选定一个来源 $`s(l)`$；块间独立使每个块单项式乘积的期望可以分解。

*Claim 3（网格上的线性无关）.* 在 $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上，单项式 $`\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace`$ 作为 $`\mathcal Y`$ 上的函数线性无关；特别地，$`\lvert\alpha\rvert\le N`$ 的 $`\binom{k+N}N`$ 个单项式线性无关。

- $`k=1`$：次数不超过 $`N`$ 的多项式若在 $`N+1`$ 个不同点为零，则恒为零。
- $`k>1`$：写 $`f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)`$，设 $`f`$ 在 $`\mathcal Y`$ 上为零。对子网格中每个固定的 $`(p_2,\dots,p_k)`$，$`p_1\mapsto f`$ 次数不超过 $`N`$、有 $`N+1`$ 个零点，所以每个 $`f_a`$ 在 $`k-1`$ 维子网格上为零，由归纳假设是零多项式。
- 总次数单项式是其子集，个数为 $`\binom{k+N}N`$（隔板法：$`\lvert\alpha\rvert\le N`$ 的 $`\alpha\in\mathbb N^k`$ 与 $`\lvert\alpha\rvert=N`$ 的 $`\alpha\in\mathbb N^{k+1}`$ 一一对应）。

*(a).* 对部署树的节点 $`u`$，令 $`V_u(p)`$ 为从 $`u`$ 出发的期望终止奖励，$`n(u)`$ 为从 $`u`$ 到叶的路径上决策节点数的最大值。对 $`u`$ 的高度归纳，证明 $`\deg V_u\le n(u)`$、$`\lVert V_u\rVert_{\rm coef}\le3^{n(u)}`$。

- *叶节点.* $`V_u=r\in[0,1]`$ 为常数，范数至多 $`1=3^0`$。
- *机会节点.* $`V_u=\sum_e\pi_eV_{u_e}`$，概率与 $`p`$ 无关，且 $`n(u_e)\le n(u)`$。次数至多 $`n(u)`$；由 Claim 1，范数至多 $`\max_e3^{n(u_e)}\le3^{n(u)}`$。
- *context 为 $`j`$ 的决策节点.* 子节点为 $`u_1`$（动作 1）与 $`u_0`$（动作 0），$`V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}`$。次数至多 $`1+(n(u)-1)=n(u)`$；由 Claim 1，范数至多 $`3\cdot3^{n(u)-1}`$。

根节点处 $`n\le N`$。$`V_u`$ 是 $`[0,1]`$ 中奖励的平均，所以 $`J_E\in[0,1]`$。唯一性：若两个多项式在 $`[0,1]^k`$ 上相等，它们的差在一个内部非空的集合上为零，于是在内点处所有偏导数为零，所有系数相同。

*(b).*（第一条 ⇒ 第二条）由 (a)，$`\mathbb EJ_E(p_R)=\sum_{\lvert\alpha\rvert\le N}c_\alpha\,\mathbb Ep_R^\alpha`$。(4.4) 中出现的每个块矩都满足 $`\lvert\beta_i\rvert\le\lvert\alpha\rvert\le N`$，所以在 $`P`$ 与 $`Q`$ 下相同。

（第二条 ⇒ 第一条）固定 $`1\le\lvert\alpha\rvert\le N`$ 的 $`\alpha`$（$`\alpha=0`$ 平凡）。按固定顺序访问 context $`j`$ 恰好 $`\alpha_j`$ 次、当且仅当所有动作都为 1 时回报 1 的部署，含 $`\lvert\alpha\rvert\le N`$ 个决策，价值为 $`p^\alpha`$。取 $`R=1`$、$`w_0=0`$、$`w_1=1`$，学习价值在 $`P`$ 下为 $`m_P(\alpha)`$，在 $`Q`$ 下为 $`m_Q(\alpha)`$，所以第二条迫使两者相等。

*边缘分布不够.* $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$ 与 $`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$ 下两个坐标都是 $`\mathrm{Bernoulli}(\tfrac12)`$，但 $`m_P(1,1)=\tfrac12`$、$`m_Q(1,1)=0`$。任务"两个 context 各访问一次，两个动作都为 1 时回报 1"使训练出的学生相差 $`\tfrac12`$。

*(c).* 令 $`(\mathsf Ef)(x)=f(x+1)`$、$`\Delta=\mathsf E-I`$。由可交换算子 $`\mathsf E`$、$`I`$ 的二项式定理，$`(\Delta^Nf)(0)=\sum_{i=0}^N(-1)^{N-i}\binom Nif(i)`$。若 $`f`$ 次数为 $`d\ge1`$、首项系数为 $`c`$，则 $`\Delta f`$ 次数为 $`d-1`$、首项系数为 $`cd`$；所以 $`\deg f<N`$ 时 $`\Delta^Nf=0`$，且 $`\Delta^Nx^N=N!`$。

- *标量见证，总质量.* $`N\ge1`$ 时 $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$，因为两者之差为 $`(1-1)^N=0`$，之和为 $`2^N`$。
- *标量见证，矩.* 对 $`\ell\ge0`$，

```math
m_{P_N}(\ell)-m_{Q_N}(\ell)=\frac{1}{2^{N-1}N^\ell}\sum_{i=0}^N(-1)^i\binom Ni\,i^\ell=\frac{(-1)^N}{2^{N-1}N^\ell}\big(\Delta^Nx^\ell\big)(0),
```

- 它在 $`\ell<N`$ 时为零，在 $`\ell=N`$ 时等于 $`(-1)^NN!/(2^{N-1}N^N)`$。
- *标量见证，实现.* 访问同一 context $`N`$ 次、仅当全部动作为 1 时回报 1，得到 $`J(p)=p^N`$；配合 $`p=Z`$，学习差距为 $`\Delta_N`$。
- *Parity 见证，真子集上均匀.* 设 $`S\subsetneq\lbrace1,\dots,N\rbrace`$、$`x_S\in\lbrace0,1\rbrace^S`$。补集 $`S^c\ne\varnothing`$ 有 $`2^{N-\lvert S\rvert}`$ 种补全，奇偶各占一半，因为翻转 $`S^c`$ 中某个固定坐标是改变 parity 的双射。所以 $`P(Z_S=x_S)=Q(Z_S=x_S)=2^{-\lvert S\rvert}`$。
- *Parity 见证，矩.* $`\lvert\alpha\rvert<N`$ 时，支撑 $`S=\lbrace j:\alpha_j>0\rbrace`$ 满足 $`\lvert S\rvert<N`$。在二元向量上，$`\alpha_j\ge1`$ 时 $`Z_j^{\alpha_j}=Z_j`$，所以 $`Z^\alpha=\prod_{j\in S}Z_j`$ 只依赖于 $`Z_S`$，两种分布下期望相同。
- *Parity 见证，价值.* 对每个 context 访问一次，动作 $`a_j\sim\mathrm{Bernoulli}(p_j)`$ 相互独立，$`\sum_ja_j`$ 为偶数时回报 1。则 $`\Pr(\text{even})=\tfrac12\big(1+\mathbb E(-1)^{\sum_ja_j}\big)=\tfrac12\big(1+\prod_j(1-2p_j)\big)=J_{\rm even}(p)`$。当 $`p=Z`$ 为二元向量时动作就等于 $`Z`$，所以 $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$，它在 $`P`$ 下期望为 $`1`$，在 $`Q`$ 下为 $`0`$。

*(d)，网格计数.*

- *上界.* 由 (b)，$`\binom{k+N}N-1`$ 个非常数矩 $`\lbrace m(\alpha):1\le\lvert\alpha\rvert\le N\rbrace`$ 已经充分。
- *下界.* 一步 learner $`p=Z`$ 配合 (b) 中的单项式任务，把所有 $`\lvert\alpha\rvert\le N`$ 的单项式 $`p^\alpha`$（限制在 $`\mathcal Y`$ 上）放进 $`\mathcal G`$。有限网格上的所有分布都允许，所以由 [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)（特征均值相等对所有分布推出 response 均值相等，当且仅当 response 落在特征的 span 中），充分的接口必须满足 $`\mathrm{span}\mathcal G\subseteq H_\phi`$。由 Claim 3，$`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\ge\binom{k+N}N`$，因此 $`d\ge\binom{k+N}N-1`$。

*(d)，二元壁垒.*

- *每张奖励表都是 response.* 取 $`p=Z`$，部署对 $`k\le N`$ 个 context 各访问一次，动作向量就等于 $`Z\in\lbrace0,1\rbrace^k`$，所以任意终止奖励 $`f:\lbrace0,1\rbrace^k\to[0,1]`$ 都实现为 $`g=f`$。因此 $`\mathcal G=[0,1]^{\mathcal Y}`$。
- *少于 $`2^k-1`$ 个特征时差距为 1.* 若 $`d<2^k-1`$，则 $`\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert`$，存在 $`v\in H_\phi^\perp\setminus\lbrace0\rbrace`$。由于 $`\mathbf 1\in H_\phi`$，$`v_\pm`$ 质量相同，记为 $`s>0`$。分布 $`P=v_+/s`$、$`Q=v_-/s`$ 支撑不相交，且 $`(P-Q)\phi_j=v^\top\phi_j/s=0`$。取 $`f=\mathbf 1_{\mathrm{supp}\,v_+}\in\mathcal G`$，得 $`(P-Q)f=1`$。response 都在 $`[0,1]`$ 中，差距不会超过 1。
- *用 $`2^k-1`$ 个特征时差距为 0.* 若 $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace`$ 线性无关，则 $`H_\phi=\mathbb R^{\mathcal Y}`$，特征均值相等就推出 $`P=Q`$。$`\square`$

## 附注

- **全称量词.** (b) 的必要性是对整个任务与 learner 族而言的。固定的任务或固定的更新配方可能只需要少得多的矩组合。
- **保留的是什么.** (b) 保留的是期望最终任务回报，不是训练所得参数的分布，也不是任务类中没有出现的尾部事件。
- **见证的大小.** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ 指数衰减，所以标量见证证明的是每一阶的精确必要性，而不是固定容差下的困难性。parity 见证在每一阶都给出常数差距，代价是需要 $`k=N`$ 个坐标。[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)（光滑的价值近似地只需约 $`\sqrt N`$ 阶，parity 需要全部 $`N`$ 阶）展开这一对比。
- **(4.3) 中的 $`3^N`$ 是保守的.** 对抗构造的部署离它有多近，见[实验页](../experiments/04-04_affine_learners_moment_order_ZH.md)。
- **假设在哪里起作用.**
  - 分解 (4.4) 需要独立同分布、与状态无关的块；依赖当前学生的 on-policy 反馈会破坏它。
  - 展开需要事先固定的非负凸权重；依赖数据的权重会破坏它。
  - 次数是在概率坐标 $`p`$ 中度量的。$`p=\sigma(\theta)`$ 这样的重参数化一般会破坏有限次数。
- **(d) 的二元壁垒有意取得很宽.** 任意终止奖励使每个函数都成为 response。结构化的任务族维数可以小得多。

## 相关工作

- 有限差分、parity 与真坐标子集的独立性、网格上单项式的线性无关性都是经典工具，在此注明，不作为新结果。
- 本页新增的是用它们计算：对凸仿射 learner 与至多 $`N`$ 次决策的部署，一个反馈来源必须携带的精确信息。
