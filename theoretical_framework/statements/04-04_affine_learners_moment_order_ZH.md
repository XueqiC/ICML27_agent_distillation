# Theorem 4.4 · 仿射 learner：矩阶充分且必要

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-04_affine_learners_moment_order_ZH.md) · [证明](../proofs/04-04_affine_learners_moment_order_ZH.md) · [实验](../experiments/04-04_affine_learners_moment_order_ZH.md) · [English](04-04_affine_learners_moment_order.md)

**一段话.** 学生做二元决策，部署至多含 $`N`$ 个决策，learner 对独立的经验块取平均。两个经验来源究竟要共享哪些信息，才能让每个这样的任务与 learner 都给出相同的期望结果？这个定理给出完整的回答：块的直到 $`N`$ 阶的全部联合矩，不能少，也不需要多。每一阶都必要；在网格上精确个数是一个二项式系数；在二元块上、奖励任意时，少于整个分布的接口都无济于事。它是 [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) 中一般 span 判据的完全算出的实例。

**本页用到的设定**（[Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3)）.

- 学生有 $`k`$ 个二元动作 context，参数为 $`p\in[0,1]^k`$；在 context 为 $`j`$ 的决策点，它以概率 $`p_j`$ 选择动作 1。
- 部署是由机会节点、决策节点与终止节点构成的有限树，其概率、context 标签与 $`[0,1]`$ 中的奖励都与 $`p`$ 无关；每条路径至多 $`N`$ 个决策，$`J_E(p)`$ 是期望奖励。
- 经验块 $`Z_1,\dots,Z_R\in[0,1]^k`$ IID 于来源分布 $`P`$（或 $`Q`$）。凸仿射 learner 为 $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，权重 $`w_i\ge0`$、$`\sum_iw_i=1`$ 固定，$`p_0`$ 固定。
- 对多重指标 $`\alpha\in\mathbb N^k`$，$`p^\alpha=\prod_jp_j^{\alpha_j}`$，$`\lvert\alpha\rvert=\sum_j\alpha_j`$，$`m_P(\alpha)=\mathbb E_PZ^\alpha`$ 是块的联合矩。一个来源的学习价值为 $`\mathbb EJ_E(p_R)`$。
- 块的*线性期望统计量*是对固定函数 $`\phi`$ 的 $`\mathbb E\phi(Z)`$；由这类统计量构成的接口及其未分辨差距如 [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)。

<a id="s4-4"></a>

**Theorem 4.4（仿射 learner：矩阶充分且必要）.** 在 Definition 4.3 的设定下：

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

**说了什么.**

- (a) 至多 $`N`$ 次决策的部署，其期望回报是学生动作概率的至多 $`N`$ 次多项式，系数总量至多 $`3^N`$。
- (b) 两个来源对每个这样的部署与每个平均化 learner 给出相同的学习价值，当且仅当它们直到 $`N`$ 阶的联合矩都相等。边缘分布不够，交叉矩也起作用。
- (c) 任何一阶都不能省：对每个 $`N`$，都有在所有更低阶矩上一致、却仍然不同的来源。标量见证的差距随 $`N`$ 指数缩小，parity 见证的差距保持为 1。
- (d) 在网格上个数恰为 $`\binom{k+N}N-1`$。在二元块上、奖励任意时，任何比整个分布小的接口都可能被完全骗过。

**直觉.** 部署价值是 $`p`$ 的 $`N`$ 次多项式，对独立块取平均把这个多项式的期望变成不超过 $`N`$ 阶的块矩的组合。反过来，一个恰好询问 context $`j`$ 共 $`\alpha_j`$ 次、只在每次回答都是 1 时付费的部署，读出的正是单个矩 $`m(\alpha)`$。

**在框架中的作用.** 它在一个一切都可计算的模型里说明：反馈来源必须携带的信息由部署决定，部署长度 $`N`$ 确定矩的阶数。Theorem 4.5 加上速率，Proposition 4.6 把精确阶与近似阶分开，Theorem 4.7 说明其他 learner 如何改变阶数。

**说明.**

- (b) 的必要性是对整个任务与 learner 族而言的。固定的任务或固定的更新配方可能只需要少得多的矩组合。
- (b) 保留的是期望最终任务回报，不是训练所得参数的分布，也不是任务类中没有出现的尾部事件。
- 由于 $`\lvert\Delta_N\rvert\sim2\sqrt{2\pi N}(2e)^{-N}`$ 指数衰减，标量见证证明的是每一阶的精确必要性，而不是固定容差下的困难性。parity 见证在每一阶都给出常数差距，代价是需要 $`k=N`$ 个坐标。
- 假设在哪里起作用：(b) 背后的分解需要独立同分布、与状态无关的块（依赖当前学生的 on-policy 反馈会破坏它），需要事先固定的凸权重（依赖数据的权重会破坏它），并且次数是在概率坐标 $`p`$ 中度量的（$`p=\sigma(\theta)`$ 这样的重参数化一般会破坏有限次数）。
- (d) 的二元壁垒有意取得很宽：任意终止奖励使每个函数都成为 response。结构化的任务族维数可以小得多。

**推导要点.**

- (a) 在部署树上反向归纳：决策节点乘以 $`p_j`$ 或 $`1-p_j`$，次数加一，系数范数至多乘 3；机会节点取平均。
- (b) 用分配律展开 $`p_R^\alpha`$，得到不超过 $`\lvert\alpha\rvert`$ 阶块矩之积的概率加权和；块独立使期望可以分解。反过来，任务"访问 context $`j`$ 恰好 $`\alpha_j`$ 次，全部动作为 1 时回报 1"的价值为 $`p^\alpha`$，所以一步 learner $`p=Z`$ 能读出每一个矩。
- (c) $`N`$ 阶有限差分把低于 $`N`$ 次的多项式化为零，作用在 $`x^N`$ 上给出 $`N!`$；parity 分布在每个真坐标子集上都是均匀的。
- (d) 次数不超过 $`N`$ 的单项式在网格上线性无关，由 Theorem 4.2(a) 的 span 判据得到计数。在二元块上每张奖励表都是 response，任何维数低于 $`2^k`$ 的接口都有一个非零的零化方向，它分成两个支撑不相交的分布。

**例示.** 两个坐标，$`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$、$`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$。两者的边缘分布都是 $`\mathrm{Bernoulli}(\tfrac12)`$，但两次决策的任务"两个动作都为 1 时回报 1"让训练出的学生相差 $`\tfrac12`$，因为 $`m_P(1,1)=\tfrac12`$、$`m_Q(1,1)=0`$。所需的信息不是边缘分布，而是直到 $`N=2`$ 阶的联合矩。

**用到.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) — (d) 中计数所用的 span 判据。
- [Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3) — 学生、部署与仿射 learner。

**被用于.**

- [Theorem 4.5](04-05_round_independent_bound_averaging_rates_ZH.md#s4-5) — 由 (a)–(c) 得到定量界与速率。
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) — 标量见证与 parity 见证。
- [Theorem 4.7](04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7) — 次数复合的基本情形。
