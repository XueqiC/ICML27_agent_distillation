# Definition 4.3 · 二元 context 学生与凸仿射 learner

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-03_binary_context_student_affine_learner_ZH.md) · [English](04-03_binary_context_student_affine_learner.md)

**一段话.** [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) 说明，两个 feedback law 必须共享的信息是 response 类的 span。为了看清对实际的 learner 与任务这个 span 是什么，这个定义固定一个一切都能算出来的模型：学生在少数几个 context 中做二元决策，部署是有限决策树，经验以独立的块到达，learner 对这些块取事先固定的凸平均。§4 在这个模型里算出精确的信息阶数、速率，以及非线性更新的影响。

**本页用到的设定.**

- 环境的部署回报在 $`[0,1]`$ 中，$`J_E(\pi)`$ 是一个 policy 的期望回报（[Definition 1.1](../README_ZH.md#s1-1)）。这里 policy 由学生的动作概率 $`p`$ 参数化。
- "独立同分布"简称 IID。

<a id="s4-3"></a>

**Definition 4.3（二元 context 学生与凸仿射 learner）.**

- **学生.** 学生有 $`k`$ 个二元动作 context，参数为 $`p\in[0,1]^k`$。在 context 为 $`j`$ 的决策点，给定全部历史，它以条件概率 $`p_j`$ 选择动作 1。
- **部署.** 部署过程是一棵有限树，含三类节点：
  - *机会节点*：转移概率与 $`p`$ 无关；
  - *决策节点*：各带一个 context 标签 $`j`$，标签可以依赖历史，但不能直接依赖 $`p`$；
  - *终止节点*：奖励在 $`[0,1]`$ 中，且与 $`p`$ 无关。
- 每条根到叶的路径至多含 $`N`$ 个决策节点，同一 context 可在一条路径上重复出现。$`J_E(p)`$ 是期望终止奖励。
- **经验.** 一个经验块为 $`Z\in[0,1]^k`$，块内各坐标可任意相关。块 $`Z_1,\dots,Z_R`$ IID 于 $`P`$（或 $`Q`$）。
- **凸仿射 learner.** 设 $`p_0\in[0,1]^k`$ 公共且固定，权重事先固定、与数据无关：

```math
p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1 .
```

- **记号.** 对 $`\alpha\in\mathbb N^k`$，记 $`p^\alpha=\prod_jp_j^{\alpha_j}`$、$`\lvert\alpha\rvert=\sum_j\alpha_j`$、$`m_P(\alpha)=\mathbb E_PZ^\alpha`$。对多项式 $`f=\sum_\alpha c_\alpha p^\alpha`$，记 $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$。
- 于是 $`p_R\in[0,1]^k`$，一个数据源的*学习价值*为 $`\mathbb E\,J_E(p_R)`$。

**说了什么.**

- 学生是一张动作概率表，每个 context 一项；一次部署要求它至多行动 $`N`$ 次。
- 经验的来源是块上的一个分布 $`P`$；learner 用事先选定的权重对 $`R`$ 个独立块取平均。
- 一个数据源的学习价值，是它训练出的学生的期望部署回报。

**直觉.** 这是结果不只取决于数据均值的最简单的 learner：一次部署可以多次查询相同或不同的 context，所以部署价值关于 $`p`$ 是非线性的。

**在框架中的作用.**

- 它是一般理论的可计算实例：对这个模型，[Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) 精确地把 response span 识别为直到 $`N`$ 阶的联合矩。
- 这些假设是有意取得很严的。依赖当前学生的 on-policy 反馈、依赖数据的权重、$`p=\sigma(\theta)`$ 这样的重参数化都会离开这个模型；§4 说明离开它会改变信息阶数（[Theorem 4.7](04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7)），§5 处理真实的 optimizer。

**用到.**

- [Definition 1.1](../README_ZH.md#s1-1) — 环境、policy 与回报。

**被用于.**

- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 矩阶充分且必要。
- [Theorem 4.5](04-05_round_independent_bound_averaging_rates_ZH.md#s4-5) — 与轮数无关的界与速率。
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) — $`p^N`$ 与 parity 的近似阶。
- [Theorem 4.7](04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7) — 非线性与复合的更新。

## Related work

- 部署是一个带机会节点的单人 extensive-form 博弈（[Kuhn, 1953](#ref-kuhn1953extensive)），context 扮演 information set 的角色，$`p`$ 是 behaviour strategy：每个 context 一个动作概率，凡在该 context 下作决策都用它。由于同一个 context 可以在一条路径上重复出现，$`J_E`$ 在单个坐标 $`p_j`$ 上的次数可以大于 1。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-kuhn1953extensive"></a>

**Harold W. Kuhn (1953).** Extensive games and the problem of information. In *Contributions to the Theory of Games, Volume II*, edited by H. W. Kuhn and A. W. Tucker, Annals of Mathematics Studies 28, 193–216, Princeton University Press, Princeton, NJ. DOI [10.1515/9781400881970-012](https://doi.org/10.1515/9781400881970-012).
