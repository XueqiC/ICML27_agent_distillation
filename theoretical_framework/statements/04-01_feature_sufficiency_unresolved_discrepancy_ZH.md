# Definition 4.1 · 特征充分性与未分辨差距

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md) · [English](04-01_feature_sufficiency_unresolved_discrepancy.md)

**一段话.** 两个 feedback law 在学习上可以互换，当且仅当它们在 response 类中的每个 response 上一致（[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)）。实际中人们只通过少数几个汇总统计量比较 law，例如消息的某些特征的均值。这个定义给出提问"需要多少个这样的统计量"所用的词汇：*特征接口*、它分不开的分布对，以及这些分布对仍能藏住的最大 response 差。

**本页用到的设定.**

- $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。一条消息是 learner 在一轮中消耗的一个反馈单元。
- $`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量；$`\mathbf 1`$ 是全一向量。$`\mathcal Y`$ 上的概率分布是 $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$ 中的向量，$`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$ 是 $`g`$ 在 $`P`$ 与 $`Q`$ 下的期望之差。
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是一个 *response 类*。在框架中，它收集单边界 response $`g_{t,z}(y)`$：learner 在状态 $`z`$ 收到消息 $`y`$、执行实际更新并完成声明的参考训练之后的期望最终任务回报（[Definition 1.4](../README_ZH.md#s1-4)）。这里只用到 response 在 $`\mathcal Y`$ 上的取值。
- 除非另有说明，$`\mathcal Y`$ 上的所有概率分布都允许。

<a id="s4-1"></a>

**Definition 4.1（特征充分性与未分辨差距）.**

- *特征接口*传递固定特征 $`\phi_1,\dots,\phi_d:\mathcal Y\to\mathbb R`$ 的均值。记 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace\subseteq\mathbb R^m`$，$`H_\phi^\perp=\lbrace v\in\mathbb R^m:v^\top h=0\ \forall h\in H_\phi\rbrace`$。任意 $`v\in H_\phi^\perp`$ 都满足 $`\mathbf 1^\top v=0`$。
- *特征匹配的分布对*为 $`\mathcal M_\phi=\lbrace(P,Q)\in\Delta(\mathcal Y)^2:(P-Q)\phi_j=0,\ j=1,\dots,d\rbrace`$。
- 接口在 $`\mathcal G`$ 上的*未分辨差距*为

```math
\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\ \sup_{g\in\mathcal G}\ \lvert(P-Q)g\rvert .
```

- *精确维数* $`d_0(\mathcal G)`$ 是满足下述条件的最小 $`d`$：存在特征 $`\phi_1,\dots,\phi_d`$，使得对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$。*近似维数* $`d_\varepsilon(\mathcal G)`$ 是使某 $`d`$ 个特征满足 $`\mathfrak D_\phi(\mathcal G)\le\varepsilon`$ 的最小 $`d`$。
- 对 $`g\in\mathbb R^m`$ 与子空间 $`H`$，记 $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$。

**说了什么.**

- 特征接口用有限个期望来汇总一个 feedback law。汇总相同的两个分布构成一个特征匹配的分布对。
- 未分辨差距是特征匹配的分布对仍能造成的最坏 response 差：汇总看不见的风险。
- $`d_0`$ 计的是使这个风险恰为零所需的统计量个数；$`d_\varepsilon`$ 计的是使它不超过 $`\varepsilon`$ 所需的个数。

**直觉.** 如果汇总统计量相等，而某个 response 仍然不同，说明汇总丢掉了 learner 与任务要用的信息。

**在框架中的作用.**

- 这些维数计的是固定的线性期望统计量个数，不是比特复杂度、样本量或查询次数；从有限数据估计这些统计量是 §8 的内容（[Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1)）。
- §4 中的每个计数，都是某个特定 response 类的 $`d_0`$ 或 $`d_\varepsilon`$。

**用到.**

- [Definition 1.4](../README_ZH.md#s1-4) — 单边界 response 与 response 类。

**被用于.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) — 精确算出 $`d_0`$ 与 $`\mathfrak D_\phi`$。
- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 仿射 learner 的精确计数。
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) — 近似阶与精确阶。
- [Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1) — 带误差地测量特征的代价。
