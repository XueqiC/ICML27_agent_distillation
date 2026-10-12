# Theorem 4.2 · 充分性对偶与迁移

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-02_sufficiency_duality_transfer_ZH.md) · [证明](../proofs/04-02_sufficiency_duality_transfer_ZH.md) · [实验](../experiments/04-02_sufficiency_duality_transfer_ZH.md) · [English](04-02_sufficiency_duality_transfer.md)

**一段话.** 设两个 feedback law 只通过消息的少数几个特征的均值来比较。特征均值相等何时能保证学习结果相等？不能保证时，结果的差能有多大？这个定理用线性代数回答这两个问题：精确充分性是一个 span 条件；最坏的隐藏差恰为最佳一致逼近误差的两倍；系数有界的分解把特征均值的误差传递为 response 的误差；不受限的 learner 需要整个分布。它是 §4 中每一个信息计数背后的一般计数工具。

**本页用到的设定.**

- $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。$`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量；$`\mathbf 1`$ 是全一向量。分布是 $`\Delta(\mathcal Y)=\lbrace P\ge0:\mathbf 1^\top P=1\rbrace`$ 中的向量，$`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$。
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是 response 类：单边界 response $`g(y)`$ = 收到消息 $`y`$、执行实际更新并完成参考训练之后的期望最终任务回报（[Definition 1.4](../README_ZH.md#s1-4)）。由 [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)，两个消息分布可以互换，当且仅当对每个 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$。
- 特征接口传递 $`\phi_1,\dots,\phi_d`$ 的均值；$`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$，$`H_\phi^\perp`$ 是它的正交补（[Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)）。
- $`\mathcal M_\phi`$ 是特征匹配的分布对 $`(P,Q)`$ 的集合，即对所有 $`j`$ 有 $`(P-Q)\phi_j=0`$。未分辨差距为 $`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$，$`d_0(\mathcal G)`$ 是使 $`\mathfrak D_\phi(\mathcal G)=0`$ 的最少特征个数。
- $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$，$`\mathbb E_P\phi=(\mathbb E_P\phi_1,\dots,\mathbb E_P\phi_d)`$，$`\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$ 是全变差距离。

<a id="s4-2"></a>

**Theorem 4.2（充分性对偶与迁移）.**

- **(a) 精确充分性.** "对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$"当且仅当 $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$。因此

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1. \qquad(4.1)
```

- **(b) 近似充分性.**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \mathrm{dist}_\infty(g,H_\phi). \qquad(4.2)
```

- 因此用 $`d`$ 个特征能达到的最小未分辨差距是 $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$，下确界取遍满足 $`H\ni\mathbf 1`$、$`\dim H\le d+1`$ 的子空间。
- **(c) 迁移.** 若目标类中的每个 $`g`$ 都能写成 $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$，且 $`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则对所有 $`P,Q\in\Delta(\mathcal Y)`$，

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

- **(d) 不受限的 learner.** 固定一个任务，它有两个 policy，价值分别为 $`a`$ 与 $`a+\gamma`$，$`\gamma>0`$。假设对每个事件 $`A\subseteq\mathcal Y`$，learner 族都含有一步规则"当且仅当 $`Y\in A`$ 时部署较好的 policy"。则

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1 .
```

**说了什么.**

- (a) 特征均值相等能对每一对分布推出 response 均值相等，当且仅当每个 response 都是常数加特征的组合。所需统计量个数是 response 的 span 维数，不计常数。
- (b) 若特征张不成 response，最坏的隐藏差等于特征逼近最坏 response 的最佳一致误差的两倍。这是等式，不只是上界。
- (c) 若每个 response 都是"系数总大小至多 $`B`$ 的特征组合，加上大小至多 $`\tau`$ 的残差"，那么大小为 $`\epsilon`$ 的特征均值误差至多造成 $`B\epsilon+2\tau`$ 的结果差。
- (d) 能对消息的任意事件作出反应的 learner，迫使接口携带整个分布：$`m-1`$ 个统计量。

**直觉.** 特征均值相等只排除了特征所张方向上的差异。response 落在这个 span 之外的部分，就是两个分布可以不被察觉地相差的方向。

**在框架中的作用.** 它把"两个 feedback law 必须共享多少信息"变成可计算的量：一个 span 维数或一个逼近宽度。§4 后面的每个计数、§5 的一阶统计量和 §8 的审计界都是它的应用。

**说明.**

- (b) 中的上确界取遍所有共享某组特征均值的分布对。如果参考分布事先固定，或要求两个分布落在同一个预先指定的特征均值纤维上，可行的差分集合会变小，(4.2) 只是上界。
- 若只有某个分布族可实现，(a) 应读作：$`(P-Q)g`$ 在特征匹配对上恒为零，当且仅当该族内每个特征匹配的差分都零化 $`g`$。
- 残差 $`\tau`$ 小而系数大小 $`B`$ 大时，特征估计误差会被放大；[Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1) 把这一权衡精确化。
- (d) 中 $`d_0=m-1`$ 要求 learner 能实现每个单点示性函数。实际的 learner 与任务能实现的 response 少得多；(d) 只说明：不限制任务与 learner 类，就不存在一致小的接口。

**推导要点.**

- (a) 若 $`g\in H_\phi`$，它是常数加特征的组合，两部分在匹配的分布下均值都相等。若 $`g\notin H_\phi`$，它与 $`H_\phi`$ 正交的分量 $`v`$ 给出一个方向：沿它倾斜均匀分布，所有特征均值都不动，而 $`(P-Q)g`$ 按 $`\lVert v\rVert_2^2`$ 的比例变化。
- (b) Hölder 不等式对每个 $`h\in H_\phi`$ 给出 $`\lvert(P-Q)(g-h)\rvert\le2\lVert g-h\rVert_\infty`$。可达性：线性规划对偶把上确界范数距离写成 $`\sigma^\top g`$ 在 $`\sigma\perp H_\phi`$、$`\lVert\sigma\rVert_1\le1`$ 上的最大值，而每个这样的 $`2\sigma`$ 都是两个分布之差。
- (c) 常数项抵消，Hölder 把特征项界为 $`B`$ 乘以特征均值之差，残差至多贡献 $`2\tau`$。
- (d) response $`a+\gamma\mathbf 1_A`$ 实现了全变差的事件形式，单点事件张成 $`\mathbb R^m`$。

**例示.** 取 $`\mathcal Y=\lbrace1,2,3\rbrace`$、单个特征 $`\phi(y)=y`$。分布 $`P=(\tfrac12,0,\tfrac12)`$ 与 $`Q=(0,1,0)`$ 均值相同，都是 2。response $`g(y)=y^2`$ 把它们分开 1：$`\mathbb E_Pg=5`$、$`\mathbb E_Qg=4`$。特征 $`\phi`$ 不充分，因为在三个点上 $`y^2\notin\mathrm{span}\lbrace1,y\rbrace`$。

**用到.**

- [Definition 1.4](../README_ZH.md#s1-4) — response 与 response 类。
- [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) — 可互换即在每个 response 上一致。
- [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1) — 特征接口、匹配对、未分辨差距。

**被用于.**

- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 仿射 learner 的精确矩计数。
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) — 由 (b) 得到 parity 的逼近误差。
- [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) — 一阶统计量的极小性。
- [Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1) — 带测量容差的对偶。
