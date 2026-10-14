# Proposition 8.13 · 跨 checkpoint 共享的消息特征

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](08-13_shared_features_across_checkpoints_ZH.md) · [证明](../proofs/08-13_shared_features_across_checkpoints_ZH.md) · [实验](../experiments/08-13_shared_features_across_checkpoints_ZH.md) · [English](08-13_shared_features_across_checkpoints.md)

**一段话概括.** 一次训练会经过很多 checkpoint，在每个 checkpoint 上都估计一次 response operator 负担不起。把多个 checkpoint 的 response 行堆起来、从中学一组共享的消息特征，是可以的，前提是分清什么被共享。被共享的是特征，不是反馈差，因为参考反馈与替代反馈都随状态变化。输入随状态而变时，正确的误差界是堆叠算子奇异值平方的尾部*之和*，而不是第一个被丢弃的奇异值的平方；截断后的奇异基也只对按状态平均的目标最优。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 所有考虑的 checkpoint 使用同一个消息空间与同一个参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$（中心化的平方可积函数）。
- 声明的测试集 $`\mathcal A`$ 与权重 $`\omega`$，在每个 checkpoint 上含义相同。不同的剩余预算属于测试标签的一部分：含义不同的两行永远不当作同一个测试。
- checkpoint $`z`$ 带概率权重 $`\nu`$。在每个 checkpoint 上，$`\mathcal R_z:\mathcal H\to L^2(\omega)`$ 是 [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator，因此 $`\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$。
- $`z`$ 处的实际反馈差为 $`h_z=dP_z/d\rho-dQ_z/d\rho\in\mathcal H`$，它依赖于 $`z`$。如果合法消息依赖于状态，就用一个共同的合法池或事先声明的拒绝规则；支持集一旦改变，$`\rho`$ 的覆盖就不再成立。
- *堆叠算子* $`\mathcal S:\mathcal H\to L^2(\nu\otimes\omega)`$ 为

```math
(\mathcal Sh)(z,a)=(\mathcal R_zh)(a).
```

它是 Hilbert–Schmidt 算子，$`\lVert\mathcal S\rVert_{\rm HS}^2=\mathbb E_{z\sim\nu}\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$；奇异值为 $`\sigma_1(\mathcal S)\ge\sigma_2(\mathcal S)\ge\dots`$，$`\Pi_r`$ 投影到它的前 $`r`$ 个右奇异向量上。*秩 $`r`$ 的共享线性消息接口* 是 $`\mathcal H`$ 上一个秩为 $`r`$ 的正交投影 $`\Pi`$，也就是 $`r`$ 个共享的消息特征。

<a id="s8-13"></a>

**Proposition 8.13（跨 checkpoint 共享的消息特征）.** 在上述设定下：

**(a) 条件输入的界.** 若对 $`\nu`$-几乎所有 $`z`$ 有 $`\lVert h_z\rVert_\rho\le B`$，则

```math
\mathbb E_{z\sim\nu}\,\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2 .
```

**(b) 对平均目标最优.** 在所有秩为 $`r`$ 的共享接口中，$`\Pi_r`$ 使 $`\mathbb E_{z\sim\nu}\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2`$ 最小，最小值为 $`\sum_{j>r}\sigma_j(\mathcal S)^2`$。这是对 $`\nu`$-平均的最优，不是对 checkpoint 分布的 minimax 陈述。

**(c) 尾部之和不能换成第一个被丢弃的奇异值.** 存在两个等权 checkpoint、有界分数与满足 $`\lVert h_z\rVert_\rho=B`$ 的输入，使得取 $`r=0`$（不保留任何特征）时，(a) 的左边等于 $`B^2c^2`$，而 $`B^2\sigma_1(\mathcal S)^2=B^2c^2/2`$；尾部之和等于 $`B^2c^2`$。分数有界、输入为概率分布，要求 $`0<c\le1/(2\lVert e\rVert_\infty)`$ 且 $`B\le2/\lVert e\rVert_\infty`$，其中 $`e_1,e_2`$ 是构造中所用的正交基；两个界之间的 2 倍与 $`c`$、$`B`$ 无关。

**(d) 从加权到实际访问的 checkpoint.** 如果实际访问的 checkpoint 服从分布 $`d`$，$`d\ll\nu`$ 且 $`dd/d\nu\le\kappa`$，那么在 $`d`$ 下 (a) 的左边至多为 $`\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$。没有这样的界时，$`\nu`$-平均对实际访问的 checkpoint 什么也说明不了；这时应使用 [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 的逐 checkpoint 保证，或者支付 [Proposition 3.4(d)](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 的退出项。

**说了什么.**

- 从多个 checkpoint 学一组消息特征是允许的；喂给这些特征的反馈差可以随状态变化。
- 输入随状态变化的代价是堆叠谱的整个尾部。只用 $`\sigma_{r+1}(\mathcal S)`$ 来界是错的，误差倍数可以和 checkpoint 个数一样大。
- 平均意义上的精度不会自动传到训练实际访问的状态，除非实际访问的分布被加权分布控制住。

**直觉.** 一个共同的输入每次只从一个方向看堆叠算子，所以最坏情形是它的最大奇异值。随状态变化的输入可以在每个 checkpoint 上对准该 checkpoint 自己的主方向；这时误差在各 checkpoint 上累加，只有奇异值平方之和才控制得住。

**在框架中的作用.**

- 它确定了跨 checkpoint 复用 response 接口时应使用的范数，这是在持续训练中维护充分性的第一步。
- [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 给出逐 checkpoint 的补充，[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 使用逐 checkpoint 的形式。

**关于陈述的说明.**

- **被共享的是什么.** 消息特征 $`\phi(y)`$，而不是 feedback law，也不是 $`h_z`$。在一个 checkpoint 上特征均值相等，不意味着在另一个 checkpoint 上也相等。
- **反例的直观说法.** 两个 checkpoint 对正交的方向作出响应，每个 checkpoint 的反馈差都指向它自己的方向。每个输入的范数都是 1，每个 checkpoint 的 response 也是 1，但堆叠算子的最大奇异值只有它的 $`1/\sqrt2`$（[Check 8.19](../experiments/08-13_shared_features_across_checkpoints_ZH.md#c8-19)）。

**推导要点.**

- (a)：在每个 checkpoint 上用 $`\lVert Ah\rVert\le\lVert A\rVert_{\rm HS}\lVert h\rVert`$，再对 $`\nu`$ 积分；$`\mathcal R_z(I-\Pi)`$ 的平均 Hilbert–Schmidt 范数就是 $`\mathcal S(I-\Pi)`$ 的 Hilbert–Schmidt 范数，在 $`\Pi_r`$ 处取值为尾部之和。
- (b)：$`\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\mathrm{tr}(\Pi\mathcal S^*\mathcal S\Pi)`$，再用 Ky Fan 极大值原理在秩 $`r`$ 投影上最大化迹。
- (c)：在三条消息上用有界分数实现两个正交的一维 response。
- (d)：测度变换。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8) — 单个 checkpoint 上的 response operator
- [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) — 单个 checkpoint 上的截断

**被用于.**

- [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) — 逐 checkpoint 复用
- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) — 组合保证
- [Remark 8.17](08-17_open_problems_ZH.md#s8-17) — 跨格复用

## 相关工作

- (b) 就是把 Eckart–Young 定理（[Eckart & Young, 1936](#ref-eckart1936approximation)）用到堆叠算子上。Datamodels 在许多测试样本上拟合同一个"输出对训练子集指示变量"的线性模型（[Ilyas et al., 2022](#ref-ilyas2022datamodels)）；这里的行来自不同的 learner state，要点是它们的输入随状态不同。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
