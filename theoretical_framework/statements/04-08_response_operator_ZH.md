# Definition 4.8 · 检查点上的 response operator

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-08_response_operator_ZH.md) · [English](04-08_response_operator.md)

**一段话.** [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) 在一致范数下计数：对整个 response 类，需要匹配多少个特征均值。若要度量 feedback 差异的每个方向*有多重要*，并在一个明确的度量下做压缩，就把同一个检查点上许多声明测试的 response 收进一个线性算子：它把两个 feedback law 之间的差映射为该差在每个测试上造成的得分差向量。本页固定这个算子、决定其几何的测度，以及使它成为带奇异值分解的 Hilbert–Schmidt 算子的几个基本事实。它的谱在 [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 中研究。

**本页用到的设定.**

- 检查点 $`(t,z)`$：第 $`t`$ 轮与 complete learner state $`z`$，包括 optimizer state、缓冲区与历史（[Definition 1.3](../README_ZH.md#s1-3)）。消息空间 $`\mathcal Y`$ 是标准 Borel 空间；实际更新为 $`U_t(z,y,\xi_t)`$。
- 一个声明的*测试*集合 $`\mathcal A`$，同样是标准 Borel 空间。测试 $`a\in\mathcal A`$ 由一个声明的任务得分 $`F_a\in[0,1]`$ 加上第 $`t`$ 轮之后一个声明的 continuation 协议组成。不同测试可以有不同的 continuation 长度或不同的合法训练计划；剩余预算不同的测试必须标为不同的测试。
- 测试 $`a`$ 的 *response* 为 $`g_a(y)=\mathbb E[\text{走完测试 }a\text{ 的 continuation 后的 }F_a\mid z,\ \text{本轮消息 }y]\in[0,1]`$，即 [Definition 1.4](../README_ZH.md#s1-4) 中的单边界 response，只是用测试自己的 continuation。$`(a,y)\mapsto g_a(y)`$ 联合可测。
- *参考消息测度* $`\rho`$（在 $`\mathcal Y`$ 上）与*测试权重* $`\omega`$（在 $`\mathcal A`$ 上），都是概率测度，在选任何 feedback 之前声明。

<a id="s4-8"></a>

**Definition 4.8（检查点上的 response operator）.**

- 输入空间为 $`H=L_0^2(\rho)=\lbrace h:\mathbb E_\rho h=0,\ \mathbb E_\rho h^2<\infty\rbrace`$，内积 $`\langle h,h'\rangle_\rho=\mathbb E_\rho[hh']`$。中心化的 response 为 $`\tilde g_a=g_a-\mathbb E_\rho g_a`$。
- *response operator* $`\mathcal R_{t,z}:H\to L^2(\omega)`$ 定义为

```math
(\mathcal R_{t,z}h)(a)=\langle\tilde g_a,h\rangle_\rho . \qquad(4.10)
```

- 它的奇异值分解（由下面的事实保证存在）记为 $`\mathcal R_{t,z}h=\sum_{j\ge1}\sigma_j\langle v_j,h\rangle_\rho\,u_j`$，其中 $`\sigma_1\ge\sigma_2\ge\dots\ge0`$，*输入模式* $`v_j\in H`$ 与*输出模式* $`u_j\in L^2(\omega)`$ 各自正交归一。检查点固定时省略下标 $`(t,z)`$。

**随定义而来的事实.** 每条都有一行理由。

- *精确的得分差.* 若 $`P,Q`$ 的密度 $`p,q\in L^2(\rho)`$，则 $`h=p-q\in H`$，且对每个测试 $`a`$ 有 $`(\mathcal Rh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$。理由：$`\mathbb E_\rho h=0`$，所以 $`\langle\tilde g_a,h\rangle_\rho=\int g_a\,dP-\int g_a\,dQ`$。
- *线性从哪里来.* $`\mathcal R`$ 是线性的，因为期望对消息的分布是线性的。更新 $`U_t`$、optimizer state 以及之后的每一步都可以任意非线性；没有任何东西被线性化。

*Hilbert–Schmidt.* $`\mathcal R`$ 是核为 $`\tilde g_a(y)`$ 的积分算子，所以

```math
\lVert\mathcal R\rVert_{\rm HS}^2=\mathbb E_{a\sim\omega}\mathrm{Var}_{Y\sim\rho}\,g_a(Y)\le\tfrac14, \qquad(4.11)
```

因为取值于 $`[0,1]`$ 的变量方差至多为 $`\tfrac14`$。因此 $`\mathcal R`$ 是紧算子，奇异值分解存在（标准 Borel 空间上的 $`L^2`$ 空间可分），且 $`\sum_j\sigma_j^2\le\tfrac14`$。这里没有假设有限秩。

- *秩的上限.* 一个测试时秩至多为 1；$`p`$ 个测试时至多为 $`p`$；$`m`$ 个消息时至多为 $`m-1`$。因为声明的测试少而得到的低秩，不包含任何结构信息。
- *混合参考测度.* 对有限个候选分布 $`P^1,\dots,P^J`$，取 $`\rho=J^{-1}\sum_jP^j`$，则每个候选及其任意凸混合的密度都在 $`[0,J]`$ 内，任意两个这样的分布之差满足 $`\lVert h\rVert_\rho^2\le2J`$，因为 $`\mathbb E_\rho(p-q)^2\le\mathbb E_\rho(p^2+q^2)\le J(\mathbb E_\rho p+\mathbb E_\rho q)=2J`$。两个候选等权混合时为 $`\lVert h\rVert_\rho^2\le4`$。声明目录之外的消息、候选或支撑，需要新的 $`\rho`$，或者显式付出 coverage 代价。
- *算子依赖什么.* $`\rho`$、$`\omega`$、声明的测试、检查点（参数与 optimizer state）以及声明的 continuation 协议。改变 $`\omega`$ 会改变奇异值，必须报告。对参数做一个可逆重参数化，只要实际过程与测试不变，$`\mathcal R`$ 就不变：这个算子由可观测的后果定义。
- *多个检查点.* 不同检查点收到的 feedback 差 $`h_z`$ 一般不同。不能把不同检查点的 $`\mathcal R_{t,z}`$ 的行堆在一起，当作有一个公共输入 $`h`$ 同时作用在它们上面；正确的条件版本见 [Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13)。

**说了什么.**

- 在固定检查点上，把一个 feedback law 换成另一个、对每个声明测试造成的影响，是同一个线性映射作用在密度差上。
- 这个映射是 Hilbert–Schmidt 的，总平方增益至多为 $`\tfrac14`$，所以它的奇异值按造成的得分变化大小，给 feedback 差异的各个方向排序。

**直觉.** 这个算子问的是：两个 feedback law 可能在哪些方向上不同；在每个方向上，learner 消化本轮消息并走完声明的训练之后，每个声明测试会动多少。

**在框架中的作用.**

- 它的谱给出 §4 充分性计数在明确度量下的版本（[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)），它的动态对应物是 [Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10) 中的 Hankel 矩阵，它的估计、复用与成本是 [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 的主题。
- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12) 把它分解为经过下一步 predictive state 期望变化的复合。

**用到.**

- [Definitions 1.3–1.4](../README_ZH.md#s1-3)：complete learner state 与单边界 response.
- [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)：response 是正确的不变量.

**被用于.**

- [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)：谱充分性与线性接口复杂度.
- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)：经过 predictive state 的分解.
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)：在有限目录上估计谱.
- [Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13)：跨检查点共享消息特征.

## 相关工作

- 预测训练数据变化时输出如何变化，是 datamodels 研究的问题 ([Ilyas et al., 2022](#ref-ilyas2022datamodels))；response operator 是那个对象的一个按状态条件化、单轮、基于 continuation 的近亲，反事实矩阵本身并不新。
- 紧算子的奇异值分解与截断的最优性，在矩阵情形可追溯到 Eckart 与 Young ([Eckart & Young, 1936](#ref-eckart1936approximation))。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
