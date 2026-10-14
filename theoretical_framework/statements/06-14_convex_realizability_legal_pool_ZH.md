# Proposition 6.14 · 有限合法消息池上的凸可实现性

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](06-14_convex_realizability_legal_pool_ZH.md) · [证明](../proofs/06-14_convex_realizability_legal_pool_ZH.md) · [实验](../experiments/06-14_convex_realizability_legal_pool_ZH.md) · [English](06-14_convex_realizability_legal_pool.md)

**一段话概括.** 知道一个 feedback law 应当有哪些特征均值，不等于能够发出这样的 law。在 agent 蒸馏里，蒸馏器只能拿已经生成并核验过的消息来训练，所以它真正能发出的，是一个有限合法消息池的正权重混合。这个命题说明这样的混合何时能匹配目标特征均值：恰好当目标落在消息池特征向量的凸包里，而且这时至多需要 $`r+1`$ 条消息。它也说明这种稀疏混合的代价：权重集中在少数消息上，会放大控制 [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 谱误差的那个密度界。

**本页用到的设定与记号.**

- **合法消息池.** 有限个已核验的合法消息 $`\lbrace y_1,\dots,y_m\rbrace`$（训练配方可以使用的轨迹、标签或 batch）。*合法混合*是概率向量 $`q\in\Delta_m`$，即 $`q_i\ge0`$、$`\sum_iq_i=1`$；它以概率 $`q_i`$ 发出 $`y_i`$。负权重与未经核验的生成消息都不合法。
- **保留的特征.** 固定的特征映射 $`\phi(y)=(\phi_1(y),\dots,\phi_r(y))\in\mathbb R^r`$，例如 [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 的前几个消息模式，或 [Definition 4.1](04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1) 中任意冻结的接口。记 $`\Phi=[\phi(y_1),\dots,\phi(y_m)]\in\mathbb R^{r\times m}`$，于是混合 $`q`$ 的特征均值为 $`\Phi q`$。
- **目标.** 编译出的反馈应当具有的特征均值向量 $`\mu_*\in\mathbb R^r`$，例如被审计状态上 oracle 律的特征均值，或由 [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16) 选出的点。
- **参照测度与密度.** 消息池上的概率向量 $`\rho`$，每个 $`\rho_i>0`$（例如 [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的混合测度）。混合 $`q`$ 的密度为 $`dQ/d\rho=(q_i/\rho_i)_i`$，$`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$。对消息池上的目标律 $`P`$，$`B=\lVert dP/d\rho-dQ/d\rho\rVert_\rho`$ 就是 [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 中乘在谱尾部前面的密度界。
- 对 $`S\subseteq\lbrace1,\dots,m\rbrace`$，$`\rho(S)=\sum_{i\in S}\rho_i`$；$`q`$ 的支撑是 $`\lbrace i:q_i>0\rbrace`$。

<a id="s6-14"></a>

**Proposition 6.14（有限合法消息池上的凸可实现性）.** 在上述设定下：

**(a) 存在性.** 存在满足 $`\Phi q=\mu_*`$ 的合法混合，当且仅当 $`\mu_*\in\mathrm{conv}\lbrace\phi(y_1),\dots,\phi(y_m)\rbrace`$。

**(b) 至多 $`r+1`$ 条消息.** 若存在，则存在至多 $`r+1`$ 个非零权重的解。多面体 $`\lbrace q\ge0:\Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$ 的每个顶点（例如单纯形法返回的基本解）都有这个性质。

**(c) 凸包之外.** 若 $`\mu_*\notin\mathrm{conv}\lbrace\phi(y_i)\rbrace`$，则对 $`\mathbb R^r`$ 上任意范数 $`\lVert\cdot\rVert`$，

```math
\min_{q\in\Delta_m}\lVert\Phi q-\mu_*\rVert=\mathrm{dist}\big(\mu_*,\mathrm{conv}\lbrace\phi(y_i)\rbrace\big)>0 ,
```

并且最小值可以取到。这个消息池的任何合法混合都做不得更好；只有更大的消息池、新的证据或不同的训练接口才能降低这个误差。

**(d) 稀疏混合的密度代价.** 支撑为 $`S`$ 的每个混合 $`q`$ 都满足

```math
\Big\lVert\frac{dQ}{d\rho}\Big\rVert_\rho^2=\sum_i\frac{q_i^2}{\rho_i}\ \ge\ \frac1{\rho(S)},\qquad\text{hence}\qquad B=\Big\lVert\frac{dP}{d\rho}-\frac{dQ}{d\rho}\Big\rVert_\rho\ \ge\ \rho(S)^{-1/2}-\Big\lVert\frac{dP}{d\rho}\Big\rVert_\rho .
```

当 $`\rho`$ 均匀且 $`\lvert S\rvert=r+1`$ 时，右边为 $`\sqrt{m/(r+1)}-\lVert dP/d\rho\rVert_\rho`$。

**(e) 带约束的编译.** 加上密度上限 $`q_i\le\kappa\rho_i`$ 或 $`\chi^2`$ 约束 $`\sum_iq_i^2/\rho_i\le1+\beta^2`$ 之后，可行集仍是紧凸集，编译仍是线性规划或凸二次规划。在密度上限下，每个可行混合都有 $`\rho(S)\ge1/\kappa`$；$`\rho`$ 均匀时至少要用 $`m/\kappa`$ 条消息，只要 $`\kappa<m/(r+1)`$ 就多于 $`r+1`$。

**说了什么.**

- 把目标特征均值向量编译成合法反馈，是消息池上的一个凸可行性问题。它可解，恰好当目标落在消息池特征向量的凸包中，这时 $`r+1`$ 条消息就够。
- 在凸包之外，误差有一个精确的下限，即到凸包的距离；对同一个消息池重新加权去不掉它。
- 稀疏解并非没有代价：只用少数消息的混合，相对参照测度的密度很大。不检查密度，就不能同时宣称"只需 $`r+1`$ 条消息"和"[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 的谱误差界不变"。

**直觉.**

- 正权重混合只能对消息池里已有的东西取平均，所以它的特征均值被限制在消息池的凸包里。
- 把混合集中在少数消息上，就像用很大的 importance weight：被匹配的特征是精确的，但每个没被匹配的特征，现在都是在一个远离参照测度的分布下度量的。

**在框架中的作用.**

- 当蒸馏器只能发出已核验消息池的混合时，它代替"可以生成 oracle 律"这条假设（[Theorem 6.5](06-05_achievability_ZH.md#s6-5) 的 (R)）。后者只保留为一个条件性的特例。
- 它为 [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16) 提供合法集合，也为 [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 提供编译一步。
- 它保证的是特征均值被匹配，不保证消息在文本上正确，也不保证学习结果无损。让误差账目保持诚实的三种做法：
  - 加上 (e) 中的密度上限或 $`\chi^2`$ 约束，接受更大的支撑；
  - 用 [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) 的一致范数界，它不需要密度条件：若每个 response 都能被特征一致逼近到 $`\varepsilon`$，特征均值相同的两个分布之差不超过 $`2\varepsilon`$；
  - 报告编译出的分布实际的 $`B`$ 与逐测试的残差。
- 若目标落在凸包之外，就报告不可实现，或把 (c) 中的距离作为特征失配计入误差。负权重或未经核验的生成消息不能作为可达性的证明。

**推导要点.**

- (a) 就是有限点集凸包的定义。
- (b) 是线性规划形式的 Carathéodory 定理：在顶点处，被用到的列 $`(\phi(y_i),1)\in\mathbb R^{r+1}`$ 线性无关，所以至多有 $`r+1`$ 列。
- (c) 用到凸包的紧性；(d) 是在支撑上用 Cauchy–Schwarz；(e) 把单纯形与凸约束求交，再在支撑上对密度上限求和。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8)：参照测度。
- [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)：谱误差中的密度界 $`B`$。
- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)：一致范数的替代方案。

**被用于.**

- [Theorem 6.5](06-05_achievability_ZH.md#s6-5)：可实现性假设的适用范围。
- [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)：合法集合。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：候选 feedback law 的编译。

**状态.** 已证明。

## 相关工作

- (b) 是 $`\mathbb R^r`$ 中凸包的 Carathéodory 定理（[Rockafellar, 1970](#ref-rockafellar1970convex)），这里用的是它的线性规划形式。
- 为一个已知学习器挑选能引导它的小训练集，是 machine teaching 的研究内容（[Zhu, 2015](#ref-zhu2015machine)）。这里的候选被限制在一个固定的已核验消息池里，目标是特征均值向量，于是问题变成带有明确密度代价的凸可行性问题。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-rockafellar1970convex"></a>

**R. Tyrrell Rockafellar (1970).** Convex Analysis. Princeton University Press, Princeton, NJ (Princeton Mathematical Series 28). DOI [10.1515/9781400873173](https://doi.org/10.1515/9781400873173).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
