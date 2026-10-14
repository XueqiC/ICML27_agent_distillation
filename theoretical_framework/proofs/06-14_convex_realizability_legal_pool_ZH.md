# Proposition 6.14 · 有限合法消息池上的凸可实现性

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-14_convex_realizability_legal_pool_ZH.md) · [证明](06-14_convex_realizability_legal_pool_ZH.md) · [实验](../experiments/06-14_convex_realizability_legal_pool_ZH.md) · [English](06-14_convex_realizability_legal_pool.md)

**本页证明什么.** 全部五个部分。(a) 与 (c) 是有限点集的凸几何，(b) 是线性规划的顶点论证，(d)–(e) 是 Cauchy–Schwarz 不等式与在支撑上求和。

**本页用到的设定与记号.**

- **合法消息池.** 有限个已核验的合法消息 $`\lbrace y_1,\dots,y_m\rbrace`$。*合法混合*是概率向量 $`q\in\Delta_m`$（$`q_i\ge0`$，$`\sum_iq_i=1`$）。
- **保留的特征.** $`\phi(y)\in\mathbb R^r`$，$`\Phi=[\phi(y_1),\dots,\phi(y_m)]\in\mathbb R^{r\times m}`$；混合 $`q`$ 的特征均值为 $`\Phi q`$。
- **目标.** $`\mu_*\in\mathbb R^r`$。
- **参照测度.** 概率向量 $`\rho`$，每个 $`\rho_i>0`$；$`dQ/d\rho=(q_i/\rho_i)_i`$，$`\lVert f\rVert_\rho^2=\sum_i\rho_if_i^2`$；对消息池上的目标律 $`P`$，$`B=\lVert dP/d\rho-dQ/d\rho\rVert_\rho`$（[Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9)）。对 $`S\subseteq\lbrace1,\dots,m\rbrace`$，$`\rho(S)=\sum_{i\in S}\rho_i`$。

**陈述（[Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14)）.** 在上述设定下：(a) 存在满足 $`\Phi q=\mu_*`$ 的合法混合，当且仅当 $`\mu_*\in\mathrm{conv}\lbrace\phi(y_i)\rbrace`$；(b) 若存在，则存在至多 $`r+1`$ 个非零权重的解，且 $`\lbrace q\ge0:\Phi q=\mu_*,\mathbf 1^\top q=1\rbrace`$ 的每个顶点都有此性质；(c) 若 $`\mu_*\notin\mathrm{conv}\lbrace\phi(y_i)\rbrace`$，则对任意范数，$`\min_{q\in\Delta_m}\lVert\Phi q-\mu_*\rVert=\mathrm{dist}(\mu_*,\mathrm{conv}\lbrace\phi(y_i)\rbrace)>0`$，且可以取到；(d) 支撑为 $`S`$ 的每个混合都有 $`\lVert dQ/d\rho\rVert_\rho^2\ge1/\rho(S)`$ 与 $`B\ge\rho(S)^{-1/2}-\lVert dP/d\rho\rVert_\rho`$；(e) 密度上限 $`q_i\le\kappa\rho_i`$ 或 $`\chi^2`$ 约束 $`\sum_iq_i^2/\rho_i\le1+\beta^2`$ 使可行集保持紧凸，且在密度上限下 $`\rho(S)\ge1/\kappa`$。

**证明用到的结果.** 除设定中的定义外不需要其他结果。

**用到的经典工具.** 线性规划形式的 Carathéodory 定理（[Rockafellar, 1970](#ref-rockafellar1970convex)）；Cauchy–Schwarz 不等式。

**证明.**

**(a)** 有限个点的凸包，就是这些点所有凸组合构成的集合。合法混合恰好是一组凸组合权重，它的特征均值就是相应的组合 $`\Phi q=\sum_iq_i\phi(y_i)`$。所以能够达到的特征均值集合正是 $`\mathrm{conv}\lbrace\phi(y_i)\rbrace`$。

**(b)** 记 $`\mathcal F=\lbrace q\in\mathbb R^m:q\ge0,\ \Phi q=\mu_*,\ \mathbf 1^\top q=1\rbrace`$，按假设它非空。它是包含在单纯形中的多面体，因而是有界多胞形，而非空多胞形至少有一个顶点。设 $`q`$ 是顶点，$`S`$ 是它的支撑。假如列向量 $`a_i=(\phi(y_i),1)\in\mathbb R^{r+1}`$（$`i\in S`$）线性相关，即 $`\sum_{i\in S}w_ia_i=0`$，$`w\ne0`$。把 $`w`$ 在 $`S`$ 之外补零。则 $`\Phi w=0`$、$`\mathbf 1^\top w=0`$，并且对 $`\varepsilon=\min_{i\in S}q_i/\max_i\lvert w_i\rvert>0`$，$`q+\varepsilon w`$ 与 $`q-\varepsilon w`$ 都属于 $`\mathcal F`$。于是 $`q`$ 是 $`\mathcal F`$ 中两个不同点的中点，与它是顶点矛盾。所以 $`S`$ 所标记的列是 $`\mathbb R^{r+1}`$ 中线性无关的向量，$`\lvert S\rvert\le r+1`$。

**(c)** $`\mathrm{conv}\lbrace\phi(y_i)\rbrace`$ 是紧单纯形在连续线性映射 $`q\mapsto\Phi q`$ 下的像，因而是紧凸集。函数 $`x\mapsto\lVert x-\mu_*\rVert`$ 连续，所以在这个集合上取到最小值；由于 $`\mu_*`$ 在这个闭集之外，最小值为正。由 (a)，对 $`q\in\Delta_m`$ 取最小与对凸包取最小是一回事。

**(d)** 设 $`S`$ 是 $`q`$ 的支撑。由 Cauchy–Schwarz，

```math
1=\sum_{i\in S}q_i=\sum_{i\in S}\frac{q_i}{\sqrt{\rho_i}}\sqrt{\rho_i}\le\Big(\sum_{i\in S}\frac{q_i^2}{\rho_i}\Big)^{1/2}\Big(\sum_{i\in S}\rho_i\Big)^{1/2}=\Big\lVert\frac{dQ}{d\rho}\Big\rVert_\rho\,\rho(S)^{1/2}.
```

两边平方即得第一个不等式。$`L^2(\rho)`$ 中的三角不等式给出 $`B\ge\lVert dQ/d\rho\rVert_\rho-\lVert dP/d\rho\rVert_\rho\ge\rho(S)^{-1/2}-\lVert dP/d\rho\rVert_\rho`$。$`\rho`$ 均匀且 $`\lvert S\rvert=r+1`$ 时，$`\rho(S)=(r+1)/m`$。

**(e)** 集合 $`\lbrace q:q_i\le\kappa\rho_i\ \forall i\rbrace`$ 与 $`\lbrace q:\sum_iq_i^2/\rho_i\le1+\beta^2\rbrace`$ 都是闭凸集，它们与 (b) 中的紧凸可行集之交仍是紧凸集。线性约束保持线性规划；$`\chi^2`$ 约束是凸二次约束。在密度上限下，$`1=\sum_{i\in S}q_i\le\kappa\sum_{i\in S}\rho_i=\kappa\,\rho(S)`$；$`\rho`$ 均匀时这就是 $`\lvert S\rvert\ge m/\kappa`$。$`\square`$

**附注.**

- $`\chi^2`$ 约束直接给出 $`\lVert dQ/d\rho\rVert_\rho^2\le1+\beta^2`$，所以加上它之后，[Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9) 中的密度界经由三角不等式由 $`\beta`$ 与 $`\lVert dP/d\rho\rVert_\rho`$ 控制。
- (a) 的可行性由一个线性规划判定；不可行的证书是一个分离超平面 $`(w,w_0)`$，对每个 $`i`$ 都有 $`w^\top\phi(y_i)\le w_0<w^\top\mu_*`$（Farkas 引理）。

## 相关工作

- Carathéodory 定理以及附注中用到的分离定理都是经典凸分析的内容（[Rockafellar, 1970](#ref-rockafellar1970convex)）。
- 挑选能引导已知学习器的训练样本，是 machine teaching 的研究对象（[Zhu, 2015](#ref-zhu2015machine)）；这里的挑选是固定已核验消息池上的混合，并带有明确的密度代价。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-rockafellar1970convex"></a>

**R. Tyrrell Rockafellar (1970).** Convex Analysis. Princeton University Press, Princeton, NJ (Princeton Mathematical Series 28). DOI [10.1515/9781400873173](https://doi.org/10.1515/9781400873173).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
