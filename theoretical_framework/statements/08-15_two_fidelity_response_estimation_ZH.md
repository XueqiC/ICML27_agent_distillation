# Proposition 8.15 · 双保真 response 估计及其成本条件

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](08-15_two_fidelity_response_estimation_ZH.md) · [证明](../proofs/08-15_two_fidelity_response_estimation_ZH.md) · [实验](../experiments/08-15_two_fidelity_response_estimation_ZH.md) · [English](08-15_two_fidelity_response_estimation.md)

**一段话概括.** response 目录的每一格都需要完整续训，代价很高。完整结果的便宜预测器（例如短续训）不必无偏也能帮上忙：少量与便宜预测配对的完整运行可以估计它的偏差，校正后的估计量精确无偏。这是经典的控制变量与多保真思想，这里用在 response 上。命题给出方差、成本最优分配，以及节省计算的精确条件。在便宜运行成本为十分之一时做同口径比较：若每个便宜输出都计费，系数固定时便宜结果与完整结果的相关系数需要约 0.787，用嵌套多保真估计的最优系数则为 0.575；若每个配对样本中的便宜输出被复用，两个门槛分别为 0.766 与约 0.316。有两个结构性的限制：它减少的是一格*之内*的复现数，而不是格子数；真实模型上还没有测过短续训与完整续训的相关性。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 固定一格：一个 checkpoint、一个候选 feedback law、一个由 $`p`$ 个测试组成的向量，权重 $`\omega`$；$`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$。
- $`X\in[0,1]^p`$ 是一次*完整*续训的输出；目标是 $`f=\mathbb EX`$，即这一格的 response 向量。
- $`Y\in[0,1]^p`$ 是一个*便宜*估计器的输出，这个估计器在抽取下面的数据之前就已冻结：短续训、截断的代理，或使用 learner state 的预测器。$`Y`$ 可以有偏。
- *两组相互独立的样本.* 第 0 组有 $`n_0`$ 个独立同分布的便宜输出 $`Y_i`$。第 1 组有 $`n_1`$ 个独立同分布的配对 $`(X'_i,Y'_i)`$，$`X'_i`$ 与 $`X`$ 同分布，$`Y'_i`$ 与 $`Y`$ 同分布，二者是耦合的（例如短运行就是完整运行的前缀，或二者共享随机数）。
- *方差.* $`v_0=\mathbb E\lVert Y-\mathbb EY\rVert_\omega^2`$，$`v_1=\mathbb E\lVert(X-Y)-\mathbb E(X-Y)\rVert_\omega^2`$（在第 1 组的耦合下），$`v_H=\mathbb E\lVert X-\mathbb EX\rVert_\omega^2`$。
- *成本.* 每个便宜样本 $`c_0`$；每个配对样本 $`c_1`$，含其中的便宜部分（按复用计算之后的实际成本）；单独一个完整样本 $`c_H`$。

<a id="s8-15"></a>

**Proposition 8.15（双保真 response 估计及其成本条件）.** 在上述设定下令

```math
\hat f=\frac1{n_0}\sum_{i=1}^{n_0}Y_i+\frac1{n_1}\sum_{i=1}^{n_1}\big(X'_i-Y'_i\big).
```

**(a) 无偏，方差可加.** $`\mathbb E\hat f=f`$，且 $`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$。

**(b) 成本最优分配.** 忽略取整，在 $`\mathbb E\lVert\hat f-f\rVert_\omega^2\le\varepsilon^2`$ 的约束下，最小成本 $`n_0c_0+n_1c_1`$ 为

```math
C_{\rm MF}=\frac{S^2}{\varepsilon^2},\qquad S=\sqrt{v_0c_0}+\sqrt{v_1c_1},\qquad n_\ell=\frac S{\varepsilon^2}\sqrt{\frac{v_\ell}{c_\ell}} .
```

把每个 $`n_\ell`$ 上取整为 $`\max\lbrace1,\lceil n_\ell\rceil\rbrace`$，至多多花 $`c_0+c_1`$。

**(c) 何时省计算.** 与只用 $`n_H`$ 个完整样本相比，在相同均方误差、相同连续分配口径下，双保真估计更便宜当且仅当

```math
\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H} .
```

对标量测试，若 $`v_0=v_H`$ 且 $`X`$ 与 $`Y`$ 的相关系数为 $`\varrho`$，则 $`v_1=2v_H(1-\varrho)`$。取 $`c_0=c_H/10`$：若配对中的便宜部分被复用（$`c_1=c_H`$），条件为 $`\varrho>0.766`$；若不复用（$`c_1=1.1c_H`$），条件为 $`\varrho>0.787`$。

**(d) 最优系数的嵌套估计量.** 对标量测试，令 $`n_L\ge n_H`$ 个便宜输出中的前 $`n_H`$ 个与 $`n_H`$ 个完整输出配对，$`\hat f_\alpha=\bar X_{n_H}+\alpha(\bar Y_{n_L}-\bar Y_{n_H})`$，每个完整输出成本 $`c_H`$，每个便宜输出成本 $`c_L=c\,c_H`$。取 $`\alpha=\varrho\sqrt{v_H/v_0}`$ 与最优比例 $`n_L/n_H=\sqrt{\varrho^2/(c(1-\varrho^2))}`$（当 $`\varrho^2\ge c/(1+c)`$ 时可行），在相同均方误差下相对于只用完整样本的成本为

```math
\big(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c\big)^2 ,
```

它小于 1 当且仅当 $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$：$`c=0.1`$ 时为 0.575。这个成本模型对每个便宜输出都计费。若每个配对样本中的便宜输出被复用，成本为 $`c_Hn_H+c_L(n_L-n_H)`$，门槛更低，$`c=0.1`$ 时约为 $`\sqrt c\approx0.316`$（数值结果，[Check 8.22](../experiments/08-15_two_fidelity_response_estimation_ZH.md#c8-22)）；它应与同一复用模型下的固定系数门槛 0.766 比较，0.575 则与 0.787 比较。实际中最优 $`\alpha`$ 未知；用同一批样本估计它会引入 $`1/n_H`$ 量级的偏差，所以精确无偏的版本要用 pilot 冻结 $`\alpha`$。

**(e) 减的是复现，不是格子.** 每一格都至少需要一次完整续训（$`n_1\ge1`$，或 $`n_H\ge1`$），所以相对于每格 $`n_H`$ 次完整复现，节省至多 $`n_H`$ 倍。要减少需要完整续训的*格子数*，就得在格子之间共享校正项 $`\mathbb E(X-Y)`$，这是一种迁移假设，[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 并不提供。

**(f) 方差未知.** 对取值范围长度已知为 $`L`$ 的标量 $`Z`$，用 pilot 中 $`k`$ 个独立的对 $`(Z_{2i-1},Z_{2i})`$，$`\hat v=\frac1k\sum_i(Z_{2i}-Z_{2i-1})^2/2`$ 以至少 $`1-\delta_v`$ 的概率满足

```math
\mathrm{Var}\,Z\le\hat v+\frac{L^2}2\sqrt{\frac{\log(1/\delta_v)}{2k}} .
```

要认证 (c) 成立，需要它左边的上界、右边的下界以及联合覆盖；比较点估计并不能认证节省。pilot 的成本计入预算。

**(g) 不用方差信息的严格半径.** 对 $`N`$ 个冻结的标量 response，以至少 $`1-\delta`$ 的概率同时有

```math
\lvert\hat f-f\rvert\le\eta=\sqrt{\frac{\log(4N/\delta)}{2n_0}}+2\sqrt{\frac{\log(4N/\delta)}{2n_1}} .
```

如果用同一批样本挑选代理、系数或分配，再把它们当作已冻结，(a)、(f)、(g) 都不再成立。

**说了什么.**

- 一个有偏但相关的完整续训便宜估计，经过少量配对完整运行校正，就给出无偏的 response 估计，方差与成本最优分配都有显式形式。
- 它省计算的充要条件是成本加权的标准差满足 (c)：在成本比为十分之一时，需要很强的配对相关：同口径比较，每个便宜输出都计费时，系数固定为 0.787、系数最优为 0.575；每对中的便宜输出被复用时，分别为 0.766 与约 0.316。
- 它削减一格之内的复现，但每格至少一次完整续训；格子数不变。

**直觉.** 大部分抽样由便宜估计承担，完整运行只负责估计它错了多少。如果两者同涨同落，差的方差很小，几次完整运行就够；如果不是，校正项和完整结果本身一样嘈杂，什么也省不下。

**在框架中的作用.**

- 这是框架中唯一一个以降低 response 目录成本为目的的估计量，而且它的条件可以在投入预算之前检查。
- 用 (g) 的半径代替单保真的 $`\varepsilon_n`$，[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 的谱界照样成立；[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 的锚点误差 $`\epsilon_j`$ 也可以这样得到。
- 它的前提，即真实学习器上短续训与完整续训强相关，尚未检验：这是 [Remark 8.17](08-17_open_problems_ZH.md#s8-17) 的一个未解决问题。

**关于陈述的说明.**

- **低秩在这里帮不上忙.** 低秩描述的是各格之间的均值 response。如果对固定的消息，便宜预测是常数，它完全不会降低完整结果在运行之间的条件方差。
- **摊销.** 成本为 $`C_{\rm fit}`$ 的拟合预测器被用于 $`M_{\rm use}`$ 个决策时，每个决策分摊 $`C_{\rm fit}/M_{\rm use}`$；应报告实际使用次数，而不是假定无限摊销。
- **合成算例.** 在 $`v_H=v_0=0.25`$、$`v_1=0.0025`$（相关系数约 0.995）、$`c_0=0.01`$、$`c_1=1.01`$、$`c_H=1`$ 时，不计 pilot 成本的成本比为 0.0402（[Check 8.22](../experiments/08-15_two_fidelity_response_estimation_ZH.md#c8-22)）。这只是公式的演示，不是实测的加速。

**推导要点.**

- (a)：$`\mathbb EX=\mathbb EY+\mathbb E(X-Y)`$；两组均值独立，交叉项均值为零。
- (b)：Cauchy–Schwarz，$`(\sum_\ell c_\ell n_\ell)(\sum_\ell v_\ell/n_\ell)\ge(\sum_\ell\sqrt{v_\ell c_\ell})^2`$，在 $`n_\ell\propto\sqrt{v_\ell/c_\ell}`$ 时取等号。
- (c)：把 $`C_{\rm MF}`$ 与 $`v_Hc_H/\varepsilon^2`$ 比较；门槛由 $`\sqrt c+\sqrt{2(1-\varrho)c_1/c_H}=1`$ 解出。
- (d)：嵌套估计量的方差，对 $`\alpha`$ 求最小，再对 $`(n_H,n_L)`$ 用 Cauchy–Schwarz。
- (e)：$`C_{\rm MF}\ge c_1\ge c_H`$，对比 $`n_Hc_H`$。
- (f)、(g)：对配对差平方的一半与两组均值用 Hoeffding 界，再取 union bound。

**用到.**

- [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) — 要估计其格子的目录

**被用于.**

- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) — 锚点估计误差
- [Remark 8.17](08-17_open_problems_ZH.md#s8-17) — 真实学习器上短续训与完整续训的相关性

## 相关工作

- 多保真 Monte Carlo 的最优模型管理（[Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)）与多层 Monte Carlo（[Giles, 2015](#ref-giles2015multilevel)）给出了嵌套估计量、它的最优系数与分配，以及 (d) 的成本比。控制变量在模拟中是经典方法（[Glasserman, 2004](#ref-glasserman2004monte)）。这里特有的只是对象：持续训练目录中一格的 response，以短续训作为低保真模型。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).
