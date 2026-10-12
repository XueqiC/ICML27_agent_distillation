# §4 充分性的复杂度 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/04_complexity_of_sufficiency_ZH.md) · [Experiments](../experiments/04_complexity_of_sufficiency_ZH.md) · [English](04_complexity_of_sufficiency.md)

### §4 · 充分性的复杂度：精确陈述

[证明](../proofs/04_complexity_of_sufficiency_ZH.md) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md)

**本节做什么.**

- 两个 feedback law 可以在学习上互换，意思是它们在 response 类 $`\mathcal G`$ 上一致（[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)）。本节度量这种一致需要它们共享多少信息。
- 一般的回答是 $`\mathcal G`$ 的 span 维数或逼近宽度（[Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)）。
- 对凸仿射 learner 可以完全算出来：直到部署长度的联合矩，以及随独立块数变化的速率（[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4) 与 [4.5](../proofs/04_rates_and_composition_ZH.md#s4-5)）；近似复杂度可以远低于精确阶数（[Proposition 4.6](../proofs/04_rates_and_composition_ZH.md#s4-6)）。
- 非线性或复合的更新会改变所需的阶数，所以必须对 learner 本身建模（[Theorem 4.7](../proofs/04_rates_and_composition_ZH.md#s4-7)）。

**设定（Definitions [4.1](../proofs/04_complexity_of_sufficiency_ZH.md#s4-1) 与 [4.3](../proofs/04_complexity_of_sufficiency_ZH.md#s4-3)）.**

- $`\mathcal Y`$ 是含 $`m`$ 个元素的有限消息字母表；$`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是一个 response 类。特征接口传递 $`\phi_1,\dots,\phi_d`$ 的均值；$`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。
- $`\mathcal M_\phi`$ 是特征匹配的分布对集合，未分辨差距为 $`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$。精确维数 $`d_0`$ 与近似维数 $`d_\varepsilon`$ 分别是使差距为 $`0`$、不超过 $`\varepsilon`$ 所需的最少特征个数。
- §4 中可计算的模型：学生 $`p\in[0,1]^k`$ 在 context $`j`$ 以概率 $`p_j`$ 选动作 1；部署是一棵有限树，每条路径至多 $`N`$ 个决策，价值为 $`J_E(p)`$；经验块 $`Z_i\in[0,1]^k`$ 独立同分布；凸仿射 learner 为 $`p_R=w_0p_0+\sum_iw_iZ_i`$，权重事先固定。

#### Theorem 4.2 · 充分性对偶与迁移

**(a)** 特征均值相等能对所有分布推出 response 均值相等，当且仅当 $`\mathrm{span}\,\mathcal G\subseteq H_\phi`$；因此 $`d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$。

**(b)** 未分辨差距恰为最坏的最佳逼近误差的两倍：

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\inf_{h\in H_\phi}\lVert g-h\rVert_\infty .
```

**(c)** 若每个目标 response 都可写成 $`g=c+b^\top\phi+r`$，$`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则 $`\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$。

**(d)** 若 learner 族能在任意事件 $`A\subseteq\mathcal Y`$ 上部署两个 policy（价值 $`a`$ 与 $`a+\gamma`$）中较好的一个，则差距为 $`\gamma\,\mathrm{TV}(P,Q)`$，且 $`d_0=m-1`$。

*推导要点.*

- (a) 若 $`g\in H_\phi`$，它是常数加特征的组合，两部分的均值都相等。若 $`g\notin H_\phi`$，它与 $`H_\phi`$ 正交的分量 $`v`$ 给出一个方向：沿它倾斜均匀分布，所有特征均值都不动，而 $`(P-Q)g`$ 按 $`\lVert v\rVert_2^2`$ 的比例变化。
- (b) Hölder 不等式对每个 $`h\in H_\phi`$ 给出 $`\lvert(P-Q)(g-h)\rvert\le2\lVert g-h\rVert_\infty`$。可达性：LP 对偶把上确界范数距离写成 $`\sigma^\top g`$ 在 $`\sigma\perp H_\phi`$、$`\lVert\sigma\rVert_1\le1`$ 上的最大值，而每个这样的 $`2\sigma`$ 都是两个分布之差。
- (c) 常数项抵消，Hölder 把特征项界为 $`B`$ 乘以特征均值之差，残差至多贡献 $`2\tau`$。
- (d) response $`a+\gamma\mathbf 1_A`$ 实现了全变差的事件形式，单点事件张成 $`\mathbb R^m`$。

*例.* 取 $`\mathcal Y=\lbrace1,2,3\rbrace`$、单个特征 $`\phi(y)=y`$。分布 $`P=(\tfrac12,0,\tfrac12)`$ 与 $`Q=(0,1,0)`$ 均值相同。response $`g(y)=y^2`$ 把它们分开 1：$`\mathbb E_Pg=5`$、$`\mathbb E_Qg=4`$。特征 $`\phi`$ 不充分，因为在三个点上 $`y^2\notin\mathrm{span}\lbrace1,y\rbrace`$。

*用到* [Definition 1.4](../README_ZH.md#s1-4)、[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)。*被用于* [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)、[Proposition 4.6](../proofs/04_rates_and_composition_ZH.md#s4-6)、[Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Theorem 8.1](../proofs/08_verification_ZH.md#s8-1)。

[证明](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md#c4-1) · 已证明

#### Theorem 4.4 · 仿射 learner：矩阶充分且必要

**(a)** $`J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha`$ 的表示唯一，且 $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$。

**(b)** 对所有 $`\lvert\alpha\rvert\le N`$ 有 $`\mathbb E_PZ^\alpha=\mathbb E_QZ^\alpha`$，当且仅当对每个至多 $`N`$ 次决策的部署与每个凸仿射 learner 都有 $`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$。

**(c)** 每一阶都必要。$`\lbrace0,\tfrac1N,\dots,1\rbrace`$ 上的标量分布 $`P_N,Q_N`$（二项式权重分别放在偶数点与奇数点上）在低于 $`N`$ 阶时一致，在 $`N`$ 阶相差 $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$；$`\lbrace0,1\rbrace^N`$ 上的偶、奇 parity 分布在低于 $`N`$ 阶时一致，在 parity 任务上的价值分别为 $`1`$ 与 $`0`$。

**(d)** 在网格 $`\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上恰好需要 $`\binom{k+N}N-1`$ 个统计量。在 $`\lbrace0,1\rbrace^k`$ 上，若 $`N\ge k`$ 且奖励任意，少于 $`2^k-1`$ 个统计量会留下大小为 $`1`$ 的差距。

*推导要点.*

- (a) 在部署树上反向归纳：决策节点乘以 $`p_j`$ 或 $`1-p_j`$，次数加一，系数范数至多乘 3；机会节点取平均。
- (b) 用分配律展开 $`p_R^\alpha`$，得到不超过 $`\lvert\alpha\rvert`$ 阶块矩之积的概率加权和（块独立使期望可以分解）。反过来，任务"访问 context $`j`$ 恰好 $`\alpha_j`$ 次，全部动作为 1 时回报 1"的价值为 $`p^\alpha`$，所以一步 learner $`p=Z`$ 能读出每一个矩。
- (c) $`N`$ 阶有限差分把低于 $`N`$ 次的多项式化为零，作用在 $`x^N`$ 上给出 $`N!`$；parity 分布在每个真坐标子集上都是均匀的。
- (d) 次数不超过 $`N`$ 的单项式在网格上线性无关，由 Theorem 4.2(a) 得到计数；在二元块上每张奖励表都是 response，任何维数低于 $`2^k`$ 的接口都有一个非零的零化方向，它分成两个支撑不相交的分布。

*例.* 两个坐标，$`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$、$`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$：两者的边缘分布都是 $`\mathrm{Bernoulli}(\tfrac12)`$，但两次决策的任务"两个动作都为 1 时回报 1"让训练出的学生相差 $`\tfrac12`$。所需的信息不是边缘分布，而是直到 $`N=2`$ 阶的联合矩。

*用到* [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)。*被用于* [Theorem 4.5](../proofs/04_rates_and_composition_ZH.md#s4-5)、[Proposition 4.6](../proofs/04_rates_and_composition_ZH.md#s4-6)、[Theorem 4.7](../proofs/04_rates_and_composition_ZH.md#s4-7)。

[证明](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md#c4-2) · 已证明

#### Theorem 4.5 · 与轮数无关的界与平均化速率

**(a)** 对每个凸仿射 learner，$`\lvert\mathbb E_{P^R}J(p_R)-\mathbb E_{Q^R}J(p_R)\rvert\le C_J\,\delta_N(P,Q)\le N3^N\delta_N(P,Q)`$，其中 $`\delta_N`$ 是直到 $`N`$ 阶的最大矩差，$`C_J=\sum_\alpha\lvert c_\alpha\rvert\lvert\alpha\rvert`$。这个界与 $`R`$ 无关。

**(b)** 对 $`n`$ 个独立块的平均，若均值相等、低于 $`r`$ 阶的联合 cumulant 都匹配，

```math
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)=\frac{\langle D^rJ(\mu),\kappa_r(P)-\kappa_r(Q)\rangle}{r!\,n^{r-1}}+O(n^{-r}).
```

**(c)** 对标量见证，这个展开是精确的：对每个 $`n\ge1`$，差距为 $`\Delta_N/n^{N-1}`$。

**(d)** 均值不同时，差距以 $`O(n^{-1})`$ 的速率趋于 $`J(\mu_P)-J(\mu_Q)`$。

**(e)** 把一个块重放 $`n`$ 次，或者用它的 $`n`$ 个确定性派生 $`T(Z_1)`$ 训练，对每个 $`n`$ 差距都是 $`\mathbb E_PJ(T(Z))-\mathbb E_QJ(T(Z))`$。

*推导要点.*

- (a) 在块矩展开中，对每个 $`R`$ 系数都构成一个概率向量；至多 $`\lvert\alpha\rvert`$ 个 $`[0,1]`$ 中块矩之积的变化，不超过各块矩变化之和（乘积的 telescoping）。
- (b) 在公共均值处对多项式 $`J`$ 作 Taylor 展开，并用 cumulant 写出 $`\bar p_n`$ 的中心矩。一个有 $`q`$ 个块、总大小为 $`s`$ 的划分按 $`n^{q-s}`$ 缩放，阶数为 $`n^{1-r}`$ 的不同项只有大小为 $`r`$ 的单块。
- (c) 对 $`J=p^N`$ 不存在其他不同的划分。(d) 块大小都至少为 2 的划分是 $`O(n^{-1})`$。(e) 相同副本的平均就是副本本身。

*例.* 两个数据源均值与方差相同、偏度不同（$`r=3`$）。只要部署价值在均值处的三阶导数与偏度之差的缩并不为零，对 $`n`$ 个独立块取平均时，差距就按 $`1/n^2`$ 缩小；而在同一个块上训练十个 epoch，单块时的差距原封不动。

*用到* [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)。*被用于* [Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/04_rates_and_composition_ZH.md#s4-5) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md#c4-6) · 已证明

#### Proposition 4.6 · 精确阶不等于近似复杂度

**(a)** 把 $`p^N`$ 的 Chebyshev 展开截断到 $`q<N`$ 阶，上确界误差恰为 $`\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$，$`B\sim\mathrm{Bin}(2N,\tfrac12)`$。所以矩匹配到 $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ 阶的分布（或凸仿射 learner 的学习价值）在 $`p^N`$ 上至多相差 $`\varepsilon`$。

**(b)** 对 $`[0,1]^N`$ 上的 parity 价值 $`J_{\rm even}`$，每个 $`q<N`$ 都有 $`\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_\infty=\tfrac12`$。

*推导要点.*

- (a) 令 $`2p-1=\cos\theta`$，$`p^N=\cos^{2N}(\theta/2)`$ 展开成以二项式系数为系数的 Chebyshev 多项式。非负系数的尾部就是上确界误差（在 $`p=1`$ 处取到），它是一个二项分布的尾概率，用 Hoeffding 界住。
- (b) 偶、奇 parity 分布在每个次数低于 $`N`$ 的多项式上都一致，在 $`J_{\rm even}`$ 上却相差 1；由 Theorem 4.2(b)，逼近误差至少为 $`\tfrac12`$。

*例.* $`N=100`$、$`\varepsilon=0.01`$ 时，大约 $`\sqrt{100\log400}\approx25`$ 个匹配的矩就已经能分辨 $`p^{100}`$；而 100 个 context 上的 parity 需要全部 100 阶。

*用到* [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)、[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)。*被用于* [Theorem 8.1](../proofs/08_verification_ZH.md#s8-1)。

[证明](../proofs/04_rates_and_composition_ZH.md#s4-6) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md#c4-9) · 已证明

#### Theorem 4.7 · 更新复合放大信息需求

**(a)** 若更新 $`p_i=U_i(p_{i-1},Z_i)`$ 关于 $`p_{i-1}`$ 是至多 $`a_i`$ 次、关于 $`Z_i`$ 是至多 $`b_i`$ 次的多项式，则 $`J(p_R)`$ 关于块 $`Z_i`$ 的次数至多为 $`D_i=Nb_i\prod_{t>i}a_t`$，把每个块的矩匹配到 $`D_i`$ 阶就够了。

**(b)** 这个界可以达到：逐坐标平方的更新把 $`Z_1`$ 变成 $`Z_1^{2^{R-1}}`$。

**(c)** 阈值 learner $`p=\mathbf 1\lbrace Z>\tfrac12\rbrace`$ 在 $`N=1`$ 时就能分开两个均值相同的分布。

**(d)** 对光滑 SGD 的一步 $`\theta\mapsto\theta-\eta G`$，

```math
(K^P-K^Q)V=-\eta\nabla V^\top(\mu_P-\mu_Q)+\tfrac{\eta^2}{2}\langle\nabla^2V,M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\tfrac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big).
```

**(e)** 满足 $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ 的逐步误差累积为 $`C\eta^2\sum_{s<R}(1+\eta L)^s`$；只有当 $`\eta LR=O(1)`$ 或误差映射是压缩的，它才关于 $`R`$ 线性。

*推导要点.*

- (a) 把 $`J`$ 沿更新反向拉回：把一个 $`(a_t,b_t)`$ 次映射代入 $`d_t`$ 次多项式，关于 $`p`$ 的次数乘以 $`a_t`$，并产生关于 $`Z_t`$ 的 $`b_td_t`$ 次项；再由独立性把期望分解成各块的矩。
- (b) 先取 $`U_1(p,z)=z`$，之后逐坐标平方，配合 Theorem 4.4(c) 中 $`2^{R-1}`$ 阶的标量见证。
- (c) 直接计算：$`P(0)=\tfrac13`$、$`P(\tfrac34)=\tfrac23`$ 对比 $`Q(\tfrac12)=1`$。
- (d) 对 $`V(\theta-\eta g)`$ 作带积分余项的三阶 Taylor 展开，再取期望。
- (e) 归纳加几何级数；递推 $`e_{t+1}=\tfrac32e_t+\tfrac14`$ 说明线性化的简化可以失效。

*例.* 把同样两个块分布分别喂给平均化 learner，和每读一个块就把参数平方一次的 learner。前者需要直到部署长度的矩；后者需要直到"部署长度乘 $`2^{R-1}`$"的矩。信息需求是 learner 的性质，而不只是数据的性质。

*用到* [Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)、[Definition 1.4](../README_ZH.md#s1-4)。*被用于* [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3)。

[证明](../proofs/04_rates_and_composition_ZH.md#s4-7) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md#c4-11) · 已证明
