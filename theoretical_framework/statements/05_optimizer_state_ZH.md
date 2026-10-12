# §5 Optimizer state 与一阶表示 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/05_optimizer_state_ZH.md) · [Experiments](../experiments/05_optimizer_state_ZH.md) · [English](05_optimizer_state.md)

### §5 · Optimizer state 与一阶表示：精确陈述

[证明](../proofs/05_optimizer_state_ZH.md) · [实验](../experiments/05_optimizer_state_ZH.md)

**本节做什么.**

- 把每个带状态的一阶 optimizer 写成完整 learner state 上的增量映射，并按增量的各阶矩展开一个 feedback law 的效应（Proposition 5.1，工具）。
- 对 Adam 算出一阶统计量如何依赖 optimizer state：cold 状态下是符号分布，warm 状态下是 Adam 增量的精确均值，以及冻结分母近似和它的误差（Proposition 5.3–5.7）。
- 给出 $`\beta_2`$ 固定时 warmth 的精确演化（Proposition 5.8），以及任意激活率下裁剪的精确统计量（Proposition 5.9）。
- 列出各种常见学习器的一阶统计量（Theorem 5.10）。

**设定.** 完整 learner state 为 $`z=(\theta,o)`$，$`o`$ 是 optimizer state。optimizer 是 consumed gradient $`g`$ 的增量映射 $`z^+=z+\Delta_t(z,g)`$，包括裁剪。对 feedback law $`P`$，$`\mu^P_t(z)=\mathbb E_P\Delta_t(z,G)`$ 是平均增量，$`m^P_{k,t}`$ 是增量的 $`k`$ 阶绝对矩。$`V^P_{t+1}`$ 是 continuation value；对声明族 $`\mathcal V`$ 记 $`U=\mathrm{span}\lbrace\nabla V(z):V\in\mathcal V\rbrace`$；条件 $`(\mathrm S_k)`$ 要求 continuation value 在可达线段上属于 $`C^{k+1}`$，且 $`\lvert\nabla^{k+1}V[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}`$。对计数器为 $`t`$ 的 Adam，坐标 $`i`$ 的 warmth 为 $`\kappa_i=(1-\beta_2)G_i^2/(\beta_2v_i)`$，保留的分母为 $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^t)}`$，$`u_i=\lvert n_i\rvert/(a_i+\epsilon)`$（$`n`$ 是 bias-corrected momentum）；frozen-denominator surrogate 把分母换成 $`a_i+\epsilon`$，即带固定 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$ 的一步 momentum SGD（Definition 5.4）。

#### [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1) · optimizer 变换后的展开

- **(a)** 在 $`(\mathrm S_k)`$ 下，$`(K_t^P-K_t^Q)V(z)=\sum_{j=1}^{k}\tfrac1{j!}\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle+r_k`$，$`\lvert r_k\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$。
- **(b)** 在整个训练过程上，记 $`d_t^Q`$ 为 $`Q`$ 下的状态分布，差距如下式；由此得到对偶范数形式的证书，以及多出一个离开概率 $`\rho`$ 的停止版本。

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big\langle\nabla V^P_{t+1}(Z_t),\mu_t^P(Z_t)-\mu_t^Q(Z_t)\big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big).
```

- **(c)** 匹配 $`\Pi_U\mu`$ 一阶充分；只要某个 $`V\in\mathcal V`$ 的一阶项超过 $`\tfrac{M_2}2(m_2^P+m_2^Q)`$，它就能察觉失配；一阶统计量的最少个数为 $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})-1\le\min\lbrace\dim U,m-1\rbrace`$。
- **(d)** 若 $`j\le k`$ 的 $`\mathbb E\Delta^{\otimes j}`$ 都相同，每轮差距至多 $`\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$，对 SGD 为 $`O(\eta^{k+1})`$。

*推导要点.*

- 沿 $`z`$ 到 $`z+\Delta`$ 的线段对 $`V(z+\Delta)`$ 做带 Lagrange 余项的 Taylor 展开，在每个分布下积分；常数抵消，期望可以穿过多重线性项。
- 对整个训练过程，把这个展开代入 Theorem 3.2 telescoping 恒等式的每一项。
- 充分性与必要性来自 $`\nabla V(z)\in U`$；最小性是 Theorem 4.2 的充分性对偶用于线性化的 response 类。

*用到:* [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)。*被用于:* Theorem 5.10、[Definition 6.2](../proofs/06_learner_relative_information_ZH.md#s6-2)、[Theorem 7.1](../proofs/07_second_moment_channel_ZH.md#s7-1)、[Proposition 8.4](../proofs/08_certificate_ZH.md#s8-4)、[Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)。

[证明](../proofs/05_optimizer_state_ZH.md#s5-1) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Remark 5.2](../proofs/05_optimizer_state_ZH.md#s5-2) · 与 stochastic modified equations、TracIn、influence function 和 metagradient 的关系

- Proposition 5.1 与 stochastic modified equations、TracIn、influence function、metagradient 以及 In-Run Data Shapley 属于同一类计算；我们不主张它的新颖性。
- 它显式写出的是完整状态这个对象（包括 $`m`$ 与 $`v`$），以及余项的分块结构：参数块与 momentum 块在 $`\eta`$ 的二阶上起作用，而 Adam 的二阶矩块在 $`\eta`$ 的一阶上起作用，相对大小与 warmth 同阶。

[证明](../proofs/05_optimizer_state_ZH.md#s5-2) · remark

#### [Proposition 5.3](../proofs/05_optimizer_state_ZH.md#s5-3) · Cold-start Adam：符号商与 optimizer 记忆

从 cold 状态出发：

- **(a)** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$，其中 $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$，$`h_0=\mathrm{sign}`$。
- **(b)** $`\epsilon=0`$ 时，对所有有界即时参数效用相等，等价于联合符号分布相等；有 $`s`$ 个可达符号模式时，$`s-1`$ 个统计量既必要又充分；逐坐标的符号边缘分布不够。
- **(c)** $`\epsilon>0`$ 时，$`h_\epsilon`$ 是到 $`(-1,1)`$ 的双射，所以需要完整的梯度分布（$`m`$ 点字母表上 $`m-1`$ 个统计量）。
- **(d)** 对 Lipschitz 效用，把 $`h_\epsilon`$ 换成符号映射的代价至多为 $`\eta\sum_iL_i[\beta_{P,i}(a)+\beta_{Q,i}(a)+2\epsilon/(a+\epsilon)]`$，$`\beta_{P,i}(a)`$ 是梯度落在 $`0<\lvert G_i\rvert<a`$ 的质量。
- **(e)** 第一个状态保留 $`m_1=(1-\beta_1)G`$；历史 $`(1,-\tfrac34)`$ 与 $`(2,-\tfrac34)`$ 的第一步和符号都相同，第二步却方向相反。$`\beta_1=0`$ 时不存在这样的一对。

*推导要点.*

- cold 状态下 bias-corrected 矩为 $`G`$ 与 $`G^{\odot2}`$，所以这一步是 $`G/(\lvert G\rvert+\epsilon)`$。
- 第一步是符号向量（$`\epsilon=0`$）或 $`G`$（$`\epsilon>0`$）的单射函数；统计量个数是在纤维上取常值的函数空间的维数，由 Theorem 4.2 得到。
- margin 界用到 $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\epsilon/(\lvert g_i\rvert+\epsilon)`$。
- 记忆的反例是精确的两步计算；momentum 项 $`\beta_1(1-\beta_1)G_1`$ 翻转了 $`m_2`$ 的符号。

*例示.* 两个坐标的梯度以相同概率取 $`(1,1)`$ 或 $`(-1,-1)`$，对比以相同概率取 $`(1,-1)`$ 或 $`(-1,1)`$：每个坐标的符号边缘分布相同，平均步长都是 $`0`$；但前者两个参数总是一起动，后者从不一起动。

*用到:* [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)。*被用于:* Proposition 5.7、Theorem 5.10。

[证明](../proofs/05_optimizer_state_ZH.md#s5-3) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Proposition 5.5](../proofs/05_warm_adam_ZH.md#s5-5) · Adam 的精确一阶统计量与冻结分母误差

- **(a)** Adam 的一阶统计量是精确的平均增量 $`\mu=\mathbb E\,\Delta_{\rm Adam}`$。它的参数块是梯度分布的非线性泛函：$`\mathbb EG`$ 与 $`\mathbb E\,G^{\odot2}`$ 都相同的两个分布，$`\mu_\theta`$ 可以不同。
- **(b)** 对每个坐标，$`\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\,u_i\min\lbrace\kappa_i/2,1\rbrace`$。
- **(c)** 冻结分母下的平均增量由 $`(\mathbb EG,\mathbb E\,G^{\odot2})`$ 精确决定。所以"warm Adam 只需要这两个矩"依赖两个不同的近似：Proposition 5.1 的一阶展开，以及冻结分母，后者的误差就是 (b)。

*推导要点.*

- 由 $`v^+=\beta_2v(1+\kappa)`$，真实分母为 $`a\sqrt{1+\kappa}`$；比较 $`1/(a\sqrt{1+\kappa}+\epsilon)`$ 与 $`1/(a+\epsilon)`$ 得到因子 $`1-(1+\kappa)^{-1/2}\le\min\lbrace\kappa/2,1\rbrace`$。
- Adam 的一步是每个梯度坐标的有界非常数函数，因此不是二次多项式；矩映射在四个点上的核向量给出前两阶矩相同、平均步长不同的两个分布。

*用到:* Proposition 5.1。*被用于:* Proposition 5.6、5.8，Theorem 5.10，[Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)。

[证明](../proofs/05_warm_adam_ZH.md#s5-5) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Proposition 5.6](../proofs/05_warm_adam_ZH.md#s5-6) · warm Adam 关于 consumed gradient 一阶仿射

warm 状态下，在参数上的 Lipschitz 条件 (L)、二阶矩敏感度条件 $`(\mathrm S_v)`$，以及 $`F(g)=V(\tilde z^+(g))`$ 的曲率界 (H) 之下，

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i\min\lbrace\tfrac{\kappa_i}2,1\rbrace+L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace+\tfrac12H_G\lVert G\rVert_\infty^2=:r(y),
```

其中 $`b=(1-\beta_1)[-\eta\Phi\nabla_\theta V/(1-\beta_1^t)+\nabla_mV]`$。平均 consumed gradient 相同的分布，response 之差至多为 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。

*推导要点.*

- 把 response 拆成三个差：真实与冻结的参数步（由 Proposition 5.5(b) 控制），真实与冻结的二阶矩（由 $`(\mathrm S_v)`$ 控制），以及 $`F`$ 的 Taylor 余项（由 (H) 控制）。

*用到:* Proposition 5.5。*被用于:* Proposition 5.7、5.8，Theorem 5.10。

[证明](../proofs/05_warm_adam_ZH.md#s5-6) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Proposition 5.7](../proofs/05_warm_adam_ZH.md#s5-7) · cold 步在一阶就分开均值相同的反馈

在 cold 状态、符号映射与仿射效用 $`V(\theta)=\tfrac12+w\theta`$ 下：

- **(a)** $`P_0(G=1)=\tfrac14`$、$`P_0(G=-\tfrac13)=\tfrac34`$ 与 $`P_1=\delta_0`$ 均值相同，response 相差 $`\lvert w\rvert\eta/2`$。
- **(b)** 对每个 $`N`$，所有低于 $`N`$ 阶的矩都相同的两个梯度分布，response 相差 $`\lvert w\rvert\eta\,2^{2-N}`$。

*推导要点.* cold 步是 $`-\eta\,\mathrm{sign}(G)`$，所以 response 关于 $`\mathbb E\,\mathrm{sign}(G)`$ 仿射；Theorem 4.4 的二项奇偶分布在 $`G>0`$ 的唯一那个点上放的质量不同。

*用到:* Proposition 5.3、[Theorem 4.4](../proofs/04_complexity_of_sufficiency_ZH.md#s4-4)。*被用于:* Proposition 5.8(c)。

[证明](../proofs/05_warm_adam_ZH.md#s5-7) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Proposition 5.8](../proofs/05_warm_adam_ZH.md#s5-8) · warmth 的演化

- **(a)** 若梯度二阶矩平稳，则 $`\mathbb E v_s=\sigma^2(1-\beta_2^s)`$，mean-field warmth 如下式。它递减到平台 $`(1-\beta_2)/\beta_2>0`$；近似 $`1/(\beta_2s)`$ 的相对误差至多为 $`s(1-\beta_2)`$。

```math
\bar\kappa(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad\frac{\max\lbrace1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- **(b)** bias correction 不改变 warmth。
- **(c)** 若一段带 $`W\ge1`$ 次 warm-up 计算的训练中，warm 更新上 $`\max_i\kappa_i\le c\big(1/(W+s)+(1-\beta_2)\big)`$，则均值相同的两个来源之间差距中由 warmth 控制的部分至多为 $`\bar A\,[c\log(1+R/W)+c(1-\beta_2)R]`$，再加上每个 cold 更新一个单位，以及曲率项。
- **(d)** 学习率为零的 warm-up 推进 $`m`$、$`v`$ 与计数器的方式与真实步完全相同；在 mean-field 近似且 $`G_i^2\le\bar\gamma\sigma_i^2`$ 时，(c) 的假设以 $`c=2\bar\gamma/\beta_2`$ 成立。

*推导要点.*

- 展开指数滑动平均并取期望；两侧界由 Bernoulli 不等式与 $`e^{-y}(1+y)\le1`$ 得到。
- 对 (c)，telescoping 恒等式中每个 warm 项用 Proposition 5.6 控制，再用 $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$ 与 $`\sum_s1/(W+s)\le\log(1+R/W)`$。

*例示.* $`\beta_2=0.999`$ 时，前一百次计算里典型梯度的 warmth 约为 $`1/s`$，但它从不会降到 $`0.001`$ 以下太多：几千步之后，每个新梯度仍然把二阶矩扰动约千分之一，这个下限给累积误差加上一个关于更新次数线性的项。

*用到:* Proposition 5.6、5.7，[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)。*被用于:* Theorem 5.10、Remark 5.11。

[证明](../proofs/05_warm_adam_ZH.md#s5-8) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明；(d) 中的充分条件在所述 mean-field 近似下成立

#### [Proposition 5.9](../proofs/05_learner_statistics_ZH.md#s5-9) · 部分激活的梯度裁剪

记 $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$，

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad\Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c(1-p_{\rm act}).
```

*推导要点.* 按事件 $`\lbrace\lVert G\rVert\le c\rbrace`$ 拆分；在该事件上 $`\lVert G-cG/\lVert G\rVert\rVert=c-\lVert G\rVert\le c`$。

*被用于:* Theorem 5.10(e)。

[证明](../proofs/05_learner_statistics_ZH.md#s5-9) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10) · 各学习器的一阶统计量

| 部分 | 学习器 | 反馈在一阶上必须匹配的统计量 |
|---|---|---|
| **(a)** | SGD | $`\mathbb EG`$ |
| **(b)** | heavy-ball、Nesterov | $`\mathbb EG`$，经由 $`\theta`$ 与 $`m`$ 进入 |
| **(c)** | Adam/AdamW，warm | 精确地是 Adam 增量的均值；冻结分母近似下为 $`(\mathbb EG,\ \mathbb E\,G^{\odot2})`$，误差至多 $`\eta(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$；二阶矩经由 $`v`$ 以权重 $`(1-\beta_2)\nabla_vV`$ 进入 |
| **(d)** | Adam，cold 步 | $`\theta`$ 中为符号分布 $`\mathbb E\,h_\epsilon(G)`$；$`m,v`$ 中保留 $`\mathbb EG`$ 与 $`\mathbb E\,G^{\odot2}`$ |
| **(e)** | 范数 $`c`$ 的裁剪 | 精确地是裁剪后均值；平均单位方向在每个分布下准确到 $`c(1-p_{\rm act})`$，始终激活时精确 |
| **(f)** | Lion / 带 momentum 的 sign | 翻转概率；momentum 主导时 $`\theta`$ 中无依赖 |
| **(g)** | $`B=0`$ 的 LoRA | 第 1 步不改变 $`A`$；反馈只通过 rank-$`r`$ sketch $`\Gamma A_0^\top`$ 进入，cold Adam 下只通过它的符号分布 |

*推导要点.* 对每条更新规则算出增量，去掉只依赖状态的项，取期望之差。(c) 行用到 Proposition 5.5；(e) 行用到 Proposition 5.9；(g) 行用到 $`B=0`$ 时 $`G_A=\alpha B^\top\Gamma=0`$。

*例示.* 两个平均梯度相同的标签分布，一个集中、一个分散，对 SGD 在一阶上可以互换。在裁剪下，分散的那个会被阈值削掉大梯度；在 warm Adam 下，它更大的二阶矩抬高 $`v`$，使之后每一步都变小。两种效应都是一阶的。

*用到:* Proposition 5.1、5.3、5.5、5.9。*被用于:* [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)、[Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)、[Theorem 7.1](../proofs/07_second_moment_channel_ZH.md#s7-1)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/05_learner_statistics_ZH.md#s5-10) · [实验](../experiments/05_optimizer_state_ZH.md) · 已证明

#### [Remark 5.11](../proofs/05_learner_statistics_ZH.md#s5-11) · 在真实模型上检验 warm 状态的条件

- Proposition 5.6 的误差是按敏感度加权的和再加一个最大值，所以 warmth 的分位数验证不了它。应报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$、其中最大的几项以及 $`\max_i\kappa_i`$，任何替代量 $`\hat L_i`$ 都要标明是 proxy。
- 更新映射的非可加性是 optimizer 的性质，不是任务 response 的性质；response 必须直接测量，例如用 Definition 8.6 的配对替换分数。

[证明](../proofs/05_learner_statistics_ZH.md#s5-11) · remark
