# Corollary 8.9 · 一阶版本、它的 margin 判据与上限

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-09_first_order_version_margin_criterion_ZH.md) · [证明](../proofs/08-09_first_order_version_margin_criterion_ZH.md) · [实验](../experiments/08-09_first_order_version_margin_criterion_ZH.md) · [English](08-09_first_order_version_margin_criterion.md)

**一段话概括.** 基于梯度的分数按效应的一阶项给 feedback law 排序。这个推论精确说明：这种排序什么时候有保证（margin 判据），实测的一阶证书在统计半径之外还要付出什么，以及哪些候选对它永远认证不了（上限）。它还把只有受控实验才有的、用精确任务 continuation 梯度算出的分数，与可测量的证书区分开。

**本页用到的设定与记号.**

全节沿用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 轮次 $`t=0,\dots,R-1`$，$`z`$ 是 complete learner state。对一阶 optimizer，实际更新为 $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。
- 在第 $`t`$ 轮状态 $`z`$ 处，参照发出 $`Y\sim P^0_{t,z}`$，候选 $`j\in\lbrace1,\dots,M\rbrace`$ 发出 $`Y\sim P^j_{t,z}`$。候选 $`j`$ 的 kernel 记为 $`K^j_t`$。
- 参照 continuation 为 $`V^0_{t+1}`$，$`V^0_R=F_E\in[0,1]`$。response 为 $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$。对消息积分得到 kernel 恒等式

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- 对由候选 $`j`$ 驱动的过程，$`L_E(j)=\mathbb E\,F_E(Z^j_R)`$。所有过程从相同的 $`Z_0`$ 初始律出发。
- 候选 $`j`$ 在 $`(t,z)`$ 处的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- 它表示：把一轮参照替换成一轮候选 $`j`$，之后走参照尾段，对最终得分的影响。由 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)，整次训练的差距就是这些项在替代过程到达的状态上的和。

*配对替换分数*（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）.

- 在产生它的历史条件下，固定第 $`t`$ 轮的一个被 audit 状态 $`z`$（complete learner state，含 optimizer state）。候选为 $`j=0,1,\dots,M`$，$`M\ge1`$，其中 $`j=0`$ 是参照。候选 $`j`$ 的消息律为 $`P^j=P^j_t(\cdot\mid z)`$，消息 $`Y`$ 产生增量 $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$（含更新随机性）。
- **surrogate 尾段** $`T_\omega`$ 用随机性 $`\omega`$ 运行一段声明的（可以截短的）参照尾段，把一个状态映到最终状态。**surrogate 目标** $`\tilde F`$ 给最终状态打分（例如留出成功轨迹的 teacher-forced log-likelihood，映到 $`[0,1]`$）。记

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

其中 $`\omega`$ 与 $`\Delta^j`$ 独立。$`\tilde L(P^j)`$ 是"这一轮用候选 $`j`$、之后走 surrogate 尾段"的 surrogate 结果。

- **副本** $`\ell=1,\dots,r`$，$`r\ge2`$：抽一个新的 $`\omega_\ell`$，并为每个候选抽一条消息，得到增量 $`\Delta^j_\ell`$。同一副本内所有候选共用 $`\omega_\ell`$（**common random numbers**）；它们的消息也可以用共同的随机数耦合，只要每个 $`\Delta^j_\ell`$ 的边缘分布正确。不同副本相互独立。
- 若候选的消息律支撑有限，可以用支撑上的精确加权平均代替抽样，此时随机性只来自 $`\omega`$。
- 有序对 $`(j,k)`$ 的分数为

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- 任务一侧，$`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ 是"这一轮用候选 $`j`$、之后走完整参照尾段、按任务目标打分"的结果。用设定中的记号，$`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$，所以 $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。

*它的假设*（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)）.

- **(I) 副本独立.** 在产生 $`z`$ 的历史条件下，向量 $`(Z^{jk}_\ell)_{j,k}`$（$`\ell=1,\dots,r`$）独立同分布；$`r`$ 在观察分数之前固定。
- **(B) 分数有界.** 对每一对，$`Z^{jk}_\ell`$ 几乎必然落在一个已知的、长度为 $`B`$ 的区间里。例如 $`\tilde F\in[0,1]`$ 时 $`B\le2`$。
- **(A) surrogate 对齐.** 存在 $`\alpha>0`$ 与 $`\varepsilon_{\rm sur}\ge0`$，使对所有候选对，

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ 同时吸收了 surrogate 与任务目标之间的差别，以及尾段截短带来的差别。

<a id="s8-9"></a>

**Corollary 8.9（一阶版本、它的 margin 判据与上限）.** 沿用 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 的设定，令 $`\mu^j=\mathbb E\Delta^j`$，$`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$。

- **(a) margin 判据.** 设 continuation $`V`$ 沿到达的线段二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$。若 $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$，则 $`(K^j_t-K^k_t)V(z)`$ 与 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ 同号。整次训练层面，令 $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ 与 $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ 为 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 中以 $`P_j`$ 作 $`P`$ 时的一阶项与余项界；这里 $`d^k_t`$ 是由 $`P_k`$ 驱动的过程在第 $`t`$ 轮的状态分布。若 $`\lvert T_{jk}\rvert>B_{jk}`$，则 $`L(P_j)-L(P_k)`$ 与 $`T_{jk}`$ 同号。
- **(b) 测得的一阶证书.** 设 $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ 对到达的增量 $`u`$ 沿 $`\lbrace su:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$；令 $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$。对步长 $`h\in(0,1]`$，定义 $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$，其平均为 $`\hat S^{jk}(h)`$；设 (I)、(A) 成立，(B) 以区间长度 $`B_h`$ 成立。记 $`e^{jk}_{\rm stat}(h)`$ 为把 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的半径中 $`B`$ 换成 $`B_h`$ 得到的量，并令

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

则以至少 $`1-\delta`$ 的概率，对所有有序对同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) 上限.** 在 (b) 的事件上，一个候选对能被 (b) 认证，仅当 $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$。一阶项不超过二阶余项界的候选对，无论真实差距多大，在一阶意义下都不会被认证。特别地，若两个候选的平均增量沿 $`\nabla\tilde V(0)`$ 方向一致，它们只在增量的二阶及更高阶矩上不同，这样的候选对永远不会被一阶证书认证；[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 则可以认证它们。按总体量来说，(b) 认证的是下界 $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$，而右端正是 Theorem 8.8 直接估计的量。
- **(d) oracle 分数不是证书.** 用任务的精确 continuation 梯度算出的分数 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$（例如受控实验中对整个参照尾段求导得到的），在 margin 成立的地方满足 (a)。但它不是可测量的证书：$`\nabla V(z)`$ 与 $`M_2`$ 不能从反馈样本中得到，而检查 margin 两者都需要。一个便宜的分数要按 (a) 排对，需要：(i) 分数之差等于用实际学习器的 optimizer 变换后平均增量（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）在实际状态上算出的一阶项；(ii) margin 成立。用原始平均梯度构造的分数，或在不同状态与任务之间共用系数的分数，违反 (i)；在 margin 不成立处使用的任何分数，违反 (ii)。

**说了什么.** 只要两个候选的一阶差超过二阶余项，一阶分数就能把它们排对。用有限差分测量时，一阶证书在统计半径之外还要付出有限差分项与余项；对一阶项低于余项的候选对，无论真实差距多大，它都永远认证不了。

**直觉.** 主要靠增量二阶矩不同的候选（Adam 下正是如此），对任何一阶证书都是不可见的。

**在框架中的作用.** 它标出基于梯度的分数在哪里可信，以及为什么 Theorem 8.8 改为测量有限替换；它也把 oracle 诊断与可测量的证书区分开。

**关于陈述的说明.**

- **一阶版本为什么弱.** 两个效应叠加。$`Z(h)`$ 的先验取值范围随 $`1/h`$ 增长（$`\tilde F\in[0,1]`$ 时 $`B_h\le2/h`$），所以 $`h`$ 变小时 $`e_{\rm stat}(h)`$ 的范围项变大，而 $`e_{\rm fd}(h)`$ 随 $`h`$ 变大。并且余项 $`\tilde e_2`$ 完全不随 $`r`$ 缩小。
- **margin 在哪里成立.** 若 continuation 有曲率的每个坐标都只移动 $`O(\eta)`$（参数块上的 SGD、momentum、sign 类更新），则每一轮的一阶项是 $`O(\eta)`$、余项是 $`O(\eta^2)`$，步长小时 margin 成立。在 Adam 下，moment 增量 $`\Delta m=(1-\beta_1)(G-m)`$ 与 $`\Delta v=(1-\beta_2)(G\odot G-v)`$ 是 $`O(1)`$ 的，而 continuation 对 $`(m,v)`$ 的依赖只通过大小为 $`\eta`$ 的步长体现。于是一阶项与余项都是 $`O(\eta)`$，步长小并不意味着 margin 成立。这正是 §7 中二阶矩通道所在的区域（[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)）。

**推导要点.**

- margin 判据就是二阶 Taylor 界：余项小于一阶项，翻不了符号。
- 测得的版本沿 $`s\mapsto\tilde V(shu)`$ 用 Taylor 定理，得到 $`h`$ 阶的有限差分偏差，以及 $`h=1`$ 时的余项；抽样误差如 Theorem 8.8 一样由 empirical Bernstein 控制。
- 在置信事件上把这些不等式串起来，就得到上限。

**示例.** 在 Adam 下，moment 增量是 $`O(1)`$ 的，而 continuation 对它们的依赖只通过大小为 $`\eta`$ 的步长体现；于是一阶项与余项同阶，学习率小并不能让 margin 成立。

**用到.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) — 一阶展开及其余项
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — optimizer 变换后的平均增量
- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) — 一个状态上的 Taylor 界
- [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) — 配对替换分数
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) — 与之比较的有限证书

**被用于.**

- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5) — 一阶排序器在哪里失效
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3) — 什么时候允许一阶筛选

## Related work

- 一阶的数据归因分数，如 influence functions（[Koh & Liang, 2017](../references_ZH.md#koh2017understanding)）、TracIn（[Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)）、In-Run Data Shapley（[Wang et al., 2025](../references_ZH.md#wang2025data)）与 metagradient（[Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing)），使用目标关于训练数据或超参数的梯度。[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) 说明这类一阶排序何时有保证，以及为什么一阶项低于余项的候选对无法被认证。Datamodels（[Ilyas et al., 2022](../references_ZH.md#ilyas2022datamodels)）通过对大量训练运行做回归来估计有限的重训练效应；Theorem 8.8 则在固定状态上用配对尾段测量一轮的有限效应。

## 参考文献

- [Koh & Liang, 2017](../references_ZH.md#koh2017understanding). Koh, P. W., Liang, P. (2017). Understanding black-box predictions via influence functions. ICML.
- [Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating). Pruthi, G., Liu, F., Kale, S., Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. NeurIPS.
- [Wang et al., 2025](../references_ZH.md#wang2025data). Wang, J. T., Mittal, P., Song, D., Jia, R. (2025). Data Shapley in one training run. ICLR.
- [Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing). Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.
- [Ilyas et al., 2022](../references_ZH.md#ilyas2022datamodels). Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., Madry, A. (2022). Datamodels: Understanding predictions with data and data with predictions. ICML.
