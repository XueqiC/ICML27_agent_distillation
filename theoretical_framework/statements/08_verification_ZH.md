# §8 验证与 Theorem C — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/08_verification_ZH.md) · [Experiments](../experiments/08_verification_ZH.md) · [English](08_verification.md)

### §8 · 验证与 Theorem C：精确陈述

[证明](../proofs/08_verification_ZH.md) · [实验](../experiments/08_verification_ZH.md)

**本节做什么.**

- 本节问：只用有限次测量，怎样认证一个 feedback law 足够好，或者给候选 law 排序。它比较四类测量：冻结 cell 上的特征均值、实际状态上完整尾段的配对对比、optimizer 平均增量，以及 surrogate 尾段上的 CRN 配对替换分数。
- 它说明冻结的 audit 需要显式的 transport 项，而在实际状态上用完整尾段做 audit 虽然可靠，代价却很高。
- 主要结果 **Theorem C**（[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)）直接测量一轮替换的有限效应，所以证书不需要 continuation 梯度，也没有二阶余项。
- 它精确地说明一阶分数在哪里可信（[Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)），以及还有哪些问题没有解决（[Remark 8.12](../proofs/08_certificate_ZH.md#s8-12)）。

**设定.** 轮次 $`t<R`$，complete learner state $`z`$，候选 kernel $`K^j_t`$ 与参照 $`K^0_t`$，参照 continuation $`V^0_{t+1}`$ 与 response $`g_{t,z}`$，均同 [Definitions 1.3–1.4](../README_ZH.md#s1-3)。候选 $`j`$ 的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1],
```

即把一轮参照替换成一轮候选 $`j`$ 对最终得分的影响。由 [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)，整次训练的差距是这些项在到达状态上的和。

#### [Theorem 8.1](../proofs/08_verification_ZH.md#s8-1) · 稳定性–近似的权衡

对有限消息字母表上的 response $`g`$、特征 $`\phi_1,\dots,\phi_d`$（矩阵 $`\Phi`$）与容差 $`\epsilon_j\ge0`$，令 $`W_{\mathbf{\epsilon}}(g)`$ 为所有满足 $`\lvert(P-Q)\phi_j\rvert\le\epsilon_j`$（对所有 $`j`$）的律对上 $`\lvert(P-Q)g\rvert`$ 的最大值。则

```math
W_{\mathbf{\epsilon}}(g)=\min_{b,c}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert\Big]=\min_b\Big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].
```

- $`\mathbf{\epsilon}=0`$ 时，它是用 $`\mathrm{span}\lbrace\mathbf 1,\phi\rbrace`$ 做最佳一致逼近的误差的两倍。
- 若对非常数的二值 $`g`$ 取 $`\phi_1=\alpha g`$，则 $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$。
- 在特征均值测量的同时置信事件上，取 $`\epsilon_j`$ 为测得的差加上两侧的置信半径，定理即可使用。

*推导要点.*

- 概率律之差恰好是和为零、$`\ell_1`$ 范数不超过 $`2`$ 的向量，而这个集合的支撑函数就是振幅（oscillation）。
- 特征约束把 $`W_{\mathbf{\epsilon}}`$ 变成线性规划；弱对偶对每个系数向量 $`b`$ 给出上界，对 Lagrangian 用 Sion minimax 定理得到等号。
- 对偶目标是凸多面体函数且有下界，所以最小值能取到。

*示例.* 一个等于 response 乘以 $`\alpha=0.01`$ 的特征能精确表示 response，但在容差 $`\epsilon=0.005`$ 下只能认证 $`W=0.5`$：系数 $`1/\alpha`$ 放大了测量误差。

*用到* [Theorem 4.2(b)](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)、[Proposition 4.6](../proofs/04_rates_and_composition_ZH.md#s4-6)。*被用于* [Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/08_verification_ZH.md#s8-1) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Theorem 8.2](../proofs/08_verification_ZH.md#s8-2) · 带 transport 的冻结有限接口选择

冻结 $`M`$ 个候选、一个有限的 audit cell 集合，以及每个 cell 上 $`d`$ 个取值于 $`[0,1]`$ 的特征。令 $`\widehat D_j`$ 为候选 $`j`$ 相对参照测得的最大特征均值差距，由 Hoeffding 与 union bound 得到半径 $`e_0,e_S`$，选择使 $`\widehat D_j`$ 最小的 $`\widehat j`$。设每个被认证的状态 $`z\in\mathcal C_t`$ 附带一个 cell、一个 transport 界 $`a_{t,j}(z)`$（$`z`$ 处的特征差距与其 cell 中的特征差距相差多少），以及系数界为 $`B_t`$、残差为 $`\tau_t`$ 的 response 分解。令 $`\sigma`$ 为首次离开认证状态集的时刻，$`\rho`$ 为其概率的上界。则以至少 $`1-\delta`$ 的概率，

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- 测得的差距一致地集中，所以按它选择至多损失 $`2(e_0+e_S)`$。
- transport 精确时，或者在凸仿射更新、IID block、以矩为特征的情形下，界会简化；后一种情形不需要 transport。
- transport 项是必要的：存在 audit 差距为零、response 分解精确、学习差距却为 $`1`$ 的实例。

*推导要点.*

- 对所有 cell、特征与候选做 Hoeffding 加 union bound，得到同时成立的集中性。
- 在被认证的状态上，一轮的效应是律之差作用在 response 上；response 分解把它拆成常数（抵消）、线性部分（由系数范数乘以经 transport 后的差距控制）与残差。
- 构造一个首次退出后改用参照的切换过程，它与实际训练过程至多差 $`\rho`$，而它的 telescoping 差距（[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)）只涉及退出前、被认证的状态。

*示例.* 在一个两律发出相同消息的状态上做 audit，然后从一个两律发出不同消息的状态开始训练。audit 报告零差距，而学习差距为 $`1`$；只有加上值为 $`1`$ 的 transport 项，或者把未被覆盖的状态计为退出，界才是对的。

*用到* [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Theorem 4.5](../proofs/04_rates_and_composition_ZH.md#s4-5)、[Theorem 8.1](../proofs/08_verification_ZH.md#s8-1)。*被用于* [Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)、[Remark 8.11](../proofs/08_certificate_ZH.md#s8-11)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/08_verification_ZH.md#s8-2) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Theorem 8.3](../proofs/08_verification_ZH.md#s8-3) · 实际状态上的自适应 audit；保真与改进

在每个被 audit 的轮次，以历史为条件，冻结实际状态 $`Z_t`$、$`M_t`$ 个候选、$`n_t`$ 个完整尾段的配对副本，以及置信份额 $`\delta_t`$（$`\sum_t\delta_t\le\delta`$）。按置信下界 $`\ell_t(j)=\widehat A_t(j)-e_t`$ 选择，其中 $`e_t=\sqrt{2\log(2M_t/\delta_t)/n_t}`$，$`\ell_t(0)=0`$；记被选中的下界为 $`\Gamma_t`$。则：

- **保真.** 若参照训练相对初始学生提升 $`\gamma`$，而候选协议相对参照的损失不超过 $`\varepsilon`$，则候选的提升至少为 $`\gamma-\varepsilon`$。
- **带符号的恒等式.** 对每个依赖历史的 provider 策略，$`L_E(S)-L_E(0)=\sum_t\mathbb E_{d^S_t}A_t(Z_t,J_t)`$。
- **不退化.** 以 $`1-\delta`$ 的概率 $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$，并且无条件地

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **代价.** 半径 $`e`$ 需要每个候选 $`\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ 个配对副本；$`M=4`$、$`\delta=0.05`$、$`e=0.1`$ 时为 $`1016`$。

*推导要点.*

- 以 audit 前的历史为条件，配对对比独立同分布、取值于 $`[-1,1]`$、均值为 $`A_t(Z_t,j)`$；Hoeffding 与对候选的 union bound 给出每轮的事件，沿训练失败概率之和不超过 $`\delta`$。
- 在每轮的事件上，被选候选的优势至少为 $`\Gamma_t\ge0`$。
- 新鲜更新使 $`A_t`$ 成为 $`V^0`$ 的条件漂移，于是终值可以 telescoping 成漂移加鞅差；取期望即得无条件的界。

*示例.* 即使每次 audit 都用精确的完整尾段，按点估计而不是按下界选择，期望上也可能亏：略差的候选在带噪声的比较中赢的次数足够多。

*用到* [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)。*被用于* [Proposition 8.4](../proofs/08_certificate_ZH.md#s8-4)、[Proposition 8.10](../proofs/08_certificate_ZH.md#s8-10)、[Remark 8.11](../proofs/08_certificate_ZH.md#s8-11)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/08_verification_ZH.md#s8-3) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Proposition 8.4](../proofs/08_certificate_ZH.md#s8-4) · 在实际状态上测量平均增量

设增量在状态空间的 Hilbert 范数下有界，$`\lVert\Delta\rVert\le b`$。

- 对 $`n`$ 个独立同分布的增量，以 $`1-\delta`$ 的概率

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big),
```

两个独立样本之差也有类似的界。

- 若 continuation $`V`$ 在 $`z`$ 处梯度范数为 $`\Lambda`$，沿增量的曲率不超过 $`M_2`$，则以 $`1-\delta`$ 的概率，一轮的效应 $`\lvert(K^P_t-K^Q_t)V(z)\rvert`$ 不超过 $`\Lambda(\lVert\bar X_P-\bar X_Q\rVert+e_1)+\tfrac{M_2}2(\hat m^P_2+\hat m^Q_2+e_2)`$，半径 $`e_1,e_2`$ 显式给出。
- 在 $`J`$ 次抽样运行的每个访问状态上应用它，就能界定整次训练的差距，另加 Monte Carlo 项 $`R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$。
- 样本只测增量；$`\Lambda`$ 与 $`M_2`$ 属于 continuation，必须另行给界或校准。

*推导要点.*

- 在 Hilbert 空间中 $`\lVert\bar X-\mu\rVert^2`$ 的交叉项为零，所以平均偏差不超过 $`b/\sqrt n`$；McDiarmid 不等式再补上偏离项。
- 二阶 Taylor 界把一轮的效应写成一阶项加余项，Cauchy–Schwarz 用 $`\Lambda`$ 乘以平均增量之差控制一阶项。
- 整次训练的版本结合了 telescoping 恒等式、对各次 audit 的 union bound，以及对运行次数的 Hoeffding。

*用到* [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)、[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)。*被用于* [Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)。

[证明](../proofs/08_certificate_ZH.md#s8-4) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Remark 8.5](../proofs/08_certificate_ZH.md#s8-5) · 范数与常数

- 集中界在 Hilbert 范数下成立，不能以同样的常数推广到任意范数（$`\mathbb R^p`$ 的 sup 范数下多一个 $`\sqrt{\log p}`$ 因子）。
- 界中不显含维数，但 $`b`$ 可以随维数增长：cold-start Adam 的一步让每个坐标移动约 $`\eta`$，所以 $`\lVert\Delta\theta\rVert`$ 可以达到 $`\eta\sqrt p`$。
- 排序只需要标量投影 $`\langle\nabla V,\Delta\rangle`$；Theorem 8.8 则完全不需要 $`\lVert\Delta\rVert`$ 的界。

[证明](../proofs/08_certificate_ZH.md#s8-5) · remark

#### [Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8) · CRN 配对替换证书；Theorem C

*设定*（[Definition 8.6](../proofs/08_certificate_ZH.md#s8-6)、[Assumption 8.7](../proofs/08_certificate_ZH.md#s8-7)）. 在被 audit 的状态 $`z`$ 上，候选 $`j=0,\dots,M`$（$`0`$ 为参照）发出消息，增量为 $`\Delta^j`$。surrogate 尾段 $`T_\omega`$ 与目标 $`\tilde F`$ 给出 $`\varphi_\omega(u)=\tilde F(T_\omega(z+u))`$ 与 $`\tilde L(P^j)=\mathbb E\varphi_\omega(\Delta^j)`$。在 $`r`$ 个相互独立的副本中，每个副本内所有候选共用一个 $`\omega_\ell`$（common random numbers），并令

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_\ell Z^{jk}_\ell .
```

分数落在已知的、长度为 $`B`$ 的区间里；surrogate 与任务在 $`(\alpha,\varepsilon_{\rm sur})`$ 意义下对齐：任务差 $`L_F(P^j)-L_F(P^k)`$ 等于 $`\alpha`$ 乘以 surrogate 差，误差不超过 $`\alpha\varepsilon_{\rm sur}`$。这里 $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。

*陈述.* $`\mathbb EZ^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$ 精确成立。记 $`\hat V^{jk}`$ 为样本方差，

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)},
```

则以至少 $`1-\delta`$ 的概率，对所有有序对同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- 只要 $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ 就判定 $`j`$ 优于 $`k`$，则以 $`1-\delta`$ 的概率不会做出错误判定；取 $`k=0`$ 即认证 $`A_t(z,j)>0`$。
- 不涉及 continuation 梯度、曲率常数或二阶余项；只需校准 $`(\alpha,\varepsilon_{\rm sur})`$。

*推导要点.*

- 精确性来自期望的线性：每一项只依赖于一个候选增量的边缘分布与独立的尾段随机性，所以候选之间怎么耦合都不改变均值。
- 把 Maurer 与 Pontil 对 $`[0,1]`$ 变量的 empirical Bernstein 界缩放到长度为 $`B`$ 的区间，得到每个单侧界；有序对至多 $`2M^2`$ 个，union bound 后得到 $`\log(4M^2/\delta)`$。
- 对齐假设把 surrogate 上的界转到任务上。

*示例.* 两个候选的平均增量相同、二阶矩不同，它们的一阶差为零；Theorem 8.8 仍然看得见它们的有限差，因为没有做任何线性化。

*用到* [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)（校准）。*被用于* [Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)、[Proposition 8.10](../proofs/08_certificate_ZH.md#s8-10)、[Remark 8.11](../proofs/08_certificate_ZH.md#s8-11)、[Remark 7.5](../proofs/07_second_moment_channel_ZH.md#s7-5)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/08_certificate_ZH.md#s8-8) · [实验](../experiments/08_verification_ZH.md) · 已证明（具体使用时的严格性依赖于已知的范围 $`B`$ 与校准过的 $`(\alpha,\varepsilon_{\rm sur})`$，见 Remark 8.12）

#### [Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9) · 一阶版本、它的 margin 判据与上限

令 $`\mu^j=\mathbb E\Delta^j`$，$`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$。

- **margin 判据.** 若曲率不超过 $`M_2`$ 的 continuation $`V`$ 满足 $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}2(m^j_2+m^k_2)`$，则一轮的效应与一阶项同号；整次训练有类似的结论。
- **测得的一阶证书.** 把增量按步长 $`h`$ 缩放、配对分数除以 $`h`$，估计的是一阶 surrogate 差 $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$。surrogate 曲率为 $`\tilde M`$ 时，被认证的界为

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\Big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-\tfrac12h\tilde M(m^j_2+m^k_2)-\tfrac12\tilde M(m^j_2+m^k_2)-\varepsilon_{\rm sur}\Big).
```

- **上限.** 这样认证一个候选对，必须 $`\tilde s^{jk}>\tfrac12\tilde M(m^j_2+m^k_2)+\varepsilon_{\rm sur}`$。一阶项低于余项界的候选对（例如平均增量沿 $`\nabla\tilde V(0)`$ 一致的两个候选），无论真实差距多大，在一阶意义下都不会被认证。
- **oracle 分数.** 用任务的精确 continuation 梯度算出的一阶分数满足 margin 判据，但不是可测量的证书，因为 $`\nabla V`$ 与 $`M_2`$ 都不能从反馈样本中得到。

*推导要点.*

- margin 判据就是二阶 Taylor 界：余项小于一阶项，翻不了符号。
- 测得的版本沿 $`s\mapsto\tilde V(shu)`$ 用 Taylor 定理，得到 $`h`$ 阶的有限差分偏差，以及 $`h=1`$ 时的余项；抽样误差如 Theorem 8.8 一样由 empirical Bernstein 控制。
- 在置信事件上把这些不等式串起来，就得到上限。

*示例.* 在 Adam 下，moment 增量是 $`O(1)`$ 的，而 continuation 对它们的依赖只通过大小为 $`\eta`$ 的步长体现；于是一阶项与余项同阶，学习率小并不能让 margin 成立。

*用到* [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)、[Proposition 8.4](../proofs/08_certificate_ZH.md#s8-4)、[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)。*被用于* [Remark 7.5](../proofs/07_second_moment_channel_ZH.md#s7-5)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/08_certificate_ZH.md#s8-9) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Proposition 8.10](../proofs/08_certificate_ZH.md#s8-10) · 代价，以及 common random numbers 带来的好处

- 一个被 audit 的状态需要 $`(M+1)r`$ 条 surrogate 尾段，外加校准尾段。
- $`e_{\rm stat}`$ 的范围项不超过 $`\zeta B`$，当且仅当 $`r\ge1+7\log(4M^2/\delta)/(3\zeta)`$；$`M=8`$、$`\delta=0.05`$、$`\zeta=0.2`$ 时即 $`r\ge101`$。
- 若 $`\varphi_\omega`$ 是常数为 $`\lambda_\omega`$ 的 pathwise Lipschitz 函数，则 $`\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\lVert\Delta^j_\ell-\Delta^k_\ell\rVert`$，$`\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E[\lambda_\omega^2\lVert\Delta^j-\Delta^k\rVert^2]`$；而独立尾段的方差中始终含有尾段噪声。一致界 $`\lambda_\omega\le\bar\lambda`$ 与 $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$ 给出合法的范围 $`B=2\bar\lambda b_{jk}`$。

*推导要点.* 计数；整理范围项；在配对增量处用 Lipschitz 条件，对独立的两臂用全方差公式。

*用到* [Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)、[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)。*被用于* [Remark 8.12](../proofs/08_certificate_ZH.md#s8-12)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/08_certificate_ZH.md#s8-10) · [实验](../experiments/08_verification_ZH.md) · 已证明

#### [Remark 8.11](../proofs/08_certificate_ZH.md#s8-11) · surrogate 的校准；沿训练的迁移

- $`(\alpha,\varepsilon_{\rm sur})`$ 的拟合方法：在一组校准对上，把完整尾段的任务配对对比对 surrogate 分数做回归。把它们用到新的候选对上，假设了这些候选对来自同一族；这是假设，不是定理。
- 在 Theorem 8.3 的自适应 audit 中，可以用 Theorem 8.8 的被认证下界代替完整尾段的下界，Theorem 8.3(c) 原样成立。
- 证书在被 audit 的状态上成立；之后继续使用这个选择，需要 Theorem 8.2 的 transport 项，或者在下一个检查点重新 audit。

[证明](../proofs/08_certificate_ZH.md#s8-11) · remark（校准的迁移是一个假设）

#### [Remark 8.12](../proofs/08_certificate_ZH.md#s8-12) · 未解决的问题

- **可证明的取值范围.** 用先验范围（$`B\le2`$）时，量级为 $`B\log(M^2/\delta)/r`$ 的范围项对任何负担得起的 $`r`$ 都可能超过相邻候选之间的差距，而分数的观测范围不能代替先验范围。一个可证明的、小得多的范围（例如来自被证明的 pathwise Lipschitz 常数）还没有得到。
- **校准的迁移.** 使校准过的 $`(\alpha,\varepsilon_{\rm sur})`$ 在新的候选对与之后的状态上仍然有效的可检验条件，还没有得到。
- 在这两点解决之前，Theorem 8.8 在实践中是无偏、低方差的排序分数；只有在 $`B`$ 与 $`(\alpha,\varepsilon_{\rm sur})`$ 被独立论证的地方，它才是严格的证书。

[证明](../proofs/08_certificate_ZH.md#s8-12) · 未解决
