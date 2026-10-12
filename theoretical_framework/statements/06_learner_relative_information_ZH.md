# §6 Theorem A：学习器相对的信息界 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/06_learner_relative_information_ZH.md) · [Experiments](../experiments/06_learner_relative_information_ZH.md) · [English](06_learner_relative_information.md)

### §6 · Theorem A：学习器相对的信息界：精确陈述

[证明](../proofs/06_learner_relative_information_ZH.md) · [实验](../experiments/06_learner_relative_information_ZH.md)

**本节做什么.**

- 度量 $`K`$ 条示范与 $`n`$ 次查询在"实际学习器与声明任务所在意的那个方向"上揭示了多少关于 oracle 反馈的信息，并证明这个 learner-relative information norm 决定了任何蒸馏器所能达到的最优一阶 fidelity，相差一个介于 $`1/4`$ 与 $`1`$ 之间的因子（Theorem A：[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3) 与 [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)）。
- 指出最优查询是一个 c-optimal design，其目标向量由 optimizer 决定（[Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4)）。
- 以学习器相对的形式推出混叠、replay 的零价值，以及示范与查询的兑换率（[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)、[Corollary 6.8](../proofs/06_learner_relative_information_ZH.md#s6-8)）。
- 针对 agent 蒸馏的状态覆盖结构，给出显式公式、water-filling 查询分配与 SGD、warm Adam 的学习器权重，并说明同一批示范可以对一个学习器充分、对另一个不充分（[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)–[Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)）。

**设定（[Definition 6.1](../proofs/06_learner_relative_information_ZH.md#s6-1)、[Definition 6.2](../proofs/06_learner_relative_information_ZH.md#s6-2)）.** 环境 $`E_\theta`$，$`\theta\in\Theta\subseteq\mathbb R^p`$，其 oracle feedback law 为 $`P^*_\theta`$。数据是来自 $`P^D_\theta`$ 的 $`K`$ 条独立同分布示范，以及 $`n`$ 次可以自适应的查询，结果服从 $`P^Q_\theta(\cdot\mid x)`$，$`x\in\mathcal X_Q`$。蒸馏器是从 transcript 与不依赖于 $`\theta`$ 的私有随机性到 feedback law $`Q`$ 的任何映射；replay、改写、合成都是蒸馏器。在固定的被审计 complete learner state $`z_1,\dots,z_T`$ 上，oracle 平均增量为 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，学习器 Jacobian 为 $`J_t=D_\theta\mu^*_t`$，声明的任务给出 continuation value 梯度 $`g_t=\nabla V_{t+1}(z_t)`$。于是

```math
\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle,\qquad c=\nabla\Psi=\sum_tJ_t^*g_t,\qquad \mathrm{Err}_\theta(Q)=\Big\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\Big\rvert,
```

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}} .
```

$`\mathrm{Err}`$ 是被审计状态上 $`Q`$ 与 oracle 反馈之间结果差距的一阶部分（[Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)）；$`\mathfrak n_{K,n}`$ 就是 learner-relative information norm。部分渐近结论假设增长条件 (G)：对单位向量 $`h`$ 有 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$，$`N\to\infty`$。

#### Theorem 6.3 · 学习器相对的下界

**(a)** 对任意 $`\theta_0,\theta_1`$ 与任意蒸馏器，

```math
\max_{i}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big).
```

**(b)** 在局部正则性（differentiability in quadratic mean、一致的 KL 展开）下，对满足 $`c^\top h\neq0`$ 的单位方向 $`h`$，取 $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}\to0`$，

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,(1-o(1)).
```

**(c)** 在 (G) 下，局部 minimax 下界为 $`\tfrac14\mathfrak n_{K,n}(\theta_0)(1-o(1))`$。

*推导要点.*

- 蒸馏器自身的贡献 $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ 与 $`\theta`$ 无关，所以由三角不等式，两个世界中的误差之和至少是 $`\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。
- Le Cam 两点论证：在两条 transcript 分布重叠的部分，没有蒸馏器能在两个世界里都对，所以期望误差之和至少是差距乘以 $`1-\mathrm{TV}`$。
- Pinsker 不等式加上沿自适应 transcript 的 KL 链式法则：示范贡献 $`K\,\mathrm{KL}_D`$，每次查询至多贡献 $`\sup_x\mathrm{KL}_Q(x)`$，查询的选择在两个世界中遵循同一规则，贡献为零。
- 局部地，KL 是位移的二次型，系数是 Fisher information；取 $`\tau=\mathbb I^{-1/2}`$ 使总变差不超过 $`1/2`$，由此得到常数 $`1/4`$。

*示例.* 一维参数、$`\Psi(\theta)=c\theta`$，每条示范的 Fisher information 为 $`i_D`$，每次查询为 $`i_Q`$，则 $`\mathfrak n_{K,n}=\lvert c\rvert/\sqrt{Ki_D+ni_Q}`$。敏感度 $`\lvert c\rvert`$ 越大的学习器，要达到同样的误差就需要成比例更多的信息。

*用到* [Definition 6.2](../proofs/06_learner_relative_information_ZH.md#s6-2)、[Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)。*被用于* [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)、[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)、[Corollary 6.8](../proofs/06_learner_relative_information_ZH.md#s6-8)、[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-3) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-2) · 已证明

#### Lemma 6.4 · 设计的 minimax 等式

查询空间紧、$`I_Q(\cdot)`$ 连续，approximate design $`\xi`$，$`M(\xi)=\int I_Q(x)\,\xi(\mathrm dx)`$，$`A(\xi)=KI_D+nM(\xi)\succ0`$，则

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

并且下确界由目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design $`\xi^*`$ 取到。当 $`n=0`$ 且 $`I_D`$ 奇异时：若 $`c\in\mathrm{range}(I_D)`$，两边都是 $`c^\top(KI_D)^+c`$；否则都是 $`+\infty`$。

*推导要点.*

- 固定 design 时，Cauchy–Schwarz 把 $`c^\top A^{-1}c`$ 写成对方向 $`h`$ 取 $`(c^\top h)^2/h^\top Ah`$ 的上确界。
- 固定方向时，最好的 design 把全部质量放在信息量最大的查询点上，这就是 $`\sup_x`$ 项的来源。
- $`h^\top A(\xi)h`$ 关于 design 线性、关于方向凸，design 构成紧凸集，所以 Sion minimax 定理可以交换两个优化。

*用到* [Definition 6.2](../proofs/06_learner_relative_information_ZH.md#s6-2)。*被用于* [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)、[Remark 6.6](../proofs/06_learner_relative_information_ZH.md#s6-6)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-4) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-1) · 已证明

#### Theorem 6.5 · 可达性；标量情形紧到常数

设 (G)、[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)(b) 的正则性、在任意 $`\theta'`$ 处 oracle 律的可实现性 (R)、某个 design 下的可辨识性 (I)、$`\Psi`$ 的光滑性 (S)，以及有效的两阶段估计 (E) 成立。用正则化的 c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$（$`\varepsilon_n\to0`$，$`\varepsilon_n\sqrt n\to\infty`$）查询，输出 $`Q=P^*_{\hat\theta}`$。则

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,(1+o(1))+o(N^{-1/2}),
```

当 $`c\neq0`$ 时最后一项是 $`o(\mathfrak n_{K,n})`$。结合 [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)(c)，最小可达一阶误差介于 $`\tfrac14\mathfrak n_{K,n}`$ 与 $`\mathfrak n_{K,n}`$ 之间（相差 $`1\pm o(1)`$ 的因子）。

*推导要点.*

- plug-in 输出的误差是 $`\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$，它不超过线性项 $`\lvert c^\top(\hat\theta-\theta)\rvert`$ 加一个二次余项。
- 正则化 design 只让 $`c^\top A^{-1}c`$ 损失一个因子 $`1/(1-\varepsilon_n)`$，所以由 Jensen 不等式与 [Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4)，线性项为 $`\mathfrak n_{K,n}(1+o(1))`$。
- 那一小部分非退化的 design 让 $`\hat\theta`$ 在每个方向上都相合，所以余项是 $`o(N^{-1/2})`$；单独的 c-optimal design 往往是奇异的，保证不了这一点。

*示例.* 在高斯线性实验中，最优 design 下 plug-in 误差恰为 $`\sqrt{2/\pi}\,\mathfrak n_{K,n}\approx0.80\,\mathfrak n_{K,n}`$，落在 $`[1/4,1]`$ 之内。

*用到* [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)、[Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4)。*被用于* [Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-5) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-2) · 已证明

#### Remark 6.6 · 对声明类一致的 fidelity

若要求 fidelity 对一个声明的 continuation value 类 $`\mathcal V`$ 一致成立，下界变为 $`\tfrac14\sup_{V\in\mathcal V}\mathfrak n_{K,n}(c_V)`$；用若干 c-optimal design 的混合，plug-in 蒸馏器可以达到 $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)(1+o(1))`$，其中 $`d=\dim\mathrm{span}\lbrace c_V\rbrace\le\dim\mathrm{span}\,\mathcal G`$。上下界至多差 $`4d^{3/2}`$ 倍。

*推导要点.* 三角不等式对类上的上确界成立，由此得到下界。上界方面，John 定理把类的半范数夹在一个椭球范数与它的 $`\sqrt d`$ 倍之间；对椭球 $`d`$ 个轴的 c-optimal design 取均匀混合，每个方差至多损失 $`d`$ 倍。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-6) · 没有单独的检验 · 已证明（因子为 $`4d^{3/2}`$）

#### Corollary 6.7 · 混叠与 replay 的零价值

若 $`P^D_{\theta_0}=P^D_{\theta_1}`$、$`\Psi(\theta_0)\neq\Psi(\theta_1)`$，且没有查询（或查询的分布也相同），则对每个 $`K`$，任何蒸馏器的最坏期望误差至少为 $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。replay、改写、合成都是蒸馏器，不能降低这个下界。示范是否充分，取决于示范看不见的方向在这个学习器的 $`c`$ 上有没有分量。

*推导要点.* transcript 分布相同，[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)(a) 中的总变差为零。

*用到* [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)。是 [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)(a) 与 [Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4) 的学习器相对形式。*被用于* [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)、[Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-7) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-3) · 已证明

#### Corollary 6.8 · 示范与查询的兑换率

记 $`e_D(h)=h^\top I_Dh`$，$`e_Q(h)=\sup_xh^\top I_Q(x)h`$。一个局部最坏一阶误差不超过 $`\varepsilon`$ 的蒸馏器，对每个方向 $`h`$ 必须满足

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c^\top h)^2}{16\,\varepsilon^2}\,(1-o(1)).
```

在起约束作用的方向 $`h^\star`$ 上，一条示范值 $`e_D(h^\star)/e_Q(h^\star)`$ 次查询；$`h^\star`$ 以及兑换率都取决于学习器。

*推导要点.* 对 [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)(b) 逐方向移项。

*用到* [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)。是 [Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11) 的学习器相对形式。*被用于* [Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-8) · 没有单独的检验（由 Theorem 6.3 移项得到） · 已证明

#### Assumption 6.9、Proposition 6.10 · 状态覆盖与 water-filling

**[Assumption 6.9](../proofs/06_learner_relative_information_ZH.md#s6-9).** 有限个 agent 状态 $`s`$；在状态 $`s`$ 上 oracle 消息的分布为 $`P^*_s`$，各状态的分布独立地变化，$`\Psi`$ 是 $`(P^*_s)_s`$ 的可微泛函，influence function 为 $`f_s`$。状态 $`s`$ 得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 个来自示范（期望为 $`KH\,d_T(s)`$，$`d_T`$ 是 teacher 的状态访问分布），$`n_s`$ 个来自查询。

**[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10).** 记 $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$，即状态 $`s`$ 的学习器相对权重。

- **(a)** 任何蒸馏器的局部 minimax 一阶误差至少为 $`\tfrac14\big(\sum_sW_s/N_s\big)^{1/2}(1-o(1))`$。
- **(b)** 在每个状态上 replay 获取到的消息（逐状态的经验分布），可以达到 $`\big(\sum_sW_s/N_s\big)^{1/2}(1+o(1))`$。
- **(c)** $`n`$ 次查询的最优分配是 water-filling：

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \textstyle\sum_sn_s^\star=n .
```

- **(d)** $`W_s>0`$ 而 $`k_s=n_s=0`$ 的状态留下任何 $`K`$ 都去不掉的误差下限。

*推导要点.*

- (a) tilting 子模型 $`(1+\tau h_sf_s)P^*_s`$ 的导数为 $`\sum_sh_sW_s`$，Fisher information 为 $`\sum_sN_sh_s^2W_s`$；由 [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3) 与 Cauchy–Schwarz，取最优的 tilt $`h_s=1/N_s`$ 即得下界。
- (b) plug-in 误差是 influence function 的各状态独立样本均值之和，加上 $`1/N`$ 阶的余项。
- (c) 目标 $`\sum_sW_s/(k_s+n_s)`$ 是凸的；KKT 条件让所有被查询状态上的边际收益 $`W_s/(k_s+n_s)^2`$ 相等。
- (d) 没有任何观测依赖于 $`P^*_s`$，所以 [Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7) 适用。

*示例.* 两个状态的 teacher 覆盖相同，其中一个的权重是另一个的四倍：water-filling 给权重大的状态两倍（$`\sqrt4=2`$）而不是四倍的总样本数。

*用到* [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)、[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)。*被用于* [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)、[Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-10) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-5) · 已证明

#### Proposition 6.11 · SGD 与 warm Adam 的学习器权重

设参数坐标为 state-owned，每步访问一个状态 $`s_k\sim d`$，梯度 $`g=G(s,Y)`$ 与参数无关（线性 response 区间），结果 $`F=\langle\zeta,\theta_R\rangle`$，$`m_1=\mathbb Eg`$，$`m_2=\mathbb Eg^2`$。在首项意义下：

- **(a) SGD.** $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$。
- **(b) warm Adam，频繁访问区间**（每个二阶矩窗口内访问很多，$`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$）：

```math
W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big),
```

  其中第二项是二阶矩通道。

- **(c) warm Adam，孤立访问区间**（$`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$）：每次访问让坐标移动 $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$，$`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}(1-\beta_1/\sqrt{\beta_2})\big)`$，所以 $`W_s\propto d(s)^2\mathrm{Var}(\mathrm{sign}\,g)`$。有 bias correction 时，第 $`t`$ 步的第一次访问多带一个因子 $`\sqrt{1-\beta_2^{\,t}}`$。
- **(d)** 对处在频繁访问区间的两个状态，相对于高频状态，warm Adam 对稀有状态的权重是 SGD 的 $`d(s)/d(s')`$ 倍。

*推导要点.*

- SGD 的期望位移关于该状态的消息分布线性，所以 influence function 正比于 $`d(s)(g-m_1)`$。
- 频繁访问区间中二阶矩平均为 $`d(s)m_2`$，期望位移为 $`-\eta R\sqrt{d(s)}\,m_1/\sqrt{m_2}`$；对 $`m_1`$、$`m_2`$ 求导（delta method）得到 influence function，其中对 $`m_2`$ 的导数就是二阶矩通道。
- 孤立访问区间中，两个矩在每次访问之前都已衰减殆尽；访问本身带来的动量与二阶矩随后按几何速度衰减，它们的比值求和得到 $`\kappa_{\rm iso}\,\mathrm{sign}(g)`$。
- 在 $`d(s)\approx1-\beta_2`$ 附近，两个闭式都不适用。

*示例.* [Example 6.13](../proofs/06_learner_relative_information_ZH.md#s6-13)：一个常见状态以概率 $`1-q`$ 被访问，$`N`$ 个长尾状态各以概率 $`q/N`$ 被访问，都处在频繁访问区间。相对于常见状态，长尾在 $`\sum_sW_s/N_s`$ 中的份额在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍；例如 $`N=20`$、$`q=0.2`$ 时为 $`80`$ 倍。

*用到* [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)、[Proposition 5.5](../proofs/05_warm_adam_ZH.md#s5-5)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)。*被用于* [Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)、[Corollary 7.2](../proofs/07_second_moment_channel_ZH.md#s7-2)、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-11) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-4) · 在所述区间内首项意义下已证明

#### Corollary 6.12 · 学习器之间的分离

- **(a)** 设 oracle 的 consumed gradient 为 $`b\,G_\alpha(Y)`$，尺度 $`b>0`$ 未知，示范只依赖于 $`\alpha`$。对始终激活的裁剪 SGD（或 normalized SGD），oracle 平均增量与 $`b`$ 无关，误差随 $`K\to\infty`$ 趋于零。对 SGD，学习器相对泛函依赖于 $`b`$，示范却不依赖，任何 $`K`$ 都不够。同一批示范对一个学习器充分，对另一个不充分。
- **(b)** 当学生的状态分布有大量低频状态（处在 warm Adam 的频繁访问区间）、而 teacher 很少到达它们时，这些状态在 warm Adam 下承担的信息需求份额比在 SGD 下更大。

*推导要点.* (a) 始终激活的裁剪去掉了尺度，所以 $`c`$ 在 $`b`$ 方向上没有分量；对 SGD，尺度改变 $`\Psi`$ 而示范分布不变，这正是 [Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)。若裁剪只以概率 $`p_{\rm act}`$ 激活，[Proposition 5.9](../proofs/05_learner_statistics_ZH.md#s5-9) 把剩下的尺度依赖控制在 $`c_{\rm clip}(1-p_{\rm act})`$ 以内。(b) 把 [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10) 与 [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11) 的权重结合。

*用到* [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)、[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)、[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)、[Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)、[Proposition 5.9](../proofs/05_learner_statistics_ZH.md#s5-9)。*被用于* [Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/06_learner_relative_information_ZH.md#s6-12) · [实验](../experiments/06_learner_relative_information_ZH.md#c6-6) · 已证明；(b) 在所述区间内首项意义下成立
