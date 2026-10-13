# Proposition 6.10 · 显式信息界与 water-filling 最优查询分配

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-10_water_filling_allocation_ZH.md) · [证明](../proofs/06-10_water_filling_allocation_ZH.md) · [实验](../experiments/06-10_water_filling_allocation_ZH.md) · [English](06-10_water_filling_allocation.md)

**一段话概括.** 在状态覆盖模型中，learner-relative information norm 有显式形式：在 $`1/4`$ 到 $`1`$ 的因子内等于 $`(\sum_sW_s/N_s)^{1/2}`$；最优的查询花法是按学习器相对权重做 water-filling。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **agent 状态** $`s\in\mathcal S`$，有限集；本页中 $`s`$ 表示 agent 的状态（[Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)）。
- **逐状态的 oracle 分布.** 在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在该状态的动作或修正），取值于有限字母表；各状态的分布互不相关，不假设状态之间有泛化。
- **influence function** $`f_s(y)`$：$`\Psi`$ 在 $`s`$ 处、把 $`P^*_s`$ 向 $`y`$ 处点质量移动时的导数；在 $`P^*_s`$ 下均值为零。
- **学习器相对权重** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$：在 $`s`$ 上多一个 oracle 样本，对估计 $`\Psi`$ 值多少。
- **样本数.** 状态 $`s`$ 共得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 为轨迹长度，$`d_T`$ 为 teacher 的状态访问分布），$`n_s`$ 来自查询，$`\sum_sn_s=n`$。

<a id="s6-10"></a>

**Proposition 6.10（显式信息界与 water-filling 最优查询分配）.** 在 [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9) 下，记 $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$，称为状态 $`s`$ 的**学习器相对权重**（learner-relative weight）。

- **(a) 下界.** 设在 $`W_s>0`$ 的状态上样本数按比例增长（$`N_s=a_sN`$，$`N\to\infty`$）。任何蒸馏器的局部 minimax 一阶误差至少为

```math
\tfrac14\Big(\sum_{s}\frac{W_s}{N_s}\Big)^{1/2}\big(1-o(1)\big).
```

- **(b) 上界.** 设 $`\Psi`$ 关于概率向量 $`(P_s)_s`$ 二阶连续可微。逐状态的经验分布，即在每个状态上 replay 获取到的消息，达到

```math
\mathbb E\lvert\hat\Psi-\Psi\rvert\le\Big(\sum_s\frac{W_s}{N_s}\Big)^{1/2}\big(1+o(1)\big).
```

- **(c) 最优查询（water-filling）.** 在 $`n_s\ge0`$、$`\sum_sn_s=n`$ 下最小化 $`\sum_sW_s/(k_s+n_s)`$（连续松弛），解为

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \text{其中 }\lambda>0\text{ 由 }\textstyle\sum_sn_s^\star=n\text{ 确定}.
```

  查询流向学习器相对权重高、teacher 覆盖低的状态。

- **(d) 不可约部分.** 若某个状态 $`W_s>0`$ 而 $`k_s=n_s=0`$（学生会到、teacher 不到、也从不查询），它留下的误差下限不能被任何 $`K`$ 消除（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）。

**说了什么.**

- 当示范覆盖 teacher 的状态、训练发生在学生的状态上时，误差由 $`\sum_sW_s/N_s`$ 决定，其中 $`W_s`$ 是状态 $`s`$ 的学习器相对权重，$`N_s`$ 是它的样本数。最优查询分配是 water-filling，$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$；权重为正而完全未被覆盖的状态留下不可约的误差。

**直觉.**

- 查询应当流向学习器敏感、而 teacher 没去过的状态。

**在框架中的作用.**

- 它让 Theorem A 在 agent 蒸馏中可以直接使用，并给方法的获取步骤提供分配规则（[Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)）。

**推导要点.**

- (a) tilting 子模型 $`(1+\tau h_sf_s)P^*_s`$ 的导数为 $`\sum_sh_sW_s`$，Fisher information 为 $`\sum_sN_sh_s^2W_s`$；由 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3) 与 Cauchy–Schwarz，取最优的 tilt $`h_s=1/N_s`$ 即得下界。
- (b) plug-in 误差是 influence function 的各状态独立样本均值之和，加上 $`1/N`$ 阶的余项。
- (c) 目标 $`\sum_sW_s/(k_s+n_s)`$ 是凸的；KKT 条件让所有被查询状态上的边际收益 $`W_s/(k_s+n_s)^2`$ 相等。
- (d) 没有任何观测依赖于 $`P^*_s`$，所以 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 适用。

**示例.** 两个状态的 teacher 覆盖相同，其中一个的权重是另一个的四倍：water-filling 给权重大的状态两倍（$`\sqrt4=2`$）而不是四倍的总样本数。

**用到.**

- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：状态覆盖模型。
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：沿 tilting 子模型的两点论证。
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：不可约部分。

**被用于.**

- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)：代入公式的权重。
- [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)：长尾分离。
- [Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)：算例。
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)：CED 的获取步骤。

**状态.** 已证明。

## 相关工作

- 没有 teacher 覆盖（$`k_s=0`$）时，最优解 $`n_s\propto\sqrt{W_s}`$ 就是分层抽样中的 Neyman 分配（[Neyman, 1934](#ref-neyman1934two)）。有覆盖时它变成 water-filling，与并行 Gaussian 信道上功率分配的解形状相同（[Cover & Thomas, 2006](#ref-cover2006elements)）。
- DAgger 在学生访问的状态上查询 teacher，查询量与学生的访问频率成正比（[Ross et al., 2011](#ref-ross2011reduction)）；信息意义下的最优分配则按学习器相对权重与已有覆盖来定。

## 参考文献

- [Ross, Gordon & Bagnell, 2011](../references_ZH.md#ross2011reduction)
