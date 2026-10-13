# Proposition 6.10 · 显式信息界与 water-filling 最优查询分配

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-10_water_filling_allocation_ZH.md) · [证明](../proofs/06-10_water_filling_allocation_ZH.md) · [实验](../experiments/06-10_water_filling_allocation_ZH.md) · [English](06-10_water_filling_allocation.md)

**本页证明什么.** 下界 $`\tfrac14(\sum_sW_s/N_s)^{1/2}`$、逐状态经验分布达到的匹配上界、water-filling 分配，以及不可约部分。

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

**陈述（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）.** 在 [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9) 下，记 $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$，称为状态 $`s`$ 的**学习器相对权重**（learner-relative weight）。

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

**证明用到的结果.**

- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：状态覆盖模型。
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：两点论证，用在一维 tilting 子模型上。
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：用于 (d)。

**用到的经典工具.** Cauchy–Schwarz 不等式、凸规划的 KKT 条件，以及沿 influence function 的一维子模型；后者是给泛函求信息界的标准手段（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）。

**证明.** (a) 固定 $`h\in\mathbb R^{\mathcal S}`$，取 tilting 子模型 $`P_{s,\tau}=(1+\tau h_sf_s)\,P^*_s`$。$`f_s`$ 在有限字母表上有界且均值为零，所以 $`\lvert\tau\rvert`$ 足够小时它是概率分布。

- 由 influence function 的定义，沿子模型 $`\frac{\mathrm d}{\mathrm d\tau}\Psi=\sum_s\int f_s\,h_sf_s\,\mathrm dP^*_s=\sum_sh_sW_s`$。
- $`s`$ 上一个样本的 score 为 $`h_sf_s`$，Fisher information 为 $`h_s^2W_s`$；所以对标量参数 $`\tau`$，整条 transcript 的信息为 $`\sum_sN_sh_s^2W_s`$。score 有界，KL 展开因而一致。

[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3) 的证明只用到 KL 的链式法则，它对固定的逐状态样本数同样适用，给出下界 $`\tfrac14\lvert\sum_sh_sW_s\rvert/(\sum_sN_sh_s^2W_s)^{1/2}(1-o(1))`$。由 Cauchy–Schwarz，

```math
\Big(\sum_sh_sW_s\Big)^2=\Big(\sum_s h_s\sqrt{N_sW_s}\cdot\sqrt{W_s/N_s}\Big)^2\le\Big(\sum_sN_sh_s^2W_s\Big)\Big(\sum_sW_s/N_s\Big),
```

等号在 $`h_s=1/N_s`$ 处成立，即得 (a)。

(b) 记 $`\hat P_s`$ 为 $`s`$ 上 $`N_s`$ 个样本的经验分布。二阶展开给出 $`\Psi(\hat P)-\Psi(P^*)=\sum_s\frac1{N_s}\sum_{i\le N_s}f_s(Y_{s,i})+\mathrm{Rem}`$，其中 $`\mathbb E\lvert\mathrm{Rem}\rvert\le C\sum_s\mathbb E\lVert\hat P_s-P^*_s\rVert^2=O(\sum_s1/N_s)=O(1/N)`$。线性项均值为零、方差为 $`\sum_sW_s/N_s`$，由 Jensen 不等式其期望绝对值不超过方差的平方根；在按比例增长下 $`O(1/N)=o\big((\sum_sW_s/N_s)^{1/2}\big)`$。输出 $`Q=\hat P`$ 可以由 replay 实现。

(c) 目标函数在 $`n_s\ge0`$ 上是凸的，在 $`W_s>0`$ 的坐标上严格凸。KKT 条件为：$`n_s>0`$ 时 $`W_s/(k_s+n_s)^2=\lambda`$，$`n_s=0`$ 时 $`W_s/k_s^2\le\lambda`$，这正是所列的公式；预算方程左边在活跃范围内关于 $`\lambda`$ 连续且严格递减，所以 $`\lambda`$ 唯一。

(d) 没有任何观测依赖于 $`P^*_s`$，而 $`\Psi`$ 依赖于它（$`W_s>0`$）。只在 $`s`$ 上不同的两个 oracle 分布给出相同的 transcript 分布、不同的 $`\Psi`$，于是 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 适用。$`\square`$

## 相关工作

- **模仿学习中的主动获取.** DAgger（[Ross, Gordon & Bagnell, 2011](../references_ZH.md#ross2011reduction)）在学生访问的状态上查询 teacher；用本命题的语言，它按学生的访问频率分配查询。信息意义下的最优分配是按学习器相对权重做 water-filling，而这些权重随 optimizer 改变（[Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)）。
- **分配规则.** 没有 teacher 覆盖时，最优解 $`n_s\propto\sqrt{W_s}`$ 就是分层抽样中的 Neyman 分配（[Neyman, 1934](#ref-neyman1934two)）；有覆盖 $`k_s`$ 时它是 water-filling，与并行 Gaussian 信道上功率分配的解形状相同（[Cover & Thomas, 2006](#ref-cover2006elements)）。

## 参考文献

- [Ross, Gordon & Bagnell, 2011](../references_ZH.md#ross2011reduction)
