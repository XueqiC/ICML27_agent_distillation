# Definition 6.2 · 学习器相对泛函与信息范数

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-02_learner_relative_functional_information_norm_ZH.md) · [English](06-02_learner_relative_functional_information_norm.md)

**一段话概括.** 这个定义把"对这个学习器和这个任务而言，蒸馏器复现 oracle 反馈复现得多好"变成一个标量泛函 $`\Psi`$，并用 learner-relative information norm $`\mathfrak n_{K,n}`$ 度量数据关于它揭示了多少信息。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **平均增量** $`\mu^P_t(z)`$：消费一条来自 $`P`$ 的消息时，complete learner state $`z`$（参数与 optimizer state）的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。
- **continuation value** $`V_{t+1}`$：从某个状态出发、按声明的参考尾段把训练走完后的期望最终任务得分（[Definition 1.4](../README_ZH.md#s1-4)）。
- **一阶表示**（[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)）：一轮反馈对最终得分的效应，一阶上等于 $`\nabla V_{t+1}`$ 与平均增量的内积。

<a id="s6-2"></a>

**Definition 6.2（学习器相对泛函与信息范数）.**

- **学习器 Jacobian.** 被审计状态上的 oracle 平均增量为 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$。学习器 Jacobian $`J_t(\theta)=D_\theta\mu^*_t(\theta)`$ 是从 $`\mathbb R^p`$ 到状态空间 $`\mathcal Z`$ 的线性映射。它同时依赖于环境（通过 $`P^*_\theta`$）和学习器（通过增量映射 $`\Delta_t`$）。
- **学习器相对泛函（learner-relative functional）.** 声明的任务给出方向 $`g_t=\nabla V_{t+1}(z_t)\in\mathcal Z`$，即参考尾段在被审计状态上的 continuation value 梯度，与 $`\theta`$ 无关。定义

```math
\Psi(\theta)=\sum_{t=1}^{T}\big\langle g_t,\,\mu^*_t(\theta)\big\rangle,\qquad c(\theta)=\nabla_\theta\Psi(\theta)=\sum_{t=1}^{T}J_t(\theta)^{*}g_t\in\mathbb R^p .
```

- **一阶 fidelity 误差.** 对蒸馏器的输出 $`Q`$，

```math
\mathrm{Err}_\theta(Q)=\Big\lvert\Psi(\theta)-\sum_{t=1}^{T}\big\langle g_t,\,\mu^Q_t(z_t)\big\rangle\Big\rvert .
```

- **关于定义的说明.**
  - consumed gradient $`G_t(z,y)`$ 与增量映射 $`\Delta_t`$ 只依赖于学习器状态和消息，所以 $`\mu^Q_t(z_t)`$ 与 $`\theta`$ 无关；环境只通过"oracle 律会发出哪些消息"进入。
  - 由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)(a)（取 $`k=1`$），$`\langle g_t,\mu^*_t(\theta)-\mu^Q_t(z_t)\rangle`$ 是"在 $`z_t`$ 上把一轮 $`Q`$ 反馈换成 oracle 反馈、之后走参考尾段"这一效应的一阶部分。$`\mathrm{Err}_\theta`$ 把这些一阶效应在被审计状态上加起来。
- **信息形式与范数.** 记 $`I_D(\theta)`$ 为一条示范的 Fisher information，$`I_Q(\theta;x)`$ 为在 $`x\in\mathcal X_Q`$ 处一次查询的 Fisher information。对方向 $`h\in\mathbb R^p`$，

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_D(\theta)h+n\sup_{x\in\mathcal X_Q}h^\top I_Q(\theta;x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c(\theta)^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}},
```

  约定 $`0/0=0`$，$`a/0=\infty`$（$`a>0`$）。$`\mathfrak n_{K,n}`$ 称为 **learner-relative information norm**：它是学习器相对梯度 $`c`$ 相对于信息规范 $`\sqrt{\mathbb I_{K,n}}`$ 的对偶范数。

- **增长条件 (G).** 下面部分渐近结论假设信息非退化地增长：存在 $`N\to\infty`$ 与常数 $`0<a\le b<\infty`$，使所有单位向量 $`h`$ 满足 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$。例如 $`K`$ 与 $`n`$ 按比例增长、且 $`I_D+\sup_xI_Q(x)`$ 在每个方向上正定时，(G) 成立。

**是什么.**

- $`\Psi`$ 把各被审计状态上任务的 continuation value 梯度与 oracle 平均增量的内积加起来：oracle 反馈对最终任务得分的一阶效应。
- $`c=\nabla\Psi=\sum_tJ_t^*g_t`$ 把学习器的平均增量随 $`\theta`$ 的变化（Jacobian $`J_t`$）与任务奖励的方向（$`g_t`$）合在一起。
- $`\mathfrak n_{K,n}`$ 是 $`c`$ 在信息尺度下的对偶范数：数据在 $`\Psi`$ 信息最少的方向上留下的不确定性。

**在框架中的作用.**

- §6 的每个界都用 $`\mathfrak n_{K,n}`$ 表达；增长条件 (G) 是这些界在渐近意义下紧所需的正则性。

**用到.**

- [Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)：环境、数据、蒸馏器、被审计状态。
- [Definition 1.3](../README_ZH.md#s1-3)：complete learner state 与平均增量。
- [Definition 1.4](../README_ZH.md#s1-4)：continuation value。
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)：Err 是一阶的结果差距。

**被用于.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：用该范数表达的下界。
- [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)：design 值等于范数平方。
- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：可达性。
- [Remark 6.6](../statements/06-06_class_uniform_fidelity_ZH.md#s6-6)：对声明类一致的版本。
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8)：兑换率。
- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：状态覆盖模型。

**状态.** 定义。

## 相关工作

- 当 $`n=0`$ 时，$`\mathfrak n_{K,0}^2=c^\top(K\,I_D)^{-1}c`$ 正是光滑泛函 $`\Psi`$ 的正则估计量方差的经典局部渐近 minimax 下界（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）；有查询时，最优 design 进入其中（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)）。
- $`\Psi`$ 在被审计的检查点上累加任务梯度与学习器更新的内积，这与 TracIn 等沿梯度轨迹的影响估计结构相同（[Pruthi et al., 2020](#ref-pruthi2020estimating)）；在这里它是蒸馏器必须复现的量，而不是给数据排序的分数。
