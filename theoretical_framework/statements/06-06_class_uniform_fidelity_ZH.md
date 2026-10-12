# Remark 6.6 · 对声明类一致的 fidelity

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-06_class_uniform_fidelity_ZH.md) · [证明](../proofs/06-06_class_uniform_fidelity_ZH.md) · [English](06-06_class_uniform_fidelity.md)

**一段话概括.** 当蒸馏得到的反馈必须对一整个声明的任务类都保真、而不只对一个任务保真时，Theorem A 的两半都可以推广，两者之间的差距随该类的维数 $`d`$ 至多按 $`4d^{3/2}`$ 增长。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **Fisher information**：一条示范的 $`I_D(\theta)`$，在 $`x\in\mathcal X_Q`$ 处一次查询的 $`I_Q(\theta;x)`$。对方向 $`h\in\mathbb R^p`$，$`\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h`$。
- **learner-relative information norm** $`\mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\lvert c(\theta)^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta;h)}`$，约定 $`0/0=0`$，$`a>0`$ 时 $`a/0=\infty`$。
- **增长条件 (G)**：存在 $`N\to\infty`$ 与 $`0<a\le b<\infty`$，使对每个单位向量 $`h`$ 有 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$。
- **design.** approximate design $`\xi`$ 是查询空间上的概率测度，表示把 $`n`$ 次查询怎样分摊到各查询点上。$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ 是平均查询信息，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$ 是示范与查询合起来的总信息。

<a id="s6-6"></a>

**Remark 6.6（对声明类一致的 fidelity）.** 若要求 fidelity 对一个声明的 continuation value 类 $`\mathcal V`$ 一致成立，每个 $`V\in\mathcal V`$ 给出方向 $`g^V_t=\nabla V_{t+1}(z_t)`$、泛函 $`\Psi_V`$ 与梯度 $`c_V`$；误差为 $`\mathrm{Err}^{\mathcal V}_\theta(Q)=\sup_{V\in\mathcal V}\lvert\Psi_V(\theta)-\sum_t\langle g^V_t,\mu^Q_t(z_t)\rangle\rvert`$。

- **下界.** [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3) 的证明照样成立，只需把 $`\lvert c^\top h\rvert`$ 换成 $`\sigma(h)=\sup_V\lvert c_V^\top h\rvert`$，因为 $`\sup_V\lvert a_V\rvert+\sup_V\lvert b_V\rvert\ge\sup_V\lvert a_V-b_V\rvert`$。交换两个上确界，局部下界为 $`\tfrac14\sup_V\mathfrak n_{K,n}(c_V)`$。
- **上界.** 记 $`d`$ 为 $`\mathrm{span}\lbrace c_V\rbrace`$ 的维数（$`d\le\dim\mathrm{span}\,\mathcal G`$）。在 (E) 对 $`d`$ 个线性泛函的类似假设下，配合合适的混合 design，plug-in 蒸馏器的误差不超过 $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1))`$。
- 所以上下界之间至多差 $`4d^{3/2}`$ 倍，只依赖于声明类的维数。这里不主张更紧的常数。

**说了什么.**

- 下界：$`\tfrac14\sup_{V\in\mathcal V}\mathfrak n_{K,n}(c_V)`$，即类中最难的单个任务。
- 上界：用若干 c-optimal design 的均匀混合，plug-in 蒸馏器可达 $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)(1+o(1))`$。

**直觉.**

- 对每个任务都保真的蒸馏器，必须在这些任务用到的每个方向上都有信息；John 椭球把最坏的方向与 $`d`$ 个基方向联系起来。

**在框架中的作用.**

- 它覆盖声明的 continuation value 类；不主张更紧的常数。

**推导要点.**

- 三角不等式对类上的上确界成立，由此得到下界。上界方面，John 定理把类的半范数夹在一个椭球范数与它的 $`\sqrt d`$ 倍之间；对椭球 $`d`$ 个轴的 c-optimal design 取均匀混合，每个方差至多损失 $`d`$ 倍。

**用到.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：对上确界的两点论证。
- [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)：每个轴上的 c-optimal design。
- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：二阶项的处理。

**被用于.**

- 无

**状态.** 已证明（因子为 $`4d^{3/2}`$）。

## 参考文献

- [John, 1948](../references_ZH.md#john1948extremum)
