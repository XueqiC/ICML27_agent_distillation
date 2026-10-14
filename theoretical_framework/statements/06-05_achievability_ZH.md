# Theorem 6.5 · 可达性；标量情形紧到常数

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-05_achievability_ZH.md) · [证明](../proofs/06-05_achievability_ZH.md) · [实验](../experiments/06-05_achievability_ZH.md) · [English](06-05_achievability.md)

**一段话概括.** 用略加正则化的 c-optimal design 查询、再在估计值处发出 oracle 律的 plug-in 蒸馏器，误差在 $`1+o(1)`$ 因子内达到信息范数。与 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3) 合起来，最优一阶误差被夹在 $`\mathfrak n_{K,n}/4`$ 与 $`\mathfrak n_{K,n}`$ 之间；这一对结果就是 Theorem A。

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

<a id="s6-5"></a>

**Theorem 6.5（可达性；标量情形紧到常数）.** 设 (G) 与 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(b) 的正则性成立，并且：

- **(R) 可实现性.** 对任意 $`\theta'\in\Theta`$，蒸馏器能在被审计状态上发出 oracle 律 $`P^*_{\theta'}`$，例如在 $`\theta'`$ 下运行模拟器或 teacher 模型，或对查询得到的消息重新加权。这是一条假设：在每个候选环境下都能生成 oracle 律。只能使用有限个已核验合法消息的蒸馏器通常做不到；它只能输出这个消息池的正权重混合，[Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14) 说明这种混合何时能匹配所需的特征均值、匹配不了时要付出什么。
- **(I) 可辨识性.** 存在某个 design $`\xi_0`$，使 $`\lambda_{\min}(M(\xi_0))=\lambda_0>0`$。
- **(S) 光滑性.** $`\Psi`$ 在 $`\theta_0`$ 的一个凸邻域 $`U`$ 上属于 $`C^2`$，且在 $`U`$ 上 $`\lVert\nabla^2\Psi\rVert\le H_\Psi`$。
- **(T) 尾部控制.** 对 (E) 中收缩邻域内的 $`\theta`$ 一致地有

```math
\mathbb E_\theta\big[\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\,\mathbf 1\lbrace\hat\theta\notin U\rbrace\big]=o\big(N^{-1/2}\big).
```

  只有当 $`\theta`$ 到 $`\hat\theta`$ 的线段留在 $`U`$ 内时，(S) 的 Taylor 界才控制得住误差；其余部分由 (T) 控制。例如 $`\Psi`$ 在 $`\Theta`$ 上有界时它成立：(E) 的邻域收缩到内点 $`\theta_0`$，所以存在 $`r>0`$，由 Chebyshev 不等式与 (E) 得 $`\mathsf P_\theta(\hat\theta\notin U)\le C/(r^2\lambda_{\min}(A(\xi_n)))=o(N^{-1/2})`$（证明中给出 $`\lambda_{\min}(A(\xi_n))\gg\sqrt N`$）。另一种做法是用一个按构造取值于 $`U`$、并经验证满足 (E) 的估计量。
- **Design.** 使用正则化的 c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$，其中 $`\xi^*`$ 来自 [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)，$`\varepsilon_n\to0`$ 且 $`\varepsilon_n\sqrt n\to\infty`$（例如 $`\varepsilon_n=n^{-1/4}`$）。
- **(E) 有效的两阶段估计.** $`\xi_n`$ 在一个用 $`o(n)`$ 次查询得到的相合 pilot 估计处计算，并舍入成整数次数。最终估计量 $`\hat\theta`$ 在 $`\theta_0`$ 的一个收缩邻域上对 $`\theta`$ 一致地满足（$`\xi_n`$ 在真值 $`\theta`$ 处取值）

```math
\mathbb E_\theta\big[(c(\theta)^\top(\hat\theta-\theta))^2\big]\le c(\theta)^\top A(\xi_n)^{-1}c(\theta)\,(1+o(1)),\qquad \mathbb E_\theta\lVert\hat\theta-\theta\rVert^2\le\frac{C}{\lambda_{\min}(A(\xi_n))} .
```

令蒸馏器输出 $`Q=P^*_{\hat\theta}`$。则

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,\big(1+o(1)\big)+o\big(N^{-1/2}\big),
```

并且只要 $`c(\theta)\neq0`$，最后一项就是 $`o(\mathfrak n_{K,n}(\theta))`$。与 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(c) 合起来，在局部 minimax 意义下，最小可达一阶误差介于 $`\tfrac14\mathfrak n_{K,n}`$ 与 $`\mathfrak n_{K,n}`$ 之间（相差 $`1\pm o(1)`$ 的因子）。最优查询服从目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design。

**说了什么.**

- 用（略加正则化的）c-optimal 查询 design 估计环境，再在估计值处发出 oracle 律，误差可以达到 $`\mathfrak n_{K,n}(1+o(1))`$。与下界合起来，最优一阶误差被夹在 $`\mathfrak n_{K,n}/4`$ 与 $`\mathfrak n_{K,n}`$ 之间。

**直觉.**

- 信息一旦按正确的方向收集到，plug-in 蒸馏器最多只损失一个常数因子。

**在框架中的作用.**

- 它是 Theorem A 的上半部分：信息范数就是衡量"一批经验对给定学习器是否充分"的正确尺度。
- 这个界是能够生成 $`P^*_{\hat\theta}`$ 的蒸馏器的上界。没有 (R)，它就不是真实 agent 蒸馏的默认上界：在有限合法消息池下，plug-in 一步要换成 [Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14) 的可实现性误差与 [Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16) 的信息—编译半径。

**推导要点.**

- plug-in 输出的误差是 $`\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$，它不超过线性项 $`\lvert c^\top(\hat\theta-\theta)\rvert`$ 加一个二次余项。
- 正则化 design 只让 [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4) 的值 $`\Phi_c(A)`$ 损失一个因子 $`1/(1-\varepsilon_n)`$；c-optimal design 可以奇异，所以 $`\Phi_c`$ 用伪逆定义。于是由 Jensen 不等式，线性项为 $`\mathfrak n_{K,n}(1+o(1))`$。
- Taylor 界只在 $`\hat\theta`$ 留在 $`\Psi`$ 光滑的邻域内这一事件上使用；由 (T)，其余事件的贡献是 $`o(N^{-1/2})`$。
- 那一小部分非退化的 design 让 $`\hat\theta`$ 在每个方向上都相合，所以余项是 $`o(N^{-1/2})`$；单独的 c-optimal design 往往是奇异的（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)(b) 恰好允许这一点），保证不了这一点。

**示例.** 在高斯线性实验中，最优 design 下 plug-in 误差恰为 $`\sqrt{2/\pi}\,\mathfrak n_{K,n}\approx0.80\,\mathfrak n_{K,n}`$，落在 $`[1/4,1]`$ 之内。

**用到.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：匹配的下界。
- [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)：c-optimal design 的取值与存在性。

**被用于.**

- [Remark 6.6](../statements/06-06_class_uniform_fidelity_ZH.md#s6-6)：一致版本中的二阶项。
- [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)：裁剪学习器的分离。
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)：CED 的获取步骤。
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3)：设计规则：先获取、再编译。

**状态.** 已证明（局部渐近 minimax）；常数比较在 (G) 下成立。

## 相关工作

- c-optimal design 常常是奇异的，所以这里用一个非退化分量对 design 做正则化；奇异的 c-optimal design 以及 $`c^\top\theta`$ 的可估性见 [Pukelsheim, 1993](#ref-pukelsheim1993optimal)。
- (E) 要求一个有效估计量，例如从一致的 pilot 估计出发的 one-step 估计量；plug-in 输出就是 $`\Psi`$ 的代入估计量，与局部渐近理论中的做法相同（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
