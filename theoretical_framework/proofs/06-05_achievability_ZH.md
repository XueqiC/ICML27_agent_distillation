# Theorem 6.5 · 可达性；标量情形紧到常数

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-05_achievability_ZH.md) · [证明](../proofs/06-05_achievability_ZH.md) · [实验](../experiments/06-05_achievability_ZH.md) · [English](06-05_achievability.md)

**本页证明什么.** 使用正则化 c-optimal design 的 plug-in 蒸馏器，期望一阶误差至多为 $`\mathfrak n_{K,n}(1+o(1))+o(N^{-1/2})`$。

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

**陈述（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)）.** 设 (G) 与 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(b) 的正则性成立，并且：

- **(R) 可实现性.** 对任意 $`\theta'\in\Theta`$，蒸馏器能在被审计状态上发出 oracle 律 $`P^*_{\theta'}`$，例如在 $`\theta'`$ 下运行模拟器或 teacher 模型，或对查询得到的消息重新加权。
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

**证明用到的结果.**

- [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)：$`c^\top A(\xi^*)^{-1}c=\mathfrak n_{K,n}^2`$ 以及 $`\xi^*`$ 的存在性。
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：(c)，匹配的下界。
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：增长条件 (G)。

**用到的经典工具.** Jensen 不等式与二阶 Taylor 界。

**证明.** 由 (R)，输出 $`P^*_{\hat\theta}`$ 是可得的，且 $`\mathrm{Err}_\theta(P^*_{\hat\theta})=\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$。若 $`\mathfrak n_{K,n}(\theta)=\infty`$，无需证明，故设它有限。按事件 $`\lbrace\hat\theta\in U\rbrace`$ 拆分：

```math
\mathbb E_\theta\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert=\mathbb E_\theta\big[\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\,\mathbf 1\lbrace\hat\theta\in U\rbrace\big]+\mathbb E_\theta\big[\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\,\mathbf 1\lbrace\hat\theta\notin U\rbrace\big].
```

由 (T)，第二项是 $`o(N^{-1/2})`$。在 $`\lbrace\hat\theta\in U\rbrace`$ 上，$`\theta`$（$`N`$ 足够大时，因为 (E) 的邻域收缩到 $`\theta_0`$）与 $`\hat\theta`$ 都在凸集 $`U`$ 内，两者之间的线段也在 $`U`$ 内，于是由 Taylor 定理与 (S)，

```math
\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\le\lvert c(\theta)^\top(\hat\theta-\theta)\rvert+\tfrac12H_\Psi\lVert\hat\theta-\theta\rVert^2 .
```

去掉示性函数，第一项不超过右边的期望。

- *线性项.* $`A(\xi_n)=KI_D+n(1-\varepsilon_n)M(\xi^*)+n\varepsilon_nM(\xi_0)\succeq(1-\varepsilon_n)A(\xi^*)`$，其中 $`A(\xi^*)`$ 可以奇异。由下一条中 $`\lambda_{\min}(A(\xi_n))`$ 的下界，$`N`$ 足够大时 $`A(\xi_n)`$ 本身可逆，所以 $`c^\top A(\xi_n)^{-1}c=\Phi_c(A(\xi_n))`$，$`\Phi_c`$ 如 [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4) 所定义。由它的变分形式 $`\Phi_c(A)=\sup_h(2c^\top h-h^\top Ah)`$，$`\Phi_c`$ 关于 Loewner 序单调递减，且对 $`\alpha>0`$ 有 $`\Phi_c(\alpha A)=\Phi_c(A)/\alpha`$。因此 $`\Phi_c(A(\xi_n))\le\Phi_c(A(\xi^*))/(1-\varepsilon_n)=\mathfrak n^2_{K,n}(1+o(1))`$，其中的等式是 Lemma 6.4 中最优 design 的值。由 Jensen 不等式与 (E)，$`\mathbb E\lvert c^\top(\hat\theta-\theta)\rvert\le\mathfrak n_{K,n}(1+o(1))`$。
- *二次项.* 由 (E)，它不超过 $`\tfrac12H_\Psi C/\lambda_{\min}(A(\xi_n))`$，而 $`\lambda_{\min}(A(\xi_n))\ge\max\lbrace K\lambda_{\min}(I_D),\,n\varepsilon_n\lambda_0\rbrace`$。这个量远大于 $`\sqrt N`$：
  - 若对某个 $`\delta>0`$ 有 $`n\ge\delta N`$，则 $`n\varepsilon_n\gg\sqrt n\ge\sqrt{\delta N}`$；
  - 否则沿某个子列 $`n=o(N)`$，于是对 $`h\in\ker I_D`$ 有 $`\mathbb I_{K,n}(h)\le n\sup_xh^\top I_Qh=o(N)`$，与 (G) 矛盾；所以 $`I_D\succ0`$，$`K`$ 与 $`N`$ 同阶，$`K\lambda_{\min}(I_D)\asymp N`$。

  所以二次项是 $`o(N^{-1/2})`$。
- *比较.* 若 $`c(\theta)\neq0`$，由 (G) 有 $`\mathfrak n_{K,n}\ge\lVert c\rVert/\sqrt{bN}`$，所以 $`o(N^{-1/2})=o(\mathfrak n_{K,n})`$。$`\square`$

**附注.**

- 正则化 $`\varepsilon_n`$ 是必要的：c-optimal design 往往只支撑在少数查询点上，$`A(\xi^*)`$ 在 $`c`$ 看不到的方向上可以奇异；而 $`\Psi`$ 的二阶项仍然要求 $`\hat\theta`$ 在每个方向上相合。
- 在高斯线性实验中，$`\Psi`$ 是线性的（所以 (T) 在 $`U=\mathbb R^p`$ 时成立），有效估计量精确服从高斯分布，plug-in 误差恰为 $`\sqrt{2/\pi}\,\sqrt{\Phi_c(A)}`$；在最优 design 处约为 $`0.80\,\mathfrak n_{K,n}`$。

## 相关工作

- **正则化的设计与两阶段估计** 在最优设计（[Pukelsheim, 1993](#ref-pukelsheim1993optimal)）与局部渐近理论（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）中都是标准做法；这里需要正则化，是因为学习器相对的 c-optimal design 通常是奇异的。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
