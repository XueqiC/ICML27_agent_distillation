# Proposition 2.3 · 稳健歧义的对偶

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-03_robust_ambiguity_duality_ZH.md) · [证明](../proofs/02-03_robust_ambiguity_duality_ZH.md) · [实验](../experiments/02-03_robust_ambiguity_duality_ZH.md) · [English](02-03_robust_ambiguity_duality.md)

**一段话.** 当若干环境无法被任何允许的实验区分时，学习器只能在它们之间对冲。这个命题用两种等价形式给出对冲的代价：对混合策略取最小的最坏 regret，以及在这些环境的先验上取最大的信息 regret。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 歧义类 $`C\subseteq\mathcal E`$ 是有限的环境集合；$`\Delta(C)`$ 是它上面的先验集合，$`\mathbb E_{\pi\sim\rho}`$ 表示对策略混合 $`\rho\in\Delta(\Pi)`$ 取平均。

<a id="s2-3"></a>

**Proposition 2.3（稳健歧义的对偶）.** 对有限的歧义类 $`C\subseteq\mathcal E`$，令

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(2.3)
```

则

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(2.4)
```

此外：

- **(a)** 若每个允许的实验在所有 $`E\in C`$ 下分布相同，则任何学习器在 $`C`$ 上的最坏期望 regret 至少为 $`r(C)`$。若 $`C`$ 已知，minimax 混合策略可以达到 $`r(C)`$。
- **(b)** $`r(C)=0`$ 当且仅当存在某个 $`\pi\in\Pi`$ 在所有 $`E\in C`$ 中都最优。

**说了什么.**

- 在一个歧义类上对冲的 minimax regret，等于在该类的先验上取最大的 Bayes 信息 regret。
- 若实验分不清该类的成员，任何学习器在最坏情况下都做不得更好，而 minimax 混合策略能达到这个值。这个值为零，恰好当有一个策略在所有成员中都最优。

**直觉.**

- 无法区分的世界迫使学习器对冲。对冲的代价是对抗最不利先验的零和博弈的值。

**在框架中的作用.**

- 它给信息误差一个可计算的 minimax 形式，并给出衡量其大小的证书（一个最不利先验）。[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6) 的两世界族就是一个值为 $`\Delta`$ 的实例。

**推导要点.**

- regret 矩阵 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ 定义了一个有限零和博弈：学习器选策略的混合，对手选 $`C`$ 上的先验。
- 由 von Neumann minimax 定理交换 min 与 max；内层最小值在纯策略处取到，即得对偶形式。
- 当实验分不清 $`C`$ 中的成员时，学习器的输出分布在它们中都相同，所以其最坏 regret 至少等于博弈的值。

**例示.** 两个环境偏好相反的策略，regret 矩阵为 $`\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$。minimax 混合是 $`(\tfrac12,\tfrac12)`$，最不利先验也是 $`(\tfrac12,\tfrac12)`$，两边都等于 $`0.1`$。这是混合平均的 regret：单个抽样策略的 regret 是 0 或 0.2。

**说明.**

- **(2.3) 的概率含义.** 设置信集 $`C_D`$ 以概率 $`1-\delta`$ 包含真实环境，某个混合策略在 $`C_D`$ 上的最坏差距不超过 $`\varepsilon`$。这界定的是该混合策略在这一数据事件上的*期望* regret，而不是单个抽样策略的 regret。

**用到.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1): 三项 regret.

**被用于.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): 一个值为 Δ 的实例.

## 相关工作

- (2.3) 与 (2.4) 相等，就是 von Neumann 关于有限零和博弈的 minimax 定理，博弈的支付为 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ ([von Neumann, 1928](#ref-vonneumann1928zur))；取到最大值的先验就是这个博弈的最不利先验。
