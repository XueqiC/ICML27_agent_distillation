# Proposition 2.3 · 稳健歧义的对偶

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-03_robust_ambiguity_duality_ZH.md) · [证明](02-03_robust_ambiguity_duality_ZH.md) · [实验](../experiments/02-03_robust_ambiguity_duality_ZH.md) · [English](02-03_robust_ambiguity_duality.md)

**本页证明什么.** 把稳健歧义值写成学习器（策略混合）与对手（先验）之间有限零和博弈的值，用 von Neumann minimax 定理交换 min 与 max。(a) 与 (b) 由博弈形式直接得到。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 歧义类 $`C\subseteq\mathcal E`$ 是有限的环境集合；$`\Delta(C)`$ 是它上面的先验集合。

**陈述（[Proposition 2.3](../statements/02-03_robust_ambiguity_duality_ZH.md#s2-3)）.** 对有限的歧义类 $`C\subseteq\mathcal E`$，令

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

**证明.**

定义 regret 矩阵 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$。对混合策略 $`\rho`$，映射 $`\mu\mapsto\mu^\top M\rho`$ 是线性的，它在 $`\Delta(C)`$ 上的最大值在顶点处取到，所以

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho .
```

这是一个有限二人零和博弈的值。由 von Neumann minimax 定理，可以交换 min 与 max：

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi .
```

单纯形上的内层最小值在纯策略处取到，而 $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$，即得 (2.4)。

- **(a)** 若没有实验能区分 $`C`$ 中的成员，则 $`D`$ 的分布在每个 $`E\in C`$ 下都相同，输出分布 $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$ 也相同。学习器在 $`E`$ 中的期望 regret 是 $`(M\bar\rho)_E`$，所以它在 $`C`$ 上的最坏情况至少为 $`\min_\rho\max_E(M\rho)_E=r(C)`$。若 $`C`$ 已知，输出一个最小化者 $`\rho^\star`$ 即可；$`\Delta(\Pi)`$ 紧且目标连续，所以最小化者存在。
- **(b)** 若 $`\pi^\circ`$ 在每个 $`E\in C`$ 中都最优，则 $`\pi^\circ`$ 处的点质量最坏 regret 为 0。反之，若 $`r(C)=0`$ 由 $`\rho^\star`$ 取到，则对每个 $`E\in C`$ 有 $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$。由于 $`M\ge0`$，$`\rho^\star`$ 支撑中的每个 $`\pi`$ 对所有 $`E\in C`$ 都有 $`M_{E,\pi}=0`$，因此是共同最优。$`\square`$

## 相关工作

- min 与 max 的交换是有限零和博弈的 von Neumann minimax 定理 ([von Neumann, 1928](#ref-vonneumann1928zur)).

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-vonneumann1928zur"></a>

**John von Neumann (1928).** Zur Theorie der Gesellschaftsspiele. *Mathematische Annalen* 100, no. 1, 295–320. DOI [10.1007/BF01448847](https://doi.org/10.1007/BF01448847).
