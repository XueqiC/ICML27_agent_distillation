# Definition 2.1 · Bayes 值与三项 regret

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-01_bayes_value_three_regret_terms_ZH.md) · [English](02-01_bayes_value_three_regret_terms.md)

**一段话.** 这个定义给出 §2 用来回答"现有数据能否确定学生该怎么做"的基本对象。它引入 Bayes 值 $`B(D)`$，即由 transcript 能达到的最好期望回报，并把学生的不足拆成三项 regret：学生类表示不了的部分、数据确定不了的部分、学习器没有用上的部分。

**本页用到的设定.**

- [Definition 1.1](../README_ZH.md#s1-1)：环境 $`E`$、取值 $`J_E(\pi)\in[0,1]`$、学生类 $`\Pi`$ 与参照类 $`\Pi_{\rm ref}`$。
- [Definition 1.2](../README_ZH.md#s1-2)：公共初始信息 $`I_0`$ 与后续实验的 transcript $`D`$。
- $`\Delta(\Pi)`$ 是 $`\Pi`$ 上概率分布的集合；Markov kernel $`\lambda(\cdot\mid D)`$ 给每条 transcript 指定一个这样的分布。

<a id="s2-1"></a>

**Definition 2.1（Bayes 值与三项 regret）.** 考虑一个有限决策问题。

- $`\mathcal E`$ 是有限的环境集合，先验为 $`\mu`$。先验以公共初始信息 $`I_0`$ 为条件：预训练模型、已知规则和 $`K`$ 条可信示范。
- $`\Pi\subseteq\Pi_{\rm ref}`$ 是有限策略类，取值 $`J_E(\pi)\in[0,1]`$。$`\Pi`$ 是学生的部署类，$`\Pi_{\rm ref}`$ 是参照类。
- transcript $`D`$ 取值于有限字母表 $`\mathcal D`$，分布为 $`P(D\mid E)`$。学习器可以做的所有实验，无论是否自适应，都包含在 $`D`$ 中。
- 学习器是一个 Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$，输出 $`\widehat\pi\sim\lambda(\cdot\mid D)`$。给定 $`D`$ 时，它的私有随机性与 $`E`$ 条件独立。
- $`\mu(E\mid D)`$ 是后验。

定义

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D],
```

以及表示、信息、算法三项 regret

```math
\mathcal R_{\rm rep}=\mathbb E[J_E^{\rm ref}-J_E^\Pi],\qquad \mathcal R_{\rm info}=\mathbb E J_E^\Pi-\mathbb E_D B(D),\qquad \mathcal R_{\rm alg}=\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big].
```

策略类有限，所以所有最大值都能取到。$`B(D)`$ 是 Bayes 值：给定 transcript 时，$`\Pi`$ 中的学生能达到的最好期望回报。

**说了什么.**

- $`B(D)`$ 是一个 Bayes 决策者用这条 transcript、只用学生能执行的策略所能做出的最好决策的值。
- $`\mathcal R_{\rm rep}`$ 把学生类与参照类比较，$`\mathcal R_{\rm info}`$ 把 transcript 与完全知道环境相比较，$`\mathcal R_{\rm alg}`$ 把学习器与 Bayes 值比较。

**直觉.**

- 表示：学生到底能不能做到？信息：数据有没有说清该怎么做？算法：学习器有没有照数据说的去做？

**在框架中的作用.**

- 这三项是整个 §2 的记账方式：[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 证明它们精确相加，后面的结果在具体数据下界定 $`\mathcal R_{\rm info}`$。

**用到.**

- 无

**被用于.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2): 精确的三项分解.
- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality_ZH.md#s2-3): 信息项的稳健版本.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4): 信息项的 data processing.
