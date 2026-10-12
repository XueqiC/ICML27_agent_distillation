# Proposition 2.4 · 后处理不能降低信息 regret

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-04_post_processing_cannot_reduce_information_ZH.md) · [证明](../proofs/02-04_post_processing_cannot_reduce_information_ZH.md) · [实验](../experiments/02-04_post_processing_cannot_reduce_information_ZH.md) · [English](02-04_post_processing_cannot_reduce_information.md)

**一段话.** replay、转述、改写和合成生成，都是对学习器已有数据的变换。这个命题说明，任何这样的变换都不能降低 regret 中的信息项，因为它不能带来关于环境的新证据；它只能让已有的证据更容易学到。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 后处理是一个 kernel $`\kappa(\cdot\mid D)`$，它不看环境，把 transcript 变成新的 transcript $`D'`$，所以 $`E\to D\to D'`$ 是 Markov 链。$`\mathcal R_{\rm info}(D)`$ 是用 transcript $`D`$ 算出的信息项。

<a id="s2-4"></a>

**Proposition 2.4（后处理不能降低信息 regret）.** 设 $`D'\sim\kappa(\cdot\mid D)`$ 是随机后处理，kernel $`\kappa`$ 不依赖于 $`E`$，即 $`E\to D\to D'`$ 是 Markov 链。则

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。若 $`Z\perp E\mid(D,I_0)`$，则 $`B(D,Z)=B(D)`$：把 $`Z`$ 加进 transcript 不改变 $`\mathcal R_{\rm info}`$。

**说了什么.**

- 任何不依赖于环境的 transcript 随机变换，都只会提高信息项或使它不变。
- 加入给定 transcript 时与环境独立的数据，不改变 Bayes 值，也不改变信息项。

**直觉.**

- 加工数据可以让已有信息更容易学到，从而降低算法误差，但造不出缺失的事实。

**在框架中的作用.**

- 它排除了用生成数据代替新实验的可能。[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 对学习器相对的一阶效应、对每个蒸馏器给出同样的结论。

**推导要点.**

- $`E\to D\to D'`$ 是 Markov 链，所以 $`\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D]\,\big|\,D'\big]`$。
- 条件期望的最大值不超过最大值的条件期望，所以 $`D'`$ 下的 Bayes 值不超过 $`D`$ 下的。

**例示.** 用与环境无关的流程对 $`K`$ 条示范做 replay、转述或改写，就是这样的 $`\kappa`$。它可以让学生更好地拟合已有信息（降低 $`\mathcal R_{\rm alg}`$），但不能降低 $`\mathcal R_{\rm info}`$。

**说明.**

- **生成的数据.** 由 Proposition 2.4，对 $`D`$ 的合成改写、转述和 replay 都不能降低 $`\mathcal R_{\rm info}`$。它们可以让已有信息更容易学到，从而降低 $`\mathcal R_{\rm alg}`$。预训练 teacher 内部已有的知识属于 $`I_0`$；新的外部查询才扩大 $`D`$。

**用到.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1): Bayes 值与信息项.

**被用于.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): 被动学习器恢复不了符号.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10): replay 不改变信息下限.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7): 对每个蒸馏器的学习器相对版本.
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3): 生成数据代替不了获取.
