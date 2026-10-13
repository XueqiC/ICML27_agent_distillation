# Proposition 2.4 · 后处理不能降低信息 regret

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-04_post_processing_cannot_reduce_information_ZH.md) · [证明](02-04_post_processing_cannot_reduce_information_ZH.md) · [实验](../experiments/02-04_post_processing_cannot_reduce_information_ZH.md) · [English](02-04_post_processing_cannot_reduce_information.md)

**本页证明什么.** 与环境无关的后处理之后的 Bayes 值，是条件期望之最大值的期望，它不超过原 Bayes 值的期望。用独立噪声扩充 transcript 不改变任何后验，因而不改变 Bayes 值。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。
- 后处理是不依赖于 $`E`$ 的 kernel $`\kappa(\cdot\mid D)`$，所以 $`E\to D\to D'`$ 是 Markov 链。

**陈述（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)）.** 设 $`D'\sim\kappa(\cdot\mid D)`$ 是随机后处理，kernel $`\kappa`$ 不依赖于 $`E`$，即 $`E\to D\to D'`$ 是 Markov 链。则

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。若 $`Z\perp E\mid(D,I_0)`$，则 $`B(D,Z)=B(D)`$：把 $`Z`$ 加进 transcript 不改变 $`\mathcal R_{\rm info}`$。

**证明.**

令 $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$。由 Markov 性质（$`E\perp D'\mid D`$）与 tower property，

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'] .
```

条件期望的最大值不超过最大值的条件期望：

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'] .
```

对 $`D'`$ 取期望即得 (2.5)。$`\mathbb EJ_E^\Pi`$ 与 transcript 无关，所以 (2.5) 等价于 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。对于加入 $`Z`$：$`Z\perp E\mid D`$ 给出 $`\mu(E\mid D,Z)=\mu(E\mid D)`$，因此对每个 $`\pi`$ 有 $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$，从而 $`B(D,Z)=B(D)`$。$`\square`$

## 相关工作

- garbling transcript 只会降低每个决策问题的 Bayes 值，这是 Blackwell 实验比较理论中容易的那个方向 ([Blackwell, 1953](#ref-blackwell1953equivalent)).

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).
