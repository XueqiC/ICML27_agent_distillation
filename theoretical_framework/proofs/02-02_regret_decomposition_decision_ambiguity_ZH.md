# Theorem 2.2 · regret 分解与决策歧义

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md) · [证明](02-02_regret_decomposition_decision_ambiguity_ZH.md) · [实验](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md) · [English](02-02_regret_decomposition_decision_ambiguity.md)

**本页证明什么.** 分解是一个精确恒等式，再逐项证明三项都非负。零信息判据来自把信息项写成一组非负的后验括号之和。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。

**陈述（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）.** 在 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1) 下：

**(a) 精确分解.** 相对参照最优的期望 regret 为

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) 决策歧义.** $`\mathcal R_{\rm info}=0`$，当且仅当对每个满足 $`\Pr(D)>0`$ 的 $`D`$，

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

**证明.**

**(a)** 加减 $`\mathbb EJ_E^\Pi`$ 与 $`\mathbb E_DB(D)`$ 即得恒等式。下面逐项证明非负。

- $`\mathcal R_{\rm rep}\ge0`$：由 $`\Pi\subseteq\Pi_{\rm ref}`$，对每个 $`E`$ 有 $`J_E^\Pi\le J_E^{\rm ref}`$。
- $`\mathcal R_{\rm info}\ge0`$：固定满足 $`\Pr(D)>0`$ 的 $`D`$。逐点有 $`J_E(\pi)\le J_E^\Pi`$，所以对每个 $`\pi\in\Pi`$ 有 $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$。对 $`\pi`$ 取最大得 $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$。对 $`D`$ 取期望并用 tower property，得 $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$。
- $`\mathcal R_{\rm alg}\ge0`$：给定 $`D`$ 时 $`\widehat\pi`$ 与 $`E`$ 条件独立，所以

```math
\mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D),
```

因为凸组合不会超过其中最大的一项。

**(b)** 由 tower property，

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

由 (a)，每个括号都非负，所以 $`\mathcal R_{\rm info}=0`$ 当且仅当每个括号都为零。固定这样的 $`D`$，取 $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$，则

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

这是一组非负项之和。

- ($`\Leftarrow`$) 若 $`\pi^\circ`$ 属于交集 (2.2)，则 $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$，括号为零。
- ($`\Rightarrow`$) 若括号为零，则每个 $`\mu(E\mid D)>0`$ 的求和项都为零。于是 $`\pi_D`$ 在每个后验概率为正的环境中都最优，属于交集。$`\square`$

## 相关工作

- (b) 中的括号 $`\mathbb E[J_E^\Pi\mid D]-B(D)`$ 是给定 $`D`$ 时关于 $`E`$ 的完全信息的期望价值 ([Howard, 1966](#ref-howard1966information))；证明表明，它为零当且仅当某个后验最优策略在每个后验概率为正的环境中都最优。
