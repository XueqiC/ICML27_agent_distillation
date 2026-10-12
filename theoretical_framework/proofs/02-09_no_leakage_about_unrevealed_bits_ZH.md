# Lemma 2.9 · 示范不泄露未揭示比特

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [证明](02-09_no_leakage_about_unrevealed_bits_ZH.md) · [实验](../experiments/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [English](02-09_no_leakage_about_unrevealed_bits.md)

**本页证明什么.** 揭示集合、比特与 transcript 其余内容的联合分布可以分解，未揭示的比特只通过常数 $`2^{-d}`$ 出现；因此给定 transcript 时它们的条件分布是均匀的。

**本页用到的设定.**

- 未知比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j`$ 出现（$`\sum_jw_j=1`$），其正确决策为 $`\Theta_j`$；学生能学任何"上下文到动作"的表，所以没有表示误差。
- $`K`$ 条示范中的每一条揭示一个随机的比特集合 $`S_k`$ 及其值；这些集合独立同分布，与比特独立，$`q_j=\Pr(j\in S)`$ 是一条示范揭示比特 $`j`$ 的概率。
- 示范之后，硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特；选择可以依赖于此前看到的一切。
- $`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。$`R_{\rm info}`$ 表示可达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中的信息项）。
- 完整定义：[Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8)。

**陈述（[Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9)）.** 在 [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8) 的模型中，给定完整 transcript $`T`$，未揭示的比特 $`(\Theta_j)_{j\in U_K}`$ 独立同分布、均匀，并且与 $`T`$ 以及学习器的私有随机性独立。

**证明.**

联合分布分解为

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

其中 $`X`$ 是 transcript 的其余内容。固定观察到的 $`(S_{1:K},\Theta_{\cup S_k},X)`$，唯一依赖于 $`\Theta_{U_K}`$ 的因子是常数 $`2^{-d}`$。所以给定 $`T`$ 时 $`\Theta_{U_K}`$ 在 $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ 上均匀。学习器的随机性与其余一切独立。$`\square`$
