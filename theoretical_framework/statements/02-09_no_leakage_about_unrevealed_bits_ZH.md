# Lemma 2.9 · 示范不泄露未揭示比特

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-09_no_leakage_about_unrevealed_bits_ZH.md) · [证明](../proofs/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [实验](../experiments/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [English](02-09_no_leakage_about_unrevealed_bits.md)

**一段话.** 在隐藏比特模型中，示范对它们没揭示的事实有没有透露什么？这个引理说明没有：给定观察到的一切，未揭示的比特仍是独立的均匀硬币。它是精确信息下限与贪心查询最优性的关键一步。

**本页用到的设定.**

- 未知比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j`$ 出现（$`\sum_jw_j=1`$），其正确决策为 $`\Theta_j`$；学生能学任何"上下文到动作"的表，所以没有表示误差。
- $`K`$ 条示范中的每一条揭示一个随机的比特集合 $`S_k`$ 及其值；这些集合独立同分布，与比特独立，$`q_j=\Pr(j\in S)`$ 是一条示范揭示比特 $`j`$ 的概率。
- 示范之后，硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特；选择可以依赖于此前看到的一切。
- $`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。$`R_{\rm info}`$ 表示可达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中的信息项）。
- 完整定义：[Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8)。

<a id="s2-9"></a>

**Lemma 2.9（示范不泄露未揭示比特）.** 在 [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8) 的模型中，给定完整 transcript $`T`$，未揭示的比特 $`(\Theta_j)_{j\in U_K}`$ 独立同分布、均匀，并且与 $`T`$ 以及学习器的私有随机性独立。

**说了什么.**

- 给定完整 transcript，没有被任何示范揭示的比特相互独立、均匀，且与 transcript 和学习器的随机性独立。

**直觉.**

- 揭示机制与事实本身无关时，示范揭示了一部分事实，却不提供关于其余事实的证据。

**在框架中的作用.**

- 它使每个未揭示比特对任何学习器都是掷硬币，这给出 [Theorem 2.10(a)](../statements/02-10_few_shot_information_floor_ZH.md#s2-10) 的信息下限；它也使 [Theorem 2.10(b)](../statements/02-10_few_shot_information_floor_ZH.md#s2-10) 中按值自适应的查询没有用处。

**推导要点.**

- 揭示集合、比特与 transcript 其余内容的联合分布可以分解，唯一依赖于未揭示比特的因子是常数 $`2^{-d}`$。

**用到.**

- [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8): 隐藏比特模型.

**被用于.**

- [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10): 精确下限与贪心最优.
