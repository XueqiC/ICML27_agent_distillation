# Lemma 2.9 · 示范不泄露未揭示比特

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [证明](../proofs/02-09_no_leakage_about_unrevealed_bits_ZH.md) · [实验](02-09_no_leakage_about_unrevealed_bits_ZH.md) · [English](02-09_no_leakage_about_unrevealed_bits.md)

**本页检验什么.** 这个引理本身没有单独的数值陈述；[Check 2.4](../experiments/02-10_few_shot_information_floor_ZH.md#c2-4) 用穷举检验它的推论。在全部 100 个穷举情形中（其中 80 个揭示集合含多个相关比特），Bayes 最优风险都等于由引理推出的闭式，最优的按值自适应查询树也不比贪心查询更好（通过）。

**本页用到的设定.**

- 未知比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j`$ 出现（$`\sum_jw_j=1`$），其正确决策为 $`\Theta_j`$；学生能学任何"上下文到动作"的表，所以没有表示误差。
- $`K`$ 条示范中的每一条揭示一个随机的比特集合 $`S_k`$ 及其值；这些集合独立同分布，与比特独立，$`q_j=\Pr(j\in S)`$ 是一条示范揭示比特 $`j`$ 的概率。
- 示范之后，硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特；选择可以依赖于此前看到的一切。
- $`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。$`R_{\rm info}`$ 表示可达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中的信息项）。
- 完整定义：[Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8)。

**被检验的陈述（[Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9)）.** 在 [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8) 的模型中，给定完整 transcript $`T`$，未揭示的比特 $`(\Theta_j)_{j\in U_K}`$ 独立同分布、均匀，并且与 $`T`$ 以及学习器的私有随机性独立。

### [Check 2.4](../experiments/02-10_few_shot_information_floor_ZH.md#c2-4) · 隐藏比特的 coverage、硬查询预算与自适应性

- *对本引理检验什么.* 若给定 transcript 时未揭示的比特仍是独立的均匀硬币，则 (i) 每个学习器在一个未见比特上的风险恰为一半，所以 Bayes 风险是期望未见权重的一半；(ii) 按揭示出的值调整查询没有帮助。穷举后验若在任何未揭示比特上放了信息，两条都会被打破。
- *设置.* 100 个情形：$`d\le5`$ 个比特、$`K\le4`$ 条示范、$`n\le3`$ 次查询，随机有理权重，随机的位掩码揭示集合分布（允许同一条轨迹内的相关）；精确枚举全部 $`2^d`$ 个世界与所有有序揭示历史，按标签的查询树精确最小化。
- *结果.* Bayes 风险在 100/100 个情形中等于闭式；自适应最优在 100/100 个情形中等于贪心；80 个情形的揭示集合含多个坐标。
- *结论.* 通过。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5 --output theory/v015_checks/quick/sec2.json
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：`v015:V5` 的条目；代码 `theory/v015_checks/mathematics.py`（`check_V5`）。
