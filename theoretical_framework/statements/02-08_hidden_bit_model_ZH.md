# Definition 2.8 · 隐藏比特模型

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-08_hidden_bit_model_ZH.md) · [English](02-08_hidden_bit_model.md)

**一段话.** 这个定义把示范数 $`K`$ 显式写进模型。每条示范以一个与事实本身无关的机制揭示学生需要的部分事实；硬预算内的查询可以揭示更多。模型足够简单，$`K`$ 条示范之后剩下的信息、查询的最优用法都能精确算出。

**本页用到的设定.**

- [Definition 1.2](../README_ZH.md#s1-2)：$`K`$ 条可信示范与后续实验（查询）。
- 策略的效用是按部署权重加权的、在每个上下文选对动作的概率。
- 信息项 $`R_{\rm info}`$ 是任何学习器由数据能达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）。

<a id="s2-8"></a>

**Definition 2.8（隐藏比特模型）.** 所有概率都以一个固定的、声明过的预训练信息状态 $`I_{\rm pre}`$ 为条件。额外的 teacher 反馈从不计入 $`K`$ 条示范。

- **比特与部署.** 未知比特 $`\Theta=(\Theta_1,\dots,\Theta_d)`$ 相互独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j\ge0`$ 出现，$`\sum_jw_j=1`$；上下文 $`j`$ 处的正确决策是 $`\Theta_j`$。学生类包含所有"上下文到动作"的表，所以 $`\mathcal R_{\rm rep}=0`$。效用为 $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$，知情最优为 1。
- **不泄露的示范.** 第 $`k`$ 条轨迹揭示一个随机子集 $`S_k\subseteq[d]`$ 及其标签 $`\Theta_{S_k}`$。$`S_1,\dots,S_K`$ 独立同分布于 $`L`$，且与 $`\Theta`$ 独立；同一条轨迹内揭示的坐标之间可以相关。transcript 的其余内容的条件分布只依赖于已揭示的标签和独立的随机性。记 $`q_j=\Pr(j\in S)`$。
- **硬查询预算.** 示范之后，学习器至多做 $`n`$ 次无噪声、单位成本的查询，每次揭示一个选定的比特。选择可以依赖于此前观察到的一切。$`n`$ 是每条 transcript 的硬预算，不是期望预算。
- $`U_K`$ 是 $`K`$ 条轨迹都没有揭示的比特集合，$`M_K=\lvert U_K\rvert`$。

**说了什么.**

- 事实是独立的均匀比特，每个上下文一个；示范揭示它们的随机子集；查询在硬预算 $`n`$ 内每次揭示一个选定的比特。

**直觉.**

- 一条示范是事实的一个随机样本；它漏掉的事实完全未知。

**在框架中的作用.**

- 它是 [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9) 与 [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10) 的设定，也是 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 中学习器相对 coverage 模型的模型情形。

**用到.**

- 无

**被用于.**

- [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9): 未揭示比特保持均匀.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10): 信息下限与最优查询.

## 相关工作

- 揭示集合与比特独立，所以未揭示的标签在 [Rubin, 1976](#ref-rubin1976inference) 的意义下是 *missing completely at random*。[Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9) 把这一假设变成"不泄露"；若揭示集合依据比特来选，缺失本身就会携带信息。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-rubin1976inference"></a>

**Donald B. Rubin (1976).** Inference and missing data. *Biometrika* 63, no. 3, 581–592. DOI [10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581).
