# Theorem 2.2 · regret 分解与决策歧义

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-02_regret_decomposition_decision_ambiguity_ZH.md) · [证明](../proofs/02-02_regret_decomposition_decision_ambiguity_ZH.md) · [实验](../experiments/02-02_regret_decomposition_decision_ambiguity_ZH.md) · [English](02-02_regret_decomposition_decision_ambiguity.md)

**一段话.** 在问学生学得好不好之前，先要问：数据到底能不能告诉学生该怎么做。这个定理把学生的期望 regret 精确拆成表示项、信息项和算法项，并刻画信息项何时为零：恰好当数据排除不了的那些环境有共同的最优策略时。

**本页用到的设定.**

- 有限的环境集合 $`\mathcal E`$，先验为 $`\mu`$，以公共初始信息 $`I_0`$（预训练模型、已知规则、$`K`$ 条可信示范）为条件。
- 有限策略类 $`\Pi\subseteq\Pi_{\rm ref}`$，取值 $`J_E(\pi)\in[0,1]`$：$`\Pi`$ 是学生能部署的策略，$`\Pi_{\rm ref}`$ 是参照类（例如 teacher 能执行的策略）。
- transcript $`D`$：学习器在 $`I_0`$ 之外观察到的一切，包括它可以做的每个实验；给定 $`E`$ 时其分布为 $`P(D\mid E)`$，$`\mu(E\mid D)`$ 是后验。
- 学习器：一个 kernel $`\lambda(\cdot\mid D)`$，输出策略 $`\widehat\pi\in\Pi`$；给定 $`D`$ 时，它的私有随机性与 $`E`$ 独立。
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ 与 $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ 是环境 $`E`$ 中的最好值；$`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 是 Bayes 值，即给定 $`D`$ 时 $`\Pi`$ 中的学生能达到的最好期望回报。
- 三项 regret $`\mathcal R_{\rm rep}`$、$`\mathcal R_{\rm info}`$、$`\mathcal R_{\rm alg}`$ 的定义见 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1)。

<a id="s2-2"></a>

**Theorem 2.2（regret 分解与决策歧义）.** 在 [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1) 下：

**(a) 精确分解.** 相对参照最优的期望 regret 为

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) 决策歧义.** $`\mathcal R_{\rm info}=0`$，当且仅当对每个满足 $`\Pr(D)>0`$ 的 $`D`$，

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

**说了什么.**

- 相对最好参照策略的期望 regret 是三个非负项之和：表示误差（学生类达不到参照最优）、信息误差（数据分不清环境）、算法误差（学习器没有用好数据）。
- 信息误差为零，当且仅当对每条概率为正的 transcript，所有后验概率为正的环境有共同的最优策略。

**直觉.**

- 信息误差是即使学习器完美也会损失的部分，因为数据让正确的决策悬而未决。更多数据只能降低它；更好的学习器只能降低算法误差。

**在框架中的作用.**

- 它把"信息不够"与"学得不好"分开。框架中每个信息论结果都在界定这些项中的某一项；[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6) 与 [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10) 在显式的族中算出信息项。

**推导要点.**

- 加减 $`\mathbb EJ_E^\Pi`$ 与 $`\mathbb E_DB(D)`$ 即得恒等式。
- 表示项非负，因为 $`\Pi\subseteq\Pi_{\rm ref}`$。信息项非负，因为对每个策略都有 $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$。算法项非负，因为给定 $`D`$ 时学习器的输出是策略的凸组合，不会超过其中最好的那个。
- 对 (b)，$`\mathcal R_{\rm info}`$ 是对 transcript 求和的一组非负括号 $`\sum_E\mu(E\mid D)(J_E^\Pi-J_E(\pi_D))`$；一个括号为零，恰好当后验最优策略在每个后验概率为正的环境中都最优。

**说明.**

- **每一项分离出什么.**
  - $`\mathcal R_{\rm rep}`$ 是任何信号都消除不了的部分：学生执行不了参照最优策略。
  - $`\mathcal R_{\rm info}`$ 是 transcript 无法确定一个共同最优决策时剩下的部分。
  - $`\mathcal R_{\rm alg}`$ 是学习器没有用好已有信息。因此某一个学习器的失败不构成信息下界；对固定的学习器，额外信息也未必有帮助，因为它只能降低 $`\mathcal R_{\rm info}`$。
- **等式只在 Bayes 意义下可加.** 对三项分别取最大（例如逐项取最坏情况）不保持 (2.1)。
- **范围.** 有限性保证最大值可取到、条件期望有定义；一般可测空间需要可测选择与可积性条件。这个分解是经典的决策论恒等式，它在这里的作用是把三种误差来源分开，框架后面分别处理它们。

**用到.**

- [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms_ZH.md#s2-1): Bayes 值与三项 regret.

**被用于.**

- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality_ZH.md#s2-3): 信息项的稳健版本.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4): 后处理会提高信息项.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): teacher 最优时的信息缺口.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10): K 条示范之后的信息项.
- [Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1): 表示之外的学习差距.
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3): 先获取再编译.

## 相关工作

- (b) 推导中的括号 $`\mathbb E[J_E^\Pi\mid D]-B(D)`$ 是给定 $`D`$ 时关于 $`E`$ 的完全信息的期望价值（expected value of perfect information）([Howard, 1966](#ref-howard1966information))。(b) 说的是：这一价值对每条 transcript 都为零，当且仅当 transcript 无法排除的环境有共同的最优策略。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-howard1966information"></a>

**Ronald A. Howard (1966).** Information value theory. *IEEE Transactions on Systems Science and Cybernetics* 2, no. 1, 22–26. DOI [10.1109/TSSC.1966.300074](https://doi.org/10.1109/TSSC.1966.300074).
