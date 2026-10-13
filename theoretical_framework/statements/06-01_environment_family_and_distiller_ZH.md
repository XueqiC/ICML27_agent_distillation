# Definition 6.1 · 环境族、数据实验、蒸馏器、被审计状态

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-01_environment_family_and_distiller_ZH.md) · [English](06-01_environment_family_and_distiller.md)

**一段话概括.** 这个定义固定了 Theorem A 的对象：由未知参数刻画的环境族，蒸馏器能得到的数据（示范与查询），什么算作蒸馏器，以及在哪些学习器状态上审计反馈的效应。§6 的每个结果都是针对这些对象陈述的。

**本页用到的设定与记号.**

- **complete learner state** $`z=(\theta,o)`$：参数与 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；**feedback law** 发出学习器要消费的消息，实际更新把状态与消息映射到下一个状态。
- **信息与蒸馏器**（[Definition 1.2](../README_ZH.md#s1-2)）：公共初始信息（预训练学生、已知规则、$`K`$ 条可信示范）与允许实验的 transcript；蒸馏器连同私有随机性，把它们映射为一个反馈协议。

<a id="s6-1"></a>

**Definition 6.1（环境族、数据实验、蒸馏器、被审计状态）.**

- **环境族.** $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知。
- **oracle 反馈.** $`P^*_\theta=(P^*_{\theta,t})_t`$ 是知道 $`\theta`$ 时应给的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据实验.**
  - $`K`$ 条示范 $`O_1,\dots,O_K`$ 独立同分布于 $`P^D_\theta`$。
  - $`n`$ 次查询可以自适应：查询点 $`X_i\in\mathcal X_Q`$（查询空间）由历史和私有随机性决定，结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。
  - 整条 transcript $`D`$ 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 任何从 $`(D,U)`$ 到 feedback law $`Q=(Q_t)_t`$ 的可测映射，其中私有随机性 $`U`$ 的分布与 $`\theta`$ 无关。
  - replay、改写、合成都是蒸馏器，只要它们不再获取依赖于 $`\theta`$ 的信息。
  - teacher 模型的预训练知识属于公共初始信息，对所有 $`\theta`$ 相同；新调用 teacher 去观察一个新状态，算作一次查询。
- **被审计状态.** 固定的 complete learner state $`z_1,\dots,z_T`$（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在若干检查点到达的状态。它们不依赖于 $`\theta`$。

**是什么.**

- $`\theta`$ 刻画候选环境之间所有不同之处；teacher 模型的预训练知识对每个 $`\theta`$ 都相同。
- 蒸馏器可以对 transcript 做任何事（replay、改写、合成），但除了通过 transcript，无法得知关于 $`\theta`$ 的任何信息。
- 观察新状态的一次 teacher 调用是一次查询，因此属于 transcript。

**在框架中的作用.**

- 它把"replay、改写、合成不能创造信息"变成关于 transcript 的所有可测映射的陈述，[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3) 再对它定量。

**用到.**

- [Definition 1.2](../README_ZH.md#s1-2)：信息与蒸馏器。
- [Definition 1.3](../README_ZH.md#s1-3)：complete learner state 与平均增量。

**被用于.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：在这些对象上构造泛函。
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：对每个蒸馏器的下界。
- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：达到下界的 plug-in 蒸馏器。
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：混叠与 replay。
- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：状态覆盖模型。

**状态.** 定义。

## 相关工作

- 蒸馏器是 transcript 与独立随机性的可测映射，所以它的输出是数据实验在 Blackwell 意义下的 garbling；对任何决策问题，garbling 都不会比原实验更有信息（[Blackwell, 1953](#ref-blackwell1953equivalent)）。
- 把在新状态上新调用一次 teacher 算作查询、而不是免费的合成，与交互式模仿学习一致：在那里 expert 为学习器访问到的状态打标签（[Ross et al., 2011](#ref-ross2011reduction)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).
