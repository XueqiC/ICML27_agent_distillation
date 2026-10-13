# Remark 2.12 · 学习器相对的版本

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-12_learner_relative_versions_ZH.md) · [English](02-12_learner_relative_versions.md)

**一段话.** §2 度量的是学生策略的决策 regret。蒸馏关心的东西更具体：一个反馈律对一个声明的学习器造成的一阶后果。这个说明指向 §6 中的结果，它们把混叠界与示范–查询预算改写成学习器相对的形式。

**本页用到的设定.**

- $`\Psi`$ 是 [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2) 中的学习器相关泛函：在实际学习器看来，oracle 反馈对声明任务的一阶效应。
- $`e_D(h)`$ 与 $`e_Q(h)`$ 是单条示范与单次查询在方向 $`h`$ 上的 Fisher information，$`c`$ 是 $`\Psi`$ 的梯度，由学习器决定。

<a id="s2-12"></a>

**Remark 2.12（学习器相对的版本）.** §6 针对蒸馏真正关心的量重新表述本节的结果：一个反馈律对一个声明的学习器造成的一阶后果。

- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 推广了 [Theorem 2.6(a)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6) 与 [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)。若两个环境下示范的分布相同，而它们在学习器相关泛函 $`\Psi`$ 上不同，则任何蒸馏器（包括 replay、改写与合成）的最坏误差至少为 $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$，对每个 $`K`$ 都成立。
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8) 推广了 [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)。渐近地，对每个方向 $`h`$ 有 $`K\,e_D(h)+n\,e_Q(h)\ge(c^\top h)^2/(16\varepsilon^2)`$，其中 $`c`$ 由学习器决定。所以示范替代查询的比率取决于学习器，而不只取决于数据。
- 对象上的区别：本节度量学生策略的决策 regret；§6 度量反馈律相对于学习器更新的 fidelity。两者都说明，缺失的信息只能由新的实验补上。

**说了什么.**

- [Theorem 2.6(a)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6) 的被动不可能与 [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4) 的 data processing 结论，变成 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：若示范分不清两个在 $`\Psi`$ 上不同的环境，则任何蒸馏器（包括 replay、改写与合成）都不能做得比差值的一半更好，对任何 $`K`$ 都如此。
- [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11) 的预算变成 [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8) 中逐方向的预算，方向由学习器加权。

**在框架中的作用.**

- 它把 §2 的决策论结果与 §6 的学习器相对信息界（Theorem A）连接起来。

**用到.**

- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4): 后处理.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): 被动不可能.
- [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11): 共用预算.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7): 学习器相对的混叠界.
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8): 学习器相对的兑换率.

**被用于.**

- 无

## 相关工作

- 两个 learner-relative 版本都是 Le Cam 框架下的下界：[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 是两点界 ([Le Cam, 1973](#ref-lecam1973convergence))，[Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8) 是在每个方向 $`h`$ 上用 Fisher information 写出的局部界，与 local asymptotic minimax 理论中的形式相同 ([van der Vaart, 1998](#ref-vandervaart1998asymptotic))。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
