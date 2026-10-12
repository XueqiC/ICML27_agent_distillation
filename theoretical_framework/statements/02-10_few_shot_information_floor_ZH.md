# Theorem 2.10 · 少样本信息下限与硬预算下的最优查询

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-10_few_shot_information_floor_ZH.md) · [证明](../proofs/02-10_few_shot_information_floor_ZH.md) · [实验](../experiments/02-10_few_shot_information_floor_ZH.md) · [English](02-10_few_shot_information_floor.md)

**一段话.** $`K`$ 条示范之后还有多少未知？硬查询预算应该怎么花？在隐藏比特模型中，这个定理给出 $`K`$ 条示范之后精确的信息下限，并说明最优查询是先查权重最大的未见事实，根据查到的值调整查询没有收益。

**本页用到的设定.**

- 未知比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j`$ 出现（$`\sum_jw_j=1`$），其正确决策为 $`\Theta_j`$；学生能学任何"上下文到动作"的表，所以没有表示误差。
- $`K`$ 条示范中的每一条揭示一个随机的比特集合 $`S_k`$ 及其值；这些集合独立同分布，与比特独立，$`q_j=\Pr(j\in S)`$ 是一条示范揭示比特 $`j`$ 的概率。
- 示范之后，硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特；选择可以依赖于此前看到的一切。
- $`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。$`R_{\rm info}`$ 表示可达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中的信息项）。
- 完整定义：[Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8)。

<a id="s2-10"></a>

**Theorem 2.10（少样本信息下限与硬预算下的最优查询）.** 在 [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8) 的模型中：

**(a) 信息下限.** 不查询时，$`K`$ 条轨迹之后可达到的最小期望 regret 为

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

约定 $`(1-q)^0=1`$。再多一条轨迹的边际价值为

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) 最优查询.** 把未揭示比特的权重排序为 $`w_{(1)}\ge\dots\ge w_{(M_K)}`$，则

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

并且查询权重最大的 $`n`$ 个未揭示比特是最优的；根据揭示出的值调整查询没有帮助。在均匀情形（$`w_j=1/d`$，每条轨迹揭示一个均匀抽取的上下文）下，

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

一般有 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$，且不等号可以严格。

**说了什么.**

- $`K`$ 条示范之后，信息下限为 $`\tfrac12\sum_jw_j(1-q_j)^K`$：没有被任何示范揭示的事实的期望权重的一半。
- 硬预算为 $`n`$ 次查询时，先查权重最大的未见事实是最优的，按揭示出的值调整查询没有帮助。期望必须留在正部外面。

**直觉.**

- 每条示范以一定概率揭示一个事实；没揭示的只能去查，而且先查最值钱的缺口。

**在框架中的作用.**

- 它把 $`K`$ 写进理论，是 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 中查询分配与 [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) 中获取步骤的模型情形。

**推导要点.**

- *(a)* 任何动作猜对一个未揭示比特的概率都是一半（[Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9)），已揭示的比特可以照抄，所以条件 regret 是未见比特权重的一半。比特 $`j`$ 在 $`K`$ 条独立示范中都没被揭示的概率是 $`(1-q_j)^K`$。
- *(b)* 对剩余预算归纳。一次查询之后，由 Lemma 2.9，剩下的问题与查到的值无关，所以按值分支没有收益，每一步贪心地查权重最大的未见比特就是最优的。$`x\mapsto(x-n)_+`$ 是凸函数，由 Jensen 不等式，两个表达式有大小关系，在小例子中是严格的。

**例示.** $`d=8`$ 个等权上下文、每条示范均匀地揭示其中一个时，下限为 $`\tfrac12(7/8)^K`$：一个事实以 $`(7/8)^K`$ 的概率一直躲过示范；一次定向查询必定消除一个未见事实，而多一条示范只以 $`1/8`$ 的概率消除每个未见事实。

**说明.**

- **replay 的价值为零.** 重新展示已揭示的标签不改变 $`U_K`$，因而不改变 $`R_{\rm info}`$。这是 [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4) 在这个模型中的特例。replay 仍可能降低学习器自身的误差 $`\mathcal R_{\rm alg}`$。
- **关于 $`K`$ 没有普适速率.** $`q_j=0`$ 的坐标留下不可约的被动误差 $`\tfrac12w_j`$，很小的 $`q_j>0`$ 会让收敛任意慢。任何关于 $`K`$ 的速率都需要对 $`(q_j)`$ 的 coverage 假设。
- **每个假设的作用.**
  - 比特独立使查询之间互不影响。比特相关时，一次查询会提供关于另一个比特的信息，贪心结论随之改变。
  - 硬预算必不可少：期望预算约束允许在不同 transcript 之间调配预算。
  - 不泄露的揭示机制必不可少：若揭示集合依赖于比特本身，它就会泄露信息。
- **是下限，不是实际风险.** $`R_{\rm info}(K)`$ 是 Bayes 意义的信息下限。实际学习器的风险可以更高；超出的部分是 [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中它的算法项 $`\mathcal R_{\rm alg}`$。

**用到.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2): 信息项.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4): replay 不增加信息.
- [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8): 隐藏比特模型.
- [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9): 未揭示比特保持均匀.

**被用于.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10): 学习器相对的 coverage 模型.
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2): 先分配查询再编译.
