# Theorem 2.6 · 被动不可能，主动可达

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-06_passive_impossibility_active_attainability_ZH.md) · [证明](../proofs/02-06_passive_impossibility_active_attainability_ZH.md) · [实验](../experiments/02-06_passive_impossibility_active_attainability_ZH.md) · [English](02-06_passive_impossibility_active_attainability.md)

**一段话.** 完美的示范也可能不够吗？在两世界族中，teacher 是最优的，它的示范在两个世界里分布相同，但混叠学生的最优动作相反。这个定理说明：只看示范的学习器总会在某一个世界里至少损失 $`\Delta`$；reset 查询可以补上这个缺口，所需查询次数为 $`\Delta^{-2}\log(1/\delta)`$ 量级，而这个量级也是必要的。

**本页用到的设定.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一；差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$，两者都以概率 1 成功，所以 teacher 在两个世界中都最优。
- 学生是混叠的（看不到 $`s`$），以概率 $`p`$ 选 $`A`$。另外两个状态–动作对的成功概率为 $`\tfrac12\mp2v\Delta`$，所以学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最好值为 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- *被动接口：* 学习器看到 teacher 示范（可能附带 $`s`$）及对它们的后处理。*查询接口：* 它还可以自适应地 reset 到 $`s=1`$、执行 $`A`$，观察 Bernoulli $`(\tfrac12+2v\Delta)`$ 结果，查询次数是停时 $`N`$。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度。
- 完整定义：[Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5)。

<a id="s2-6"></a>

**Theorem 2.6（被动不可能，主动可达）.** 在 [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5) 的族中：

- **(a) 被动不可能.** 在被动接口下，每个学习器都满足 $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$。
- **(b) 主动下界.** 在查询接口下，设 $`0<\delta<1/2`$，并设在每个世界中，学习器以至少 $`1-\delta`$ 的概率输出 regret 不超过 $`\Delta/2`$ 的学生。则对每个 $`v`$，

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) 主动上界.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 次独立查询足以在每个世界中以至少 $`1-\delta`$ 的概率输出最优端点 $`p\in\lbrace 0,1\rbrace`$。

**说了什么.**

- 被动学习器无论怎样处理示范，都会在某一个世界里至少损失 $`\Delta`$。
- 有 reset 查询时，要在两个世界中都以概率 $`1-\delta`$ 输出近似最优的学生，大约 $`\log(1/\delta)/\Delta^2`$ 次查询既必要又充分（差一个常数因子）。

**直觉.**

- 完美的示范展示的是 teacher 做了什么，而不是能力不同的学生应该怎么做；学生得自己去试。

**在框架中的作用.**

- 它说明方法为什么需要主动获取这一步，以及示范质量为什么不等于信息充分。它的 change of measure 论证在 [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11) 中再次使用，被动部分由 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7) 推广。

**推导要点.**

- (a) 在两个世界中，一条示范都以各一半的概率是"在 $`s=0`$ 选 $`A`$ 并成功"或"在 $`s=1`$ 选 $`B`$ 并成功"，所以示范的分布、进而学习器输出的分布都与 $`v`$ 无关。值关于 $`p`$ 是仿射的；两个 regret 分别为 $`2\Delta(1-m)`$ 与 $`2\Delta m`$（$`m`$ 是平均输出），最大值不小于平均值 $`\Delta`$。
- (b) 把"在两个世界中都 $`\Delta/2`$ 最优"化为对事件 $`\lbrace\widehat p>1/2\rbrace`$ 的二元检验。对停时用 Wald 恒等式，整条 transcript 的 KL 散度等于期望查询次数乘以单次查询的 KL；data processing 使它至少为 $`\mathrm{kl}(1-\delta,\delta)`$。常数 64 来自 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$。
- (c) 对独立的查询结果取平均，用单侧 Hoeffding 不等式。

**说明.**

- **完美示范也可能信息不足.** teacher 在两个世界中都以概率 1 成功，但对每个被动学习器都有 $`\mathcal R_{\rm info}\ge\Delta`$（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）。缺失的事实是 $`v`$ 的符号，即使示范揭示了 $`s`$，它也不在示范里：示范展示的是 teacher 做了什么，而不是能力不同的学生应该怎么做。
- **紧性.** [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability_ZH.md#c2-3) 的精确计算表明，调好参数的 SPRT 在部分设定上恰好达到 (2.6)，所以一般而言它的常数不能改进。
- **用到的假设.** $`\Delta`$ 已知用于 (c) 的停止规则。(b) 需要声明的查询接口、与世界无关的算法以及 $`\mathbb E_vN<\infty`$。论证是标准的 change of measure，作用在学生–teacher 的信息错配上；它不是新的检验不等式。

**用到.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2): regret 的信息项.
- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality_ZH.md#s2-3): 不可区分世界的对冲值.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4): 后处理不增加信息.
- [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5): 两世界族.

**被用于.**

- [Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7): 编译获取到的符号.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10): K 条示范后的信息下限.
- [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11): 同样的 change of measure 论证.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7): 学习器相对的混叠界.
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3): 先获取再编译.

## 相关工作

- (a) 是 imitation gap 的一个精确实例：teacher 用到学生看不到的信息，模仿只能把它边缘化 ([Weihs et al., 2021](#ref-weihs2021bridging))。Weihs et al. 通过混入基于奖励的强化学习来弥合这个 gap；这里学生自己的实验是 reset 查询，其次数在常数倍之内被确定。
- (b) 是 best-arm identification 中的 change of measure 论证 ([Kaufmann et al., 2016](#ref-kaufmann2016complexity))，停时部分用 Wald 恒等式 ([Wald, 1944](#ref-wald1944cumulative))。在部分设定上达到 (2.6) 的序贯概率比检验（SPRT）来自 Wald ([Wald, 1945](#ref-wald1945sequential))。
- (c) 是有界独立变量的 Hoeffding 不等式 ([Hoeffding, 1963](#ref-hoeffding1963probability))。
