# Proposition 2.7 · 编译代价

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-07_compilation_cost_ZH.md) · [证明](../proofs/02-07_compilation_cost_ZH.md) · [实验](../experiments/02-07_compilation_cost_ZH.md) · [English](02-07_compilation_cost.md)

**一段话.** 弄清该做什么与教会学生去做，是两件不同的事。两世界族的符号估计出来之后，还要把学生训练到正确的动作上；这个命题说明，这一编译步骤所需的合成标签数只依赖于目标置信度，与差距 $`\Delta`$ 无关。

**本页用到的设定.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一；差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$，两者都以概率 1 成功，所以 teacher 在两个世界中都最优。
- 学生是混叠的（看不到 $`s`$），以概率 $`p`$ 选 $`A`$。另外两个状态–动作对的成功概率为 $`\tfrac12\mp2v\Delta`$，所以学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最好值为 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- *被动接口：* 学习器看到 teacher 示范（可能附带 $`s`$）及对它们的后处理。*查询接口：* 它还可以自适应地 reset 到 $`s=1`$、执行 $`A`$，观察 Bernoulli $`(\tfrac12+2v\Delta)`$ 结果，查询次数是停时 $`N`$。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度。
- 完整定义：[Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5)。
- $`\widehat v`$ 是学习器对符号的估计，例如来自 [Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6) 的查询检验。*频率学习器*把学生的 $`\widehat p`$ 设为 $`m`$ 个训练标签中 $`A`$ 的经验频率；$`\delta_L`$ 是允许的编译失败概率。

<a id="s2-7"></a>

**Proposition 2.7（编译代价）.** 在 [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5) 的族中，符号估计为 $`\widehat v`$ 之后：

- 若 $`\widehat v=+1`$，取目标 $`q=7/8`$，否则取 $`q=1/8`$；
- 生成 $`m`$ 个 Bernoulli $`(q)`$ 目标标签，令学生为它们的经验频率 $`\widehat p`$。

若 $`m\ge\lceil32\log(2/\delta_L)\rceil`$，则以至少 $`1-\delta_L`$ 的概率有 $`\lvert\widehat p-q\rvert\le1/8`$。若 $`\widehat v`$ 以至少 $`1-\delta`$ 的概率正确（[Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)），则编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率 regret 不超过 $`\Delta/2`$。

**说了什么.**

- 估计出符号之后，朝估计的最优方向抽取 $`O(\log(1/\delta_L))`$ 个合成标签，就能以至少 $`1-\delta_L`$ 的概率把频率学习器训练到离目标 $`\tfrac18`$ 以内，与 $`\Delta`$ 无关。
- 加上正确的符号，就以至少 $`1-\delta-\delta_L`$ 的概率得到 regret 不超过 $`\Delta/2`$ 的学生。

**直觉.**

- 获取事实要花真实的环境查询，随 $`\Delta^{-2}`$ 增长；编译它只花便宜的合成标签，不随 $`1/\Delta`$ 增长。

**在框架中的作用.**

- 在方法的记账中（[Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)），它把获取预算（真实查询）与编译预算（合成标签）分开。

**推导要点.**

- 编译标签的经验频率由双侧 Hoeffding 不等式控制，再对识别失败与编译失败取 union bound。标签数与 $`\Delta`$ 无关。

**例示.** 识别消耗的是真实的环境查询，量级 $`\Delta^{-2}\log(1/\delta)`$；编译消耗的是合成标签，量级 $`\log(1/\delta_L)`$。$`\Delta`$ 小时，预算几乎全部花在获取上，两者必须分开记账。

**说明.**

- **获取与编译是两项独立的代价.**
  - 识别需要 $`\Theta(\Delta^{-2}\log(1/\delta))`$ 次真实环境查询；[Theorem 2.6(b)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)、(c) 的下界与上界在常数意义下吻合。
  - 编译还需要 $`O(\log(1/\delta_L))`$ 个合成目标标签，与 $`\Delta`$ 无关。这些标签是编译出来的，不是新的 teacher 执行，要单独计数。

**用到.**

- [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5): 两世界族.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): 用查询识别符号.

**被用于.**

- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2): 获取与编译分开记账.

## 相关工作

- 识别与编译两步都由有界独立变量的 Hoeffding 不等式控制 ([Hoeffding, 1963](#ref-hoeffding1963probability))。编译只需要固定精度 $`\tfrac18`$ 下的双边形式，所以标签数与 $`\Delta`$ 无关。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
