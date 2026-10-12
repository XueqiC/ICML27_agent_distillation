# Proposition 2.7 · 编译代价

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-07_compilation_cost_ZH.md) · [证明](02-07_compilation_cost_ZH.md) · [实验](../experiments/02-07_compilation_cost_ZH.md) · [English](02-07_compilation_cost.md)

**本页证明什么.** Hoeffding 不等式控制编译标签的经验频率，union bound 把识别失败与编译失败合在一起。

**本页用到的设定.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一；差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$，两者都以概率 1 成功，所以 teacher 在两个世界中都最优。
- 学生是混叠的（看不到 $`s`$），以概率 $`p`$ 选 $`A`$。另外两个状态–动作对的成功概率为 $`\tfrac12\mp2v\Delta`$，所以学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最好值为 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- *被动接口：* 学习器看到 teacher 示范（可能附带 $`s`$）及对它们的后处理。*查询接口：* 它还可以自适应地 reset 到 $`s=1`$、执行 $`A`$，观察 Bernoulli $`(\tfrac12+2v\Delta)`$ 结果，查询次数是停时 $`N`$。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度。
- 完整定义：[Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5)。
- $`\widehat v`$ 是估计出的符号；频率学习器把 $`\widehat p`$ 设为 $`m`$ 个 Bernoulli $`(q)`$ 标签的经验频率。

**陈述（[Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7)）.** 在 [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5) 的族中，符号估计为 $`\widehat v`$ 之后：

- 若 $`\widehat v=+1`$，取目标 $`q=7/8`$，否则取 $`q=1/8`$；
- 生成 $`m`$ 个 Bernoulli $`(q)`$ 目标标签，令学生为它们的经验频率 $`\widehat p`$。

若 $`m\ge\lceil32\log(2/\delta_L)\rceil`$，则以至少 $`1-\delta_L`$ 的概率有 $`\lvert\widehat p-q\rvert\le1/8`$。若 $`\widehat v`$ 以至少 $`1-\delta`$ 的概率正确（[Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)），则编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率 regret 不超过 $`\Delta/2`$。

**证明.**

双侧 Hoeffding 不等式给出

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

只要 $`m\ge32\log(2/\delta_L)`$。设 $`\widehat v=v=+1`$ 且 $`\lvert\widehat p-q\rvert\le\tfrac18`$，则 $`\widehat p\ge\tfrac34`$，regret 为 $`2\Delta(1-\widehat p)\le\Delta/2`$。$`v=-1`$ 时对称，此时 $`\widehat p\le\tfrac14`$。对识别失败（$`\le\delta`$）与编译失败（$`\le\delta_L`$）取 union bound 即得所述概率。$`\square`$

## 相关工作

- 充分性的界用的是有界独立变量的 Hoeffding 不等式 ([Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)).

## 参考文献

- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)
