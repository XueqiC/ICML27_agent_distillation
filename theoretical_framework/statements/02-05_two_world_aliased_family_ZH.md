# Definition 2.5 · 两世界混叠族

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](02-05_two_world_aliased_family_ZH.md) · [English](02-05_two_world_aliased_family.md)

**一段话.** 这个定义构造一个最小的族，在其中完美 teacher 的示范也不够。teacher 在两个世界里都最优，而且行为完全相同，但混叠学生的最优动作在两个世界里不同，所以学生只能靠自己行动才能弄清楚。

**本页用到的设定.**

- [Definition 1.1](../README_ZH.md#s1-1)：部署策略的值 $`J(\pi)\in[0,1]`$；[Definition 1.2](../README_ZH.md#s1-2)：示范与后续实验。
- *混叠学生：* 学生只有一个上下文，所以无论隐藏状态是什么，它都用同一个动作分布。

<a id="s2-5"></a>

**Definition 2.5（两世界混叠族）.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定世界。差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。
- 学生只有一个混叠的上下文，看不到 $`s`$，做一次决策：以概率 $`p`$ 选动作 $`A`$，否则选 $`B`$。
- teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$。
- 成功概率：

| 状态 | $`A`$ 的成功概率 | $`B`$ 的成功概率 |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

考虑两种接口。

- **被动接口.** 学习器只能看到 teacher 示范（可能附带揭示 $`s`$）以及对它们的随机后处理。
- **查询接口.** 此外，学习器可以 reset 到 $`s=1`$，执行 $`A`$，观察 Bernoulli 结果。查询可以自适应地选择。查询次数 $`N`$ 是满足 $`\mathbb E_vN<\infty`$ 的停时。学习器的算法与私有随机性在两个世界中相同。

记二元 KL 散度为 $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$。

*这一族中的值.* 对 $`s`$ 取平均，

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big]=\tfrac34+v\Delta(2p-1).
```

- $`J_v`$ 关于 $`p`$ 线性，所以 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- teacher 总是选成功概率为 1 的动作。其余状态–动作的成功概率都等于 $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$，因为 $`2\Delta\le\tfrac14`$。所以 teacher 在两个世界中都严格最优。

**说了什么.**

- 两个世界只在"teacher 不用某个动作的那个状态里，这个动作有多好"上不同。
- teacher 从不暴露这个差别，因为它总是选必定成功的动作。

**直觉.**

- teacher 的完美行为恰好藏住了学生需要的事实：对一个看不到状态的学生来说，两个不完美的动作哪个更好。

**在框架中的作用.**

- 它是 [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)（被动不可能、查询复杂度）与 [Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7)（编译代价）的设定。

**用到.**

- 无

**被用于.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6): 被动不可能与查询复杂度.
- [Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7): 编译所获事实的代价.

## 相关工作

- 这个族是 Le Cam 方法意义下的两点构造：被动数据分不开的两个世界，最优决策却不同 ([Le Cam, 1973](#ref-lecam1973convergence))。
- teacher 依据学生观察不到的信息行动，模仿时这部分信息被边缘化，这就是 [Weihs et al., 2021](#ref-weihs2021bridging) 所说的 *imitation gap*。这个族是它的一个最小实例，缺失的事实只是一个符号。
