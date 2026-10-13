# Proposition 9.1 · 耦合宽度

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-01_coupling_width_ZH.md) · [证明](../proofs/09-01_coupling_width_ZH.md) · [实验](../experiments/09-01_coupling_width_ZH.md) · [English](09-01_coupling_width.md)

**一段话概括.** 训练数据是按 packet 消耗的：一次训练更新同时看到若干个单元。设单元的内容固定，即填入 packet 每个槽位的边缘分布固定，只改变单元怎样配对成 packet。这会改变学生学到的东西吗？本命题给出精确回答：重组永远无影响，当且仅当对 packet 的 response 是各单元贡献之和；否则可能的变化以 response 离可加有多远为上界。它还给出一个严格的零假设：在同一参数处走一步 SGD，与分组完全无关，连续几步则不然。这把"packet 的组织方式对这个学习器有没有影响"变成一个可测、并带有精确零假设的量。

**本页用到的设定与记号.**

- *单元与槽位.* $`\mathcal Y_1=\lbrace1,\dots,m\rbrace`$ 与 $`\mathcal Y_2=\lbrace1,\dots,n\rbrace`$ 是有效反馈单元的有限集合，例如填入训练 packet 两个槽位的单元。除非另有说明，每一对 $`(i,j)`$ 都是有效的 packet。
- *边缘分布.* $`\mu\in\Delta(\mathcal Y_1)`$ 与 $`\nu\in\Delta(\mathcal Y_2)`$ 是两个槽位的边缘分布，假设严格为正；若有零分量，下文所有结论在正支撑的乘积上成立。
- *耦合.* 耦合多面体（coupling polytope）为 $`\mathcal C(\mu,\nu)=\lbrace\Gamma\in\mathbb R^{m\times n}_{\ge0}:\Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace`$：在两个槽位的边缘分布（因而每个单元的内容）都固定的前提下，所有可能的配对方式。
- *packet response.* 在完整学习器状态 $`z`$ 上（[Definition 1.3](../README_ZH.md#s1-3)），$`H_z(i,j)`$ 是对联合 packet $`(i,j)`$ 执行实际更新、再走完参考尾段之后最终任务回报的期望。它是以这个 packet 为消息的一轮边界 response（[Definition 1.4](../README_ZH.md#s1-4)、[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)）。配对 $`\Gamma`$ 的期望学习价值为 $`\langle\Gamma,H_z\rangle=\sum_{i,j}\Gamma_{ij}H_z(i,j)`$。
- *耦合宽度.* $`w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle`$：仅靠重新安排哪些单元打包在一起，所能造成的期望学习价值的最大变化。
- *可加表.* $`H=a\mathbf 1^\top+\mathbf 1b^\top`$，即 $`H(i,j)=a_i+b_j`$。

<a id="s9-1"></a>

**Proposition 9.1（耦合宽度）.** 在上述设定下：

- **(a) 宽度为零.** $`w(H_z)=0`$ 当且仅当存在 $`a\in\mathbb R^m`$、$`b\in\mathbb R^n`$，使 $`H_z(i,j)=a_i+b_j`$。
- **(b) 用可加残差控制.** 对所有 $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$，

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty .
```

- (b) 中的不等式可以严格成立。
- **(c) 受限配对.** 若禁止某些配对，可行耦合构成 $`\mathcal C(\mu,\nu)`$ 的一个真面，此时可行宽度为零并不推出可加。
- **(d) 严格可加零假设.** 在同一参数处、对所有 packet 的并集、以相同的逐样本权重做一步 SGD，结果与样本怎样分组成 packet 无关。连续两步及以上一般不再不变；对二次损失，差别是 $`\eta^2`$ 阶：当 packet $`P_1`$ 先于 $`P_2`$ 时，结果中依赖分组的部分是 $`\eta^2H_{P_2}g_{P_1}(\theta_0)`$。

**说了什么.**

- 在边缘分布固定的前提下重组单元，只能通过 packet response 的不可加部分改变学习结果。
- 可能的变化至多是 response 到可加表的 sup 范数距离的两倍。
- 在同一参数处走一步是精确的零假设；一次更新的结果进入下一次更新之后，分组才有真实效应。

**直觉.** 如果学习器对每个单元"各看各的"，分组就无关紧要。分组只通过单元之间的交互起作用；而交互出现在第二个 packet 的梯度是在第一个 packet 已经移动了参数之后才计算的时候。

**在框架中的作用.**

- 它是一个带严格零假设的机制检验：任何测到的单步重组效应，都说明实现或随机性有差异。
- 它是方法中候选反馈律由其发出的 packet、而不是一袋单元来刻画的原因（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 3 步；[Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)，Rule 2）。
- 这一结果是耦合上的线性代数，不主张新的对偶。

**推导要点.**

- (a) 可加表与每个耦合的内积都等于 $`\mu^\top a+\nu^\top b`$。反过来，乘积耦合 $`\mu\nu^\top`$ 是内点，所以每个矩形移动 $`e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$ 在两个方向上都可行；宽度为零迫使 $`H_z`$ 的每个矩形和都为零，这就是可加性。
- (b) 两个耦合之差的行和与列和都为零，因此与可加表正交，且其 $`\ell_1`$ 范数至多为 2；由 Hölder 不等式得到上界。
- (c) 禁止一个配对后，这个配对的指示表对每个可行耦合都不可见，尽管它不可加。
- (d) 一步更新把同一参数处的逐样本梯度相加，与划分无关。两步时，第二个 packet 的梯度在第一次更新之后才计算，由此产生交叉项 $`\eta^2H_{P_2}g_{P_1}`$。

**示例.** 取 $`\mu=\nu=(0.9,0.1)`$、$`H=e_{11}`$，每个耦合都有 $`\Gamma_{11}=x\in[0.8,0.9]`$，所以宽度为 $`0.1`$；最佳可加近似留下大小为 $`1/4`$ 的棋盘形残差，所以 (b) 的上界是 $`0.5`$。上界成立，但很松。

**用到.**

- [Definition 1.4](../README_ZH.md#s1-4) 与 [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)：一轮边界 response，这里消息是一个 packet。
- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)：与不受限分布族的对比，那里相应的上界是等式。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 3 步：候选是 packet 律。
- [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)，Rule 2：packet 的组织方式是候选的一部分。

## Related work

- coupling 多面体 $`\mathcal C(\mu,\nu)`$ 就是线性规划中运输问题的可行集（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)）；(a)–(c) 是这个多面体上的线性代数。
- (d) 中的交叉项 $`\eta^2H_{P_2}g_{P_1}`$，是在前一次更新之后再计算梯度带来的二阶效应，也正是 SGD 的 stochastic modified equations 里漂移修正项的来源（[Li et al., 2019](#ref-li2019stochastic)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.

<a id="ref-li2019stochastic"></a>

**Qianxiao Li, Cheng Tai, Weinan E (2019).** Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research* 20, no. 40, 1–47. [jmlr.org/papers/v20/17-526](https://jmlr.org/papers/v20/17-526.html).
