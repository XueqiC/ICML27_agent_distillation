# Proposition 4.11 · 静态充分不等于动态充分；有限时域的闭合误差

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-11_static_not_dynamic_closure_error_ZH.md) · [证明](../proofs/04-11_static_not_dynamic_closure_error_ZH.md) · [实验](../experiments/04-11_static_not_dynamic_closure_error_ZH.md) · [English](04-11_static_not_dynamic_closure_error.md)

**一段话.** 一组能精确预测当前部署得分的特征，仍可能预测不了再更新一步之后的得分。这个命题给出最小的例子：两个当前得分相同的状态，一次更新后分别得 0 分和 1 分。所以一个现在充分的压缩接口，还必须检查它在下一次更新下是否闭合。闭合只近似成立时，有限时域后的误差是传播开的闭合残差的显式和；不假设收缩，传播增益很大时要如实报告，而不是藏起来。

**本页用到的设定.**

- 完整学习状态、带 Markov kernel 的控制与测试空间同 [Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)。*特征映射*是有界可测的 $`\psi:\mathcal Z\to\mathbb R^d`$；对状态分布 $`\mu`$，$`q(\mu)=\mathbb E_\mu\psi`$。
- 若存在 $`c`$ 使 $`F=c^\top\psi`$，称 $`\psi`$ 对得分 $`F`$ *静态充分*；若存在矩阵 $`A`$ 使 $`K\psi=A\psi`$，称 $`\psi`$ 在 kernel $`K`$ 下*闭合*。
- (b) 的设定：一个固定的合法控制序列，kernel 为 $`K_0,\dots,K_{H-1}`$；矩阵 $`A_t`$；残差函数 $`r_t=K_t\psi-A_t\psi`$；得分分解 $`F=c^\top\psi+e_F`$；沿链的真实 predictive state $`q_t=\mathbb E\psi(Z_t)`$；从某个 $`\hat q_0`$ 出发的模型预测 $`\hat q_{t+1}=A_t\hat q_t`$。乘积记为 $`A_{s:u}=A_sA_{s-1}\cdots A_u`$，空乘积为单位阵。

<a id="s4-11"></a>

**Proposition 4.11（静态充分不等于动态充分；有限时域的闭合误差）.**

**(a) 静态充分但不闭合的接口.** 令 $`z=(s,b)\in\lbrace0,1\rbrace^2`$，$`F(z)=s`$，确定性更新 $`U(s,b)=(b,b)`$，kernel 为 $`Kf(s,b)=f(b,b)`$。特征 $`\psi=(\mathbf 1,s)`$ 对 $`F`$ 静态充分，但 $`KF=b\notin\mathrm{span}\lbrace\mathbf 1,s\rbrace`$。状态 $`(0,0)`$ 与 $`(0,1)`$ 有相同的 $`q`$ 与相同的当前得分，一次更新后得分分别为 $`0`$ 与 $`1`$。加入 $`b`$ 得到闭合的空间 $`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$。

**(b) 近似闭合下的有限时域误差.** 在上述设定下，精确地有

```math
q_H-\hat q_H=A_{H-1:0}\,(q_0-\hat q_0)+\sum_{t=0}^{H-1}A_{H-1:t+1}\,\mathbb E\,r_t(Z_t), \qquad(4.21)
```

并且对 $`\mathbb R^d`$ 上任意范数 $`\lVert\cdot\rVert`$ 及其对偶范数 $`\lVert\cdot\rVert_*`$，

```math
\big\lvert\mathbb EF(Z_H)-c^\top\hat q_H\big\rvert\le\mathbb E\lvert e_F(Z_H)\rvert+\lVert A_{H-1:0}^\top c\rVert_*\,\lVert q_0-\hat q_0\rVert+\sum_{t<H}\lVert A_{H-1:t+1}^\top c\rVert_*\,\mathbb E\lVert r_t(Z_t)\rVert. \qquad(4.22)
```

**说了什么.**

- 精确预测现在，并不使一组特征成为合格的状态：下一次更新可能依赖特征忽略的方向，造成的未来误差可以达到最大的 1。
- 对近似闭合的线性模型，$`H`$ 步之后的 predictive 误差由初始误差与每一步的闭合残差组成，每一项都经剩余的模型映射传播。
- 界用的是实际增益 $`\lVert A_{H-1:t+1}^\top c\rVert_*`$。不假设任何增益小于 1；增益大，界就大，必须如实报告。

**直觉.** 为后续训练做的压缩，必须压缩动态，而不只是快照。一个对今天的得分不可见、却驱动明天更新的坐标，正是静态拟合会丢掉的东西。

**在框架中的作用.**

- 它是框架区分"对当前部署充分"与"对后续训练充分"的原因，也是 [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12) 要求测试对多一步合法操作封闭的原因。
- 残差 $`r_t`$ 就是进入 [Proposition 3.4(c)](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 中 Bellman 残差的特征闭合误差；$`A_t`$ 的拟合误差必须算进 $`r_t`$。

**推导要点.**

- (a) $`KF(s,b)=b`$，而在四个状态上 $`b`$ 不是 $`s`$ 的仿射函数，因为 $`s`$ 与 $`b`$ 独立变化。
- (b) 由 Markov 性质 $`q_{t+1}=\mathbb E(K_t\psi)(Z_t)=A_tq_t+\mathbb Er_t(Z_t)`$，所以误差满足 $`q_{t+1}-\hat q_{t+1}=A_t(q_t-\hat q_t)+\mathbb Er_t(Z_t)`$；展开即得 (4.21)。对 (4.22)，写 $`\mathbb EF(Z_H)-c^\top\hat q_H=\mathbb Ee_F(Z_H)+c^\top(q_H-\hat q_H)`$，逐项用 Hölder 不等式，并用 $`\lVert\mathbb Er_t\rVert\le\mathbb E\lVert r_t\rVert`$。

**说明.**

- (b) 针对一个固定的开环控制序列。观察自适应的协议需要 [Theorem 4.10(c)](04-10_hankel_rank_predictive_realization_ZH.md#s4-10) 的逐分支条件化版本，或 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 的一般 Bellman 残差形式；不加论证就把开环公式用在它们身上是不行的。
- (a) 的反例没有用任何近似、optimizer 的异常或分布尾部。静态的低秩拟合代替不了闭合检查。

**用到.**

- [Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)：predictive state 与闭合.

**被用于.**

- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)：要继续使用 predictive state 就需要闭合.
- [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)：Bellman 残差中的闭合残差.

## 相关工作

- predictive state 必须能预测自己的下一个值、而不只是当前输出，这是 predictive state representation 与 value-directed compression 的闭合要求 ([Singh, James & Rudary, 2004](#ref-singh2004predictive); [Poupart & Boutilier, 2002](#ref-poupart2002value))；(a) 是它在学习设定下的最小见证。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
