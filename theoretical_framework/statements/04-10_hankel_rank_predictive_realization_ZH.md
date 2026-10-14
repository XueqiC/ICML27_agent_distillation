# Theorem 4.10 · Hankel 秩与线性 predictive 实现

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-10_hankel_rank_predictive_realization_ZH.md) · [证明](../proofs/04-10_hankel_rank_predictive_realization_ZH.md) · [实验](../experiments/04-10_hankel_rank_predictive_realization_ZH.md) · [English](04-10_hankel_rank_predictive_realization.md)

**一段话.** 对学习过程的压缩描述，要对后续训练有用，就必须能预测自己的下一个值：只匹配测试现在看到的东西不够，下一次更新可能把过程带到描述跟不上的地方。把训练看成一个受控系统，控制是可执行的 feedback 提供者，输出是声明的部署得分。这样，正确的压缩状态是一组对未来测试的预测，经典的 predictive state representation 理论可以直接用上：测试空间对多一步控制封闭时，有限基给出精确的线性更新；任务 Hankel 矩阵的秩是这种线性预测器的最小维数；记录下来的观察通过一个归一化的线性映射更新 predictive state。这个定理是经典结论的实例化；这里特有的只是：状态是完整的学习状态，输出是训练之后的得分。

**本页用到的设定.**

- *控制.* 声明的有限控制集合 $`\mathcal B`$：feedback 提供者、查询方式或训练干预。学习率、optimizer 或任务的改变算控制、算测试的一部分还是另一个系统，必须事先声明，不能事后决定。
- *状态与 kernel.* 状态是 [Definition 1.3](../README_ZH.md#s1-3) 的完整学习状态，时钟包含在内；训练预算用完后状态被吸收，此后每个控制都作用为恒等。控制 $`b`$ 有实际的 Markov kernel $`K_b`$，作用在有界函数上。有限时域与非平稳的 schedule 都包括在内；不假设任何无限时域的稳定性。
- *控制词.* 对控制词 $`w=b_1\cdots b_\ell`$，$`K_w=K_{b_1}\cdots K_{b_\ell}`$（先执行 $`b_1`$）；空词给出恒等。对状态分布 $`\mu`$，$`\mu K_w`$ 是执行 $`w`$ 之后的分布。
- *输出与测试.* 声明的有界部署得分 $`F_1,\dots,F_k`$。测试是一对 $`(w,j)`$：执行 $`w`$，再用 $`F_j`$ 打分；它对应的函数是 $`K_wF_j`$。对声明的测试词集合 $`\mathcal W`$，测试空间为 $`\mathcal V=\mathrm{span}\lbrace\mathbf 1,K_wF_j:w\in\mathcal W,\ j\le k\rbrace`$。
- *观察.* 若控制 $`b`$ 还产生一个取值于有限集合的观察 $`o`$，令 $`T_{b,o}f(z)=\mathbb E[\mathbf 1\lbrace O=o\rbrace f(Z')\mid Z=z,b]`$；于是 $`\sum_oT_{b,o}=K_b`$。

<a id="s4-10"></a>

**Theorem 4.10（Hankel 秩与线性 predictive 实现）.**

**(a) 封闭的测试给出精确的线性预测.** 设 $`\mathcal V`$ 有有限基 $`\psi=(\psi_1,\dots,\psi_d)`$，$`\psi_1=\mathbf 1`$，包含每个 $`F_j`$，并且在每个 $`K_b`$ 下不变。则存在矩阵 $`A_b`$ 与向量 $`c_j`$，使 $`K_b\psi=A_b\psi`$、$`F_j=c_j^\top\psi`$，并且对每个状态分布 $`\mu`$，predictive state $`q(\mu)=\mathbb E_\mu\psi`$ 满足

```math
q(\mu K_b)=A_b\,q(\mu),\qquad \mathbb E_\mu F_j=c_j^\top q(\mu). \qquad(4.18)
```

只要 $`\mathcal W`$ 包含空词并且对一步前置封闭（对 $`w\in\mathcal W`$、$`b\in\mathcal B`$ 有 $`bw\in\mathcal W`$），不变性就成立。

**(b) Hankel 秩是最小线性维数.** 固定初始分布 $`\mu_0`$。设控制前缀集合 $`\mathcal P`$ 包含空词并对一步延伸封闭（$`pb\in\mathcal P`$），测试列集合 $`\mathcal T`$ 包含常数测试与每个 $`(\varnothing,j)`$，并对一步前置封闭。任务 Hankel 矩阵为 $`H(p,(w,j))=\mu_0K_pK_wF_j`$，常数列恒为 1。若 $`\mathrm{rank}\,H=d<\infty`$：

- 任何实现 $`H`$ 全部列（包括常数列）的线性实现，即向量 $`q(p)\in\mathbb R^{d'}`$ 与每一列的系数 $`c`$ 使 $`H(p,\cdot)=c^\top q(p)`$，都有 $`d'\ge d`$（不要求重现常数列时这个界可以不成立：若每个 $`F_j\equiv0`$，秩为 1，而 $`d'=0`$ 已能重现每个 $`(w,j)`$ 元素）；
- 存在 $`d`$ 维实现，满足

```math
q(pb)=A_b\,q(p),\qquad H(p,(w,j))=c_j^\top A_w\,q(p),\qquad A_w=A_{b_\ell}\cdots A_{b_1}. \qquad(4.19)
```

**(c) 以观察为条件.** 设 $`\mathrm{span}\,\psi`$（$`\psi_1=\mathbf 1`$）在每个 $`T_{b,o}`$ 下不变：$`T_{b,o}\psi=M_{b,o}\psi`$。设 $`\mu`$ 是可达的状态分布，predictive state 为 $`q=\mathbb E_\mu\psi`$，$`e_1=(1,0,\dots,0)^\top`$。则执行 $`b`$ 后观察到 $`o`$ 的概率为 $`e_1^\top M_{b,o}q`$；它为正时，后验分布的 predictive state 为

```math
q'=\frac{M_{b,o}\,q}{e_1^\top M_{b,o}\,q}. \qquad(4.20)
```

**说了什么.**

- 若声明的测试对多一步合法控制封闭、且张成有限维空间，它们当前预测值组成的向量就是精确的状态：在每个控制下线性更新，并决定每个声明的期望。
- 任务 Hankel 矩阵的秩，是这些输入–输出期望的任何线性预测器所能有的最小维数，而且能达到。
- 有观察记录时，predictive state 先经一个线性映射，再按观察的概率归一化。

**直觉.** 对声明任务而言，学习过程的状态不是它的参数，而是它在每个合法 continuation 下会得多少分。有限个这样的预测够用，恰好当它们能一步一步地互相预测。

**在框架中的作用.**

- 它是 [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 中 response operator 的动态对应物。固定检查点上的 response $`g_a(y)`$ 是这类 Hankel 对象中的一个局部"输入–后缀"块：先是本轮消息，再是一个未来测试。[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 的局部 response 谱与动态的 Hankel 谱有联系，但一般不是同一组奇异值。
- [Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11) 说明闭合不能去掉，并给出闭合只近似成立时的误差；[Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12) 把 (a) 与 response operator 联系起来。

**推导要点.**

- (a) 不变性给出系数 $`A_b`$；积分的线性性把它们移过 $`\mathbb E_\mu`$。
- (b) 任何 $`d'`$ 维实现都把 $`H`$ 分解为经过 $`\mathbb R^{d'}`$，所以秩至多为 $`d'`$。反过来，取 $`H`$ 的行空间为状态空间：前缀延伸在行上作用为列的平移 $`(w,j)\mapsto(bw,j)`$，它是线性的，并且因为列对前置封闭而定义良好；行空间的一组基给出 (4.19)。
- (c) 分子是观察事件与下一步特征的未归一化联合期望，分母是该事件的概率；$`\psi`$ 的第一个坐标是常数，所以它读出概率。

**四条边界.**

1. (b) 中的最小性，针对的是从 $`\mu_0`$ 出发、用声明的实验可达的输入–输出期望的线性表示。它不是对所有隐藏状态或非线性表示的最小性；像 (a) 那样在每个单点状态上都不变的空间可能更大。
2. 由少数观测到的行与列组成的低秩子矩阵，不能证明整个 $`H`$ 低秩；它的秩只是下界。需要后缀延伸检验与留出预测；从未被激发的方向看起来就像缺失的模式。
3. 时钟已知时，要报告每个时间层的 predictive 维数，或者说明时钟的代价。把不同时间层拼在一起而增加的秩，不是环境的复杂度。
4. 开环控制词预测的是干预后的期望。要根据新的观察在线更新状态，需要 (c)；不含观察分支的测试类，不支持任意的观察自适应协议。

**说明.**

- (4.20) 的概率解释适用于可达的 predictive state；任意 $`q\in\mathbb R^d`$ 不一定对应某个概率分布。
- 不假设观察概率有全局下界。稀有观察会放大 $`M_{b,o}`$ 的任何误差；这个代价必须显式付出，或者分析停留在未归一化的量上。
- 精确实现来自真实的概率过程，因此是合法的预测器。对估计矩阵做截断奇异值分解，不会自动保正、归一，也不会自动能由合法 feedback 实现；实现 feedback 是另一个问题（[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)）。

**用到.**

- [Definition 1.3](../README_ZH.md#s1-3)：完整学习状态与 kernel.
- [Definition 4.8](04-08_response_operator_ZH.md#s4-8)：本结果在时间上推广的局部 response operator.

**被用于.**

- [Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)：闭合是必要的；近似闭合的误差.
- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)：response 经过下一步 predictive state 分解.
- [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)：predictive-state 模型是近似价值的一个来源.

## 相关工作

- predictive state representation 用对未来测试的预测描述受控过程，并以带归一化的线性方式更新 ([Littman, Sutton & Singh, 2001](#ref-littman2001predictive); [Singh, James & Rudary, 2004](#ref-singh2004predictive))；(b) 与 (c) 是它们的实现与条件化结果在本设定下的形式。
- 通过这类 Krylov 型不变子空间压缩部分可观过程、只保留能预测价值的部分，是 value-directed compression ([Poupart & Boutilier, 2002](#ref-poupart2002value))。从数据学习这类状态是 predictive state temporal-difference learning 的主题 ([Boots & Gordon, 2010](#ref-boots2010predictive))。这里变的是系统（完整的学习过程）与输出（训练之后的得分）；构造本身不新。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
