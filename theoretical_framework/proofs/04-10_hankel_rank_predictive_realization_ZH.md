# Theorem 4.10 · Hankel 秩与线性 predictive 实现

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-10_hankel_rank_predictive_realization_ZH.md) · [证明](04-10_hankel_rank_predictive_realization_ZH.md) · [实验](../experiments/04-10_hankel_rank_predictive_realization_ZH.md) · [English](04-10_hankel_rank_predictive_realization.md)

**本页证明什么.** (a) 是不变函数空间上的线性代数。(b) 说明任务 Hankel 矩阵的秩从下方界住每个线性实现，并由行空间实现达到，其转移映射由测试列的平移诱导。(c) 是用 predictive 坐标写出的 Bayes 公式。

**本页用到的设定.**

- 控制 $`b\in\mathcal B`$，在完整学习状态的有界函数上有 Markov kernel $`K_b`$；控制词 $`w=b_1\cdots b_\ell`$，$`K_w=K_{b_1}\cdots K_{b_\ell}`$；有界输出 $`F_1,\dots,F_k`$（[Definition 1.3](../README_ZH.md#s1-3)）。
- 测试空间 $`\mathcal V=\mathrm{span}\lbrace\mathbf 1,K_wF_j:w\in\mathcal W\rbrace`$；对 (b)：前缀集合 $`\mathcal P`$（对延伸封闭）、测试列集合 $`\mathcal T`$（对前置封闭，含常数测试与每个 $`(\varnothing,j)`$），以及 $`H(p,(w,j))=\mu_0K_pK_wF_j`$。
- 观察算子 $`T_{b,o}f(z)=\mathbb E[\mathbf 1\lbrace O=o\rbrace f(Z')\mid Z=z,b]`$。

**陈述（[Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization_ZH.md#s4-10)）.** (a) 若 $`\mathcal V`$ 有有限基 $`\psi`$，$`\psi_1=\mathbf 1`$，包含各 $`F_j`$ 且 $`K_b`$-不变，则 $`K_b\psi=A_b\psi`$、$`F_j=c_j^\top\psi`$，且 (4.18) 成立；$`\mathcal W`$ 含空词并对一步前置封闭时不变性成立。(b) 若 $`\mathrm{rank}\,H=d<\infty`$，每个线性实现的维数至少为 $`d`$，(4.19) 给出一个 $`d`$ 维实现。(c) 若 $`T_{b,o}\psi=M_{b,o}\psi`$，则 $`\Pr(o\mid\mu,b)=e_1^\top M_{b,o}q`$，后验 predictive state 为 (4.20)。

**证明.**

**(a)** 对每个 $`b`$ 与 $`i`$，$`K_b\psi_i\in\mathcal V`$，所以 $`K_b\psi_i=\sum_l(A_b)_{il}\psi_l`$，系数唯一；它们组成 $`A_b`$，写成向量形式即 $`K_b\psi=A_b\psi`$。同理 $`F_j\in\mathcal V`$ 给出 $`c_j`$。对状态分布 $`\mu`$，积分是线性的，所以

```math
q(\mu K_b)=\int K_b\psi\,d\mu=A_b\int\psi\,d\mu=A_b\,q(\mu),\qquad\mathbb E_\mu F_j=c_j^\top\int\psi\,d\mu=c_j^\top q(\mu).
```

*由前置封闭得到不变性.* $`K_b`$ 是 Markov kernel，所以 $`K_b\mathbf 1=\mathbf 1`$；当 $`bw\in\mathcal W`$ 时 $`K_b(K_wF_j)=K_{bw}F_j\in\mathcal V`$。$`\mathcal V`$ 的生成元被映到 $`\mathcal V`$ 中，由线性性 $`K_b\mathcal V\subseteq\mathcal V`$。空词保证每个 $`F_j\in\mathcal V`$。

**(b)** *下界.* 设 $`q(p)\in\mathbb R^{d'}`$ 与 $`c_{w,j}`$ 实现了 $`H`$ 中由 $`\mathcal T`$ 标记的每一列，包括常数列。对每个 $`i\le d'`$，令 $`x_i\in\mathbb R^{\mathcal T}`$ 为向量 $`(c_{w,j,i})_{(w,j)\in\mathcal T}`$。每一行都是 $`H(p,\cdot)=\sum_{i\le d'}q_i(p)x_i`$，所以行空间的维数至多为 $`d'`$。$`H`$ 的秩就是行空间的维数，所以 $`d\le d'`$。

*平移.* 定义 $`S_b:\mathbb R^{\mathcal T}\to\mathbb R^{\mathcal T}`$：$`(S_bx)(w,j)=x(bw,j)`$，常数列上 $`(S_bx)(\mathbf 1)=x(\mathbf 1)`$。因为 $`\mathcal T`$ 对一步前置封闭，它定义良好，并且是线性的。对一行，

```math
(S_bH(p,\cdot))(w,j)=H(p,(bw,j))=\mu_0K_pK_bK_wF_j=H(pb,(w,j)),
```

常数列上两边都是 1。所以 $`S_bH(p,\cdot)=H(pb,\cdot)`$；由于 $`\mathcal P`$ 对延伸封闭，它仍是一行。因此 $`S_b`$ 把行的集合映到自身，由线性性把行空间 $`\mathcal L`$ 映到自身。

*实现.* 取 $`\mathcal L`$ 的一组基 $`x_1,\dots,x_d`$，令 $`q(p)\in\mathbb R^d`$ 为满足 $`H(p,\cdot)=\sum_iq_i(p)x_i`$ 的唯一坐标。令 $`A_b`$ 为 $`S_b|_{\mathcal L}`$ 在这组基下的矩阵：$`S_bx_i=\sum_l(A_b)_{li}x_l`$。于是

```math
H(pb,\cdot)=S_bH(p,\cdot)=\sum_iq_i(p)S_bx_i=\sum_l\big(A_bq(p)\big)_lx_l,
```

由坐标的唯一性得 $`q(pb)=A_bq(p)`$。令 $`c_j=(x_1(\varnothing,j),\dots,x_d(\varnothing,j))^\top`$，则 $`H(p,(\varnothing,j))=c_j^\top q(p)`$。对控制词 $`w=b_1\cdots b_\ell`$，有 $`pw\in\mathcal P`$，且 $`H(p,(w,j))=\mu_0K_{pw}F_j=H(pw,(\varnothing,j))=c_j^\top q(pw)=c_j^\top A_{b_\ell}\cdots A_{b_1}q(p)`$，这就是 (4.19)。

**(c)** 设 $`\mu`$ 为当前状态分布。因为 $`\psi_1=\mathbf 1`$，$`T_{b,o}\psi=M_{b,o}\psi`$ 的第一行是 $`T_{b,o}\mathbf 1=e_1^\top M_{b,o}\psi`$，所以

```math
\Pr(O=o\mid\mu,b)=\int T_{b,o}\mathbf 1\,d\mu=e_1^\top M_{b,o}\,q .
```

它为正时，按条件期望的定义，给定 $`O=o`$ 时下一状态的后验分布 $`\mu'`$ 对有界 $`f`$ 满足 $`\int f\,d\mu'=\int T_{b,o}f\,d\mu/\Pr(O=o\mid\mu,b)`$。逐坐标取 $`f=\psi`$，

```math
q'=\int\psi\,d\mu'=\frac{\int T_{b,o}\psi\,d\mu}{\Pr(O=o\mid\mu,b)}=\frac{M_{b,o}\,q}{e_1^\top M_{b,o}\,q}. \qquad\square
```

**关于证明的说明.**

- (b) 只用到在声明的前缀与列上整个矩阵的代数秩。对一个有限的观测子矩阵，只有当观测到的行与列本身对平移封闭时，同样的论证才给出该子矩阵的实现；否则观测到的秩只是 $`d`$ 的下界。
- 依赖时间的 kernel 可以通过把时钟放进状态来处理，这样在扩大的状态上 kernel 与时间无关；预算处的吸收保证每个控制词都在声明的训练预算之内。

## 相关工作

- (b) 中的行空间构造是 predictive state representation 与 observable operator model 的标准实现论证 ([Singh, James & Rudary, 2004](#ref-singh2004predictive))；(c) 是 [Littman, Sutton & Singh (2001)](#ref-littman2001predictive) 中带归一化的线性更新。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
