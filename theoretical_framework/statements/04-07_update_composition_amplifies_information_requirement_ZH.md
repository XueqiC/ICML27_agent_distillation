# Theorem 4.7 · 更新复合放大信息需求

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-07_update_composition_amplifies_information_requirement_ZH.md) · [证明](../proofs/04-07_update_composition_amplifies_information_requirement_ZH.md) · [实验](../experiments/04-07_update_composition_amplifies_information_requirement_ZH.md) · [English](04-07_update_composition_amplifies_information_requirement.md)

**一段话.** 对平均经验块的 learner，[Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) 发现来源必须携带的信息是直到部署长度 $`N`$ 的矩。真实的 learner 不是平均：每次更新都非线性地依赖当前参数与新数据。这个定理说明这会带来什么。复合的多项式更新使所需的矩阶相乘，并且这个界可以达到；不连续的（阈值）更新彻底破坏矩充分性；对光滑 SGD 的一步，一阶统计量是梯度均值，二阶统计量是梯度二阶矩；逐步误差只在稳定性条件下线性累加。信息需求是 learner 与任务复合的性质，而不只是数据的性质。

**本页用到的设定.**

- *多项式更新.* 学生参数为 $`p\in[0,1]^k`$，learner 从公共的 $`p_0`$ 出发，处理取值于有限集 $`\mathcal Y\subset[0,1]^k`$ 的独立块 $`Z_1,\dots,Z_R`$；在参考来源下块 $`Z_i`$ 的分布为 $`P_i`$，在备选来源下为 $`Q_i`$。更新为 $`p_i=U_i(p_{i-1},Z_i)`$。至多 $`N`$ 次决策的部署有多项式价值 $`J(p)`$，总次数不超过 $`N`$（[Theorem 4.4(a)](04-04_affine_learners_moment_order_ZH.md#s4-4)）。
- *光滑局部设定.* $`\theta\in\mathbb R^p`$ 是参数向量，$`V`$ 是 continuation value：从 $`\theta`$ 出发按参考流程继续训练时的期望最终分数（[Definition 1.4](../README_ZH.md#s1-4)）。步长为 $`\eta`$、consumed gradient 分布为 $`P`$ 的一步 SGD 把 $`V`$ 映为 $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$；$`\mu_P=\mathbb E_PG`$ 是梯度均值，$`M_P=\mathbb E_PGG^\top`$ 是梯度二阶矩。$`D^3V`$ 是三阶导数。
- *误差递推.* $`e_t`$ 是 $`t`$ 步后的累积误差；$`\eta`$、$`L`$、$`C`$ 分别是步长、每步放大常数与每步误差常数。

<a id="s4-7"></a>

**Theorem 4.7（更新复合放大信息需求）.** 设定分三部分。

- **多项式更新.** 固定 $`R\ge1`$ 与公共的 $`p_0\in[0,1]^k`$。相互独立的块 $`Z_1,\dots,Z_R`$ 取值于有限集 $`\mathcal Y\subset[0,1]^k`$；在参考数据源下块 $`Z_i`$ 的分布为 $`P_i`$，在备选数据源下为 $`Q_i`$。learner 为 $`p_i=U_i(p_{i-1},Z_i)`$，其中 $`U_i`$ 把 $`[0,1]^k\times\mathcal Y`$ 映入 $`[0,1]^k`$，$`U_i`$ 的每个坐标都是多项式，并满足
  - **(D1)** 关于 $`p_{i-1}`$ 的次数不超过 $`a_i\ge1`$（看作系数为 $`Z_i`$ 的多项式的、$`p_{i-1}`$ 的多项式）；
  - **(D2)** 关于 $`Z_i`$ 的次数不超过 $`b_i\ge1`$（看作系数为 $`p_{i-1}`$ 的多项式的、$`Z_i`$ 的多项式）。
- 部署价值 $`J:[0,1]^k\to[0,1]`$ 是总次数不超过 $`N`$ 的多项式；[Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3) 中每个至多 $`N`$ 次决策的部署都满足这一点（Theorem 4.4(a)）。
- **光滑局部设定.** $`\theta\in\mathbb R^p`$，$`V:\mathbb R^p\to\mathbb R`$ 是一个 continuation value（Definition 1.4）。consumed gradient 分布为 $`P`$ 的一步 SGD 作用为 $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$，记 $`\mu_P=\mathbb E_PG`$、$`M_P=\mathbb E_PGG^\top`$。
  - **(T3)** 对 $`P`$、$`Q`$ 支撑中的每个 $`g`$，$`V`$ 在线段 $`[\theta,\theta-\eta g]`$ 上为 $`C^3`$，并且在这些线段上 $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$。
- **误差递推.** 非负序列满足 $`e_0=0`$ 与 $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$，常数 $`\eta,L,C\ge0`$。

则：

- **(a) 次数的复合.** 在 (D1)–(D2) 下，把 $`J(p_R)`$ 看作 $`(Z_1,\dots,Z_R)`$ 的多项式，它关于第 $`i`$ 块的总次数不超过

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

- 因此，若对每个 $`i`$ 与每个 $`\lvert\alpha\rvert\le D_i`$ 都有 $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$，则 $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$。若所有块服从同一个数据源分布，匹配到 $`\max_iD_i`$ 阶即可。
- **(b) 上界可达.** 对每个 $`R\ge1`$，都存在该类中的更新与一个 $`N=1`$ 的部署，使 response 恰为 $`Z_1^{2^{R-1}}`$。[Theorem 4.4(c)](04-04_affine_learners_moment_order_ZH.md#s4-4) 中 $`M=2^{R-1}`$ 阶的标量分布在所有低于 $`M`$ 阶的矩上一致，学习价值却不同，所以 (4.8) 一般不能改进。
- **(c) 阈值 learner.** 对 learner $`p=\mathbf 1\lbrace Z>1/2\rbrace`$ 与 $`J(p)=p`$（$`N=1`$），分布 $`P(Z=0)=\tfrac13`$、$`P(Z=\tfrac34)=\tfrac23`$ 与 $`Q(Z=\tfrac12)=1`$ 满足 $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$，但 $`\mathbb E_PJ(p)=\tfrac23`$、$`\mathbb E_QJ(p)=0`$。
- **(d) SGD 的局部展开.** 在 (T3) 下，

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big). \qquad(4.9)
```

- **(e) 误差累积.** 在误差递推下，

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

- 当递推取等号时上式也取等号。$`L=0`$ 时线性界 $`e_R\le RC\eta^2`$ 成立；$`L>0`$ 时它一般不成立：取 $`\eta=\tfrac12`$、$`L=C=1`$，取等号的递推给出 $`e_4=\tfrac{65}{32}>1=RC\eta^2`$。

**说了什么.**

- (a) 当每次更新都是多项式时，一个早期块必须匹配的矩阶，等于部署长度乘以所有后续更新的次数。
- (b) 这种相乘确实会发生：平方更新把第一个块变成 $`Z_1^{2^{R-1}}`$，在低于这一阶时匹配的分布仍然不同。
- (c) 对不连续的更新，即使均值相同，学生也可以完全不同；一般没有哪一阶矩是充分的。
- (d) 对光滑 SGD 的一步，两个 feedback law 一阶上通过梯度均值相区别，二阶上通过梯度二阶矩相区别，余项是显式的三阶项。
- (e) 大小为 $`C\eta^2`$ 的逐步误差按几何级数累积，把它们线性相加需要 $`\eta LR=O(1)`$ 或压缩的误差映射。

**直觉.** 同样的数据喂给不同的 learner，需要的信息量不同。后面的非线性更新会重新加工早期块留下的一切，使早期块的高阶矩变得可见。

**在框架中的作用.** 它说明必须对实际的 optimizer 建模，这是 §5 的主题。(d) 是一般一阶表示 [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的 SGD 情形；cold 与 warm Adam 分别在 [Proposition 5.3](05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) 与 [Propositions 5.5](05-05_exact_first_order_statistic_adam_ZH.md#s5-5)–[5.6](05-06_warm_adam_first_order_affine_ZH.md#s5-6) 中处理。

**说明.**

- 只有一步凸仿射更新（$`a_i=b_i=1`$）时，(4.8) 回到 $`D_i=N`$，即 Theorem 4.4(b)。后面每一步非线性更新都会把对所有更早块的需求成倍放大。
- 次数依赖坐标：可逆的重参数化可能改变更新的多项式形式。内在的对象仍是 response 类及其近似维数（[Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)）；次数只是在所选概率坐标下一个易于处理的证书。
- (4.9) 是局部的、单步的。对带状态的 optimizer，完整学习器状态的增量取代参数步（Proposition 5.1）。
- (e) 对它的递推是紧的；关于 $`R`$ 线性的界需要 $`\eta LR=O(1)`$（因为 $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$），或一个压缩的误差映射。

**推导要点.**

- (a) 把 $`J`$ 沿更新反向拉回：把一个 $`(a_t,b_t)`$ 次映射代入 $`d_t`$ 次多项式，关于 $`p`$ 的次数乘以 $`a_t`$，并产生关于 $`Z_t`$ 的 $`b_td_t`$ 次项；再由独立性把期望分解成各块的矩。
- (b) 先取 $`U_1(p,z)=z`$，之后逐坐标平方，配合 Theorem 4.4(c) 中 $`2^{R-1}`$ 阶的标量见证。
- (c) 直接计算：$`P(0)=\tfrac13`$、$`P(\tfrac34)=\tfrac23`$ 对比 $`Q(\tfrac12)=1`$。
- (d) 对 $`V(\theta-\eta g)`$ 作带积分余项的三阶 Taylor 展开，再取期望。
- (e) 归纳加几何级数；递推 $`e_{t+1}=\tfrac32e_t+\tfrac14`$ 说明线性化的简化可以失效。

**例示.** 把同样两个块分布分别喂给平均化 learner，和每读一个块就把参数平方一次的 learner。前者需要直到部署长度的矩；后者需要直到"部署长度乘 $`2^{R-1}`$"的矩。

**用到.**

- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 多项式部署价值与标量见证。
- [Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3) — 至多 $`N`$ 次决策的部署。
- [Definition 1.4](../README_ZH.md#s1-4) — continuation value。

**被用于.**

- [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) — 任意带状态 optimizer 的展开。
- [Proposition 5.3](05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) — cold-start Adam，此时更新不是仿射的。

## Related work

- (d) 是用随机修正方程（stochastic modified equation）弱逼近 SGD 时所依据的单步矩展开（[Li et al., 2019](#ref-li2019stochastic)）：随机梯度的均值与二阶矩决定一步的期望效应，余项是 $`\eta`$ 的三阶。(e) 则是把这类单步误差累积成多步界的那一步。
