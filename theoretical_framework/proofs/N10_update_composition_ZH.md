# N10 · 更新的复合会放大信息需求 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N10_update_composition_ZH.md)

## 设定与假设

**数据块与更新。** 固定训练轮数 $`R\ge1`$ 和共同的初始参数 $`p_0\in[0,1]^k`$。相互独立的经验块 $`Z_1,\dots,Z_R`$ 取值于有限集 $`\mathcal Y\subset[0,1]^k`$。在参考数据源下第 $`i`$ 块的分布为 $`P_i`$，在备选数据源下为 $`Q_i`$。学习器按

```math
p_i=U_i(p_{i-1},Z_i),\qquad i=1,\dots,R,
```

更新。$`U_i:\mathbb R^k\times\mathbb R^k\to\mathbb R^k`$ 的每个坐标都是多项式，并满足：

- **(D1)** 固定 $`Z_i`$ 时，关于旧参数向量 $`p_{i-1}`$ 的总次数不超过 $`a_i\ge1`$；
- **(D2)** 固定 $`p_{i-1}`$ 时，关于新数据块 $`Z_i`$ 的总次数不超过 $`b_i\ge1`$。

假设 $`U_i`$ 把 $`[0,1]^k\times\mathcal Y`$ 映入 $`[0,1]^k`$，于是每一步迭代都是合法的策略参数。$`a_i,b_i`$ 只是上界，某个坐标的实际次数可以更低。

**部署。** $`J:[0,1]^k\to[0,1]`$ 是总次数不超过 $`N`$ 的多项式。由 [N07(a)](N07_affine_moment_order_ZH.md)（节点 [N07](N07_affine_moment_order_ZH.md)），任何决策数不超过 $`N`$ 的有限二元上下文部署都满足这一点。一个数据源的学习价值记为 $`\mathbb E\,J(p_R)`$。

**光滑局部设定（用于 SGD 展开）。** 参数 $`\theta\in\mathbb R^p`$，$`V:\mathbb R^p\to\mathbb R`$ 是一个 continuation value（节点 [N05](N05_continuation_response_ZH.md)）。消耗梯度分布为 $`P`$ 的单步 SGD kernel 作用为

```math
(K^PV)(\theta)=\mathbb E_{G\sim P}\,V(\theta-\eta G),
```

步长 $`\eta>0`$。记 $`\mu_P=\mathbb E_PG`$，$`M_P=\mathbb E_PGG^\top`$。

- **(T3)** 对 $`P`$、$`Q`$ 支撑中的每个 $`g`$，$`V`$ 在线段 $`[\theta,\theta-\eta g]`$ 上三阶连续可微；并且在这些线段上，对所有 $`h`$ 有 $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$。

**误差递推。** 非负序列 $`(e_t)`$ 满足 $`e_0=0`$ 以及

```math
e_{t+1}\le(1+\eta L)e_t+C\eta^2,\qquad t\ge0,
```

其中常数 $`\eta,L,C\ge0`$。

## 命题

**(a)** **Theorem（次数的复合）。** 在 (D1)–(D2) 下，把最终响应 $`J(p_R)`$ 看作 $`(Z_1,\dots,Z_R)`$ 的多项式，它关于第 $`i`$ 块的总次数不超过

```math
D_i=N\,b_i\prod_{t=i+1}^{R}a_t. \qquad(34)
```

因此，若对每个 $`i`$ 和所有满足 $`\lvert\alpha\rvert\le D_i`$ 的多重指标都有 $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$，则 $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$。若各块服从同一个数据源分布，匹配到 $`\max_iD_i`$ 阶即可。

**(a)** **Proposition（上界可达）。** 对每个 $`R\ge1`$，都存在该类中的一列更新和一个 $`N=1`$ 的部署，使响应恰为 $`Z_1^{2^{R-1}}`$。[N07(c)](N07_affine_moment_order_ZH.md) 中 $`2^{R-1}`$ 阶的标量分布在所有低于 $`2^{R-1}`$ 阶的矩上一致，学习价值却不同。所以 (34) 一般不能改进。

**(b)** **Proposition（阈值学习器）。** 考虑非多项式学习器 $`p=1\lbrace Z>1/2\rbrace`$，部署 $`J(p)=p`$（$`N=1`$），并取

```math
P(Z=0)=\tfrac13,\quad P(Z=\tfrac34)=\tfrac23,\qquad Q(Z=\tfrac12)=1.
```

则 $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$，但 $`\mathbb E_PJ(p)=\tfrac23`$，$`\mathbb E_QJ(p)=0`$。

**(c)** **Proposition（SGD 局部展开）。** 在 (T3) 下，

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E, \qquad(35)
```

```math
\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\left(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\right).
```

**(d)** **Proposition（误差累积）。** 在上述误差递推下，

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

递推取等号时上式也取等号。线性化的 $`e_R\le RC\eta^2`$ 在 $`L=0`$ 时成立，在 $`L>0`$ 时一般不成立。例如 $`\eta=\tfrac12`$、$`L=C=1`$ 的等式递推给出 $`e_4=65/32>1=RC\eta^2`$。

## 证明

**(a) 的证明。** 定义反向 continuation 多项式

```math
\Phi_R(p)=J(p),\qquad \Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t\big(U_t(p,Z_t);Z_{t+1},\dots,Z_R\big),\quad t=R,\dots,1.
```

于是 $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$。

*断言。* 对每个 $`t\in\lbrace 0,\dots,R\rbrace`$：$`\Phi_t`$ 关于 $`p`$ 的总次数不超过 $`d_t:=N\prod_{s=t+1}^Ra_s`$（其中 $`d_R=N`$）；并且对每个 $`i>t`$，$`\Phi_t`$ 关于 $`Z_i`$ 的总次数不超过 $`D_i`$。

*归纳起点 $`t=R`$：* $`\Phi_R=J`$ 关于 $`p`$ 的次数不超过 $`N`$，且不含任何数据块。

*归纳步：* 设断言在 $`t`$ 时成立。写 $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$；由归纳假设，每个系数 $`c_\gamma`$ 关于 $`Z_i`$（$`i>t`$）的次数不超过 $`D_i`$。代入 $`p=U_t(p',Z_t)`$ 后，单项式 $`p^\gamma`$ 变成 $`U_t`$ 的 $`\lvert\gamma\rvert`$ 个坐标多项式之积。由 (D1)，每个因子关于 $`p'`$ 的次数不超过 $`a_t`$；由 (D2)，关于 $`Z_t`$ 的次数不超过 $`b_t`$。乘积的次数相加，所以代入后的单项式：

- 关于 $`p'`$ 的次数不超过 $`a_t\lvert\gamma\rvert\le a_td_t=d_{t-1}`$；
- 关于 $`Z_t`$ 的次数不超过 $`b_t\lvert\gamma\rvert\le b_td_t=Nb_t\prod_{s>t}a_s=D_t`$。

系数 $`c_\gamma`$ 与 $`p'`$、$`Z_t`$ 无关，所以关于 $`Z_i`$（$`i>t`$）的次数不变。断言在 $`t-1`$ 时成立。到 $`t=0`$ 时，$`p=p_0`$ 是常数，因而 $`J(p_R)`$ 关于每个块 $`Z_i`$ 的次数不超过 $`D_i`$。

*矩的推论。* 展开 $`J(p_R)=\sum_{\alpha_1,\dots,\alpha_R}c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$，这是一个有限和，其中 $`\lvert\alpha_i\rvert\le D_i`$。各块相互独立，所以

```math
\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}.
```

在匹配假设下，每个因子在 $`P_i`$ 和 $`Q_i`$ 下相同。对有限展开用期望的线性性，得 $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$。若所有 $`P_i=P`$、$`Q_i=Q`$，匹配到 $`\max_iD_i`$ 阶就蕴含匹配到每个 $`D_i`$ 阶。$`\square`$

**(a) 的证明。** 取 $`U_1(p,z)=z`$（$`b_1=1`$），对 $`i\ge2`$ 取 $`U_i(p,z)=p\odot p`$（$`a_i=2`$，$`b_i=1`$ 作为上界）。两者都保持 $`[0,1]^k`$ 不变。部署一次，取第一个坐标 $`J(p)=p_1`$（$`N=1`$）。归纳可得 $`p_i=Z_1^{2^{i-1}}`$（逐坐标），所以响应为 $`Z_{1,1}^{2^{R-1}}`$。此时 (34) 给出 $`D_1=1\cdot1\cdot2^{R-1}`$，上界取到。

再取 [N07(c)](N07_affine_moment_order_ZH.md) 中 $`M=2^{R-1}`$ 的标量分布 $`P_M,Q_M`$（节点 [N07](N07_affine_moment_order_ZH.md)，[N07(c)](N07_affine_moment_order_ZH.md)）。它们所有 $`\ell<M`$ 阶的矩相同，而 $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M=(-1)^MM!/(2^{M-1}M^M)\ne0`$。所以两个数据源在所有低于可达次数的矩上一致，学习价值仍然不同。后面的块不被使用，这是 (D2) 允许的。$`\square`$

**(b) 的证明。** $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$。在 $`P`$ 下，$`1\lbrace Z>1/2\rbrace =1`$ 当且仅当 $`Z=3/4`$，概率为 $`2/3`$。在 $`Q`$ 下，$`Z=1/2`$ 不大于 $`1/2`$，取值为 $`0`$。$`\square`$

**(c) 的证明。** 取任一支撑中的 $`g`$，令 $`\varphi(\tau)=V(\theta-\tau\eta g)`$，$`\tau\in[0,1]`$。由 (T3)，$`\varphi\in C^3[0,1]`$。带积分余项的 Taylor 定理给出

```math
\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\frac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau.
```

各阶导数为

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

由 (T3)，$`\lvert\varphi'''(\tau)\rvert\le M_3\eta^3\lVert g\rVert^3`$；又 $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$，故

```math
V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g),\qquad\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3.
```

分别对 $`\mathbb E_P`$、$`\mathbb E_Q`$ 取期望后相减：常数 $`V(\theta)`$ 抵消；一次项和二次项分别给出 $`-\eta\nabla V^\top(\mu_P-\mu_Q)`$ 和 $`\tfrac{\eta^2}{2}\langle\nabla^2V,M_P-M_Q\rangle`$；余项 $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$ 由三角不等式控制，不超过 $`\tfrac{M_3\eta^3}{6}(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)`$。$`\square`$

**(d)** **Proposition。** 对 $`t`$ 归纳证明 $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$：

- $`t=0`$ 时是空和；
- 若在 $`t`$ 时成立，则 $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^{t}(1+\eta L)^s`$。

每一步只用到递推本身，所以递推取等号时结论也取等号。闭式就是几何级数求和。

反例：$`\eta=\tfrac12`$、$`L=C=1`$ 的等式递推为 $`e_{t+1}=\tfrac32e_t+\tfrac14`$，依次得 $`e_1=\tfrac14`$、$`e_2=\tfrac58`$、$`e_3=\tfrac{19}{16}`$、$`e_4=\tfrac{65}{32}`$，超过 $`RC\eta^2=1`$。$`\square`$

## 注记

1. **定理的操作含义。** 数据源必须匹配的信息阶数，是*学习器与部署复合之后*的性质，不只是部署本身的性质。只有一次凸仿射更新时（$`a_i=b_i=1`$），(34) 退化为 $`D_i=N`$，即节点 [N07](N07_affine_moment_order_ZH.md) 的 [N07(b)](N07_affine_moment_order_ZH.md)。后面每多一次非线性更新，对所有更早数据块的要求都会成倍增加。
2. **次数依赖于坐标。** 可逆的重新参数化会改变更新的多项式形式。真正内在的对象仍是 response class 及其逼近宽度（节点 [N06](N06_sufficiency_duality_ZH.md)）。次数只是在所选概率坐标下一个便于处理的证书。
3. **非多项式学习器。** (b) 说明，一旦更新不连续，一般来说任何矩的阶数都不够。即使只有一次决策，对任意学习器也谈不上"一阶矩足够"。
4. **(35) 的适用范围。** 这个展开是局部的、单步的。它的一阶统计量是消耗梯度的均值，二阶统计量是梯度的二阶矩。把 $`V`$ 换成训练 loss，或把它用到 Adam 上，都需要另外论证。cold Adam 的替代结论见节点 [N11](N11_adam_cold_start_ZH.md)，warm Adam 的见节点 [N12](N12_optimizer_warmth_ZH.md)。
5. **累积需要稳定性。** (d) 对其递推是紧的。要得到关于 $`R`$ 线性的界，需要 $`\eta LR=O(1)`$（因为 $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$），或者误差映射是压缩的。

## 依赖与出处

- 依赖：[N07](N07_affine_moment_order_ZH.md)（[N07(a)](N07_affine_moment_order_ZH.md) 的多项式部署、分布 [N07(c)](N07_affine_moment_order_ZH.md)），[N05](N05_continuation_response_ZH.md)（continuation kernel $`K^P`$），[N06](N06_sufficiency_duality_ZH.md)（response class 的观点）。
- 被使用于：[N11](N11_adam_cold_start_ZH.md)，[N12](N12_optimizer_warmth_ZH.md)。

**出处：** [v0.14.1 §9](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#9-learning-dynamics-can-amplify-the-information-requirement); [v0.15 §7](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#7-experiments-that-discriminate-the-theory) — Thm. 17; eqs. (34)–(35); addendum §7 recursion remark. v0.14.1 [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §9，Theorem 17，式 (34)–(35)；v0.15 附录 [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §7 关于递推的注记。
