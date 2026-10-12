# Theorem 4.7 · 更新复合放大信息需求

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-07_update_composition_amplifies_information_requirement_ZH.md) · [证明](04-07_update_composition_amplifies_information_requirement_ZH.md) · [实验](../experiments/04-07_update_composition_amplifies_information_requirement_ZH.md) · [English](04-07_update_composition_amplifies_information_requirement.md)

**本页证明什么.** 当每次训练更新都是当前参数与新数据的多项式时，把部署价值沿更新反向拉回会使它的次数相乘，于是早期块的高阶矩变得相关；平方更新使这一乘法真正达到。阈值更新使矩匹配完全失效。对光滑 SGD 的一步，三阶 Taylor 展开给出一阶与二阶统计量；一个几何级数界住逐步误差如何累积。

**本页用到的设定.**

- *多项式更新.* 学生参数为 $`p\in[0,1]^k`$；learner 从公共的 $`p_0`$ 出发，处理取值于有限集 $`\mathcal Y\subset[0,1]^k`$ 的独立块 $`Z_1,\dots,Z_R`$；在参考来源下块 $`Z_i`$ 的分布为 $`P_i`$，在备选来源下为 $`Q_i`$；更新为 $`p_i=U_i(p_{i-1},Z_i)`$。
- *部署价值.* 对至多 $`N`$ 次决策的部署，$`J`$ 是 $`p`$ 的总次数不超过 $`N`$ 的多项式（[Theorem 4.4(a)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）。
- *标量见证*（[Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）：对每个阶数 $`M\ge1`$，$`\lbrace0,\tfrac1M,\dots,1\rbrace`$ 上存在分布 $`P_M,Q_M`$，它们在所有低于 $`M`$ 阶的矩上一致，在 $`M`$ 阶相差 $`\Delta_M=(-1)^MM!/(2^{M-1}M^M)\ne0`$。
- *光滑局部设定.* $`\theta\in\mathbb R^p`$，$`V`$ 是 continuation value：从 $`\theta`$ 出发按参考流程继续训练时的期望最终分数（[Definition 1.4](../README_ZH.md#s1-4)）。步长为 $`\eta`$、consumed gradient 分布为 $`P`$ 的一步 SGD 作用为 $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$；$`\mu_P=\mathbb E_PG`$，$`M_P=\mathbb E_PGG^\top`$；$`D^3V`$ 是三阶导数。
- *误差递推.* 非负序列 $`e_t`$，常数 $`\eta,L,C\ge0`$。

**陈述（[Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7)）.** 设定分三部分。

- **多项式更新.** 固定 $`R\ge1`$ 与公共的 $`p_0\in[0,1]^k`$。相互独立的块 $`Z_1,\dots,Z_R`$ 取值于有限集 $`\mathcal Y\subset[0,1]^k`$；在参考数据源下块 $`Z_i`$ 的分布为 $`P_i`$，在备选数据源下为 $`Q_i`$。learner 为 $`p_i=U_i(p_{i-1},Z_i)`$，其中 $`U_i`$ 把 $`[0,1]^k\times\mathcal Y`$ 映入 $`[0,1]^k`$，$`U_i`$ 的每个坐标都是多项式，并满足
  - **(D1)** 关于 $`p_{i-1}`$ 的次数不超过 $`a_i\ge1`$（看作系数为 $`Z_i`$ 的多项式的、$`p_{i-1}`$ 的多项式）；
  - **(D2)** 关于 $`Z_i`$ 的次数不超过 $`b_i\ge1`$（看作系数为 $`p_{i-1}`$ 的多项式的、$`Z_i`$ 的多项式）。
- 部署价值 $`J:[0,1]^k\to[0,1]`$ 是总次数不超过 $`N`$ 的多项式；[Definition 4.3](../statements/04-03_binary_context_student_affine_learner_ZH.md#s4-3) 中每个至多 $`N`$ 次决策的部署都满足这一点（Theorem 4.4(a)）。
- **光滑局部设定.** $`\theta\in\mathbb R^p`$，$`V:\mathbb R^p\to\mathbb R`$ 是一个 continuation value（Definition 1.4）。consumed gradient 分布为 $`P`$ 的一步 SGD 作用为 $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$，记 $`\mu_P=\mathbb E_PG`$、$`M_P=\mathbb E_PGG^\top`$。
  - **(T3)** 对 $`P`$、$`Q`$ 支撑中的每个 $`g`$，$`V`$ 在线段 $`[\theta,\theta-\eta g]`$ 上为 $`C^3`$，并且在这些线段上 $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$。
- **误差递推.** 非负序列满足 $`e_0=0`$ 与 $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$，常数 $`\eta,L,C\ge0`$。

则：

- **(a) 次数的复合.** 在 (D1)–(D2) 下，把 $`J(p_R)`$ 看作 $`(Z_1,\dots,Z_R)`$ 的多项式，它关于第 $`i`$ 块的总次数不超过

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

- 因此，若对每个 $`i`$ 与每个 $`\lvert\alpha\rvert\le D_i`$ 都有 $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$，则 $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$。若所有块服从同一个数据源分布，匹配到 $`\max_iD_i`$ 阶即可。
- **(b) 上界可达.** 对每个 $`R\ge1`$，都存在该类中的更新与一个 $`N=1`$ 的部署，使 response 恰为 $`Z_1^{2^{R-1}}`$。Theorem 4.4(c) 中 $`M=2^{R-1}`$ 阶的标量分布在所有低于 $`M`$ 阶的矩上一致，学习价值却不同，所以 (4.8) 一般不能改进。
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

**证明.** *(a).* 定义反向的 continuation 多项式 $`\Phi_R(p)=J(p)`$，以及 $`t=R,\dots,1`$ 时 $`\Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t(U_t(p,Z_t);Z_{t+1},\dots,Z_R)`$，于是 $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$。用反向归纳证明：$`\Phi_t`$ 关于 $`p`$ 的次数不超过 $`d_t:=N\prod_{s=t+1}^Ra_s`$（$`d_R=N`$），并且对每个 $`i>t`$，关于 $`Z_i`$ 的次数不超过 $`D_i`$。

- *起点 $`t=R`$.* $`\Phi_R=J`$ 关于 $`p`$ 的次数不超过 $`N`$，不含任何块。
- *归纳步.* 写 $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$，其中每个 $`c_\gamma`$ 对 $`i>t`$ 关于 $`Z_i`$ 的次数不超过 $`D_i`$。代入 $`p=U_t(p',Z_t)`$。单项式 $`p^\gamma`$ 变成 $`U_t`$ 的 $`\lvert\gamma\rvert`$ 个坐标多项式之积，每个因子由 (D1) 关于 $`p'`$ 的次数至多为 $`a_t`$，由 (D2) 关于 $`Z_t`$ 的次数至多为 $`b_t`$。乘积的次数相加，所以代入后的单项式关于 $`p'`$ 的次数至多为 $`a_td_t=d_{t-1}`$，关于 $`Z_t`$ 的次数至多为 $`b_td_t=Nb_t\prod_{s>t}a_s=D_t`$。系数 $`c_\gamma`$ 不含 $`p'`$ 与 $`Z_t`$，所以关于 $`Z_i`$（$`i>t`$）的次数不变。

在 $`t=0`$ 处 $`p=p_0`$ 是常数，所以 $`J(p_R)`$ 关于每个 $`Z_i`$ 的次数都不超过 $`D_i`$。展开 $`J(p_R)=\sum c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$，这是有限和，且 $`\lvert\alpha_i\rvert\le D_i`$。由独立性，$`\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}`$；在匹配假设下，每个因子在 $`P_i`$ 与 $`Q_i`$ 下都相同。若对所有 $`i`$ 有 $`P_i=P`$、$`Q_i=Q`$，匹配到 $`\max_iD_i`$ 阶就蕴含匹配到每个 $`D_i`$ 阶。

*(b).* 令 $`U_1(p,z)=z`$，$`i\ge2`$ 时 $`U_i(p,z)=p\odot p`$，于是 $`b_1=1`$，$`i\ge2`$ 时 $`a_i=2`$（其余次数以 1 为上界）。两个映射都保持 $`[0,1]^k`$ 不变。部署一次，$`J(p)=p_1`$（$`N=1`$）。由归纳，逐坐标 $`p_i=Z_1^{2^{i-1}}`$，所以 response 为 $`Z_{1,1}^{2^{R-1}}`$，而 (4.8) 给出 $`D_1=1\cdot1\cdot2^{R-1}`$。对 $`M=2^{R-1}`$ 的分布 $`P_M,Q_M`$，低于 $`M`$ 阶的矩相等，而 $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M\ne0`$。后面的块没有用到，这是 (D2) 允许的。

*(c).* $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$。在 $`P`$ 下，$`\mathbf 1\lbrace Z>1/2\rbrace=1`$ 当且仅当 $`Z=\tfrac34`$，其概率为 $`\tfrac23`$。在 $`Q`$ 下，$`Z=\tfrac12`$ 不大于 $`\tfrac12`$，所以价值为 0。

*(d).* 固定任一支撑中的 $`g`$，令 $`\varphi(\tau)=V(\theta-\tau\eta g)`$，$`\tau\in[0,1]`$。由 (T3)，$`\varphi\in C^3[0,1]`$，带积分余项的 Taylor 定理给出 $`\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\tfrac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau`$，其中

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

由 (T3)，$`\lvert\varphi'''\rvert\le M_3\eta^3\lVert g\rVert^3`$，又 $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$。所以 $`V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g)`$，$`\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3`$。分别取 $`\mathbb E_P`$、$`\mathbb E_Q`$ 再相减：$`V(\theta)`$ 抵消，一次项与二次项给出式中的两项，而 $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$ 由三角不等式满足所述的界。

*(e).* 对 $`t`$ 归纳得 $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$：$`t=0`$ 时是空和；并且 $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^t(1+\eta L)^s`$。每一步只用到递推，所以等号会传递下去。闭式是几何级数之和。对于反例，$`e_{t+1}=\tfrac32e_t+\tfrac14`$ 给出 $`e_1=\tfrac14`$、$`e_2=\tfrac58`$、$`e_3=\tfrac{19}{16}`$、$`e_4=\tfrac{65}{32}`$。$`\square`$

## 附注

- **信息阶数属于"learner 与部署的复合".** 只有一步凸仿射更新（$`a_i=b_i=1`$）时，(4.8) 回到 $`D_i=N`$，即 [Theorem 4.4(b)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)（直到部署长度的矩是充分且必要的）。后面每一步非线性更新都会把对所有更早块的需求成倍放大。
- **次数依赖坐标.** 可逆的重参数化可能改变更新的多项式形式。内在的对象仍是 response 类及其近似维数（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 的 span 判据与逼近判据）；次数只是在所选概率坐标下一个易于处理的证书。
- **非多项式 learner.** (c) 说明，一旦更新不连续，一般就没有哪一阶矩是充分的；即使只有一次决策，所需的也不止一阶矩。
- **(4.9) 的适用范围.** 这个展开是局部的、单步的：一阶统计量是 consumed gradient 的均值，二阶统计量是梯度二阶矩。对带状态的 optimizer，完整学习器状态的增量取代参数步（[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，任意带状态 optimizer 的一阶表示）；cold 与 warm Adam 分别见 [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3) 与 [Propositions 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)–[5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)。
- **累积需要稳定性.** (e) 对它的递推是紧的。关于 $`R`$ 线性的界需要 $`\eta LR=O(1)`$（因为 $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$），或一个压缩的误差映射。

## 相关工作

- 带积分余项的 Taylor 展开与逐步误差的几何累积都是经典工具，在此注明，不作为新结果。
- 本页新增的是用它们说明：一个反馈来源必须携带的信息由 learner 与部署的复合决定，而不只由数据决定。
