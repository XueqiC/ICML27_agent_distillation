# N20 · 在实际状态上测量一阶统计量 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N20_first_order_audit_ZH.md)

## 设定与假设

沿用节点 [N18](N18_optimizer_transformed_response_ZH.md) 的设定：完整状态 $`x`$，consumed gradient $`G_t(x,Y,\xi)`$，增量 $`\Delta_t(x,g)`$，平均增量 $`\mu^P_t(x)`$，增量二阶矩 $`m^P_{2,t}(x)`$，延续值 $`V^P_{t+1}`$。本节点中 $`\mathbb R^{p+q}`$ 上取 Euclidean 范数，于是空间是实 Hilbert 空间，对偶范数也是 Euclidean 范数。

**假设 B（增量有界）.** 存在 $`b<\infty`$，使得在被 audit 的状态上、对被比较的分布，几乎必然 $`\lVert\Delta_t(x,G_t(x,Y,\xi))\rVert\le b`$。例如：

- 裁剪 SGD 的参数块：$`b=\eta c`$；
- $`p`$ 个参数上的 sign 类更新：$`b=\eta\sqrt p`$；
- 梯度以 $`\bar G`$ 为界的 SGD：$`b=\eta\bar G`$。

**audit 样本.** 在被 audit 的状态 $`x`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid x)`$ 和 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid x)`$，每条都配独立的更新随机性 $`\xi`$，并计算增量 $`X=\Delta_t(x,G_t(x,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。得到这些量需要在 $`x`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

## 陈述

**(a)** **定理（与维数无关的估计）.** 设 $`X_1,\dots,X_n`$ 是实 Hilbert 空间中的 IID 随机元，几乎必然 $`\lVert X\rVert\le b`$，均值为 $`\mu`$。则以至少 $`1-\delta`$ 的概率，

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

对大小分别为 $`n_P`$ 和 $`n_Q`$ 的两个独立样本，以至少 $`1-\delta`$ 的概率，

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

两个界都与维数无关。

**(a′)** **推论（在一个实际状态上的一阶 audit）.** 在产生某个被 audit 状态 $`x`$ 的历史条件下固定 $`x`$。设假设 B 与 $`S_1`$ 在该处成立，常数为 $`\Lambda=\lVert\nabla V^P_{t+1}(x)\rVert`$ 与 $`M_2`$。则在 audit 样本上以至少 $`1-\delta`$ 的概率，

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}2\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

其中

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(a″)** **推论（沿实际训练的整次 audit）.** 独立运行协议 $`Q`$ 共 $`J`$ 次。在每次运行的每一轮 $`t`$，在访问到的状态上以置信度 $`\delta_{\rm a}/(JR)`$ 应用 (a′)，并把右端截断到 $`1`$，记为 $`\hat\varepsilon_{j,t}`$。设在每个访问到的状态上 $`S_1`$ 与 B 都以所述常数成立，且抽取 audit 样本不改变训练过程。则以至少 $`1-\delta_{\rm a}-\delta_{\rm mc}`$ 的概率，

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**哪些便宜，哪些不便宜.** 依赖数据的部分是 $`\bar X_P-\bar X_Q`$ 和 $`\hat m_2`$，它们只需要在访问到的状态上计算 consumed gradient，每个状态 $`n_P+n_Q`$ 次。常数 $`\Lambda`$ 和 $`M_2`$ 是延续值的性质，与消息无关，(a) 不估计它们。它们必须通过解析方式给界，或者用少量完整尾段的 audit 来校准（节点 [N15](N15_adaptive_audit_ZH.md)），并把校准误差加进界里。

**(b)** **定理（按一阶 margin 选择的正确性）.**

- *单边界.* 在状态 $`x`$ 上取候选分布 $`P_j,P_k`$，以及一个满足 $`S_1`$、常数为 $`M_2`$ 的延续值 $`V`$。若

  ```math
  \big\lvert\langle\nabla V(x),\mu^{j}-\mu^{k}\rangle\big\rvert>\tfrac{M_2}2\big(m_2^{j}+m_2^{k}\big),
  ```

  则 $`\mathrm{sign}\big((K^{P_j}-K^{P_k})V(x)\big)=\mathrm{sign}\langle\nabla V(x),\mu^j-\mu^k\rangle`$。
- *整次训练.* 令 $`T_{jk}=\sum_t\mathbb E_{d_t^{k}}\langle\nabla V^{j}_{t+1}(X_t),\mu^j_t-\mu^k_t\rangle`$，$`B_{jk}=\sum_t\mathbb E_{d_t^{k}}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$，即以 $`P_j`$ 充当 $`P`$ 时[N18(b)](N18_optimizer_transformed_response_ZH.md) 的一阶项和余项界。若 $`\lvert T_{jk}\rvert>B_{jk}`$，则 $`\mathrm{sign}\big(L(P_j)-L(P_k)\big)=\mathrm{sign}(T_{jk})`$。

**(b′)** **推论（便宜的分数何时一定排对）.** 设在状态 $`x`$ 上，对延续值 $`V`$ 用分数 $`s(P)`$ 给反馈分布排序。假设：

- (i) $`s(P)-s(P_{\rm ref})=\langle\nabla V(x),\mu^P-\mu^{P_{\rm ref}}\rangle`$，即分数是节点 [N19](N19_learner_statistics_ZH.md) 中经优化器变换的平均增量，并与实际状态上的延续值梯度配对；
- (ii) 被比较的这一对满足 (b) 的 margin 条件。

则该分数对这一对排序正确。缺少任一条件的分数都没有这样的保证：

- 用原始平均梯度的分数，或在不同状态、不同任务间共用系数拟合出来的分数，违反 (i)，因为 $`\nabla V(x)`$ 随状态和任务而变；
- 在 margin 不成立处使用的任何分数，都违反 (ii)。

## 证明

**(a) 的证明。** 令 $`f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert`$。

*均值.* 由 Jensen 不等式和独立性，

```math
\mathbb Ef\le(\mathbb Ef^2)^{1/2}=\Big(\frac{\mathbb E\lVert X-\mu\rVert^2}{n}\Big)^{1/2}.
```

在 Hilbert 空间中 $`\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2`$，所以 $`\mathbb Ef\le b/\sqrt n`$。这里用到的方差恒等式，来自交叉项 $`\mathbb E\langle X_i-\mu,X_l-\mu\rangle`$（$`i\ne l`$）由独立性为零。

*偏差.* 把 $`X_i`$ 换成 $`X_i'`$，$`\bar X`$ 改变 $`(X_i'-X_i)/n`$；由三角不等式，$`f`$ 的改变至多为 $`\lVert X_i-X_i'\rVert/n\le2b/n`$。取 $`c_i=2b/n`$ 的 McDiarmid 有界差分不等式给出

```math
\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp\Big(-\frac{2s^2}{\sum_ic_i^2}\Big)=\exp\Big(-\frac{ns^2}{2b^2}\Big).
```

令其等于 $`\delta`$，得 $`s=b\sqrt{2\log(1/\delta)/n}`$。因此以至少 $`1-\delta`$ 的概率 $`f\le b/\sqrt n+b\sqrt{2\log(1/\delta)/n}`$。

*两个样本.* 对每个样本以 $`\delta/2`$ 应用单样本界，用三角不等式

```math
\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert,
```

再取 union bound。$`\square`$

**(a′) 的证明。** 把 $`\delta/2`$ 分给 (a) 的两样本界，得到 $`e_1`$，其中 $`\log(2/(\delta/2))=\log(4/\delta)`$。再给每个二阶矩各分 $`\delta/4`$。变量 $`\lVert X\rVert^2`$ 取值于 $`[0,b^2]`$，所以单边 Hoeffding 不等式给出：以至少 $`1-\delta/4`$ 的概率 $`m_2^P\le\hat m_2^P+b^2\sqrt{\log(4/\delta)/(2n_P)}`$，$`Q`$ 同理。在三个事件的交上（概率至少 $`1-\delta`$）：

- 由三角不等式，$`\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1`$；
- $`m_2^P+m_2^Q\le\hat m_2^P+\hat m_2^Q+e_2`$。

节点 N18 的[N18(a)](N18_optimizer_transformed_response_ZH.md) 给出 $`\lvert(K^P-K^Q)V(x)\rvert\le\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\rvert+\tfrac{M_2}2(m_2^P+m_2^Q)`$；Cauchy–Schwarz 给出 $`\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert`$。代入上面两个不等式即可。样本是在状态固定之后抽取的，所以论证在给定历史的条件下成立。$`\square`$

**(a″) 的证明。** 令 $`\varepsilon_t(x)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\rvert\rbrace`$。

*audit 事件.* 对每次 audit $`(j,t)`$，在给定历史的条件下 (a′) 以至少 $`1-\delta_{\rm a}/(JR)`$ 的概率成立。条件失败概率之和至多 $`\delta_{\rm a}`$，所以以至少 $`1-\delta_{\rm a}`$ 的概率所有 audit 都成功；再利用 $`\varepsilon_t\le1`$，此时对所有 $`j,t`$ 有 $`\varepsilon_t(X_{j,t})\le\hat\varepsilon_{j,t}`$。

*望远镜.* 以 $`P`$ 为参照的望远镜恒等式 [N05(a)](N05_continuation_response_ZH.md)（节点 [N05](N05_continuation_response_ZH.md)），在期望内取三角不等式，给出 $`\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d_t^Q}\varepsilon_t(X_t)`$。截断到 $`1`$ 是合法的，因为每一项都是两个取值于 $`[0,1]`$ 的期望之差。

*Monte Carlo.* 变量 $`S_j=\sum_t\varepsilon_t(X_{j,t})`$（$`j=1,\dots,J`$）IID 地取值于 $`[0,R]`$，均值为 $`\sum_t\mathbb E_{d_t^Q}\varepsilon_t`$。单边 Hoeffding 不等式给出：以至少 $`1-\delta_{\rm mc}`$ 的概率，

```math
\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\frac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}
```

在两个事件上都有 $`\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}`$，再取 union bound 即完成证明。$`\square`$

**(b)**

*单边界.* 由节点 N18 中取 $`k=1`$ 的[N18(a)](N18_optimizer_transformed_response_ZH.md)，$`(K^{P_j}-K^{P_k})V(x)=s+r`$，其中 $`s=\langle\nabla V(x),\mu^j-\mu^k\rangle`$，$`\lvert r\rvert\le\tfrac{M_2}2(m_2^j+m_2^k)<\lvert s\rvert`$。若 $`s>0`$，则 $`s+r>s-\lvert s\rvert=0`$；若 $`s<0`$，则 $`s+r<s+\lvert s\rvert=0`$。所以 $`s+r`$ 与 $`s`$ 同号。

*整次训练.* 以 $`P_j`$ 充当 $`P`$、$`P_k`$ 充当 $`Q`$ 应用[N18(b)](N18_optimizer_transformed_response_ZH.md)，得 $`L(P_j)-L(P_k)=T_{jk}+\mathcal E`$，且 $`\lvert\mathcal E\rvert\le B_{jk}<\lvert T_{jk}\rvert`$。同样的符号论证成立。$`\square`$

**(b′) 的证明。** 在 (i) 下，参照项相互抵消，$`s(P_j)-s(P_k)=\langle\nabla V(x),\mu^j-\mu^k\rangle`$。在 (ii) 下，(b) 说它与 $`(K^{P_j}-K^{P_k})V(x)`$ 同号。若 (i) 不成立，分数与一阶项之差不受 (b) 控制；若 (ii) 不成立，(b) 不适用，余项 $`r`$ 可能使符号反转。$`\square`$

## 注记

1. **代价.** 在一个状态上做完整尾段 audit，按 Hoeffding 分配，半径 $`0.1`$ 时每个候选需要 $`1016`$ 个配对训练副本（节点 [N15](N15_adaptive_audit_ZH.md)）。(a′) 的一阶 audit 每个候选只需 $`n_P+n_Q`$ 次梯度计算，其半径与参数个数无关。
   - *例：裁剪 SGD.* 当 $`b=\eta c`$ 时，半径 $`e_1`$ 是 $`\eta c/\sqrt n`$ 的固定倍数。
   - *例：sign 类增量.* 对 $`p`$ 个参数上的 sign 类增量，$`b=\eta\sqrt p`$；它们产生的一阶差别也是同一尺度，所以*相对*精度仍由 $`1/\sqrt n`$ 决定。
2. **(a′) 单独做不到的事.** 常数 $`\Lambda`$ 和 $`M_2`$ 属于延续值。(a′) 把 certificate 中数据那一侧变成了便宜的测量，但并不能免去对延续值那一侧的校准；这个校准需要的是少量完整尾段的 audit，而不是成千上万次。
3. **margin 在哪里成立.** 由[N18，注记 3](N18_optimizer_transformed_response_ZH.md#注记)，对 SGD、动量和 sign 优化器，余项每轮是 $`O(\eta^2)`$；对多轮的 Adam 类优化器，余项是 $`\eta`$ 的一阶量，并随 warm-up 减小。因此 (b) 最容易在小步长、非自适应学习器或单边界比较中适用。在其余区间，需要由实测的短尾（节点 [N17](N17_ced_method_ZH.md)）来决定。

## 依赖与出处

- **依赖：** [N18](N18_optimizer_transformed_response_ZH.md)（[N18(a)](N18_optimizer_transformed_response_ZH.md)、[N18(b)](N18_optimizer_transformed_response_ZH.md)）、[N19](N19_learner_statistics_ZH.md)（要估计哪个平均增量）、[N15](N15_adaptive_audit_ZH.md)（完整尾段的校准与代价）。
- **被用于：** [N17](N17_ced_method_ZH.md)。

**出处：** [OTLR §5](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#5-using-the-statistic-at-the-actual-states) — Theorems C5–C6; Corollaries C5′, C5″, C6′. [Optimizer_Transformed_Linear_Response_EN.md](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md)，§5（定理 C5、C6 以及对便宜 proxy 的推论）。推论 C5′ 与 C5″ 把置信度的分配写成显式形式。
