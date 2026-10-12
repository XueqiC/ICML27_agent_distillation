# §8 验证证书（Theorem C） — 证明

[← Framework](../README_ZH.md) · [Proofs](08_verification_ZH.md) · [Experiments](../experiments/08_verification_ZH.md) · [English](08_certificate.md)

本页包含 Proposition 8.4 至 Remark 8.12，包括 Theorem C；[同节页面](08_verification_ZH.md)给出前面的测量、transport 与 adaptive-audit 证明，以及沿用的[设定](08_verification_ZH.md#设定)。

## 测量平均增量

沿用 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

<a id="s8-4"></a>

**Proposition 8.4（在实际状态上测量平均增量）.**

**(a) Hilbert 空间中的集中性.** 设 $`X_1,\dots,X_n`$ 是实 Hilbert 空间中的独立同分布随机元，几乎必然 $`\lVert X\rVert\le b`$，均值为 $`\mu`$。则以至少 $`1-\delta`$ 的概率，

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

对大小为 $`n_P,n_Q`$ 的两个独立样本，以至少 $`1-\delta`$ 的概率，

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) 在一个实际状态上的一阶 audit.** 在产生被 audit 状态 $`z`$ 的历史条件下固定 $`z`$，并设 (B) 与 (S) 在该处成立。则在 audit 样本上以至少 $`1-\delta`$ 的概率，

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

其中

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) 沿实际训练的整次 audit.** 独立运行协议 $`Q`$ 共 $`J`$ 次。在每次运行的每一轮 $`t`$，在访问到的状态上以置信度 $`\delta_{\rm a}/(JR)`$ 应用 (b)，把 (b) 的右端截断到 $`1`$，记为 $`\hat\varepsilon_{j,t}`$。设 (B) 与 (S) 在每个访问到的状态上以所述常数成立，且抽取 audit 样本不改变训练过程。则以至少 $`1-\delta_{\rm a}-\delta_{\rm mc}`$ 的概率，

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**测了什么，没测什么.**

- 依赖数据的部分是 $`\bar X_P-\bar X_Q`$ 与 $`\hat m_2`$，它们只需要在访问到的状态上计算 consumed gradient。
- $`\Lambda`$ 与 $`M_2`$ 是 continuation value 的性质，与消息无关，样本不估计它们。它们必须解析地给界，或者用完整尾段的 audit 校准（[Theorem 8.3](08_verification_ZH.md#s8-3)），并把校准误差加进界里。

*Proof.*

**(a)** 令 $`f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert`$。

- *均值.* 由 Jensen 不等式与独立性，$`\mathbb Ef\le(\mathbb Ef^2)^{1/2}=(\mathbb E\lVert X-\mu\rVert^2/n)^{1/2}`$。交叉项 $`\mathbb E\langle X_i-\mu,X_l-\mu\rangle`$（$`i\ne l`$）由独立性为零，这里用到了内积。在 Hilbert 空间中 $`\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2`$，所以 $`\mathbb Ef\le b/\sqrt n`$。
- *偏离.* 把 $`X_i`$ 换成 $`X_i'`$，$`f`$ 的变化不超过 $`\lVert X_i-X_i'\rVert/n\le2b/n`$。对 $`c_i=2b/n`$ 用 McDiarmid 不等式，$`\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp(-2s^2/\sum_ic_i^2)=\exp(-ns^2/(2b^2))`$。令它等于 $`\delta`$，得 $`s=b\sqrt{2\log(1/\delta)/n}`$。
- *两个样本.* 对每个样本以 $`\delta/2`$ 用单样本的界，再用 $`\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert`$ 并取 union bound。

**(b)** 把 $`\delta/2`$ 分给 (a) 的两样本界，由 $`\log(2/(\delta/2))=\log(4/\delta)`$ 得到 $`e_1`$。把 $`\delta/4`$ 分别分给两个二阶矩：$`\lVert X\rVert^2\in[0,b^2]`$，单侧 Hoeffding 给出以至少 $`1-\delta/4`$ 的概率 $`m^P_2\le\hat m^P_2+b^2\sqrt{\log(4/\delta)/(2n_P)}`$，$`Q`$ 同理。在三个事件的交上，$`\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1`$，且 $`m^P_2+m^Q_2\le\hat m^P_2+\hat m^Q_2+e_2`$。

- 由 (S) 与带积分余项的 Taylor 定理，$`V(z+\Delta)=V(z)+\langle\nabla V(z),\Delta\rangle+\rho(\Delta)`$，$`\lvert\rho(\Delta)\rvert\le\tfrac{M_2}2\lVert\Delta\rVert^2`$。由于 $`(K^P_tV)(z)=\mathbb E_PV(z+\Delta)`$，这就是 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 的单边界情形：

```math
\big\lvert(K^P_t-K^Q_t)V(z)\big\rvert\le\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert+\tfrac{M_2}{2}\big(m^P_2+m^Q_2\big).
```

- 由 Cauchy–Schwarz，$`\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert`$；代入上面两个不等式即可。样本是在状态固定之后抽取的，所以论证在给定历史的条件下成立。

**(c)** 令 $`\varepsilon_t(z)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\rbrace`$。

- *audit 事件.* 每次 audit $`(j,t)`$ 以至少 $`1-\delta_{\rm a}/(JR)`$ 的条件概率成功；条件失败概率之和至多 $`\delta_{\rm a}`$。在全部成功的事件上，利用 $`\varepsilon_t\le1`$，对所有 $`j,t`$ 有 $`\varepsilon_t(Z_{j,t})\le\hat\varepsilon_{j,t}`$。
- *Telescoping.* 以 $`P`$ 为参照应用 [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2)，并在期望内用三角不等式，得 $`\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d^Q_t}\varepsilon_t(Z_t)`$。截断到 $`1`$ 是合法的，因为每一项都是两个取值于 $`[0,1]`$ 的期望之差。
- *Monte Carlo.* 每次运行的总和 $`S_j=\sum_t\varepsilon_t(Z_{j,t})`$ 独立同分布，取值于 $`[0,R]`$，均值为 $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t`$。单侧 Hoeffding 给出以至少 $`1-\delta_{\rm mc}`$ 的概率 $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\tfrac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$。在两个事件上 $`\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}`$，再取 union bound 即完成证明。$`\square`$

<a id="s8-5"></a>

**Remark 8.5（范数与常数）.**

- [Proposition 8.4(a)](#s8-4) 的集中界在状态空间的 Hilbert 范数下成立，证明用到了内积（交叉项为零）。它不能以同样的常数推广到任意范数；例如在 $`\mathbb R^p`$ 的 sup 范数下，相应的界带有 $`\sqrt{\log p}`$ 因子。
- 界中不显含维数，但 $`b`$ 本身可以随维数增长。cold-start Adam 下每个坐标都移动约 $`\eta`$（[Proposition 5.3](05_optimizer_state_ZH.md#s5-3)），所以 $`\lVert\Delta\theta\rVert`$ 可以达到 $`\eta\sqrt p`$。
- 做排序决策只需要标量投影 $`\langle\nabla V(z),\Delta\rangle`$，它的波动可以远小于 $`\lVert\nabla V(z)\rVert\,b`$。[Theorem 8.8](#s8-8) 更进一步：它直接测量有限替换的效应，不需要 $`b`$ 的界。
- **代价对比.** 在一个状态上做完整尾段 audit，每个候选需要 [Theorem 8.3(d)](08_verification_ZH.md#s8-3) 的副本数；(b) 每个候选只需 $`n_P+n_Q`$ 次梯度计算。(b) 单独做不到的，是给出 $`\Lambda`$ 与 $`M_2`$。

## CRN 配对替换证书（Theorem C）

<a id="s8-6"></a>

**Definition 8.6（CRN 配对替换分数）.**

- 在产生它的历史条件下，固定第 $`t`$ 轮的一个被 audit 状态 $`z`$（complete learner state，含 optimizer state）。候选为 $`j=0,1,\dots,M`$，$`M\ge1`$，其中 $`j=0`$ 是参照。候选 $`j`$ 的消息律为 $`P^j=P^j_t(\cdot\mid z)`$，消息 $`Y`$ 产生增量 $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$（含更新随机性）。
- **surrogate 尾段** $`T_\omega`$ 用随机性 $`\omega`$ 运行一段声明的（可以截短的）参照尾段，把一个状态映到最终状态。**surrogate 目标** $`\tilde F`$ 给最终状态打分（例如留出成功轨迹的 teacher-forced log-likelihood，映到 $`[0,1]`$）。记

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

其中 $`\omega`$ 与 $`\Delta^j`$ 独立。$`\tilde L(P^j)`$ 是"这一轮用候选 $`j`$、之后走 surrogate 尾段"的 surrogate 结果。

- **副本** $`\ell=1,\dots,r`$，$`r\ge2`$：抽一个新的 $`\omega_\ell`$，并为每个候选抽一条消息，得到增量 $`\Delta^j_\ell`$。同一副本内所有候选共用 $`\omega_\ell`$（**common random numbers**）；它们的消息也可以用共同的随机数耦合，只要每个 $`\Delta^j_\ell`$ 的边缘分布正确。不同副本相互独立。
- 若候选的消息律支撑有限，可以用支撑上的精确加权平均代替抽样，此时随机性只来自 $`\omega`$。
- 有序对 $`(j,k)`$ 的分数为

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- 任务一侧，$`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ 是"这一轮用候选 $`j`$、之后走完整参照尾段、按任务目标打分"的结果。用设定中的记号，$`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$，所以 $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。

<a id="s8-7"></a>

**Assumption 8.7（副本独立、分数有界、surrogate 对齐）.**

- **(I) 副本独立.** 在产生 $`z`$ 的历史条件下，向量 $`(Z^{jk}_\ell)_{j,k}`$（$`\ell=1,\dots,r`$）独立同分布；$`r`$ 在观察分数之前固定。
- **(B) 分数有界.** 对每一对，$`Z^{jk}_\ell`$ 几乎必然落在一个已知的、长度为 $`B`$ 的区间里。例如 $`\tilde F\in[0,1]`$ 时 $`B\le2`$。
- **(A) surrogate 对齐.** 存在 $`\alpha>0`$ 与 $`\varepsilon_{\rm sur}\ge0`$，使对所有候选对，

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ 同时吸收了 surrogate 与任务目标之间的差别，以及尾段截短带来的差别。

<a id="s8-8"></a>

**Theorem 8.8（CRN 配对替换证书；Theorem C）.** 在 [Assumption 8.7](#s8-7) 下，令 $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ 为无偏样本方差，

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)} .
```

- **(a) 精确性.** $`\mathbb E\,Z^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$，不涉及任何一阶近似。
- **(b) 证书.** 以至少 $`1-\delta`$ 的概率，对所有有序对 $`(j,k)`$（$`j\ne k`$）同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- **(c) 判定规则.** 只要 $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ 就判定"$`j`$ 优于 $`k`$"，则以至少 $`1-\delta`$ 的概率，所有对上都不会出现错误的判定。取 $`k=0`$，这就认证了 $`A_t(z,j)>0`$。
- 这个证书不需要 continuation 梯度 $`\nabla V`$，不需要曲率常数，也没有二阶余项。唯一需要校准的常数是 $`(\alpha,\varepsilon_{\rm sur})`$（[Remark 8.11](#s8-11)）。

*Proof.*

**(a)** 由期望的线性，$`\mathbb EZ^{jk}_\ell=\mathbb E\varphi_{\omega_\ell}(\Delta^j_\ell)-\mathbb E\varphi_{\omega_\ell}(\Delta^k_\ell)`$。每一项只依赖于 $`(\omega_\ell,\Delta^j_\ell)`$ 的联合分布，它是 $`\omega`$ 的分布与 $`\Delta^j`$ 的边缘分布的乘积，与候选之间用什么耦合无关。所以两项分别等于 $`\tilde L(P^j)`$ 与 $`\tilde L(P^k)`$。在有限支撑上精确枚举时，把抽样换成内层平均，结论相同。

**(b)** *Empirical Bernstein 界.* Maurer 与 Pontil（2009）证明了如下 empirical Bernstein 界：对 $`n\ge2`$ 个取值于 $`[0,1]`$ 的独立同分布变量 $`X_i`$，记其无偏样本方差为 $`V_n`$，则以至少 $`1-\delta'`$ 的概率，

```math
\mathbb EX-\frac1n\sum_iX_i\le\sqrt{\frac{2V_n\log(2/\delta')}{n}}+\frac{7\log(2/\delta')}{3(n-1)} .
```

- 若 $`Z`$ 取值于区间 $`[a,a+B]`$，对 $`X=(Z-a)/B\in[0,1]`$ 用上式（其样本方差为 $`V_n(Z)/B^2`$），再乘以 $`B`$，得到关于 $`Z`$ 的同一形式的界：$`\sqrt{2V_n(Z)\log(2/\delta')/n}+7B\log(2/\delta')/(3(n-1))`$。
- *事件.* 对每个有序对 $`(j,k)`$，把上述界用于 $`X=-Z^{jk}`$（取值于长度为 $`B`$ 的区间，样本方差相同），得到下尾 $`\tilde L(P^j)-\tilde L(P^k)\ge\hat S^{jk}-e^{jk}_{\rm stat}`$。有序对共有 $`(M+1)M`$ 个，而 $`M\ge1`$ 时 $`(M+1)M\le2M^2`$。每个取 $`\delta'=\delta/(2M^2)`$，于是 $`\log(2/\delta')=\log(4M^2/\delta)`$。union bound 给出全部成立的概率至少为 $`1-\delta`$。$`(k,j)`$ 的事件就是 $`(j,k)`$ 的上尾，所以对每个无序对界都是双侧的。
- *转到任务.* 在该事件上，由 (A)，$`L_F(P^j)-L_F(P^k)\ge\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)-\alpha\varepsilon_{\rm sur}\ge\alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big)`$。

**(c)** 在 (b) 的事件上，由 $`\alpha>0`$，$`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ 推出 $`L_F(P^j)-L_F(P^k)>0`$。取 $`k=0`$，$`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。$`\square`$

*关于 Theorem 8.8 的说明.*

- **估计的是什么.** 被估计的量是 [Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 中一轮替换的有限对比，在 surrogate 尾段上测量。这里没有任何线性化，所以 §7 的二阶矩通道虽然关于消息幅度是非线性、会饱和的（[Corollary 7.2](07_second_moment_channel_ZH.md#s7-2)），也能按原样被测到。
- **假设在哪里起作用.** (I) 要求在固定状态上每个副本都用新的 $`\omega_\ell`$ 与新的消息。(B) 需要一个事先已知的取值范围：分数的观测范围不是合法的 $`B`$。(A) 是关于总体差值的陈述，需要校准（[Remark 8.11](#s8-11)）。

<a id="s8-9"></a>

**Corollary 8.9（一阶版本、它的 margin 判据与上限）.** 沿用 [Definition 8.6](#s8-6) 的设定，令 $`\mu^j=\mathbb E\Delta^j`$，$`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$。

- **(a) margin 判据.** 设 continuation $`V`$ 沿到达的线段二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$。若 $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$，则 $`(K^j_t-K^k_t)V(z)`$ 与 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ 同号。整次训练层面，令 $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ 与 $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ 为 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 中以 $`P_j`$ 作 $`P`$ 时的一阶项与余项界；这里 $`d^k_t`$ 是由 $`P_k`$ 驱动的过程在第 $`t`$ 轮的状态分布。若 $`\lvert T_{jk}\rvert>B_{jk}`$，则 $`L(P_j)-L(P_k)`$ 与 $`T_{jk}`$ 同号。
- **(b) 测得的一阶证书.** 设 $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ 对到达的增量 $`u`$ 沿 $`\lbrace su:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$；令 $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$。对步长 $`h\in(0,1]`$，定义 $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$，其平均为 $`\hat S^{jk}(h)`$；设 (I)、(A) 成立，(B) 以区间长度 $`B_h`$ 成立。记 $`e^{jk}_{\rm stat}(h)`$ 为把 [Theorem 8.8](#s8-8) 的半径中 $`B`$ 换成 $`B_h`$ 得到的量，并令

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

则以至少 $`1-\delta`$ 的概率，对所有有序对同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) 上限.** 在 (b) 的事件上，一个候选对能被 (b) 认证，仅当 $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$。一阶项不超过二阶余项界的候选对，无论真实差距多大，在一阶意义下都不会被认证。特别地，若两个候选的平均增量沿 $`\nabla\tilde V(0)`$ 方向一致，它们只在增量的二阶及更高阶矩上不同，这样的候选对永远不会被一阶证书认证；[Theorem 8.8](#s8-8) 则可以认证它们。按总体量来说，(b) 认证的是下界 $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$，而右端正是 Theorem 8.8 直接估计的量。
- **(d) oracle 分数不是证书.** 用任务的精确 continuation 梯度算出的分数 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$（例如受控实验中对整个参照尾段求导得到的），在 margin 成立的地方满足 (a)。但它不是可测量的证书：$`\nabla V(z)`$ 与 $`M_2`$ 不能从反馈样本中得到，而检查 margin 两者都需要。一个便宜的分数要按 (a) 排对，需要：(i) 分数之差等于用实际学习器的 optimizer 变换后平均增量（[Theorem 5.10](05_learner_statistics_ZH.md#s5-10)）在实际状态上算出的一阶项；(ii) margin 成立。用原始平均梯度构造的分数，或在不同状态与任务之间共用系数的分数，违反 (i)；在 margin 不成立处使用的任何分数，违反 (ii)。

*Proof.*

**(a)** *单边界.* 由 [Proposition 8.4(b)](#s8-4) 证明中的 Taylor 界，$`(K^j_t-K^k_t)V(z)=s+\rho`$，其中 $`s=\langle\nabla V(z),\mu^j-\mu^k\rangle`$，$`\lvert\rho\rvert\le\tfrac{M_2}2(m^j_2+m^k_2)<\lvert s\rvert`$。若 $`s>0`$ 则 $`s+\rho>s-\lvert s\rvert=0`$；若 $`s<0`$ 则 $`s+\rho<0`$。*整次训练.* 以 $`P_j`$ 作 $`P`$、$`P_k`$ 作 $`Q`$ 应用 [Proposition 5.1(b)](05_optimizer_state_ZH.md#s5-1)，得 $`L(P_j)-L(P_k)=T_{jk}+\mathcal E`$，$`\lvert\mathcal E\rvert\le B_{jk}<\lvert T_{jk}\rvert`$；同样的符号论证适用。

**(b)** 沿 $`s\mapsto\tilde V(shu)`$ 用带积分余项的 Taylor 定理，对每个到达的增量 $`u`$，

```math
\Big\lvert\tilde V(hu)-\tilde V(0)-h\langle\nabla\tilde V(0),u\rangle\Big\rvert\le\tfrac12h^2\tilde M\lVert u\rVert^2 .
```

- *有限差分的偏差.* 与 [Theorem 8.8(a)](#s8-8) 相同，$`\mathbb EZ^{jk}_\ell(h)=\big(\mathbb E\tilde V(h\Delta^j)-\mathbb E\tilde V(h\Delta^k)\big)/h`$。对 $`u=\Delta^j`$ 与 $`u=\Delta^k`$ 的上式取期望并除以 $`h`$，得 $`\lvert\mathbb EZ^{jk}_\ell(h)-\tilde s^{jk}\rvert\le e^{jk}_{\rm fd}(h)`$。
- *余项.* 取 $`h=1`$，得 $`\tilde L(P^j)-\tilde L(P^k)=\tilde s^{jk}+\tilde\rho`$，$`\lvert\tilde\rho\rvert\le\tilde e^{jk}_2`$。
- *集中性.* 把 [Theorem 8.8(b)](#s8-8) 的论证用于取值范围为 $`B_h`$ 的 $`Z^{jk}_\ell(h)`$，得到概率至少为 $`1-\delta`$ 的事件，其上对所有有序对 $`\mathbb EZ^{jk}_\ell(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)`$。
- 在该事件上，$`\tilde L(P^j)-\tilde L(P^k)\ge\tilde s^{jk}-\tilde e^{jk}_2\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2`$；再如 Theorem 8.8 一样用 (A) 转到任务。

**(c)** 被 (b) 认证意味着 $`\hat S^{jk}(h)>e^{jk}_{\rm stat}(h)+e^{jk}_{\rm fd}(h)+\tilde e^{jk}_2+\varepsilon_{\rm sur}`$。在 (b) 的事件上，$`\tilde s^{jk}\ge\mathbb EZ^{jk}_\ell(h)-e^{jk}_{\rm fd}(h)\ge\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$。若平均增量沿 $`\nabla\tilde V(0)`$ 一致，则 $`\tilde s^{jk}=0`$，不可能被认证。(c) 的最后一句就是 (b) 中的余项步骤。

**(d)** 第一个结论就是 (a)。关于 (i)：若分数之差等于 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ 且 margin 成立，(a) 给出符号。若分数与这一项不同，(a) 不控制两者之差；若 margin 不成立，余项可以反转符号。$`\square`$

*关于 Corollary 8.9 的说明.*

- **一阶版本为什么弱.** 两个效应叠加。$`Z(h)`$ 的先验取值范围随 $`1/h`$ 增长（$`\tilde F\in[0,1]`$ 时 $`B_h\le2/h`$），所以 $`h`$ 变小时 $`e_{\rm stat}(h)`$ 的范围项变大，而 $`e_{\rm fd}(h)`$ 随 $`h`$ 变大。并且余项 $`\tilde e_2`$ 完全不随 $`r`$ 缩小。
- **margin 在哪里成立.** 若 continuation 有曲率的每个坐标都只移动 $`O(\eta)`$（参数块上的 SGD、momentum、sign 类更新），则每一轮的一阶项是 $`O(\eta)`$、余项是 $`O(\eta^2)`$，步长小时 margin 成立。在 Adam 下，moment 增量 $`\Delta m=(1-\beta_1)(G-m)`$ 与 $`\Delta v=(1-\beta_2)(G\odot G-v)`$ 是 $`O(1)`$ 的，而 continuation 对 $`(m,v)`$ 的依赖只通过大小为 $`\eta`$ 的步长体现。于是一阶项与余项都是 $`O(\eta)`$，步长小并不意味着 margin 成立。这正是 §7 中二阶矩通道所在的区域（[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1)）。

<a id="s8-10"></a>

**Proposition 8.10（代价，以及 common random numbers 带来的好处）.**

- **(a) 代价.** 一个被 audit 的状态需要 $`(M+1)r`$ 条 surrogate 尾段，外加校准用的任务评测尾段（[Remark 8.11](#s8-11)）。以 [Theorem 8.3(d)](08_verification_ZH.md#s8-3) 的半径做完整尾段的配对 audit，每个候选需要 $`2n_t`$ 条完整尾段。
- **(b) 范围项所需的副本数.** $`e_{\rm stat}`$ 的范围项不超过 $`\zeta B`$，当且仅当

```math
r\ \ge\ 1+\frac{7\log(4M^2/\delta)}{3\zeta}.
```

例如 $`M=8`$、$`\delta=0.05`$、$`\zeta=0.2`$ 时，$`\log(4M^2/\delta)=\log5120\approx8.54`$，需要 $`r\ge101`$。副本更少时，范围项会主导半径。

- **(c) common random numbers 下的方差.** 设 $`\varphi_\omega`$ 在到达的增量上 pathwise Lipschitz，常数为 $`\lambda_\omega`$：$`\lvert\varphi_\omega(u)-\varphi_\omega(u')\rvert\le\lambda_\omega\lVert u-u'\rVert`$。则

```math
\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\,\lVert\Delta^j_\ell-\Delta^k_\ell\rVert,\qquad\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E\big[\lambda_\omega^2\,\lVert\Delta^j-\Delta^k\rVert^2\big].
```

若两臂用独立的尾段 $`\omega,\omega'`$（消息也独立），则 $`\varphi_\omega(\Delta^j)-\varphi_{\omega'}(\Delta^k)`$ 的方差为 $`\mathrm{Var}\,\varphi_\omega(\Delta^j)+\mathrm{Var}\,\varphi_{\omega'}(\Delta^k)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)+\mathbb E\,\mathrm{Var}(\varphi_{\omega'}(\Delta^k)\mid\Delta^k)`$，其中含有尾段本身的噪声，不会随两个候选靠近而消失。若进一步 $`\sup_\omega\lambda_\omega\le\bar\lambda`$，且在耦合下几乎必然 $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$，则 $`B=2\bar\lambda b_{jk}`$ 是 [Assumption 8.7(B)](#s8-7) 中合法的取值范围。

*Proof.* (a) 是计数：每个副本对 $`M+1`$ 个候选各评估一次。(b) 范围项 $`7B\log(4M^2/\delta)/(3(r-1))\le\zeta B`$ 整理后即为所示不等式；对例子，$`7\cdot8.541/(3\cdot0.2)\approx99.6`$，所以 $`r\ge100.6`$。(c) 第一个不等式是 Lipschitz 条件在 $`u=\Delta^j_\ell`$、$`u'=\Delta^k_\ell`$ 处的形式；然后 $`\mathrm{Var}\,Z\le\mathbb EZ^2`$。两臂独立时差的方差是方差之和，再由全方差公式 $`\mathrm{Var}\,\varphi_\omega(\Delta^j)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)`$。在两个一致界下，$`Z^{jk}_\ell\in[-\bar\lambda b_{jk},\bar\lambda b_{jk}]`$。$`\square`$

<a id="s8-11"></a>

**Remark 8.11（surrogate 的校准；沿训练的迁移）.**

- **校准.** 取一组校准用的候选对，用任务评测的配对完整尾段估计 $`L_F(P^j)-L_F(P^k)`$（[Theorem 8.3](08_verification_ZH.md#s8-3)），对 $`\hat S^{jk}`$ 回归，读出 $`\hat\alpha`$ 与残差上界 $`\hat\varepsilon_{\rm sur}`$。把 $`(\hat\alpha,\hat\varepsilon_{\rm sur})`$ 用到其他候选对上，需要一个可交换性假设：校准对与新的候选对来自同一族。这是假设，不是定理；校准本身的统计误差也要加进去。
- **在自适应 audit 中使用.** 在被 audit 的轮次，把 [Theorem 8.3](08_verification_ZH.md#s8-3) 的完整尾段置信下界换成被选候选的 $`\Gamma_t=\min\lbrace1,\alpha(\hat S^{J_t0}-e^{J_t0}_{\rm stat}-\varepsilon_{\rm sur})\rbrace`$；没有候选相对参照被认证时令 $`\Gamma_t=0`$。若 (A) 在每个被 audit 的状态上以声明的常数成立，且置信分配之和为 $`\delta`$，则 Theorem 8.3(c) 原样成立。它的证明只用到 audit 事件上的 $`A_t\ge\Gamma_t`$ 与 $`0\le\Gamma_t\le1`$；由于 $`A_t(z,j)=L_F(P^j)-L_F(P^0)`$，前者由 [Theorem 8.8(b)](#s8-8) 给出。
- **沿训练的迁移.** 证书在被 audit 的状态上成立。要把选出的候选用到之后的状态上，需要 [Theorem 8.2](08_verification_ZH.md#s8-2) 的 transport 项，或者在下一个检查点重新 audit，核对排序是否保持（a posteriori 检查）。

<a id="s8-12"></a>

**Remark 8.12（未解决的问题）.**

- **可证明的取值范围.** [Theorem 8.8](#s8-8) 在先验范围下成立（$`\tilde F\in[0,1]`$ 时 $`B\le2`$）。这时范围项的量级为 $`7B\log(4M^2/\delta)/(3r)`$，对任何负担得起的 $`r`$，都可能远大于相邻候选之间的差距。分数的观测范围不能代替它。缺的是一个可证明的、远小于先验范围的界，例如 [Proposition 8.10(c)](#s8-10) 中的 pathwise Lipschitz 界，配上一个对真实训练尾段被证明的 $`\bar\lambda`$。
- **校准的迁移.** 需要一个可检验的条件，使在校准对上拟合的 $`(\alpha,\varepsilon_{\rm sur})`$ 在新的候选对与之后的状态上仍然有效。
- 在这两点解决之前，Theorem 8.8 在实践中是一个无偏、低方差的排序分数；只有当 $`B`$ 与 $`(\alpha,\varepsilon_{\rm sur})`$ 被独立论证时，它才是严格的证书。

## Related work

- [Theorem 8.1](08_verification_ZH.md#s8-1) 是把线性规划与范数对偶用在可操作的差距上，不主张新的对偶原理。
- [Theorems 8.2](08_verification_ZH.md#s8-2)–[8.3](08_verification_ZH.md#s8-3) 与 [Proposition 8.4](#s8-4) 用到 Hoeffding 不等式、union bound 与 McDiarmid 不等式。按置信下界选择以保证不退化，是 high-confidence policy improvement（Thomas et al., 2015）的思路。
- [Theorem 8.8](#s8-8) 把 empirical Bernstein 界（Maurer & Pontil, 2009；Audibert et al., 2009）与 common random numbers（模拟中经典的方差缩减手段，Glasserman, 2004）结合起来。它的内容在于被估计的量：实际 complete learner state 上一轮替换的有限对比，在经过校准的 surrogate 尾段上测量。
- 一阶的数据归因分数，如 influence functions（Koh & Liang, 2017）、TracIn（Pruthi et al., 2020）、In-Run Data Shapley（Wang et al., 2025）与 metagradient（Engstrom et al., 2025），使用目标关于训练数据或超参数的梯度。[Corollary 8.9](#s8-9) 说明这类一阶排序何时有保证，以及为什么它们无法认证低于余项的候选对。Datamodels（Ilyas et al., 2022）通过对大量训练运行做回归来估计有限的重训练效应；Theorem 8.8 则在固定状态上用配对尾段测量一轮的有限效应。

**参考文献.**

- Audibert, J.-Y., Munos, R., Szepesvári, C. (2009). Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, no. 19.
- Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., Madry, A. (2022). Datamodels: Predicting predictions from training data. ICML.
- Koh, P. W., Liang, P. (2017). Understanding black-box predictions via influence functions. ICML.
- Maurer, A., Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. COLT.
- Pruthi, G., Liu, F., Kale, S., Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. NeurIPS.
- Thomas, P., Theocharous, G., Ghavamzadeh, M. (2015). High confidence policy improvement. ICML.
- Wang, J. T., Mittal, P., Song, D., Jia, R. (2025). Data Shapley in one training run. ICLR.
