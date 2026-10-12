# §5 Optimizer state 与一阶表示 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/05_optimizer_state_ZH.md) · [English](05_optimizer_state.md)

本页保留 Proposition 5.1 至 Proposition 5.3 及设定；后续证明分为 [Warm Adam](05_warm_adam_ZH.md) 和[学习器统计量](05_learner_statistics_ZH.md)。

**本节概要.**

- 带状态的一阶 optimizer 是一个映射：把 consumed gradient 映成*完整* learner state $`z=(\theta,o)`$ 的增量。Proposition 5.1 按这个增量的各阶矩展开一个 feedback law 的效应。在一阶意义下，feedback law 必须携带的是平均增量 $`\mu`$ 在 continuation value 梯度方向上的部分。这是一个工具：同类计算见 stochastic modified equations、TracIn 与 metagradient（Remark 5.2）。
- 对 Adam，统计量取决于 optimizer state。
  - cold 状态下，参数步只看梯度的符号分布，而矩保留了梯度的幅度（Proposition 5.3）。
  - warm 状态下，精确统计量是 Adam 增量的均值。$`(\mathbb EG,\mathbb E\,G\odot G)`$ 只在 frozen-denominator 近似下决定它，误差由 warmth 控制（Proposition 5.5–5.7）。
  - $`\beta_2`$ 固定时 warmth 不会趋于零：早期按 $`1/(\beta_2s)`$ 下降，之后停在平台 $`(1-\beta_2)/\beta_2`$（Proposition 5.8）。
- 梯度裁剪下统计量是裁剪后的均值；只用平均方向，误差为 $`c(1-p_{\rm act})`$（Proposition 5.9）。
- Theorem 5.10 列出 SGD、momentum、warm 与 cold Adam、裁剪、Lion、LoRA 各自的一阶统计量。Remark 5.11 说明在真实模型上检验 warm 状态条件时应当测什么。
- 用到：[Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式与停止版本；[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2) 的充分性对偶；[Theorem 4.4](04_complexity_of_sufficiency_ZH.md#s4-4) 的矩匹配分布；[Theorem 4.7](04_rates_and_composition_ZH.md#s4-7) 的 SGD 展开。
- 被用于：第 6 节的 learner Jacobian（[Definition 6.2](06_learner_relative_information_ZH.md#s6-2)、[Proposition 6.11](06_learner_relative_information_ZH.md#s6-11)）；第 7 节的状态通道分解（[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1)）；第 8 节的测量结果（[Proposition 8.4](08_certificate_ZH.md#s8-4)、[Corollary 8.9](08_certificate_ZH.md#s8-9)）；第 9 节的方法。

## 设定

全节沿用 [Definition 1.3–1.4](../README_ZH.md#s1-3) 的记号。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

**条件 $`(\mathrm S_k)`$（可达线段上光滑的 continuation value）.** 令 $`\mathcal S_{t,z}`$ 为 $`z`$ 与所有点 $`z+\Delta_t(z,g)`$ 的凸包，其中 $`g`$ 取遍被比较分布下 consumed gradient 的支撑。要求：

- $`V=V_{t+1}`$ 在 $`\mathcal S_{t,z}`$ 上属于 $`C^{k+1}`$；
- 对每个 $`w\in\mathcal S_{t,z}`$ 与每个 $`h`$，$`\lvert\nabla^{k+1}V(w)[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}`$；
- 矩 $`m^P_{k+1,t}(z)`$ 与 $`m^Q_{k+1,t}(z)`$ 有限。

**$`(\mathrm S_k)`$ 的适用范围.**

- 在裁剪或 sign 更新下，多步 continuation value 一般在这样的状态处不光滑：之后某步的梯度跨过裁剪阈值，或之后某步的更新符号翻转。此时 $`(\mathrm S_k)`$ 只在避开这些 kink 的区域成立，例如裁剪始终激活，或 momentum 主导每个符号的区域。
- 这些 optimizer 的平均增量在 Theorem 5.10 中精确算出，不需要任何光滑性。下面的余项界只在 $`(\mathrm S_k)`$ 成立处才声称成立。

**声明族.** 对 Proposition 5.1(c)，固定边界 $`(t,z)`$ 和一个 continuation value 的集合 $`\mathcal V`$，每个关心的部署任务与参考尾段对应其中一个；每个都在 $`\mathcal S_{t,z}`$ 上满足 $`(\mathrm S_1)`$，并共用 Hessian 界 $`M_2`$。令

```math
U=\mathrm{span}\lbrace \nabla V(z):V\in\mathcal V\rbrace ,\qquad \mathcal G^{(1)}_{t,z}=\lbrace\,y\mapsto\langle\nabla V(z),D_t(z,y)\rangle:\ V\in\mathcal V\,\rbrace .
```

$`\mathcal G^{(1)}_{t,z}`$ 称为一阶 response 类。

**Adam 与 AdamW.** Adam 带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

## 一阶表示

<a id="s5-1"></a>

**Proposition 5.1（optimizer 变换后的展开）.**

**(a) 单个边界，$`k`$ 阶.** 在 $`(\mathrm S_k)`$ 下，

```math
(K_t^P-K_t^Q)V(z)=\sum_{j=1}^{k}\frac1{j!}\Big\langle\nabla^jV(z),\;\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\Big\rangle+r_k,\qquad\lvert r_k\rvert\le\frac{M_{k+1}}{(k+1)!}\big(m_{k+1}^P+m_{k+1}^Q\big).
```

特别地，$`k=1`$ 时

```math
(K_t^P-K_t^Q)V(z)=\big\langle\nabla V(z),\mu_t^P(z)-\mu_t^Q(z)\big\rangle+r_1,\qquad\lvert r_1\rvert\le\tfrac{M_2}2\big(m_2^P+m_2^Q\big).
```

**(b) 整个训练过程.** 从相同的初始分布出发，比较协议 $`P`$ 与 $`Q`$ 的 $`R`$ 轮训练。设在 $`Q`$ 访问的每个状态上 $`(\mathrm S_1)`$ 成立，continuation value $`V^P_{t+1}`$ 的常数为 $`M_{2,t}(z)`$。则

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\Big\langle\nabla V^P_{t+1}(Z_t),\;\mu_t^P(Z_t)-\mu_t^Q(Z_t)\Big\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\Big[\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big].
```

因此，记 $`\Lambda_t(z)=\lVert\nabla V^P_{t+1}(z)\rVert_*`$，

```math
\lvert L(P)-L(Q)\rvert\le\min\Big\lbrace 1,\ \sum_t\mathbb E_{d_t^Q}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\Big\rbrace .
```

*停止版本.* 设 $`(\mathrm S_1)`$ 只在已认证的集合 $`C_t\subseteq\mathcal Z`$ 上成立。令 $`\tau=\min\lbrace t<R:Z_t^Q\notin C_t\rbrace`$（若不存在这样的 $`t`$ 则 $`\tau=R`$），$`\rho=\Pr(\tau<R)`$。则

```math
\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\,(Z_t)+\rho.
```

**(c) 一阶的充分性、必要性与最小性.** 用 Euclid 内积，$`\Pi_U`$ 为到 $`U`$ 的正交投影。

- *充分性.* 若 $`\Pi_U(\mu^P-\mu^Q)=0`$，则对每个 $`V\in\mathcal V`$ 有 $`\lvert(K^P-K^Q)V(z)\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$。
- *必要性.* 若 $`\Pi_U(\mu^P-\mu^Q)\ne0`$，则存在 $`V\in\mathcal V`$ 使 $`\langle\nabla V(z),\mu^P-\mu^Q\rangle\ne0`$，且对每个这样的 $`V`$ 下式成立。所以只要一阶项超过余项界，这个任务就能察觉这一失配。

```math
\lvert(K^P-K^Q)V(z)\rvert\ge\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert-\tfrac{M_2}2(m_2^P+m_2^Q).
```

- *最小性.* 在大小为 $`m`$ 的有限消息字母表上，能对所有分布对决定每个一阶项 $`\langle\nabla V(z),\mu^P-\mu^Q\rangle`$（$`V\in\mathcal V`$）的线性期望统计量的最少个数为下式。

```math
d_0^{(1)}=\dim\mathrm{span}\big(\lbrace 1\rbrace \cup\mathcal G^{(1)}_{t,z}\big)-1\le\min\lbrace \dim U,\ m-1\rbrace .
```

**(d) 矩层级.** 在 $`(\mathrm S_k)`$ 下，设两个分布在某状态处对 $`j=1,\dots,k`$ 都有相同的 $`\mathbb E\Delta^{\otimes j}`$。则它们在该处的单边界差距至多为 $`\tfrac{M_{k+1}}{(k+1)!}(m_{k+1}^P+m_{k+1}^Q)`$。像 (b) 那样对访问到的状态求和，就得到整个训练过程差距的界。

- 对 SGD，$`\Delta=-\eta G`$，所以要匹配的是梯度矩 $`\mathbb E G^{\otimes j}`$，每轮的界为 $`O(\eta^{k+1})`$。
- 对 cold Adam，参数块要匹配的是 $`h_\epsilon(G)`$ 的矩（Proposition 5.3）；当 $`\epsilon\to0`$ 时就是联合符号矩。
- 在始终激活的裁剪下，参数块只涉及方向 $`G/\lVert G\rVert`$ 的矩。

*(a) 的证明.* 固定一个实现的增量 $`\Delta=\Delta_t(z,g)`$，$`g`$ 属于支撑。线段 $`\lbrace z+u\Delta:u\in[0,1]\rbrace`$ 落在 $`\mathcal S_{t,z}`$ 中，因为 $`\mathcal S_{t,z}`$ 是凸的并且含两个端点。在这条线段上 $`\varphi(u)=V(z+u\Delta)`$ 属于 $`C^{k+1}`$，且 $`\varphi^{(j)}(u)=\nabla^jV(z+u\Delta)[\Delta^{\otimes j}]`$。带 Lagrange 余项的 Taylor 定理给出某个 $`u^*\in(0,1)`$，使得

```math
V(z+\Delta)=V(z)+\sum_{j=1}^{k}\frac1{j!}\nabla^jV(z)[\Delta^{\otimes j}]+R(\Delta),\qquad R(\Delta)=\frac1{(k+1)!}\nabla^{k+1}V(z+u^*\Delta)[\Delta^{\otimes(k+1)}].
```

由 $`(\mathrm S_k)`$，$`\lvert R(\Delta)\rvert\le\tfrac{M_{k+1}}{(k+1)!}\lVert\Delta\rVert^{k+1}`$。

对 $`Y\sim P`$ 与 $`\xi`$ 积分这个恒等式，再对 $`Y\sim Q`$ 与 $`\xi`$ 分别积分。每一项都可积：$`V`$ 取值于 $`[0,1]`$；$`\lvert R\rvert`$ 被 $`\lVert\Delta\rVert^{k+1}`$ 的倍数控制，后者期望有限；每个 $`\nabla^jV(z)[\Delta^{\otimes j}]`$（$`j\le k`$）是 $`\Delta`$ 的多项式，其 $`j`$ 次矩由 Lyapunov 不等式有限。两式相减：

- 常数 $`V(z)`$ 抵消，因为两个分布都是概率测度；
- 映射 $`\Delta^{\otimes j}\mapsto\nabla^jV(z)[\Delta^{\otimes j}]`$ 是线性的，期望可以移进去：$`\mathbb E_P\nabla^jV(z)[\Delta^{\otimes j}]=\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}\rangle`$；
- 余项给出 $`r_k=\mathbb E_PR-\mathbb E_QR`$，且 $`\lvert r_k\rvert\le\mathbb E_P\lvert R\rvert+\mathbb E_Q\lvert R\rvert\le\tfrac{M_{k+1}}{(k+1)!}(m^P_{k+1}+m^Q_{k+1})`$。

当 $`k=1`$ 时，对 $`\xi`$ 用 tower property 得 $`\mathbb E_P\Delta=\mathbb E_{Y\sim P}D_t(z,Y)=\mu_t^P(z)`$。$`\square`$

*(b) 的证明.* [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式对从相同初始分布出发的任意两个协议都成立，与哪一个被叫作参考无关。取 $`P`$ 为参考、$`Q`$ 为替代，

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(Z_t)\big].
```

在每个访问到的状态 $`Z_t=z`$ 上，$`V^P_{t+1}`$ 满足 $`(\mathrm S_1)`$，于是 (a) 取 $`k=1`$ 给出

```math
(K_t^P-K_t^Q)V^P_{t+1}(z)=\langle\nabla V^P_{t+1}(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_{1,t}(z),\qquad\lvert r_{1,t}(z)\rvert\le\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big).
```

- 映射 $`z\mapsto\mu_t(z)`$、$`m_{2,t}(z)`$、$`\nabla V^P_{t+1}(z)`$ 都可测。对 $`d_t^Q`$ 积分并对 $`t`$ 求和即得表示式，其中 $`\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}`$，界如所述。
- 对证书形式，每个一阶项用 $`\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert`$，求和用三角不等式。截断在 $`1`$ 是因为 $`L(P),L(Q)\in[0,1]`$。
- 对停止版本，在 $`C_t`$ 上定义 $`\varepsilon_t(z)=\Lambda_t(z)\lVert\mu^P_t(z)-\mu^Q_t(z)\rVert+\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big)`$。由 (a) 与对偶范数不等式，对 $`z\in C_t`$ 有 $`\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\le\varepsilon_t(z)`$。以 $`P`$ 为参考应用 [Theorem 3.2(c)](03_learning_consequence_ZH.md#s3-2) 的停止版本，得 $`\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(Z_t)+\rho`$。$`\square`$

*(c) 的证明.* 记 $`w=\mu^P-\mu^Q`$。

- *充分性.* 对 $`V\in\mathcal V`$，$`\nabla V(z)\in U`$。所以 $`\langle\nabla V(z),w\rangle=\langle\nabla V(z),\Pi_Uw\rangle=0`$，(a) 取 $`k=1`$ 后只剩 $`r_1`$。
- *必要性.* 若对每个 $`V\in\mathcal V`$ 都有 $`\langle\nabla V(z),w\rangle=0`$，则 $`w`$ 与 $`U`$ 的一组张成元都正交，故 $`\Pi_Uw=0`$，矛盾。对 $`\langle\nabla V(z),w\rangle\ne0`$ 的 $`V`$，(a) 给出 $`(K^P-K^Q)V(z)=\langle\nabla V(z),w\rangle+r_1`$；用反向三角不等式和 $`\lvert r_1\rvert`$ 的界即得。
- *最小性.* 对 $`V\in\mathcal V`$，一阶项是消息分布的线性泛函：

```math
\langle\nabla V(z),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(z),D_t(z,y)\rangle\in\mathcal G^{(1)}_{t,z}.
```

- 所以"用特征 $`\phi_1,\dots,\phi_d`$ 的均值对所有分布决定每个一阶项"，就是 [Theorem 4.2(a)](04_complexity_of_sufficiency_ZH.md#s4-2) 对 response 类 $`\mathcal G^{(1)}_{t,z}`$ 的精确充分性问题。该定理给出最少个数 $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,z})-1`$。
- 上界：每个 $`g_V`$ 都形如 $`y\mapsto\langle u,D_t(z,y)\rangle`$，$`u\in U`$。映射 $`u\mapsto\langle u,D_t(z,\cdot)\rangle`$ 是从 $`U`$ 到字母表上函数的线性映射，像的维数不超过 $`\dim U`$，所以 $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1`$。$`m`$ 点字母表上的全部函数构成含常数的 $`m`$ 维空间，所以这个量也不超过 $`m`$。减去 $`1`$ 即得两个界。$`\square`$

*(d) 的证明.* 假设使 (a) 中每一项 $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle`$（$`j\le k`$）都为零，只剩 $`r_k`$。

- 对 SGD，$`\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}`$，所以匹配增量矩就是匹配梯度矩，且 $`m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}`$。
- 对 cold Adam 与始终激活的裁剪，$`\Delta`$ 的参数块中依赖消息的部分分别是 $`-\eta h_\epsilon(G)`$（Proposition 5.3(a)）与 $`-\eta c\,G/\lVert G\rVert`$，它们的张量幂就是所述的矩。$`\square`$

<a id="s5-2"></a>

**Remark 5.2（与 stochastic modified equations、TracIn、influence function 和 metagradient 的关系）.**

- Proposition 5.1 是对完整状态增量做 Taylor 展开，再代入逐轮的 telescoping 求和。它与 weak approximation / stochastic modified equations（Li, Tai & E, 2019）、TracIn（Pruthi et al., 2020）、influence function（Koh & Liang, 2017）、metagradient（Engstrom et al., 2025）以及 In-Run Data Shapley（Wang et al., 2025）属于同一类计算。我们不主张它的新颖性；它是推导后面结果的工具。
- 这套记账显式写出了：
  - *对象.* 一阶统计量是完整状态平均增量在 $`U`$ 上的投影，包括 optimizer 的矩 $`m`$ 和 $`v`$；既不是平均梯度，也不是 loss。梯度分布的哪些特征会进入，完全由 optimizer 映射 $`\Delta`$ 决定（Theorem 5.10）。
  - *特例.* 对 SGD，(a) 取 $`k=2`$ 就是 [Theorem 4.7](04_rates_and_composition_ZH.md#s4-7) 中的局部 SGD 展开；(c) 是 [Theorem 4.2(a)–(c)](04_complexity_of_sufficiency_ZH.md#s4-2) 用于线性化的 response 类；(b) 是 [Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式在每一轮插入局部展开。
- *余项衡量的是什么（量级上的启发式；界本身是 (b) 中的 $`M_2`$）.* 把增量分块。
  - *参数块.* $`\Delta^\theta=O(\eta)`$，二阶贡献 $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ 为 $`O(\eta^2)`$。
  - *一阶矩块（momentum $`m`$）.* $`\Delta^m=O(1-\beta_1)`$。continuation value 通过之后的参数增量依赖 $`m`$，主阶是线性的，系数为 $`O(\eta/(1-\beta_1))`$。所以 $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$，这一块的贡献也是 $`O(\eta^2)`$。
  - *二阶矩块（Adam 的 $`v`$）.* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$。continuation value 通过之后各步的分母依赖 $`v`$，这些分母只带一个 $`\eta`$ 因子、不再有更高次幂。所以这一块的二阶贡献是 $`\eta`$ 的**一阶**量。它相对这一块一阶项的大小由 $`\Delta v_i/v_i^{\rm next}`$ 决定，$`v^{\rm next}`$ 是下一步用到的二阶矩：warm 状态下约为 warmth $`\kappa_i`$（Definition 5.4），cold 状态下是 1 的量级。
  - 因此在 $`(\mathrm S_1)`$ 成立处，(b) 的余项对 SGD、momentum 与 sign 类 optimizer 每轮是 $`O(\eta^2)`$；对 Adam 类 optimizer、多于一轮时，它是 $`\eta`$ 的一阶量，相对大小与 warmth 同阶。$`R=1`$ 时边界之后的 continuation value 是 $`F_E(\theta)`$，不依赖 $`o`$，对任何 optimizer 余项都是 $`O(\eta^2)`$。
- *实际中的必要性.* (c) 只有在一阶项超过余项界时才能察觉失配。余项为 $`O(\eta^2)`$ 时，只要 $`\eta`$ 足够小，任何固定的失配都能被察觉。对 Adam 类的多轮 response，余项也是 $`\eta`$ 的一阶量，察觉需要 [Corollary 8.9](08_certificate_ZH.md#s8-9) 的 margin 条件；在二阶矩通道占主导的地方，需要有限替换的测量（[Theorem 8.8](08_certificate_ZH.md#s8-8)、[Remark 7.5](07_second_moment_channel_ZH.md#s7-5)）。
- *代价.* $`\mu_t^P(z)`$ 只需要在状态 $`z`$ 上计算 consumed gradient，不需要训练尾段（[Proposition 8.4](08_certificate_ZH.md#s8-4)）。常数 $`\Lambda_t`$ 与 $`M_{2,t}`$ 是 continuation value 的性质，不能靠对消息抽样来估计。

## Cold-start Adam

**符号映射.** 对 $`\epsilon>0`$，$`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$。对 $`\epsilon=0`$，*理想符号更新*为 $`h_0(g)_i=\mathrm{sign}(g_i)`$，约定 $`\mathrm{sign}(0)=0`$。这个约定是定义的一部分；字面上的浮点 Adam 在 $`\epsilon=0`$ 时会算出 $`0/0`$。

**效用.** *即时参数效用*是 $`\theta_1`$ 的有界可测函数；*完整状态效用*是下一个完整状态 $`(\theta_1,m_1,v_1,1)`$ 的有界可测函数。consumed gradient 的分布 $`P,Q`$ 是 $`\mathbb R^p`$ 上的概率分布；在有限情形的陈述中，它们支撑在有限字母表 $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$ 上。记 $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$。

<a id="s5-3"></a>

**Proposition 5.3（Cold-start Adam：符号商与 optimizer 记忆）.** 从 cold 状态 $`(\theta_0,0,0,0)`$ 启动 Adam。

- **(a) 第一步.** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$。
- **(b) $`\epsilon=0`$：联合符号分布.** 两个分布对所有即时参数效用给出相同期望，当且仅当它们的联合符号向量分布相同。在有 $`s`$ 个可达符号模式的有限字母表上，对 $`\mathcal A`$ 上不受限的分布，足以决定所有这类效用的线性期望统计量最少为 $`s-1`$ 个；这里 $`s\le3^p`$，没有零坐标时 $`s\le2^p`$。逐坐标的符号边缘分布不够：取 $`p=2`$，$`P`$ 在 $`\lbrace (1,1),(-1,-1)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace (1,-1),(-1,1)\rbrace`$ 上均匀。每个坐标的符号边缘分布相同，两者的平均步长都是 $`0`$，但效用 $`F(\theta)=1\lbrace (\theta_1-\bar\theta_1)(\theta_2-\bar\theta_2)>0\rbrace`$ 在 $`P`$ 下期望为 $`1`$，在 $`Q`$ 下为 $`0`$。
- **(c) $`\epsilon>0`$：完整分布.** $`h_\epsilon`$ 是逐坐标的双射 $`\mathbb R\to(-1,1)`$，其逆与导数如下。所以对所有即时参数效用相等，等价于 consumed gradient 的完整分布相等。在 $`\mathcal A`$ 上最少需要 $`m-1`$ 个统计量。

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}.
```

**(d) 对 margin 敏感的近似.** 设 $`F:\mathbb R^p\to[0,1]`$ 对所有 $`u=\bar\theta-\eta h_\epsilon(g)`$、$`w=\bar\theta-\eta h_0(g)`$（$`g`$ 属于支撑）满足 $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$。对 $`a>0`$ 记 $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$，并令 $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(\bar\theta-\eta h_0(G))-\mathbb E_QF(\bar\theta-\eta h_0(G))\rvert`$。则 $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$，且

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_QF(\bar\theta-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big].
```

**(e) 保留下来的 optimizer state.** 当 $`\beta_1<1`$ 时，第一个完整状态中含有 $`m_1=(1-\beta_1)G`$，它是 $`G`$ 的单射函数。

- *反例.* 取一维，$`\beta_1=\beta_2=\tfrac12`$，$`\lambda_{\rm wd}=0`$，$`\epsilon=0`$，$`\eta>0`$，cold 启动。比较梯度历史 $`(G_1,G_2)=(1,-\tfrac34)`$ 与 $`(G_1',G_2')=(2,-\tfrac34)`$。两者都给出 $`\theta_1=\theta_1'=\theta_0-\eta`$，符号历史都是 $`(+,-)`$。第 2 步的 bias-corrected 矩为 $`\widehat m_2=-\tfrac16`$、$`\widehat m_2'=+\tfrac16`$，以及 $`\widehat v_2=\tfrac{17}{24}`$、$`\widehat v_2'=\tfrac{41}{24}`$，所以下面的第二步方向相反。有界检验 $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ 把两段历史分开，差为 $`1`$。这两段历史可以由平方损失 $`\tfrac12(\theta-a)^2`$ 在 $`\theta_0=0`$ 处实现：第一步目标为 $`a=-1`$ 与 $`a=-2`$，第二步取共同目标 $`a=\theta_1+\tfrac34`$。

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}.
```

- *记忆由 momentum 携带.* 若 $`\beta_1=0`$，任意两段 $`G_2`$ 符号相同的历史，第二步方向都相同，这类反例不存在；对每个 $`\beta_1\in(0,1)`$，这类反例都存在。

*(a) 的证明.* cold 状态下 $`m_1=(1-\beta_1)G`$，$`v_1=(1-\beta_2)G^{\odot2}`$。除以 $`t=1`$ 的 bias correction 得 $`n=G`$、$`d^2=G^{\odot2}`$，所以逐坐标 $`d=\lvert G\rvert`$。代入得 $`\theta_1=\bar\theta-\eta\,G/(\lvert G\rvert+\epsilon)`$。$`\epsilon=0`$ 时由约定得 $`h_0=\mathrm{sign}`$。$`\square`$

*(b) 的证明.* 因为 $`\eta>0`$，映射 $`\sigma\mapsto\bar\theta-\eta\sigma`$（从 $`\lbrace -1,0,1\rbrace ^p`$ 到 $`\mathbb R^p`$）是单射，所以 $`\theta_1`$ 是符号向量 $`\sigma(G)`$ 的单射函数。

- *充分性.* 联合符号分布相同，则 $`\theta_1`$ 的分布相同，所有效用的期望也相同。
- *必要性.* 对每个可达模式 $`\sigma^\star`$，示性函数 $`F=1\lbrace \theta_1=\bar\theta-\eta\sigma^\star\rbrace`$ 是有界效用，期望为 $`\Pr(\sigma(G)=\sigma^\star)`$。所有这样的 $`F`$ 期望相同，就迫使各模式的概率相同。
- *维数.* 把每个效用看成 $`\mathcal A`$ 上的函数 $`u_F(g)=F(\bar\theta-\eta h_0(g))`$。族 $`\mathcal U=\lbrace u_F\rbrace`$ 恰好是 $`\mathcal A`$ 上在 $`\sigma`$ 的纤维上取常值的函数：这些函数都在其中，因为 $`F`$ 在 $`s`$ 个不同输出点上可以任意取值；其余函数都不在其中，因为 $`u_F`$ 只通过 $`\sigma(g)`$ 依赖 $`g`$。所以 $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ 由 $`s`$ 个纤维示性函数张成，维数为 $`s`$。由 [Theorem 4.2(a)](04_complexity_of_sufficiency_ZH.md#s4-2)，对 $`\mathcal A`$ 上的所有分布，均值相等能推出每个 $`u\in\mathcal U`$ 均值相等的线性统计量最少为 $`s-1`$ 个；取 $`s-1`$ 个模式的示性函数即可达到。每个坐标符号至多取三个值，所以 $`s\le3^p`$；排除 $`0`$ 时 $`s\le2^p`$。
- *边缘分布.* 在反例的两个分布下，每个坐标都以概率 $`1/2`$ 取 $`\pm1`$，且 $`\mathbb E h_0(G)=0`$。在 $`P`$ 下 $`\theta_1-\bar\theta=-\eta\sigma`$ 的两个坐标总是同号，所以 $`F=1`$；在 $`Q`$ 下总是异号，所以 $`F=0`$。$`\square`$

*(c) 的证明.* 固定 $`\epsilon>0`$。在 $`g\ge0`$ 上 $`h_\epsilon(g)=g/(g+\epsilon)`$ 连续、严格递增，从 $`0`$ 升到 $`1`$；由奇性，$`h_\epsilon`$ 是双射 $`\mathbb R\to(-1,1)`$。

- 解 $`u=g/(\lvert g\rvert+\epsilon)`$：$`\mathrm{sign}u=\mathrm{sign}g`$ 且 $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$，所以 $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$，$`g=\epsilon u/(1-\lvert u\rvert)`$。对 $`u>0`$，导数为 $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$；由对称性，$`u<0`$ 时公式相同。
- 因为 $`g\mapsto\bar\theta-\eta h_\epsilon(g)`$ 是单射，(b) 的论证在纤维为单点时同样适用：对所有有界效用相等等价于 $`G`$ 的分布相等，在 $`\mathcal A`$ 上维数为 $`m-1`$。$`\square`$

*(d) 的证明.* 固定坐标 $`i`$ 与梯度值 $`g`$。

- 若 $`g_i=0`$，则 $`h_\epsilon(g)_i=0=h_0(g)_i`$。
- 若 $`g_i\ne0`$，则 $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$。它总不超过 $`1`$，当 $`\lvert g_i\rvert\ge a`$ 时不超过 $`\epsilon/(a+\epsilon)`$。

所以 $`\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\epsilon/(a+\epsilon)`$。对每对 $`\big(\bar\theta-\eta h_\epsilon(G),\,\bar\theta-\eta h_0(G)\big)`$ 逐点用 Lipschitz 假设，

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_PF(\bar\theta-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

在 $`Q`$ 下同样成立。经过两个理想符号期望用三角不等式即得。至于 TV 的陈述：$`F(\bar\theta-\eta h_0(G))=f(\sigma(G))`$，其中 $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$，任何这样的 $`f`$ 都满足 $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$。$`\square`$

*(e) 的证明.* $`\beta_1<1`$ 时 $`m_1=(1-\beta_1)G`$ 显然是单射。对反例，$`1-\beta_1=1-\beta_2=\tfrac12`$，$`t=1`$ 时 bias correction 为 $`\tfrac12`$，$`t=2`$ 时为 $`\tfrac34`$。

- *第 1 步.* $`m_1=G_1/2`$，$`v_1=G_1^2/2`$，所以 $`\widehat m_1=G_1`$，$`\widehat v_1=G_1^2`$。两段历史都有 $`G_1>0`$，所以 $`\theta_1=\theta_0-\eta`$。
- *第 2 步.* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$：$`G_1=1`$ 时为 $`-\tfrac18`$，$`G_1=2`$ 时为 $`+\tfrac18`$；于是 $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$。同样 $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$，即 $`\tfrac{17}{32}`$ 或 $`\tfrac{41}{32}`$，所以 $`\widehat v_2=\tfrac{17}{24}`$ 或 $`\tfrac{41}{24}`$。
- $`\epsilon=0`$ 时 $`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$，即上面的式子。分母为正、分子异号，所以两步方向相反，$`F(\theta_2)=1`$，$`F(\theta_2')=0`$。
- *可实现性.* $`\tfrac12(\theta-a)^2`$ 的梯度为 $`\theta-a`$。在 $`\theta_0=0`$ 处，目标 $`a=-1,-2`$ 给出 $`G_1=1,2`$。两个第一步都落在 $`\theta_1=-\eta`$，所以共同目标 $`a=\theta_1+\tfrac34`$ 在两段历史中都给出 $`G_2=-\tfrac34`$。不需要依赖分支的未来梯度。
- *$`\epsilon>0`$.* 对同样的梯度序列，第二步的方向对每个 $`\epsilon\ge0`$ 都是 $`-\mathrm{sign}(\widehat m_2)`$，因为分母 $`\sqrt{\widehat v_2}+\epsilon`$ 为正；两个方向仍然相反。
- *momentum.* 若 $`\beta_1=0`$，则 $`m_2=G_2`$，无论第一个梯度是什么，第二步的符号都是 $`-G_2`$ 的符号。若 $`\beta_1>0`$，则 $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$。对 $`0<G_1<G_1'`$，取 $`G_2<0`$ 且 $`\lvert G_2\rvert`$ 严格介于 $`\beta_1G_1`$ 与 $`\beta_1G_1'`$ 之间，$`m_2`$ 的符号相反，而 $`\epsilon=0`$ 时两个第一步 $`-\eta\,\mathrm{sign}(G_1)`$ 相同。$`\square`$

**关于 Proposition 5.3 的讨论.**

- *三个层次的充分性.* cold 步的即时参数充分性，在 $`\epsilon=0`$ 时需要联合符号分布，在 $`\epsilon>0`$ 时需要完整分布。完整状态的充分性需要 consumed gradient 的完整分布，因为 $`m_1`$ 是单射。任务层面的充分性只需要实际 continuation 的 response 类（[Theorem 3.2](03_learning_consequence_ZH.md#s3-2)），它可能小得多。上面精确的统计量个数针对不受限的分布；受限的任务族可能需要更少。
- *$`\epsilon=0`$ 处的不连续是精确意义上的，不是操作意义上的.* 对任意可测效用，任何 $`\epsilon>0`$ 都使幅度可辨识。对 Lipschitz 效用、且梯度在零附近质量很小时，(d) 说明符号近似是准确的。(c) 中的逆在 $`\lvert u\rvert\to1`$ 时病态。
- *warm 状态会移动阈值.* warm 状态下坐标 $`i`$ 的分子在 $`G_i=-\beta_1m_i/(1-\beta_1)`$ 处变号，分母依赖 $`v_i`$ 与 $`G_i^2`$。对标量交叉熵梯度 $`G=p-Z`$，cold 的符号概率就是 $`Z`$ 在当前预测 $`p`$ 之下、之上与等于 $`p`$ 的质量。允许所有阈值就能辨识 $`Z`$ 的分布函数；没有哪个固定阈值或边缘符号特征能在状态变化时始终充分。
- *对仿射理论的含义.* 两个标签分布可以匹配所有低于 $`N`$ 阶的矩（[Theorem 4.4](04_complexity_of_sufficiency_ZH.md#s4-4)），却在关于当前预测的符号概率上不同。cold Adam 的一步就会在 $`\eta`$ 的一阶上把它们分开（Proposition 5.7）。
