# §5 Optimizer state 与一阶表示 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/05_optimizer_state_ZH.md) · [English](05_optimizer_state.md)

**本节概要.**

- 带状态的一阶 optimizer 是一个映射：把 consumed gradient 映成*完整* learner state $`z=(\theta,o)`$ 的增量。Proposition 5.1 按这个增量的各阶矩展开一个 feedback law 的效应。在一阶意义下，feedback law 必须携带的是平均增量 $`\mu`$ 在 continuation value 梯度方向上的部分。这是一个工具：同类计算见 stochastic modified equations、TracIn 与 metagradient（Remark 5.2）。
- 对 Adam，统计量取决于 optimizer state。
  - cold 状态下，参数步只看梯度的符号分布，而矩保留了梯度的幅度（Proposition 5.3）。
  - warm 状态下，精确统计量是 Adam 增量的均值。$`(\mathbb EG,\mathbb E\,G\odot G)`$ 只在 frozen-denominator 近似下决定它，误差由 warmth 控制（Proposition 5.5–5.7）。
  - $`\beta_2`$ 固定时 warmth 不会趋于零：早期按 $`1/(\beta_2s)`$ 下降，之后停在平台 $`(1-\beta_2)/\beta_2`$（Proposition 5.8）。
- 梯度裁剪下统计量是裁剪后的均值；只用平均方向，误差为 $`c(1-p_{\rm act})`$（Proposition 5.9）。
- Theorem 5.10 列出 SGD、momentum、warm 与 cold Adam、裁剪、Lion、LoRA 各自的一阶统计量。Remark 5.11 说明在真实模型上检验 warm 状态条件时应当测什么。
- 用到：[Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式与停止版本；[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2) 的充分性对偶；[Theorem 4.4](04_complexity_of_sufficiency_ZH.md#s4-4) 的矩匹配分布；[Theorem 4.7](04_complexity_of_sufficiency_ZH.md#s4-7) 的 SGD 展开。
- 被用于：第 6 节的 learner Jacobian（[Definition 6.2](06_learner_relative_information_ZH.md#s6-2)、[Proposition 6.11](06_learner_relative_information_ZH.md#s6-11)）；第 7 节的状态通道分解（[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1)）；第 8 节的测量结果（[Proposition 8.4](08_verification_ZH.md#s8-4)、[Corollary 8.9](08_verification_ZH.md#s8-9)）；第 9 节的方法。

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
  - *特例.* 对 SGD，(a) 取 $`k=2`$ 就是 [Theorem 4.7](04_complexity_of_sufficiency_ZH.md#s4-7) 中的局部 SGD 展开；(c) 是 [Theorem 4.2(a)–(c)](04_complexity_of_sufficiency_ZH.md#s4-2) 用于线性化的 response 类；(b) 是 [Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式在每一轮插入局部展开。
- *余项衡量的是什么（量级上的启发式；界本身是 (b) 中的 $`M_2`$）.* 把增量分块。
  - *参数块.* $`\Delta^\theta=O(\eta)`$，二阶贡献 $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ 为 $`O(\eta^2)`$。
  - *一阶矩块（momentum $`m`$）.* $`\Delta^m=O(1-\beta_1)`$。continuation value 通过之后的参数增量依赖 $`m`$，主阶是线性的，系数为 $`O(\eta/(1-\beta_1))`$。所以 $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$，这一块的贡献也是 $`O(\eta^2)`$。
  - *二阶矩块（Adam 的 $`v`$）.* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$。continuation value 通过之后各步的分母依赖 $`v`$，这些分母只带一个 $`\eta`$ 因子、不再有更高次幂。所以这一块的二阶贡献是 $`\eta`$ 的**一阶**量。它相对这一块一阶项的大小由 $`\Delta v_i/v_i^{\rm next}`$ 决定，$`v^{\rm next}`$ 是下一步用到的二阶矩：warm 状态下约为 warmth $`\kappa_i`$（Definition 5.4），cold 状态下是 1 的量级。
  - 因此在 $`(\mathrm S_1)`$ 成立处，(b) 的余项对 SGD、momentum 与 sign 类 optimizer 每轮是 $`O(\eta^2)`$；对 Adam 类 optimizer、多于一轮时，它是 $`\eta`$ 的一阶量，相对大小与 warmth 同阶。$`R=1`$ 时边界之后的 continuation value 是 $`F_E(\theta)`$，不依赖 $`o`$，对任何 optimizer 余项都是 $`O(\eta^2)`$。
- *实际中的必要性.* (c) 只有在一阶项超过余项界时才能察觉失配。余项为 $`O(\eta^2)`$ 时，只要 $`\eta`$ 足够小，任何固定的失配都能被察觉。对 Adam 类的多轮 response，余项也是 $`\eta`$ 的一阶量，察觉需要 [Corollary 8.9](08_verification_ZH.md#s8-9) 的 margin 条件；在二阶矩通道占主导的地方，需要有限替换的测量（[Theorem 8.8](08_verification_ZH.md#s8-8)、[Remark 7.5](07_second_moment_channel_ZH.md#s7-5)）。
- *代价.* $`\mu_t^P(z)`$ 只需要在状态 $`z`$ 上计算 consumed gradient，不需要训练尾段（[Proposition 8.4](08_verification_ZH.md#s8-4)）。常数 $`\Lambda_t`$ 与 $`M_{2,t}`$ 是 continuation value 的性质，不能靠对消息抽样来估计。

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

## Warm Adam

<a id="s5-4"></a>

**Definition 5.4（warmth 与 frozen-denominator surrogate）.** 考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

<a id="s5-5"></a>

**Proposition 5.5（Adam 的精确一阶统计量与冻结分母误差）.** 在 Definition 5.4 的设定下：

- **(a) 精确统计量.** Adam 增量为 $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$，其中 $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$，$`\Delta m=(1-\beta_1)(G-m)`$，$`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$。Proposition 5.1 中的一阶统计量就是精确均值 $`\mu=\mathbb E\,\Delta(z,G)`$。它的参数块是 $`G`$ 的分布的非线性泛函：若 $`\beta_1<1`$，且对某个坐标 $`i`$ 有 $`v_i>0`$ 或 $`\epsilon>0`$，则存在两个有限支撑的分布，$`\mathbb E\,G`$ 相同、$`\mathbb E\,G^{\odot2}`$ 也相同，但参数块 $`\mu_\theta`$ 不同。
- **(b) 冻结分母误差.** 对每个坐标与每个 $`G`$，

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) 两种近似要分开.** frozen-denominator 平均增量 $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ 由 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定，且 $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$，$`m`$ 块与 $`v`$ 块则完全相同。所以"warm Adam 只需要 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"这一说法依赖两个不同的近似：
  - Proposition 5.1 的一阶近似，余项由增量的二阶矩控制，分块结构见 Remark 5.2；
  - frozen-denominator 近似，误差由 (b) 控制，随 warmth 变小。

*(a) 的证明.* 增量就是 AdamW 的定义。对第二个结论，固定坐标 $`i`$，令 $`G=g\,e_i`$（$`g`$ 为标量，其余坐标为零）。此时坐标 $`i`$ 的参数块为 $`\mathbb E f(g)`$，其中

```math
f(g)=-\eta\,\frac{\big(\beta_1m_i+(1-\beta_1)g\big)/(1-\beta_1^{\,t})}{\sqrt{\big(\beta_2v_i+(1-\beta_2)g^2\big)/(1-\beta_2^{\,t})}+\epsilon}.
```

- $`f`$ 在 $`\mathbb R`$ 上有界：$`v_i>0`$ 或 $`\epsilon>0`$ 时分母有正的下界；当 $`g\to\pm\infty`$ 时，$`f(g)`$ 趋于 $`\mp\eta(1-\beta_1)\sqrt{1-\beta_2^{\,t}}/\big((1-\beta_1^{\,t})\sqrt{1-\beta_2}\big)`$。因为 $`\beta_1<1`$，两个极限不同，所以 $`f`$ 不是常数。
- 在 $`\mathbb R`$ 上有界的不超过二次的多项式只能是常数，所以 $`f`$ 不是这样的多项式。于是存在四个点 $`g_1<g_2<g_3<g_4`$，$`f`$ 在其上与任何不超过二次的多项式都不一致（过图像上三点的二次多项式会错过某个第四点）。
- 线性映射 $`w\mapsto(\sum_kw_k,\sum_kw_kg_k,\sum_kw_kg_k^2)`$（从 $`\mathbb R^4`$ 到 $`\mathbb R^3`$）有非零核向量 $`w^*`$。它的行空间是 $`\mathrm{span}\lbrace1,g,g^2\rbrace`$ 在这四点上的限制，所以 $`\sum_kw^*_kf(g_k)\ne0`$，否则 $`f`$ 的限制就会落在这个张成空间里。
- 对小的 $`\tau>0`$，$`P=\tfrac14\mathbf 1+\tau w^*`$ 与 $`Q=\tfrac14\mathbf 1-\tau w^*`$ 是这四点上的概率向量（因为 $`\sum_kw^*_k=0`$）。它们的 $`\mathbb E\,g`$ 与 $`\mathbb E\,g^2`$ 相同，因此 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ 相同，而 $`\mathbb E_Pf-\mathbb E_Qf=2\tau\sum_kw^*_kf(g_k)\ne0`$。$`\square`$

*(b) 的证明.*

- *$`v_i>0`$ 的情形.* $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\big\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\big\rvert`$。通分得到下面的恒等式。由 $`\epsilon\ge0`$，最后一个因子不超过 $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$；它小于 $`1`$，又因 $`\sqrt{1+\kappa}-1\le\kappa/2`$ 与 $`\sqrt{1+\kappa}\ge1`$ 而不超过 $`\kappa_i/2`$。所以该因子不超过 $`\min\lbrace \kappa_i/2,1\rbrace`$。

```math
\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
```

- *$`v_i=0`$ 且 $`\epsilon>0`$ 的情形.* 此时 $`a_i=0`$，$`\kappa_i=\infty`$，$`u_i=\lvert n_i\rvert/\epsilon`$。两个倒数 $`1/(d_i+\epsilon)`$ 与 $`1/\epsilon`$ 都在 $`[0,1/\epsilon]`$ 中，差不超过 $`1/\epsilon`$，所以左边不超过 $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$。$`\square`$

*(c) 的证明.* 由 Definition 5.4，$`\tilde\theta^+-\theta=-\eta\lambda_{\rm wd}\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta\Phi(1-\beta_1)G/(1-\beta_1^{\,t})`$ 与 $`m^+-m=(1-\beta_1)(G-m)`$ 关于 $`G`$ 仿射，系数由状态固定；$`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ 关于 $`G^{\odot2}`$ 仿射。取期望后 $`\tilde\mu`$ 是 $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ 的仿射函数。$`\Delta`$ 与 $`\tilde\Delta`$ 的 $`m`$ 块、$`v`$ 块相同，$`\theta`$ 块相差 $`\theta^+-\tilde\theta^+`$，其期望逐坐标由 (b) 控制。$`\square`$

<a id="s5-6"></a>

**Proposition 5.6（warm Adam 关于 consumed gradient 一阶仿射）.** 固定一轮 $`t\ge1`$、状态 $`z=(\theta,m,v,t-1)`$，以及有限消息集 $`Y`$，其 consumed gradient 为 $`G(y)`$。令 $`V:\mathcal Z\to[0,1]`$ 为更新之后的 continuation value（声明的参考尾段加部署，尾段随机性已积分），$`\psi(y)=V(z^+(y))`$ 为单边界 response，$`F(g)=V(\tilde z^+(g))`$。设下列条件成立，其中所有状态都由 $`Y`$ 中的消息从 $`z`$ 生成，计数器固定为 $`t`$：

- **(W)** 对每个坐标 $`v_i>0`$（warm 状态）；
- **(L)** 在 $`(m,v)=(m^+(y),v^+(y))`$ 处的点对 $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ 上，$`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$；
- **$`(\mathrm S_v)`$** 存在 $`L_v\ge0`$ 与上限 $`\bar k\in(0,\infty]`$，使得在 $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$ 处，对 $`k=\kappa(y)`$ 有 $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$（由于 $`V\in[0,1]`$，取 $`L_v\bar k\ge1`$ 时它总成立；有内容的是 $`L_v`$ 小的情形）；
- **(H)** $`F`$ 在 $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$ 的凸包 $`\mathcal C`$ 上二阶连续可微，且对所有 $`g\in\mathcal C`$、$`s\in\mathbb R^p`$ 有 $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$。

令 $`\psi_0=F(0)`$，$`b=\nabla F(0)`$，二者都与 $`y`$ 无关。则对每个 $`y\in Y`$，

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

并且对 $`Y`$ 上任意两个消息分布 $`P_0,P_1`$，

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- 特别地，平均 consumed gradient 相同的两个分布，response 之差至多为 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。
- 若 $`V`$ 关于 $`(\theta,m)`$ 可微，一阶统计量就是经过 preconditioner 的平均 consumed gradient，权重为

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

*证明.* 固定 $`y`$ 并在记号中略去。由于 $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$，

```math
\psi-\psi_0-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

- *(i)* 由 (L) 与 Proposition 5.5(b)，$`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$。
- *(ii)* 由 (W)，$`v^+=\beta_2v\odot(1+\kappa)`$，$`\kappa\ge0`$ 有限。在 $`(\tilde\theta^+,m^+)`$ 处用 $`(\mathrm S_v)`$，$`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$。
- *(iii)* 线段 $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ 落在 $`\mathcal C`$ 中。带积分余项的 Taylor 定理给出 $`F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau`$。由 (H)，被积函数的绝对值不超过 $`(1-\tau)H_G\lVert G\rVert_\infty^2`$，所以 $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$。

三个界相加即得逐点结论。对两个分布，写 $`\psi=\psi_0+b^\top G+e`$，$`\lvert e\rvert\le r`$。于是 $`\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e`$；常数 $`\psi_0`$ 抵消，因为两者都是概率测度；$`Y`$ 有限，所有期望都存在。三角不等式给出界，$`\psi\in[0,1]`$ 给出截断。对 $`b`$ 用链式法则：$`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)\Phi/(1-\beta_1^{\,t})`$，$`\partial m^+/\partial g=(1-\beta_1)I`$，$`\tilde z^+`$ 的 $`v`$ 分量与 $`g`$ 无关。$`\square`$

<a id="s5-7"></a>

**Proposition 5.7（cold 步在一阶就分开均值相同的反馈）.** 取 cold 状态（$`m=v=0`$，$`t=1`$）、理想符号映射 $`h_0`$（Proposition 5.3）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$，限制在 $`V\in[0,1]`$ 的邻域内。

- **(a) 均值相同.** 令 $`P_0(G=1)=\tfrac14`$，$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- **(b) 任意固定阶以下的矩都相同.** 对 $`N\ge1`$，令 $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数）与 $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数），$`0\le i\le N`$（即 [Theorem 4.4(c)](04_complexity_of_sufficiency_ZH.md#s4-4) 中的分布），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。两个梯度分布的所有低于 $`N`$ 阶的矩都相同，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。
- 两种情形下的分离都是 $`\eta`$ 的一阶量。在 warm 状态、同一效用下，Proposition 5.6 把均值相同的分布之间的差距控制在 $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$ 以内。

*证明.*

- *(a)* 由 Proposition 5.3(a) 与 $`h_0=\mathrm{sign}`$，$`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$。在 $`P_0`$ 下 $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$，所以 $`\mathbb E\theta_1=\theta_0+\eta/2`$。在 $`P_1`$ 下 $`\mathrm{sign}\,0=0`$，所以 $`\mathbb E\theta_1=\theta_0`$。效用是仿射的，response 差为 $`\lvert w\rvert\eta/2`$。均值相同：$`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$。
- *(b)* 因为 $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$，$`P_N`$ 与 $`Q_N`$ 都是概率分布。对 $`\ell<N`$，$`\sum_i(-1)^i\binom Ni\,i^\ell=0`$，所以它们低于 $`N`$ 阶的矩相同，仿射像 $`G=q-Z`$ 的矩也相同。取 $`q\in(0,1/N)`$：在 $`P_N`$ 下，$`G>0`$ 当且仅当 $`i=0`$，概率为 $`2^{1-N}`$；否则 $`Z\ge2/N>q`$，$`G<0`$。所以 $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$。在 $`Q_N`$ 下总有 $`Z\ge1/N>q`$，所以 $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$。第一步的期望相差 $`\eta\,2^{2-N}`$，仿射效用给出差距 $`\lvert w\rvert\eta\,2^{2-N}`$。
- *warm 对照.* 对没有尾段的 $`V(\theta)=\tfrac12+w\theta`$，$`F`$ 关于 $`g`$ 仿射，故 $`H_G=0`$；$`V`$ 不依赖 $`(m,v)`$，故 $`L_v=0`$；且 $`L=\lvert w\rvert`$。均值相同时，Proposition 5.6 给出所示的界。$`\square`$

<a id="s5-8"></a>

**Proposition 5.8（warmth 的演化）.** 设二阶矩从 $`v_0=0`$ 开始，按 $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$ 逐坐标吸收梯度计算 $`G_1,G_2,\dots`$。记 $`\gamma=1-\beta_2`$。

**(a) 平稳二阶矩：精确的 mean-field warmth.** 设对所有 $`k`$ 有 $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$。定义 mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$。则

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ 关于 $`s`$ 递减，收敛到平台 $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$，不会趋于零。
- 早期它的行为像 $`1/(\beta_2s)`$：确切地说，$`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$，所以近似 $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ 的相对误差至多为 $`s(1-\beta_2)`$，只有在 $`s(1-\beta_2)\ll1`$ 时才准确。
- 若梯度恒为常数 $`G_k=g`$，上式对 warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 本身成立，不只对它的 mean-field 版本成立。

**(b) bias correction 不改变 warmth.** 用 bias-corrected 分母 $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ 与 $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$ 时，比值 $`d_i/a_i=\sqrt{1+\kappa_i}`$ 与不带 bias correction 时相同。所以 warmth 以及 Proposition 5.5(b) 的界都不受 bias correction 影响。

**(c) 沿训练过程的累积.** 经由 [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式，比较参考协议 $`0`$ 与替代协议 $`S`$ 在 $`W`$ 次 warm-up 计算之后的 $`R`$ 次真实更新 $`s=1,\dots,R`$。令 $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ 为这样的更新：在每个访问到的状态上，Proposition 5.6 的假设以下标为 $`s`$ 的常数成立，两个分布的平均 consumed gradient 相同，并且几乎必然地对某个常数 $`c>0`$ 有下式。

```math
\max_i\kappa_i\le c\Big(\frac1{W+s}+(1-\beta_2)\Big).
```

则

```math
\lvert L(0)-L(S)\rvert\le\big\lvert\lbrace s\le R:s\notin\mathcal S_{\rm warm}\rbrace\big\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
```

其中 $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$，$`\bar u_{i,s}`$ 是第 $`s`$ 次更新时 $`u_i`$ 的上界，$`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$。当 $`W\ge1`$ 时，

```math
\sum_{s=1}^{R}\min\Big\lbrace1,\;\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace\le c\log\Big(1+\frac RW\Big)+c(1-\beta_2)R .
```

- 所以由 warmth 控制的部分每次更新都有一个量级为 $`c(1-\beta_2)`$ 的下限，不随训练变长而消失。只有当 $`(1-\beta_2)R\lesssim\log(1+R/W)`$ 时对数项才占主导。
- $`W=0`$ 时，第一次真实更新从 $`v=0`$ 开始，不满足 (W)：它是 Proposition 5.3 的 cold 步，记为至多 $`1`$；由 Proposition 5.7，它的贡献可以是 $`\eta`$ 的一阶量。若 $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$，由 warmth 控制的和至多为 $`c\log R+c(1-\beta_2)(R-1)`$。

**(d) Warm-up，以及 (c) 的假设何时成立.** 长度为 $`W`$ 的*矩 warm-up*，用声明数据的梯度做 $`W`$ 次学习率恰为 $`0`$ 的 optimizer 步。它不改变 $`\theta`$，并按与真实步相同的递推推进 $`m`$、$`v`$ 与计数器；被 $`v`$ 吸收的每个梯度，无论来自 warm-up 还是真实更新，都算一次计算。所以第 $`s`$ 次真实更新面对的二阶矩已吸收 $`W+s-1`$ 次计算。若把实际的 $`v`$ 换成其均值（mean-field 近似），并且梯度满足 $`G_{k,i}^2\le\bar\gamma\,\sigma_i^2`$，则当 $`W+s\ge2`$ 时 (c) 的假设以 $`c=2\bar\gamma/\beta_2`$ 成立。把 $`v`$ 换成均值是一个近似；它需要指数滑动平均的集中性和平稳的梯度尺度。

*(a) 的证明.* 展开递推，$`v_{s,i}=(1-\beta_2)\sum_{r=1}^s\beta_2^{\,s-r}G_{r,i}^2`$，所以 $`\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2)\sum_{j<s}\beta_2^{\,j}=\sigma_i^2(1-\beta_2^{\,s})`$。代入定义得 $`\bar\kappa_i(s)`$。关于界：

- 由 Bernoulli 不等式，$`1-(1-\gamma)^s\le s\gamma`$，所以 $`\bar\kappa\ge1/(\beta_2s)`$；又 $`1-(1-\gamma)^s\le1`$，所以 $`\bar\kappa\ge\gamma/\beta_2`$。
- 上界等价于 $`s\gamma\le(1+s\gamma)\big(1-(1-\gamma)^s\big)`$，即 $`(1-\gamma)^s(1+s\gamma)\le1`$。这由 $`(1-\gamma)^s\le e^{-s\gamma}`$ 以及对 $`y\ge0`$ 有 $`e^{-y}(1+y)\le1`$ 得到。
- 单调性是因为 $`1-\beta_2^{\,s}`$ 关于 $`s`$ 递增，极限是直接的。两侧界乘以 $`\beta_2s`$ 得 $`1\le\beta_2s\bar\kappa\le1+s\gamma`$。
- 梯度为常数时，$`v_{s,i}=g_i^2(1-\beta_2^{\,s})`$ 是确定的，$`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 就等于该公式。$`\square`$

*(b) 的证明.* $`d_i^2/a_i^2=v_i^+/(\beta_2v_i)=1+\kappa_i`$：因子 $`1/(1-\beta_2^{\,t})`$ 同时除在分子与分母上。$`\square`$

*(c) 的证明.* [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 给出

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

并且在每个访问到的状态上，$`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$，是两个取值于 $`[0,1]`$ 的期望之差。

- 对 $`s\notin\mathcal S_{\rm warm}`$，其绝对值以 $`1`$ 为界。
- 对 $`s\in\mathcal S_{\rm warm}`$，均值相同时 Proposition 5.6 把它控制在 $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$ 以内。用 $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$，以及无论 $`\kappa\le1`$ 还是 $`\kappa>1`$ 都成立的 $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$。由 $`\kappa_{\max}`$ 的假设，$`r_s`$ 的前两项至多为 $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)+c(1-\beta_2)\rbrace`$，曲率项至多为 $`\tfrac12\bar B_s`$。对两个分布求和得到因子 $`2`$。

对 $`s`$ 求和即得第一式。对第二式：当 $`a,b\ge0`$ 时 $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$，所以和至多为 $`\sum_{s=1}^R c/(W+s)+c(1-\beta_2)R`$；又 $`c/(W+s)\le\int_{s-1}^{s}c\,dz/(W+z)`$，所以 $`W\ge1`$ 时 $`\sum_{s=1}^Rc/(W+s)\le c\log(1+R/W)`$。$`W=0`$、$`s`$ 从 $`2`$ 开始时，同理 $`\sum_{s=2}^Rc/s\le c\log R`$。$`\square`$

*(d) 的证明.* 学习率为 $`0`$ 时参数更新为 $`\theta^+=(1-0\cdot\lambda_{\rm wd})\theta-0=\theta`$，而 $`m`$、$`v`$ 与计数器按与真实步相同的递推推进。所以经过 $`W`$ 次 warm-up 计算与 $`s-1`$ 次真实更新后，$`v`$ 已吸收 $`W+s-1`$ 次计算。在 mean-field 替换下，$`v_i=\sigma_i^2(1-\beta_2^{\,k})`$，$`k=W+s-1\ge1`$，所以由 (a)，$`\kappa_i=(G_i^2/\sigma_i^2)\,\bar\kappa_i(k)\le\bar\gamma\,(1/k+1-\beta_2)/\beta_2`$。当 $`W+s\ge2`$ 时 $`k\ge(W+s)/2`$，即 $`1/k\le2/(W+s)`$，因此 $`\kappa_i\le(2\bar\gamma/\beta_2)\big(1/(W+s)+(1-\beta_2)\big)`$。$`\square`$

**关于 Proposition 5.5–5.8 的讨论.**

- *切换.* 反馈来源必须匹配的统计量，随 optimizer state 而变。
  - cold Adam 状态下，要匹配 consumed gradient 的联合符号分布，以及保留在 $`m`$、$`v`$ 中的幅度（Proposition 5.3）。
  - warm Adam 状态下，精确统计量是 Adam 增量的均值；在 warmth 的一阶意义下，它是经过 preconditioner 的平均 consumed gradient，加上 $`v`$ 块中的 $`\mathbb E\,G^{\odot2}`$（Proposition 5.5–5.6）。
  - SGD 下，要匹配平均 consumed gradient，误差 $`O(\eta^2)`$（[Theorem 4.7](04_complexity_of_sufficiency_ZH.md#s4-7) 的局部 SGD 展开）。
- *warmth 的下限.* $`\beta_2`$ 固定时，典型梯度的 warmth 在此后的训练中一直停在 $`(1-\beta_2)/\beta_2`$ 附近（Proposition 5.8(a)）。所以 frozen-denominator 描述在每一步都只准确到这个相对阶，累积的非仿射贡献有一个关于 $`R`$ 线性的项（Proposition 5.8(c)）。
- *曲率项的阶.* 对光滑的尾段，$`F`$ 通过 $`\tilde\theta^+`$（被 $`\eta`$ 缩放）与 $`m^+`$（在之后的更新中被 $`\eta`$ 缩放）依赖 $`g`$。所以可以预期 $`H_G=O(\eta^2)`$。这是对尾段的预期，不是 (H) 的一部分。
- *warmth 消除不了的部分.* 曲率项 $`\tfrac12H_G\lVert G\rVert_\infty^2`$ 不依赖 $`\kappa`$，随 $`R`$ 线性累积。在长 horizon 上，均值相同的两个来源之间剩下的差别由曲率决定，而 SGD 同样有这一项。warm-up 只控制 cold 窗口。
- *消息与梯度.* Proposition 5.6 是关于 *consumed* gradient 的仿射性。当 $`G(y)`$ 本身关于 $`y`$ 非线性时，response 关于消息不是仿射的。例如序列级 teacher forcing（前缀会改变 Jacobian），以及处于激活状态的全局范数裁剪。
- *局部性.* Proposition 5.6 是单边界的结论，常数是 continuation value 的性质。要沿训练过程给出证书，还需要 [Theorem 8.2](08_verification_ZH.md#s8-2) 的 transport 与 coverage 项。

## 梯度裁剪

<a id="s5-9"></a>

**Proposition 5.9（部分激活的梯度裁剪）.** 令 $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$，$`c>0`$；约定 $`G=0`$ 时 $`G/\lVert G\rVert=0`$；令 $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ 为某个 $`G`$ 的分布下的激活率。则

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- 对两个分布，裁剪后均值与缩放后的平均方向之间的总偏差至多为 $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。例如两个分布的激活率都是 $`0.9`$ 时，用平均方向代替裁剪后均值，在差值上的误差至多为 $`0.2c`$。
- 只有在始终激活（$`p_{\rm act}=1`$）时，平均方向才恰好等于裁剪后均值。

*证明.* 第一式是按事件 $`\lbrace\lVert G\rVert\le c\rbrace`$ 拆分 $`\mathbb E\,\mathrm{clip}_c(G)`$。第二式：$`\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E[G/\lVert G\rVert]=\mathbb E\big[(G-cG/\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace\big]`$，在该事件上（包括 $`G=0`$）$`\lVert G-cG/\lVert G\rVert\rVert=\lvert\lVert G\rVert-c\rvert=c-\lVert G\rVert\le c`$。所以范数至多为 $`\mathbb E[(c-\lVert G\rVert)1\lbrace\lVert G\rVert\le c\rbrace]\le c\Pr(\lVert G\rVert\le c)=c(1-p_{\rm act})`$。两个分布的结论由三角不等式得到。$`\square`$

## 各学习器的统计量

**Optimizer.** 所有运算都按坐标进行。$`\Delta`$ 中与消息无关的项都会在 $`\mu^P-\mu^Q`$ 中抵消：weight decay、momentum 衰减，以及作用在旧状态上的 bias-correction 常数。

- *SGD：* $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta G`$。
- *Heavy-ball：* $`m^+=\beta m+G`$，$`\theta^+=\theta-\eta m^+`$。*Nesterov：* $`m^+=\beta m+G`$，$`\theta^+=\theta-\eta(G+\beta m^+)`$。
- *Adam 与 AdamW：* 见设定部分，warm Adam 记号见 Definition 5.4，并记 $`c_1=(1-\beta_1)/(1-\beta_1^{\,t})`$。
- *全局范数裁剪：* $`\mathrm{clip}_c(G)`$，在基础 optimizer 之前作用。
- *Lion 形式的 sign optimizer：* $`u=\mathrm{sign}(\beta_1m+(1-\beta_1)G)`$，$`\theta^+=\theta-\eta u`$，$`m^+=\beta_2m+(1-\beta_2)G`$。
- *LoRA：* 被适配的权重 $`W=W_0+\alpha BA`$，$`A\in\mathbb R^{r\times k}`$，$`B\in\mathbb R^{d\times r}`$，全权重梯度为 $`\Gamma=\nabla_W\ell`$。可训练参数为 $`(A,B)`$，标准初始化为 $`B_0=0`$，$`A_0`$ 随机。

对某状态处的两个分布 $`P,Q`$，记 $`\delta_1=\mathbb E_PG-\mathbb E_QG`$，$`\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}`$。由 Proposition 5.1(c)，$`\mu^P-\mu^Q`$ 在 continuation 梯度张成的 $`U`$ 上的投影，正是两个 feedback law 在一阶上必须一致的量。

<a id="s5-10"></a>

**Theorem 5.10（各学习器的一阶统计量）.**

**(a) SGD.** $`\mu^P-\mu^Q=-\eta\,\delta_1`$。一阶统计量是平均 consumed gradient；二阶统计量是 $`\mathbb E\,GG^\top`$。

**(b) Heavy-ball 与 Nesterov momentum.** 增量关于 $`G`$ 仿射：heavy-ball 为 $`\Delta^\theta=-\eta(\beta m+G)`$，$`\Delta^m=(\beta-1)m+G`$；Nesterov 为 $`\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)`$，$`\Delta^m=(\beta-1)m+G`$。所以一阶项如下。统计量是平均 consumed gradient；momentum 改变的是它的权重，不是统计量本身。

```math
\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.
```

**(c) warm 状态下的 Adam 与 AdamW.** 精确的一阶统计量是 Adam 增量的均值（Proposition 5.5(a)）。在 $`(\theta,m,v)`$ 三块中，记 $`e(G)=\theta^+-\tilde\theta^+`$，$`\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace`$（Proposition 5.5(b)），

```math
\mu^P-\mu^Q=\Big(-\eta c_1\Phi\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

- 在 frozen-denominator 近似下，$`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定 $`\mu^P-\mu^Q`$；一般情形下只决定到 frozen-denominator 误差，每个坐标至多为 $`\eta\,(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]`$。相对于步长 $`\eta u_i`$，误差至多为 $`\kappa_i/2`$；由 Proposition 5.8，$`\beta_2`$ 固定时典型梯度的 warmth 不会沿训练趋于零，而是停在 $`(1-\beta_2)/\beta_2`$ 附近。
- 逐坐标的二阶矩通过 $`v`$ 在一阶上进入，权重为 $`(1-\beta_2)\nabla_vV`$。若 $`\delta_1=0`$，一阶项等于 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle+\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle`$，其中第二项的绝对值至多为 $`\eta\sum_i\lvert\partial_{\theta_i}V\rvert(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace\kappa_i/2,1\rbrace]`$。所以均值相同的分布可以通过二阶矩通道在一阶上不同，而且这个差别不随 warmth 变小。[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1) 用 adjoint 递推算出 $`\nabla_vV`$。

**(d) cold 状态下的 Adam.** 取 Proposition 5.3 中的 $`h_\epsilon`$，

```math
\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).
```

参数块只看 $`\mathbb E\,h_\epsilon(G)`$，$`\epsilon\to0`$ 时就是符号分布。矩块保留 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$。若训练在这次更新后结束（$`R=1`$），continuation value 为 $`F_E(\theta)`$，一阶项为 $`-\eta\,\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle`$。

**(e) 梯度裁剪.** 对裁剪 SGD，精确地有 $`\mu^P-\mu^Q=-\eta\big(\mathbb E_P\mathrm{clip}_c(G)-\mathbb E_Q\mathrm{clip}_c(G)\big)`$，所以统计量是裁剪后均值。

- *反例.* 取 $`c=1`$，$`P(G=4e_1)=\tfrac14`$，$`P(G=-\tfrac43e_1)=\tfrac34`$，$`Q=\delta_0`$。此时 $`\delta_1=0`$，但 $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$，一阶项为 $`\tfrac\eta2\,\partial_{\theta_1}V`$。
- *平均方向.* 由 Proposition 5.9，每个分布下裁剪后均值与 $`c\,\mathbb E[G/\lVert G\rVert]`$ 的范数差至多为 $`c(1-p_{\rm act})`$。所以用缩放后的平均方向代替裁剪后均值，$`\mu^P_\theta-\mu^Q_\theta`$ 的变化在范数上至多为 $`\eta c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$。
- *始终激活的裁剪.* 若在两个分布下都几乎必然有 $`\lVert G\rVert>c`$，统计量恰为 $`c\,\mathbb E[G/\lVert G\rVert]`$。若两个分布还诱导出相同的 $`G/\lVert G\rVert`$ 分布，增量分布就相同，两者在每一阶上都等价。
- *带裁剪的基础 optimizer.* 对 (a)–(d) 与 (f) 中的任一基础 optimizer，统计量就是对裁剪后梯度 $`\mathrm{clip}_c(G)`$ 取的相应统计量。

**(f) 带 momentum 的 sign optimizer（Lion 形式）.** 对 $`m_i\ne0`$，零计为半次翻转，

```math
\mathbb E\,\Delta^\theta_i=-\eta\,\mathrm{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \mathrm{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\mathrm{sign}(m_i)\rbrace ,
```

且 $`\Delta^m=(1-\beta_2)(G-m)`$。参数块的统计量是翻转概率向量 $`\pi`$。

- 若 momentum 占主导，即在两个分布下都几乎必然有 $`(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert`$，则 $`\pi_i=0`$，坐标 $`i`$ 的参数增量与消息无关。
- $`m`$ 块仍携带 $`(1-\beta_2)\delta_1`$。它的一阶效应 $`\langle\nabla_mV,(1-\beta_2)\delta_1\rangle`$ 在 continuation value 关于 $`m`$ 局部为常数处为零，也就是 $`m`$ 的小变化不会改变之后任何一步更新符号的地方。

**(g) 初始化时的 LoRA.** adapter 梯度为 $`G_A=\alpha B^\top\Gamma`$，$`G_B=\alpha\Gamma A^\top`$。在 $`B=B_0=0`$ 处：

- 对每条消息 $`G_A=0`$。在 SGD、从 $`m=0`$ 开始的 momentum 以及 cold Adam 下，第一次更新对每个分布都精确地不改变 $`A`$。
- $`B`$ 块的统计量对 SGD 与 momentum 为 $`\alpha\,\mathbb E[\Gamma]A_0^\top`$，对 cold Adam 为 $`\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)`$。
- 若两个分布的平均全权重梯度只相差满足 $`\Delta\Gamma A_0^\top=0`$ 的 $`\Delta\Gamma`$，它们在 SGD 与 momentum 下第 1 步一阶一致。在 cold Adam 下，若 sketch $`\Gamma A_0^\top`$ 的符号分布相同，$`\epsilon\to0`$ 时第一次更新相同。
- 第 1 步之后 $`B\ne0`$，$`G_A=\alpha B^\top\Gamma`$ 携带第二个 sketch，所以统计量的维数随训练增长。

**汇总.**

| 部分 | 学习器与状态 | feedback law 的一阶统计量 | 经由 |
|---|---|---|---|
| **(a)** | SGD | $`\mathbb E\,G`$ | $`\theta`$ |
| **(b)** | heavy-ball、Nesterov | $`\mathbb E\,G`$ | $`\theta`$ 与 $`m`$，权重为 $`-\eta\gamma\nabla_\theta V+\nabla_mV`$ |
| **(c)** | Adam/AdamW，warm | 精确地是 Adam 增量的均值；在 frozen-denominator 近似下为 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$，误差 $`O(\eta\,u\,\kappa)`$ | $`\theta`$（经 $`\Phi`$ 预条件）、$`m`$、$`v`$ |
| **(d)** | Adam/AdamW，cold | $`\mathbb E\,h_\epsilon(G)`$（$`\epsilon\to0`$ 时为符号分布）；$`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ | $`\theta`$；$`m`$ 与 $`v`$ |
| **(e)** | 裁剪 SGD | 精确地是 $`\mathbb E\,\mathrm{clip}_c(G)`$；平均单位方向准确到 $`c(1-p_{\rm act})`$，始终激活时精确 | $`\theta`$ |
| **(e)** | 带裁剪的基础 optimizer | 对 $`\mathrm{clip}_c(G)`$ 取基础 optimizer 的统计量 | 同基础 optimizer |
| **(f)** | Lion / 带 momentum 的 sign | 翻转概率 $`\pi`$；momentum 主导时 $`\theta`$ 中无依赖 | $`\theta`$；$`m`$ |
| **(g)** | LoRA，$`B=0`$，第 1 步 | rank-$`r`$ sketch $`\mathbb E[\Gamma]A_0^\top`$（SGD）或其符号分布（cold Adam）；$`A`$ 不动 | 只有 $`B`$ |

*证明.* 全程 $`\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD`$，其中 $`D(y)=\mathbb E_\xi\Delta(z,G(y,\xi))`$。$`\Delta`$ 中是状态 $`z`$ 的固定函数的项，在两个分布下期望相同，相互抵消。每种情形都从更新规则算出 $`\Delta`$，再取期望之差。

- *(a)* $`\Delta^\theta=-\eta\lambda_{\rm wd}\theta-\eta G`$。第一项抵消，由线性得 $`-\eta\delta_1`$。二阶统计量由 Proposition 5.1(a) 取 $`k=2`$ 得到：除去只依赖状态的项，$`\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top`$。
- *(b)* heavy-ball：$`\theta^+-\theta=-\eta(\beta m+G)`$，$`m^+-m=(\beta-1)m+G`$。Nesterov：$`\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)`$。依赖消息的部分在 $`\theta`$ 中是 $`-\eta\gamma G`$，在 $`m`$ 中是 $`G`$，所以 $`\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)`$；与 $`(\nabla_\theta V,\nabla_mV)`$ 配对即得。
- *(c)* 矩块：$`m^+-m=(1-\beta_1)(G-m)`$ 与 $`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ 给出 $`(1-\beta_1)\delta_1`$ 与 $`(1-\beta_2)\delta_2`$。参数块：由 Definition 5.4，$`\tilde\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta c_1\Phi G`$，且 $`\theta^+=\tilde\theta^++e(G)`$。取期望之差得 $`\theta`$ 块；由 $`\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert`$ 与 Proposition 5.5(b) 得误差界。若 $`\delta_1=0`$，$`\theta`$ 块化为 $`\mathbb E_Pe-\mathbb E_Qe`$，$`m`$ 块为零，于是得到一阶项的所述形式与界。误差沿训练的大小是 Proposition 5.8(a)。
- *(d)* 在 $`m=v=0`$、$`t=1`$ 处，$`n=G`$，$`d=\lvert G\rvert`$，所以 $`\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta h_\epsilon(G)`$（Proposition 5.3(a)）。三个增量为 $`-\eta\lambda_{\rm wd}\theta-\eta h_\epsilon(G)`$、$`(1-\beta_1)G`$、$`(1-\beta_2)G^{\odot2}`$。$`\epsilon\to0`$ 时 $`h_\epsilon(g)_i\to\mathrm{sign}(g_i)`$，$`h_0(0)=0`$。若 $`R=1`$，continuation value 为 $`F_E(\theta)`$，故 $`\nabla_mV=\nabla_vV=0`$，只有参数块有贡献。
- *(e)* 对裁剪 SGD，除 weight decay 外 $`\Delta^\theta=-\eta\,\mathrm{clip}_c(G)`$。对反例，$`\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG`$，而 $`\mathrm{clip}_1(4e_1)=e_1`$、$`\mathrm{clip}_1(-\tfrac43e_1)=-e_1`$ 给出 $`\mathbb E_P\mathrm{clip}_1G=-\tfrac12e_1`$，于是 $`\mu^P-\mu^Q=\tfrac\eta2e_1`$。平均方向的界是对每个分布用 Proposition 5.9 再用三角不等式。若几乎必然 $`\lVert G\rVert>c`$，则 $`\mathrm{clip}_c(G)=cG/\lVert G\rVert`$；方向分布相同使增量 $`-\eta cG/\lVert G\rVert`$ 同分布，所以 Proposition 5.1(a) 的每一项以及精确 response 都相同。带裁剪的基础 optimizer 把基础更新作用在 $`\mathrm{clip}_c(G)`$ 上。
- *(f)* 记 $`c_i=\beta_1m_i+(1-\beta_1)G_i`$。对 $`m_i\ne0`$：不翻转时 $`\mathrm{sign}(c_i)=\mathrm{sign}(m_i)`$；严格翻转时 $`\mathrm{sign}(c_i)=-\mathrm{sign}(m_i)`$；$`c_i=0`$ 时 $`\mathrm{sign}(c_i)=0=\mathrm{sign}(m_i)(1-2\cdot\tfrac12)`$。所以 $`\mathbb E\,\mathrm{sign}(c_i)=\mathrm{sign}(m_i)(1-2\pi_i)`$。momentum 主导时 $`c_i`$ 几乎必然与 $`m_i`$ 同号，故 $`\pi_i=0`$。$`m`$ 块增量 $`(1-\beta_2)(G-m)`$ 给出 $`(1-\beta_2)\delta_1`$。之后的参数更新只通过之后各个 $`c`$ 的符号依赖 $`m`$；在 $`m`$ 的小扰动不改变这些符号的地方，continuation value 关于 $`m`$ 局部为常数，$`\nabla_mV=0`$。
- *(g)* 对 $`W=W_0+\alpha BA`$，链式法则给出 $`\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}`$ 与 $`\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}`$，即 $`G_B=\alpha\Gamma A^\top`$，$`G_A=\alpha B^\top\Gamma`$。在 $`B_0=0`$ 处 $`G_A`$ 恒为零：SGD 使 $`A`$ 移动 $`0`$；从 $`m=0`$ 开始的 momentum 给出 $`m^+=0`$；cold Adam 使 $`A`$ 移动 $`-\eta h_\epsilon(0)=0`$。$`B`$ 的增量在 SGD 下为 $`-\eta\alpha\Gamma A_0^\top`$，momentum 的第一步与之相同，cold Adam 下为 $`-\eta h_\epsilon(\alpha\Gamma A_0^\top)`$；取期望即得统计量。若 $`\Delta\Gamma A_0^\top=0`$，$`B`$ 块均值相同，$`A`$ 块均值为零，所以第 1 步 $`\mu^P=\mu^Q`$；在 cold Adam 下，sketch 的符号分布相同，则 $`h_0(\alpha\Gamma A_0^\top)`$ 同分布。一步之后一般 $`B_1\ne0`$，$`G_A=\alpha B_1^\top\Gamma`$ 不再为零。$`\square`$

**Theorem 5.10 的推论.**

- *要看统计量，而不只是看数据.* 同一对 feedback law，可以对一个学习器一阶等价，对另一个学习器一阶不同。均值相同的两个标签分布，对 SGD 与 momentum 一阶等价；但对 cold Adam、sign optimizer 与裁剪 SGD 可以不同，因为它们的增量看的是符号分布或裁剪后均值；对 warm Adam 也可以不同，差别经由二阶矩通道。所以 teacher 数据必须在学生实际使用的 optimizer 下评判。
- *用 AdamW 加全局范数裁剪训练的 few-shot LoRA.* 初始化时适用 (g) 行：第一次更新只通过 rank-$`r`$ sketch $`\Gamma A_0^\top`$ 的符号分布看到反馈。一旦裁剪在部分 batch 上激活，(c)–(e) 行都对裁剪后的梯度取，而"只看平均方向"的简化只准确到 $`c(1-p_{\rm act})`$。
- *与 Proposition 5.3、5.6 的关系.* (d) 行统一了 Proposition 5.3(a)–(c) 与 (e)：参数块携带符号分布，矩块携带可以使第二步反向的幅度。(c) 行是 Proposition 5.6 的线性化形式：它把 Proposition 5.6 中的残差 $`L_v\kappa`$ 细化成显式的线性项 $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle`$。

<a id="s5-11"></a>

**Remark 5.11（在真实模型上检验 warm 状态的条件）.**

- Proposition 5.6 的 warm 状态误差，是按敏感度加权的和 $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，再加上 $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$。关于 $`\kappa`$ 分位数的陈述（例如"大多数坐标的 warmth 很小"）对这两项都不构成验证：少数 $`L_iu_i\kappa_i`$ 很大的坐标就能主导这个和，而第二项取决于最大值。
- 所以在真实模型上应当报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ 及其最大的几项，以及 $`\max_i\kappa_i`$。敏感度 $`L_i`$ 是 continuation value 的性质，单看更新是观察不到的；任何替代量 $`\hat L_i`$（例如留出验证梯度的幅度）都是 proxy，必须作为 proxy 报告。
- 测得的更新映射非可加性（$`\Delta`$ 关于消息离仿射有多远）是 optimizer 映射的性质，不是任务 response 的性质。response 是否仿射必须在 response 本身上测，例如用 [Definition 8.6](08_verification_ZH.md#s8-6) 的配对替换分数。
- 由 Proposition 5.8(a)，训练后期典型坐标 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$，而不是零。

## 相关工作

- Adam 带 bias correction 的第一步及其 sign descent 行为是已知的：Kingma & Ba（2015）；Balles & Hennig（2018）把 Adam 看成带方差自适应的 sign descent；Liu et al.（RAdam，ICLR 2020）讨论早期步的方差；signSGD（Bernstein et al., 2018）与 Lion（Chen et al., 2023）。decoupled weight decay 见 Loshchilov & Hutter（2019）。
- LESS（Xia et al., 2024）在 LoRA warm-up 之后用经 Adam 预条件的梯度特征做数据选择，对应 Theorem 5.10(c) 的参数块。Adam 版 In-Run Data Shapley（Ding et al., 2026）采用 fixed-state 近似，即只算当前一步的预条件更新。
- Proposition 5.1 与 stochastic modified equations（Li, Tai & E, 2019）、TracIn（Pruthi et al., 2020）、influence function（Koh & Liang, 2017）、metagradient（Engstrom et al., 2025）和 In-Run Data Shapley（Wang et al., 2025）属于同一类计算，见 Remark 5.2。
- 本节新增的是信息层面的陈述：给定 optimizer 使 feedback law 的哪个一阶统计量变得相关——在包括 $`m`$、$`v`$ 的完整状态上，残差作为 warmth 的函数显式给出——以及 $`\beta_2`$ 固定时 warmth 的精确长期行为。
