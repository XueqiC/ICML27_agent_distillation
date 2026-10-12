# Proposition 5.1 · optimizer 变换后的展开

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-01_optimizer_transformed_expansion_ZH.md) · [证明](../proofs/05-01_optimizer_transformed_expansion_ZH.md) · [实验](../experiments/05-01_optimizer_transformed_expansion_ZH.md) · [English](05-01_optimizer_transformed_expansion.md)

**本页证明什么.** 四个部分都给出证明：(a) 单边界上反馈律效应的 Taylor 展开及显式余项；(b) 把它代入整个训练过程的 telescoping 恒等式，并给出对偶范数形式的证书与停止版本；(c) 一阶的充分性、必要性以及一阶统计量的最小个数；(d) 矩层级。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

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

**陈述（[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)）.**

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

*(b) 的证明.* [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的 telescoping 恒等式对从相同初始分布出发的任意两个协议都成立，与哪一个被叫作参考无关。取 $`P`$ 为参考、$`Q`$ 为替代，

```math
L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(Z_t)\big].
```

在每个访问到的状态 $`Z_t=z`$ 上，$`V^P_{t+1}`$ 满足 $`(\mathrm S_1)`$，于是 (a) 取 $`k=1`$ 给出

```math
(K_t^P-K_t^Q)V^P_{t+1}(z)=\langle\nabla V^P_{t+1}(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_{1,t}(z),\qquad\lvert r_{1,t}(z)\rvert\le\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big).
```

- 映射 $`z\mapsto\mu_t(z)`$、$`m_{2,t}(z)`$、$`\nabla V^P_{t+1}(z)`$ 都可测。对 $`d_t^Q`$ 积分并对 $`t`$ 求和即得表示式，其中 $`\mathcal E=\sum_t\mathbb E_{d_t^Q}r_{1,t}`$，界如所述。
- 对证书形式，每个一阶项用 $`\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert`$，求和用三角不等式。截断在 $`1`$ 是因为 $`L(P),L(Q)\in[0,1]`$。
- 对停止版本，在 $`C_t`$ 上定义 $`\varepsilon_t(z)=\Lambda_t(z)\lVert\mu^P_t(z)-\mu^Q_t(z)\rVert+\tfrac{M_{2,t}(z)}2\big(m^P_{2,t}(z)+m^Q_{2,t}(z)\big)`$。由 (a) 与对偶范数不等式，对 $`z\in C_t`$ 有 $`\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\le\varepsilon_t(z)`$。以 $`P`$ 为参考应用 [Theorem 3.2(c)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的停止版本，得 $`\lvert L(P)-L(Q)\rvert\le\mathbb E_Q\sum_{t<\tau}\varepsilon_t(Z_t)+\rho`$。$`\square`$

*(c) 的证明.* 记 $`w=\mu^P-\mu^Q`$。

- *充分性.* 对 $`V\in\mathcal V`$，$`\nabla V(z)\in U`$。所以 $`\langle\nabla V(z),w\rangle=\langle\nabla V(z),\Pi_Uw\rangle=0`$，(a) 取 $`k=1`$ 后只剩 $`r_1`$。
- *必要性.* 若对每个 $`V\in\mathcal V`$ 都有 $`\langle\nabla V(z),w\rangle=0`$，则 $`w`$ 与 $`U`$ 的一组张成元都正交，故 $`\Pi_Uw=0`$，矛盾。对 $`\langle\nabla V(z),w\rangle\ne0`$ 的 $`V`$，(a) 给出 $`(K^P-K^Q)V(z)=\langle\nabla V(z),w\rangle+r_1`$；用反向三角不等式和 $`\lvert r_1\rvert`$ 的界即得。
- *最小性.* 对 $`V\in\mathcal V`$，一阶项是消息分布的线性泛函：

```math
\langle\nabla V(z),\mu^P-\mu^Q\rangle=\sum_y\big(P(y)-Q(y)\big)\,g_V(y),\qquad g_V(y)=\langle\nabla V(z),D_t(z,y)\rangle\in\mathcal G^{(1)}_{t,z}.
```

- 所以"用特征 $`\phi_1,\dots,\phi_d`$ 的均值对所有分布决定每个一阶项"，就是 [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 对 response 类 $`\mathcal G^{(1)}_{t,z}`$ 的精确充分性问题。该定理给出最少个数 $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)}_{t,z})-1`$。
- 上界：每个 $`g_V`$ 都形如 $`y\mapsto\langle u,D_t(z,y)\rangle`$，$`u\in U`$。映射 $`u\mapsto\langle u,D_t(z,\cdot)\rangle`$ 是从 $`U`$ 到字母表上函数的线性映射，像的维数不超过 $`\dim U`$，所以 $`\dim\mathrm{span}(\lbrace1\rbrace\cup\mathcal G^{(1)})\le\dim U+1`$。$`m`$ 点字母表上的全部函数构成含常数的 $`m`$ 维空间，所以这个量也不超过 $`m`$。减去 $`1`$ 即得两个界。$`\square`$

*(d) 的证明.* 假设使 (a) 中每一项 $`\langle\nabla^jV(z),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle`$（$`j\le k`$）都为零，只剩 $`r_k`$。

- 对 SGD，$`\Delta^{\otimes j}=(-\eta)^jG^{\otimes j}`$，所以匹配增量矩就是匹配梯度矩，且 $`m_{k+1}=\eta^{k+1}\mathbb E\lVert G\rVert^{k+1}`$。
- 对 cold Adam 与始终激活的裁剪，$`\Delta`$ 的参数块中依赖消息的部分分别是 $`-\eta h_\epsilon(G)`$（Proposition 5.3(a)）与 $`-\eta c\,G/\lVert G\rVert`$，它们的张量幂就是所述的矩。$`\square`$

## 相关工作

- Proposition 5.1 与 stochastic modified equations（[Li, Tai & E, 2019](../references_ZH.md#li2019stochastic)）、TracIn（[Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)）、influence function（[Koh & Liang, 2017](../references_ZH.md#koh2017understanding)）、metagradient（[Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing)）和 In-Run Data Shapley（[Wang et al., 2025](../references_ZH.md#wang2025data)）属于同一类计算，见 Remark 5.2。

## 参考文献

- [Li, Q., Tai, C., & E, W. (2019). Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research*, vol. 20, no. 40.](../references_ZH.md#li2019stochastic)
- [Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. *NeurIPS*.](../references_ZH.md#pruthi2020estimating)
- [Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.](../references_ZH.md#koh2017understanding)
- [Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., & Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.](../references_ZH.md#engstrom2025optimizing)
- [Wang, J. T., Mittal, P., Song, D., & Jia, R. (2025). Data Shapley in one training run. *ICLR*.](../references_ZH.md#wang2025data)
