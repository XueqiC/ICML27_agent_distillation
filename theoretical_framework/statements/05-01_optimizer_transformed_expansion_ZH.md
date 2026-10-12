# Proposition 5.1 · optimizer 变换后的展开

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-01_optimizer_transformed_expansion_ZH.md) · [证明](../proofs/05-01_optimizer_transformed_expansion_ZH.md) · [实验](../experiments/05-01_optimizer_transformed_expansion_ZH.md) · [English](05-01_optimizer_transformed_expansion.md)

**一段话.** 这个命题把"用一个反馈律替换另一个"的效应，按 optimizer 对*完整*学习器状态（参数与 optimizer state 一起）的增量的各阶矩展开。后面针对具体学习器的结果都由它算出：在一阶意义下，反馈律必须携带的是完整状态增量的均值，而且只在 continuation value 敏感的方向上起作用。这类计算与 stochastic modified equations、TracIn、metagradient 属于同一类（[Remark 5.2](05-02_relation_to_prior_work_ZH.md#s5-2)），这里不主张新颖性。

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

**说了什么.** 任何带状态的一阶 optimizer（包括裁剪）都可以写成一个映射：把 consumed gradient 映成完整状态（参数、momentum、二阶矩、计数器）的增量。两个 feedback law 造成的训练结果之差，在一阶意义下等于各轮 continuation value 梯度与平均状态增量之差的内积之和，余项由增量的二阶矩控制。在 continuation value 梯度方向上匹配平均增量，一阶充分；在一阶项超过余项处也必要；统计量的个数是线性化 response 类的维数。

**直觉.** 反馈只通过它造成的状态变化影响学生。一阶意义下只有这个变化的均值起作用，而且只在 continuation value 关心的方向上起作用。

**在框架中的作用.** 一个工具，与 stochastic modified equations、TracIn、metagradient 是同类计算（Remark 5.2）。后面所有"学习器要求反馈携带什么"的结论，以及 Theorem A–C，都从它算出。

**推导要点.**

- 沿 $`z`$ 到 $`z+\Delta`$ 的线段对 $`V(z+\Delta)`$ 做带 Lagrange 余项的 Taylor 展开，在每个分布下积分；常数抵消，期望可以穿过多重线性项。
- 对整个训练过程，把这个展开代入 Theorem 3.2 telescoping 恒等式的每一项。
- 充分性与必要性来自 $`\nabla V(z)\in U`$；最小性是 Theorem 4.2 的充分性对偶用于线性化的 response 类。

**用到.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — continuation response 刻画
- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) — 充分性对偶与迁移

**被用于.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器的一阶统计量
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2) — 学习器相对泛函与信息范数
- [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) — 状态通道分解
- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) — 在实际状态上测量平均增量
- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) — 一阶版本、它的 margin 判据与上限
