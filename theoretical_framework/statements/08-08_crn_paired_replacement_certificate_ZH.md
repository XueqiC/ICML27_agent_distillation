# Theorem 8.8 · CRN 配对替换证书；Theorem C

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-08_crn_paired_replacement_certificate_ZH.md) · [证明](../proofs/08-08_crn_paired_replacement_certificate_ZH.md) · [实验](../experiments/08-08_crn_paired_replacement_certificate_ZH.md) · [English](08-08_crn_paired_replacement_certificate.md)

**一段话概括.** Theorem C 把配对替换分数变成证书。在一个被 audit 的状态上，它测量“把一轮训练替换成各候选的一轮”的有限效应，候选之间通过 common random numbers 共享 surrogate 尾段，再用 empirical Bernstein 半径认证哪些候选优于哪些。整个过程不做线性化：证书不需要 continuation 梯度、不需要曲率常数，也没有二阶余项。

**本页用到的设定与记号.**

全节沿用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 轮次 $`t=0,\dots,R-1`$，$`z`$ 是 complete learner state。对一阶 optimizer，实际更新为 $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。
- 在第 $`t`$ 轮状态 $`z`$ 处，参照发出 $`Y\sim P^0_{t,z}`$，候选 $`j\in\lbrace1,\dots,M\rbrace`$ 发出 $`Y\sim P^j_{t,z}`$。候选 $`j`$ 的 kernel 记为 $`K^j_t`$。
- 参照 continuation 为 $`V^0_{t+1}`$，$`V^0_R=F_E\in[0,1]`$。response 为 $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$。对消息积分得到 kernel 恒等式

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- 对由候选 $`j`$ 驱动的过程，$`L_E(j)=\mathbb E\,F_E(Z^j_R)`$。所有过程从相同的 $`Z_0`$ 初始律出发。
- 候选 $`j`$ 在 $`(t,z)`$ 处的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- 它表示：把一轮参照替换成一轮候选 $`j`$，之后走参照尾段，对最终得分的影响。由 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)，整次训练的差距就是这些项在替代过程到达的状态上的和。

*配对替换分数*（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）.

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

*它的假设*（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)）.

- **(I) 副本独立.** 在产生 $`z`$ 的历史条件下，向量 $`(Z^{jk}_\ell)_{j,k}`$（$`\ell=1,\dots,r`$）独立同分布；$`r`$ 在观察分数之前固定。
- **(B) 分数有界.** 对每一对，$`Z^{jk}_\ell`$ 几乎必然落在一个已知的、长度为 $`B`$ 的区间里。例如 $`\tilde F\in[0,1]`$ 时 $`B\le2`$。
- **(A) surrogate 对齐.** 存在 $`\alpha>0`$ 与 $`\varepsilon_{\rm sur}\ge0`$，使对所有候选对，

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ 同时吸收了 surrogate 与任务目标之间的差别，以及尾段截短带来的差别。

<a id="s8-8"></a>

**Theorem 8.8（CRN 配对替换证书；Theorem C）.** 在 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 下，令 $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ 为无偏样本方差，

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)} .
```

- **(a) 精确性.** $`\mathbb E\,Z^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$，不涉及任何一阶近似。
- **(b) 证书.** 以至少 $`1-\delta`$ 的概率，对所有有序对 $`(j,k)`$（$`j\ne k`$）同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- **(c) 判定规则.** 只要 $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ 就判定"$`j`$ 优于 $`k`$"，则以至少 $`1-\delta`$ 的概率，所有对上都不会出现错误的判定。取 $`k=0`$，这就认证了 $`A_t(z,j)>0`$。
- 这个证书不需要 continuation 梯度 $`\nabla V`$，不需要曲率常数，也没有二阶余项。唯一需要校准的常数是 $`(\alpha,\varepsilon_{\rm sur})`$（[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。

**说了什么.** 在被 audit 的状态上，让每个候选走一轮，然后所有候选用同一组随机数走同一个 surrogate 尾段。配对得分差的均值无偏地估计替换的有限效应；再加上 empirical Bernstein 半径与 surrogate 校准项，就得到对所有候选对同时成立的证书。

**直觉.** 与其先线性化再给线性化误差找界，不如直接测有限效应；共享随机数抵消了两次独立运行会引入的尾段噪声。

**在框架中的作用.** 它用一个测量代替 oracle continuation 梯度，也是方法中的选择步骤。它的严格性依赖已知的分数范围与校准过的 surrogate，这正是 Remark 8.17 的未解决问题。

**关于陈述的说明.**

- **估计的是什么.** 被估计的量是 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 中一轮替换的有限对比，在 surrogate 尾段上测量。这里没有任何线性化，所以 §7 的二阶矩通道虽然关于消息幅度是非线性、会饱和的（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)），也能按原样被测到。
- **假设在哪里起作用.** (I) 要求在固定状态上每个副本都用新的 $`\omega_\ell`$ 与新的消息。(B) 需要一个事先已知的取值范围：分数的观测范围不是合法的 $`B`$。(A) 是关于总体差值的陈述，需要校准（[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。

**推导要点.**

- 精确性来自期望的线性：每一项只依赖于一个候选增量的边缘分布与独立的尾段随机性，所以候选之间怎么耦合都不改变均值。
- 把 Maurer 与 Pontil 对 $`[0,1]`$ 变量的 empirical Bernstein 界缩放到长度为 $`B`$ 的区间，得到每个单侧界；有序对至多 $`2M^2`$ 个，union bound 后得到 $`\log(4M^2/\delta)`$。
- 对齐假设把 surrogate 上的界转到任务上。

**示例.** 两个候选的平均增量相同、二阶矩不同，它们的一阶差为零；Theorem 8.8 仍然看得见它们的有限差，因为没有做任何线性化。

**用到.**

- [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) — 配对替换分数
- [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) — 独立性、范围与对齐
- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — 一轮替换对比
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) — 用完整尾段 audit 做校准

**被用于.**

- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) — 一阶变体及其上限
- [Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10) — 代价与副本数
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11) — 在自适应 audit 中的用法
- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5) — 一阶排序器在哪里失效
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — 方法中的选择步骤

## Related work

- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 把 empirical Bernstein 界（[Maurer & Pontil, 2009](#ref-maurer2009empirical)；[Audibert et al., 2009](#ref-audibert2009exploration)）与 common random numbers（模拟中经典的方差缩减手段，[Glasserman, 2004](#ref-glasserman2004monte)）结合起来。它的内容在于被估计的量：实际 complete learner state 上一轮替换的有限对比，在经过校准的 surrogate 尾段上测量。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-audibert2009exploration"></a>

**Jean-Yves Audibert, Rémi Munos, Csaba Szepesvári (2009).** Exploration–exploitation tradeoff using variance estimates in multi-armed bandits. *Theoretical Computer Science* 410, no. 19, 1876–1902. DOI [10.1016/j.tcs.2009.01.016](https://doi.org/10.1016/j.tcs.2009.01.016).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
