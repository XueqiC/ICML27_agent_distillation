# Theorem 8.8 · CRN 配对替换证书；Theorem C

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-08_crn_paired_replacement_certificate_ZH.md) · [证明](../proofs/08-08_crn_paired_replacement_certificate_ZH.md) · [实验](../experiments/08-08_crn_paired_replacement_certificate_ZH.md) · [English](08-08_crn_paired_replacement_certificate.md)

**本页证明什么.** 由期望的线性得到精确性（耦合不改变边缘分布）；对每个有序候选对用缩放后的 empirical Bernstein 界，并对至多 2M² 个候选对取 union bound；再借助 surrogate 对齐把结论转到任务上。

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

**陈述（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）.** 在 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 下，令 $`\hat V^{jk}=\frac1{r-1}\sum_\ell(Z^{jk}_\ell-\hat S^{jk})^2`$ 为无偏样本方差，

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

*Proof.*

**(a)** 由期望的线性，$`\mathbb EZ^{jk}_\ell=\mathbb E\varphi_{\omega_\ell}(\Delta^j_\ell)-\mathbb E\varphi_{\omega_\ell}(\Delta^k_\ell)`$。每一项只依赖于 $`(\omega_\ell,\Delta^j_\ell)`$ 的联合分布，它是 $`\omega`$ 的分布与 $`\Delta^j`$ 的边缘分布的乘积，与候选之间用什么耦合无关。所以两项分别等于 $`\tilde L(P^j)`$ 与 $`\tilde L(P^k)`$。在有限支撑上精确枚举时，把抽样换成内层平均，结论相同。

**(b)** *Empirical Bernstein 界.* Maurer 与 Pontil（[2009](#ref-maurer2009empirical)）证明了如下 empirical Bernstein 界：对 $`n\ge2`$ 个取值于 $`[0,1]`$ 的独立同分布变量 $`X_i`$，记其无偏样本方差为 $`V_n`$，则以至少 $`1-\delta'`$ 的概率，

```math
\mathbb EX-\frac1n\sum_iX_i\le\sqrt{\frac{2V_n\log(2/\delta')}{n}}+\frac{7\log(2/\delta')}{3(n-1)} .
```

- 若 $`Z`$ 取值于区间 $`[a,a+B]`$，对 $`X=(Z-a)/B\in[0,1]`$ 用上式（其样本方差为 $`V_n(Z)/B^2`$），再乘以 $`B`$，得到关于 $`Z`$ 的同一形式的界：$`\sqrt{2V_n(Z)\log(2/\delta')/n}+7B\log(2/\delta')/(3(n-1))`$。
- *事件.* 对每个有序对 $`(j,k)`$，把上述界用于 $`X=-Z^{jk}`$（取值于长度为 $`B`$ 的区间，样本方差相同），得到下尾 $`\tilde L(P^j)-\tilde L(P^k)\ge\hat S^{jk}-e^{jk}_{\rm stat}`$。有序对共有 $`(M+1)M`$ 个，而 $`M\ge1`$ 时 $`(M+1)M\le2M^2`$。每个取 $`\delta'=\delta/(2M^2)`$，于是 $`\log(2/\delta')=\log(4M^2/\delta)`$。union bound 给出全部成立的概率至少为 $`1-\delta`$。$`(k,j)`$ 的事件就是 $`(j,k)`$ 的上尾，所以对每个无序对界都是双侧的。
- *转到任务.* 在该事件上，由 (A)，$`L_F(P^j)-L_F(P^k)\ge\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)-\alpha\varepsilon_{\rm sur}\ge\alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big)`$。

**(c)** 在 (b) 的事件上，由 $`\alpha>0`$，$`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ 推出 $`L_F(P^j)-L_F(P^k)>0`$。取 $`k=0`$，$`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。$`\square`$

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
