# Remark 8.11 · surrogate 的校准；沿训练的迁移

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-11_surrogate_calibration_and_transfer_ZH.md) · [实验](../experiments/08-11_surrogate_calibration_and_transfer_ZH.md) · [English](08-11_surrogate_calibration_and_transfer.md)

**一段话概括.** Theorem C 通过两个校准常数把 surrogate 证书转到任务上。这个 Remark 说明它们怎样拟合，说明把它们用到新的候选对上是一个假设，说明认证下界怎样在 Theorem 8.3 的自适应 audit 中代替完整尾段的下界，以及在后续状态上使用一个选择还需要什么。

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

**自适应 audit.** **provider 策略** $`S`$ 在每一轮根据历史选择一个 kernel 下标 $`J_t`$。

- 记 $`\mathcal H_t`$ 为第 $`t`$ 轮 audit 之前的历史。在 $`\mathcal H_t`$ 条件下，冻结实际状态 $`Z_t`$、$`M_t`$ 个非参照候选 kernel 组成的有限集合、副本数 $`n_t`$ 与置信分配 $`\delta_t`$，几乎处处 $`\sum_t\delta_t\le\delta`$。
- 对每个候选，从 $`Z_t`$ 出发做 $`n_t`$ 次独立的配对实验。每次实验执行候选更新或参照更新，再执行**完整的**参照尾段与真实部署。每一臂都有正确的边缘分布，配对对比落在 $`[-1,1]`$ 中。
- 记 $`\widehat A_t(j)`$ 为平均对比，并令

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- 选择 $`J_t\in\arg\max_j\ell_t(j)`$，令 $`\Gamma_t=\ell_t(J_t)\in[0,1]`$。然后从 $`K^{J_t}_t`$ 做一次**新鲜**更新，丢弃 audit 副本。未做 audit 的轮次使用参照，令 $`\Gamma_t=0`$。
- audit 不改变实际的学习器状态与环境状态，参照也不读取 audit 记录。$`\mathcal F_t\supseteq\mathcal H_t`$ 是 audit 与选择之后、新鲜更新之前的历史。

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

<a id="s8-11"></a>

**Remark 8.11（surrogate 的校准；沿训练的迁移）.**

- **校准.** 取一组校准用的候选对，用任务评测的配对完整尾段估计 $`L_F(P^j)-L_F(P^k)`$（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)），对 $`\hat S^{jk}`$ 回归，读出 $`\hat\alpha`$ 与残差上界 $`\hat\varepsilon_{\rm sur}`$。把 $`(\hat\alpha,\hat\varepsilon_{\rm sur})`$ 用到其他候选对上，需要一个可交换性假设：校准对与新的候选对来自同一族。这是假设，不是定理；校准本身的统计误差也要加进去。
- **在自适应 audit 中使用.** 在被 audit 的轮次，把 [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 的完整尾段置信下界换成被选候选的 $`\Gamma_t=\min\lbrace1,\alpha(\hat S^{J_t0}-e^{J_t0}_{\rm stat}-\varepsilon_{\rm sur})\rbrace`$；没有候选相对参照被认证时令 $`\Gamma_t=0`$。若 (A) 在每个被 audit 的状态上以声明的常数成立，且置信分配之和为 $`\delta`$，则 Theorem 8.3(c) 原样成立。它的证明只用到 audit 事件上的 $`A_t\ge\Gamma_t`$ 与 $`0\le\Gamma_t\le1`$；由于 $`A_t(z,j)=L_F(P^j)-L_F(P^0)`$，前者由 [Theorem 8.8(b)](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 给出。
- **沿训练的迁移.** 证书在被 audit 的状态上成立。要把选出的候选用到之后的状态上，需要 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的 transport 项，或者在下一个检查点重新 audit，核对排序是否保持（a posteriori 检查）。

**说了什么.** 对齐常数通过把配对完整尾段的任务对比对 surrogate 分数做回归来拟合；把它们迁移到新的候选对是可交换性假设，不是定理；在自适应 audit 中，Theorem 8.8 的认证下界可以代替完整尾段的下界，Theorem 8.3(c) 不变；在后续状态上使用选择结果需要 transport 或重新 audit。

**直觉.** 校准是 surrogate 与任务接上的地方；它需要检验，不能假定。

**在框架中的作用.** 它把 Theorem C 接到安全的自适应 audit 上，并写出 Remark 8.12 留作未解决的校准迁移条件。

**用到.**

- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) — 自适应 audit 与完整尾段校准
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) — 认证下界
- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) — 后续状态需要的 transport
- [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) — 对齐假设

**被用于.**

- [Remark 8.12](../statements/08-12_open_problems_ZH.md#s8-12) — 校准迁移这一未解决问题
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — 方法中的校准步骤

## Related work

- 只通过改进量的高置信下界来接受候选，从而保住保证，这是 high-confidence policy improvement 的安全检验（[Thomas et al., 2015](#ref-thomas2015high)）。
- 在重新训练的运行上拟合一个廉价的训练结果预测器，再在留出的运行上检验，也是 datamodels 的拟合与验证方式（[Ilyas et al., 2022](#ref-ilyas2022datamodels)）。这里的校准是在实际学习者状态上，把任务对比回归到 surrogate 分数上。
