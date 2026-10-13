# Theorem 8.3 · 实际状态上的自适应 audit；保真与改进

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [证明](../proofs/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [实验](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [English](08-03_actual_state_adaptive_audit_fidelity.md)

**一段话概括.** 除了在冻结的 cell 上 audit，还可以在训练实际到达的状态上 audit：在每个被 audit 的轮次，对每个候选与参照做配对的完整尾段实验，并按置信下界选择。这个定理证明：这样得到的训练流程相对参照的损失不超过置信水平，保留被认证的改进，所需的配对副本数显式而且很大。它还把“忠实于参照”与“相对初始学生的改进”区分开。

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

<a id="s8-3"></a>

**Theorem 8.3（实际状态上的自适应 audit；保真与改进）.**

- **(a) 保真.** 若 $`L_E(0)-J_E(\pi_0)\ge\gamma`$ 且 $`L_E(0)-L_E(S)\le\varepsilon`$，则 $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$。这里 $`J_E(\pi_0)`$ 是初始学生的价值；$`\varepsilon`$ 可以取 [Theorem 8.2(c)](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的证书，$`\gamma`$ 可以取用 $`K`$ 条示范做参照训练时测得的提升。
- **(b) 带符号的恒等式.** 对任何依赖历史、初始律相同的 provider 策略 $`S`$，

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) 自适应 audit.** (i) 以至少 $`1-\delta`$ 的概率，$`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$。(ii) 对 audit 与训练无条件地，

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) 代价.** 要使半径 $`e_t\le e`$，每个候选需要 $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ 个配对副本。$`M=4`$、$`\delta=0.05`$、$`e=0.1`$ 时为 $`\lceil2\log160/0.01\rceil=1016`$。这个数是充分的，不是必要的。

**说了什么.** 在训练实际到达的状态上，对每个候选与参照做配对的完整尾段实验，按置信下界选择。这样得到的训练流程相对参照的损失不超过置信水平，并保留被认证的改进。所需的配对副本数是显式的，而且很大。

**直觉.** 它直接测量“替换这一轮是否有帮助”，不需要外推到没见过的状态；代价是大量完整的训练尾段。

**在框架中的作用.** 它是可靠但昂贵的参照 audit，用来校准更便宜的测量，例如 Theorem 8.8 的 surrogate。

**关于陈述的说明.**

- **置信下界是必需的.** 即使 audit 用的是精确的完整尾段，按点估计 $`\widehat A_t(j)`$ 选择的期望增益也可以严格为负；一个略差的候选也可能以正概率通过置信下界的门槛。所以 (c)(ii) 中的 $`-\delta`$ 项去不掉：精确的"不退化"（$`\ge0`$）一般不成立。第一个事实的精确有限实例见[实验页](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md)。
- **无条件，而非条件.** (c)(ii) 是对 audit 结果取平均。把终值条件在"所有 audit 都成功"上，一般是不对的，因为后面的 audit 事件依赖于到达的状态；精确实例见[实验页](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md)。
- **停时版本.** 若 $`\tau`$ 是第一次失败的 audit（截断到 $`R`$），则 $`L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}\Gamma_t-\delta`$。
- **去掉了什么，没去掉什么.** 在实际状态上做 audit，去掉了 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的 transport 项，但没有去掉恢复状态、完整尾段的代价与独立性要求。把胜出的 audit 副本直接当作更新，或者用短尾段做 audit，都会改变被估计的量；短尾段需要一个显式的 surrogate 余量，这正是 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 提供的。
- **保真不等于改进.** (a) 只继承参照测得的提升 $`\gamma`$。对一个无效参照的高保真复制，并不能改进初始学生。


# §8 验证证书（Theorem C） — 证明

[← Framework](../README_ZH.md) · Proofs · [Experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · English

本页包含 Proposition 8.4 至 Remark 8.12，包括 Theorem C；同节页面给出前面的测量、transport 与 adaptive-audit 证明，以及沿用的设定。

**推导要点.**

- 以 audit 前的历史为条件，配对对比独立同分布、取值于 $`[-1,1]`$、均值为 $`A_t(Z_t,j)`$；Hoeffding 与对候选的 union bound 给出每轮的事件，沿训练失败概率之和不超过 $`\delta`$。
- 在每轮的事件上，被选候选的优势至少为 $`\Gamma_t\ge0`$。
- 新鲜更新使 $`A_t`$ 成为 $`V^0`$ 的条件漂移，于是终值可以 telescoping 成漂移加鞅差；取期望即得无条件的界。

**示例.** 即使每次 audit 都用精确的完整尾段，按点估计而不是按下界选择，期望上也可能亏：略差的候选在带噪声的比较中赢的次数足够多。

**用到.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — telescoping 成逐轮优势
- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) — 被它去掉 transport 项的冻结 audit

**被用于.**

- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) — 用完整尾段校准一阶常数
- [Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10) — 代价比较
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11) — 在自适应 audit 中使用认证下界
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — 方法中的选择步骤

## Related work

- [Theorems 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)–[8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 与 [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) 用到 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）、union bound 与 McDiarmid 不等式（[McDiarmid, 1989](#ref-mcdiarmid1989method)）。按置信下界选择以保证不退化，是 high-confidence policy improvement（[Thomas et al., 2015](../references_ZH.md#thomas2015high)）的思路。

## 参考文献

- [Thomas et al., 2015](../references_ZH.md#thomas2015high). Thomas, P. S., Theocharous, G., Ghavamzadeh, M. (2015). High confidence policy improvement. ICML.
