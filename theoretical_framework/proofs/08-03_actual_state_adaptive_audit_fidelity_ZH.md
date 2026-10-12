# Theorem 8.3 · 实际状态上的自适应 audit；保真与改进

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [证明](../proofs/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [实验](../experiments/08-03_actual_state_adaptive_audit_fidelity_ZH.md) · [English](08-03_actual_state_adaptive_audit_fidelity.md)

**本页证明什么.** 保真性；对任意依赖历史的 provider policy 的符号 telescoping 恒等式；自适应 audit 的高概率漂移界与两条无条件期望界（用条件 Hoeffding 界、沿训练的 union bound 与鞅分解）；以及副本数。

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

**陈述（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)）.**

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

*Proof.*

**(a)** $`L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon`$。这里只用了单侧的保真界，所以允许候选优于参照。

**(b)** 沿 $`S`$ 过程，由塔性质 $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t)`$，所以 $`\mathbb E[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)]=\mathbb E\,A_t(Z_t,J_t)`$。对 $`t`$ 求和即 telescoping；由相同初始律 $`\mathbb EV^0_0(Z_0)=L_E(0)`$，由 $`V^0_R=F_E`$ 有 $`\mathbb EV^0_R(Z_R)=L_E(S)`$。

**(c)** *audit 事件.* 在 $`\mathcal H_t`$ 条件下，候选 $`j`$ 的 $`n_t`$ 个对比独立同分布，取值于 $`[-1,1]`$，均值为 $`A_t(Z_t,j)`$；这里用到了正确的边缘分布、完整尾段与精确恢复。对取值范围为 $`2`$ 的变量用 Hoeffding 不等式 ([Hoeffding, 1963](../references_ZH.md#hoeffding1963probability))，

```math
\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\ \big\vert\ \mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t .
```

- 对 $`M_t`$ 个候选取 union bound，得到事件 $`\mathcal A_t`$，$`\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\delta_t`$，其上对所有 $`j`$ 有 $`\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t`$。这个半径要求 $`n_t`$ 在看到本次 audit 的结果之前就固定。
- *被选中候选的漂移.* 记 $`A_t:=A_t(Z_t,J_t)`$。若 $`J_t=0`$，则 $`A_t=0=\Gamma_t`$。否则 $`\Gamma_t=\widehat A_t(J_t)-e_t\ge0`$，而在 $`\mathcal A_t`$ 上 $`A_t\ge\widehat A_t(J_t)-e_t=\Gamma_t`$。未做 audit 的轮次 $`A_t=\Gamma_t=0`$。
- *结论 (i).* 逐次对 $`\mathcal H_t`$ 取条件，$`\Pr(\bigcup_t\mathcal A_t^c)\le\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta`$。在 $`\bigcap_t\mathcal A_t`$ 上 $`\sum_tA_t\ge\sum_t\Gamma_t`$。这对实际到达的随机状态成立，因为每次 audit 的陈述都以它自己的 audit 前历史为条件。
- *漂移恒等式.* 实际更新是从 $`K^{J_t}_t`$ 新抽的，在 $`\mathcal F_t`$ 条件下与 audit 副本独立，而且 audit 不改变状态。所以 $`A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t)`$。差 $`D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]`$ 是交替的 audit/更新 filtration 下的鞅差，telescoping 给出

```math
F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
```

- *结论 (ii).* 总有 $`A_t\in[-1,1]`$、$`\Gamma_t\in[0,1]`$，所以 $`A_t\ge-\mathbf 1_{\mathcal A_t^c}`$，且 $`A_t\ge\Gamma_t-2\cdot\mathbf 1_{\mathcal A_t^c}`$（在 $`\mathcal A_t^c`$ 上 $`A_t\ge-1\ge\Gamma_t-2`$）。对 telescoping 恒等式取期望，并用 $`\mathbb EV^0_0(Z_0)=L_E(0)`$ 与 $`\mathbb E\sum_tD_{t+1}=0`$，

```math
L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\ge-\delta ,
```

同理 $`\mathbb E\sum_tA_t\ge\mathbb E\sum_t\Gamma_t-2\delta`$。

**(d)** $`e_t\le e`$ 等价于 $`n_t\ge2\log(2M_t/\delta_t)/e^2`$。$`M=4`$、$`\delta=0.05`$ 时 $`2\log160\approx10.150`$，除以 $`0.01`$ 再向上取整得 $`1016`$。Hoeffding 不等式只是一种充分的选择；任何半径更小、对固定 $`n`$ 成立的置信界都可以替代它。$`\square`$

## 参考文献

- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability). Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58, no. 301, 13–30.
