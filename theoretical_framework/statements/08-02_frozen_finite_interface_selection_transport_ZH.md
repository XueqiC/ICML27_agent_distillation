# Theorem 8.2 · 带 transport 的冻结有限接口选择

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md) · [证明](../proofs/08-02_frozen_finite_interface_selection_transport_ZH.md) · [实验](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md) · [English](08-02_frozen_finite_interface_selection_transport.md)

**一段话概括.** 冻结的 audit 事先固定有限个 audit cell 与特征，测量每个候选 feedback law 在这些 cell 上离参照有多远，选最近的那个。这个定理回答：这样的 audit 对“用被选候选训练”的学习结果能认证什么。答案是：测得的差距，加统计误差，加训练实际访问状态上的显式 transport 项，再加离开认证状态集的概率。它还证明 transport 项去不掉。

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

**冻结的 audit.** 只用独立的 pilot 数据，在 audit 之前冻结：

- $`M`$ 个可执行的候选 feedback law；
- 一个有限的 audit cell 集合 $`\mathcal U`$，每个 cell $`u`$ 指定一个 complete learner state、一个任务、一个查询、更新方式和参照尾段（字母 $`u`$ 只在本小节表示 cell）；
- 每个 cell 上的 $`d`$ 个特征 $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$。

$`P^0_u`$ 与 $`P^j_u`$ 分别是 cell $`u`$ 中参照与候选 $`j`$ 的消息律。audit 在每个 cell 抽取 $`n_0`$ 条独立的参照消息，在每个（候选，cell）上抽取 $`n_S`$ 条独立消息。记 $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$，经验均值为 $`\widehat\mu_{j,u,\ell}`$，并令

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

被选中的候选 $`\widehat j`$ 在已知的优化误差 $`\eta_{\rm opt}`$ 之内最小化 $`\widehat D_j`$。

**被认证的状态.** 每一轮 $`t`$ 有一个由 complete state 组成的认证集 $`\mathcal C_t`$。每个 $`z\in\mathcal C_t`$ 附带：

- 一个被 audit 的 cell $`u=u(t,z)`$，其特征用 $`z`$ 处的消息坐标表示；
- 一个 **transport 界**：对 audit 事件上每个可能被选中的候选，

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z);\qquad\text{(T)}
```

- 一个 **response 分解**

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

令 $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$（若不存在则 $`\sigma=\infty`$），它是训练过程的停时。**新鲜训练**：在 audit 结果的条件下，被选中的候选用与 audit 样本独立的随机性训练，并且 $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$。

<a id="s8-2"></a>

**Theorem 8.2（带 transport 的冻结有限接口选择）.** 下面所有的界都可以截断到 $`1`$。

**(a) audit 的集中性.** 以至少 $`1-\delta`$ 的概率，对所有 $`j`$ 同时有

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) 局部 transport 界.** 对 $`z\in\mathcal C_t`$ 与 $`u=u(t,z)`$，

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) 学习价值的界.** 在 (a) 的事件上，

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) 特例.** transport 精确（$`a\equiv0`$）时，$`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$。在 [Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates_ZH.md#s4-5) 的与状态无关的 IID block 与凸仿射更新假设下，若被 audit 的特征就是矩 $`Z^\alpha`$（$`1\le\lvert\alpha\rvert\le N`$），则 $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$。
- **(e) 特征漂移.** 若依赖状态的特征 $`\phi_z`$ 满足 $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$，则 $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$；把 (D) 中的特征换成 $`\phi_z`$、把 $`a`$ 换成 $`a+2\alpha_{zu}`$，(b)–(c) 仍成立。
- **(f) transport 项的必要性.** 对任意 audit 规模，都存在一个实例：$`D_j=0`$，response 分解精确（$`B=1`$，$`\tau=0`$），而学习差距为 $`1`$。在这个实例中，不含 transport 的表达式 $`B(\widehat D+e_0+e_S)`$ 严格小于差距，(c) 的界仍然成立而且是紧的。

**说了什么.** 事先固定 audit cell 与特征，测量每个候选相对参照的特征差距，选差距最小的。被选候选的学习差距，不超过测得的差距、统计误差、训练实际访问状态上的显式 transport 项，再加上离开认证状态集的概率。transport 项去不掉。

**直觉.** audit 只看得见它覆盖的状态。训练一旦走到别处，audit 可以报告零差距，而学习差距却是最大的。

**在框架中的作用.** 它给出任何有限、冻结的 audit 的正确形式，并指出它的盲区；Theorem 8.3 与 8.8 在实际状态上做 audit，避开了这个盲区。

**关于陈述的说明.**

- **complete state.** cell、transport 与认证都针对 complete learner state。参数相同、optimizer moment 不同的两个状态，会产生不同的下一状态，response 也可以不同（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)）。只凭参数接近不能推出 (T)。
- **精确 transport 弱于分布相等.** $`a=0`$ 只要求两个特征差向量一致，不要求四个分布都相同。
- **认证的是什么.** (c) 界定的是被选中的冻结流程在新鲜训练下的总体期望值，而不是某一个训练 seed 的分数。用 audit 样本构造新候选，或者在这些样本上训练，就改变了实验，需要新的独立 audit。
- **其他置信失败.** 若 transport 界、系数或残差本身是统计估计的，它们的失败概率要加到 $`\delta`$ 上。

**推导要点.**

- 对所有 cell、特征与候选做 Hoeffding 加 union bound，得到同时成立的集中性。
- 在被认证的状态上，一轮的效应是律之差作用在 response 上；response 分解把它拆成常数（抵消）、线性部分（由系数范数乘以经 transport 后的差距控制）与残差。
- 构造一个首次退出后改用参照的切换过程，它与实际训练过程至多差 $`\rho`$，而它的 telescoping 差距（[Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)）只涉及退出前、被认证的状态。

**示例.** 在一个两律发出相同消息的状态上做 audit，然后从一个两律发出不同消息的状态开始训练。audit 报告零差距，而学习差距为 $`1`$；只有加上值为 $`1`$ 的 transport 项，或者把未被覆盖的状态计为退出，界才是对的。

**用到.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — 整次训练差距是逐轮效应之和
- [Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates_ZH.md#s4-5) — 仿射学习器的矩特征
- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1) — 按系数加权的测量误差

**被用于.**

- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) — 在实际状态上做 audit 去掉 transport
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11) — 证书沿训练的迁移
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — 方法中的覆盖步骤

## Related work

- [Theorems 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)–[8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 与 [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) 用到 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）、union bound 与 McDiarmid 不等式（[McDiarmid, 1989](#ref-mcdiarmid1989method)）。按置信下界选择以保证不退化，是 high-confidence policy improvement（[Thomas et al., 2015](../references_ZH.md#thomas2015high)）的思路。

## 参考文献

- [Thomas et al., 2015](../references_ZH.md#thomas2015high). Thomas, P. S., Theocharous, G., Ghavamzadeh, M. (2015). High confidence policy improvement. ICML.
