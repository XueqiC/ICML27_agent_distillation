# Theorem 3.2 · continuation response 刻画

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](03-02_continuation_response_characterization_ZH.md) · [证明](../proofs/03-02_continuation_response_characterization_ZH.md) · [实验](../experiments/03-02_continuation_response_characterization_ZH.md) · [English](03-02_continuation_response_characterization.md)

**一段话.** 既然静态量决定不了学生学到什么，那么什么能决定？这个定理给出不变量：continuation response，即一条消息、实际更新、再加上声明的其余训练之后的期望最终回报。两个训练过程之间的差，精确等于逐轮单次替换效应之和；两个消息分布可互换，恰好当它们在每个 response 上一致；离开认证区域的代价只付一次。

**本页用到的设定.**

- complete learner state $`z\in\mathcal Z`$：参数、optimizer state、缓冲区、采样器状态、随机性与历史（[Definition 1.3](../README_ZH.md#s1-3)）。
- 第 $`t<R`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 发出消息 $`Y_t`$，实际更新给出 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$，$`\xi_t`$ 是更新的随机性。对 $`Y_t`$ 与 $`\xi_t`$ 积分得到学习 kernel $`K_t`$，它作用在有界函数上：$`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$。
- $`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议 $`S`$，两者从同一初始分布 $`\nu`$ 出发。$`Z_t^S`$ 是由 $`K^S`$ 驱动的链，其分布 $`d_t^S`$ 是第 $`t`$ 轮的占据分布。
- $`F_E(z)\in[0,1]`$ 是从状态 $`z`$ 部署的学生在真实任务上的回报；$`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ 是协议 $`S`$ 的学习价值，$`L_E(0)`$ 是参照的学习价值。
- 参照 continuation value 为 $`V_R^0=F_E`$，$`V_t^0=K_t^0V_{t+1}^0`$：从某个状态出发、按参照协议继续训练时的期望最终得分（[Definition 1.4](../README_ZH.md#s1-4)）。
- 快照 $`(t,z)`$ 处的单边界 response 为 $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$：收到消息 $`y`$、执行实际更新、再走完声明的参照尾段之后的期望最终回报。response 类 $`\mathcal G`$ 收集声明的快照族 $`\mathcal I`$ 上所有的 $`g_{t,z}`$。
- *单边界替换* $`I(t,z,P)`$：在第 $`t`$ 轮从状态 $`z`$ 出发，抽一条消息 $`Y\sim P`$，执行实际更新，按声明的参照尾段走到第 $`R`$ 轮，并用 $`F_E`$ 打分。

<a id="s3-2"></a>

**Theorem 3.2（continuation response 刻画）.** 在上述设定下：

**(a) 精确复合.**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(3.2)
```

因此，若对 $`d_t^S`$-几乎所有 $`z`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(3.3)
```

**(b) 操作性刻画.** 两个消息分布 $`P,Q`$ 在声明族中的每个单边界替换上价值都相等，当且仅当

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{对每个 }g\in\mathcal G. \qquad(3.4)
```

**(c) 带停止的复合.** 设 $`C_t\subseteq\mathcal Z`$ 为认证集合，$`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$（若不存在这样的 $`t`$ 则 $`\tau=R`$），$`\rho=\Pr(\tau<R)`$。若对所有 $`z\in C_t`$ 有 $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(3.5)
```

**说了什么.**

- 参照训练过程与备选训练过程之间的差，精确等于逐轮"把这一轮的参照步换成备选步"的效应之和，效应用备选过程实际访问的状态上的参照 continuation value 度量。
- 两个消息分布在每个单边界替换上可互换，当且仅当它们在每个 response 上的期望相同。
- 当逐轮保证只在认证集合上成立时，离开这些集合的概率只付一次，而不是每轮付一次。

**直觉.**

- response $`g(y)`$ 回答的是："如果这一轮喂给学生 $`y`$，之后按声明的方式继续训练，最终得分是多少"；它把更新规则、其余训练和部署任务都包含在内。

**在框架中的作用.**

- 它确定了充分性的对象，即 response 类 $`\mathcal G`$。[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 度量它的大小，[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 把它展开到一阶，[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 与 [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 用有限数据检验它。

**推导要点.**

- (a) 把差写成 $`\sum_t(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S))`$。由 Markov 性质，每一项是参照 kernel 减备选 kernel 作用在 $`V_{t+1}^0`$ 上，取在备选过程实际访问的状态上。
- (b) 单边界替换抽一条消息、执行实际更新、再走参照尾段，所以它的值是 $`\mathbb E_Pg_{t,z}`$；值之差恰为 $`(P-Q)g`$，而能分开两个分布的那个替换本身就在族中。
- (c) 在首次离开时停止 telescoping。离开之前每一项都受控；离开之后，其余训练只贡献一个有界的回报差，至多为 1，发生在概率为 $`\rho`$ 的事件上。

**例示.** 各轮差距可以抵消：两个状态，参照 kernel 为恒等，备选为交换，两轮，$`F=(0,1)`$，最终差距为 0，而绝对漂移之和为 2。这就是 (b) 要对所有单边界替换取全称量词的原因：匹配单个最终均值不要求逐点匹配，但在每个替换上都可互换就要求了。

**说明.**

- **对象.** $`g_{t,z}(y)`$ 是收到 $`y`$、执行*实际*更新、再走完*声明的参照学习尾段*之后，*最终真实任务回报*的期望。它不是训练损失，不是 teacher 置信度，也不是冻结策略下的动作值。§4 研究 $`\mathcal G`$ 的大小与近似，§5–§7 研究它对 optimizer 的依赖，§8 研究如何测量它。
- **方向与符号.** (3.2) 中的占据分布是*备选*过程的 $`d_t^S`$。每个漂移项可正可负。两个协议在更新规则上的任何差别都必须包含在比较的 kernel 中。
- **离开的代价.** 第一次离开之后，不受控的尾段至多贡献一个有界回报的差，所以代价是 $`\rho`$ 而不是 $`R\rho`$。若 $`Z_0\notin C_0`$，则 $`\tau=0`$，这个事件计入 $`\rho`$。有限个存储的检查点本身并不能保证连续轨迹的 $`\rho`$ 很小；[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 把 coverage 定量化。
- **会响应的尾段.** 参照后缀必须对该分支自己当前的学生作出反应。在分支之间耦合随机种子是合法的；在协议本该查询已改变的学生之处重放相同的标签或轨迹，测到的是另一个 continuation。

**用到.**

- [Definition 1.3](../README_ZH.md#s1-3): complete learner state、feedback law、kernel.
- [Definition 1.4](../README_ZH.md#s1-4): continuation value 与 response.
- [Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1): 静态准则不够.

**被用于.**

- [Remark 3.3](../statements/03-03_response_class_derived_not_chosen_ZH.md#s3-3): response 类是推导出来的.
- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2): response 类的大小.
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1): 一阶展开.
- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2): 带 transport 的冻结 audit.
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3): 实际状态上的自适应 audit.
- [Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1): coupling width.
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3): 按实测后果选择.

## 相关工作

- 伸缩恒等式 (3.2) 与强化学习中的 performance-difference lemma 形式相同 ([Kakade & Langford, 2002](#ref-kakade2002approximately))，只是 MDP 换成了训练过程：占据分布来自备选过程，advantage 的角色由参照 continuation 下的单轮替换效应担任。
- Comments 中允许的"在分支之间耦合随机种子"，就是 common random numbers 方法 ([Glasserman, 2004](#ref-glasserman2004monte))。
