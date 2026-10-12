# §3 学习后果 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/03_learning_consequence_ZH.md) · [Experiments](../experiments/03_learning_consequence_ZH.md) · [English](03_learning_consequence.md)

### §3 · 学习后果：精确陈述

[证明](../proofs/03_learning_consequence_ZH.md) · [实验](../experiments/03_learning_consequence_ZH.md)

**本节做什么.**

- 它说明反馈的静态度量（teacher 质量、策略值、标签边缘分布、平均梯度）决定不了学生学到什么（[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)）。
- 它找出正确的对象：continuation response，即一条消息、实际更新、再加上声明的其余训练之后，最终任务回报的期望（[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)）。
- 它说明两个消息分布可互换，恰好当它们在每个 response 上一致；coverage 的失败只付一次代价（[Theorem 3.2(b),(c)](../proofs/03_learning_consequence_ZH.md#s3-2)）。
- 它把 response 类 $`\mathcal G`$ 确定为 §4–§8 要度量的对象（[Remark 3.3](../proofs/03_learning_consequence_ZH.md#s3-3)）。

#### Proposition 3.1 · 静态价值不等于学习价值

- **(a)** 存在四状态上的两个来源（teacher 最优，学生 $`\pi_p`$ 混叠），它们让每个学生得到相同的值 $`J(\pi_p)=\tfrac12+p/20`$，也让 teacher 得到相同的值 $`\tfrac35`$。由 $`m\ge1`$ 条示范做频率蒸馏，期望回报分别为 $`\tfrac{43}{80}`$ 与 $`\tfrac{41}{80}`$，对每个 $`m`$ 差距都是 $`\tfrac1{40}`$。
- **(b)** 存在两个标签包，边缘分布、平均梯度与平均更新都相同，但在 $`J(p)=p^2`$ 下一步 SGD 之后的学习价值相差 $`\eta^2/8`$；在 $`J'(p)=1-p^2`$ 下符号相反。

*推导要点.*

- (a) 两个状态分布给动作 $`A`$ 的平均奖励都是 $`\tfrac{11}{20}`$，所以每个混叠学生的值都相同。但 teacher 在一个分布下以 $`\tfrac34`$ 的频率选 $`A`$，在另一个分布下以 $`\tfrac14`$，而频率蒸馏在期望上照抄这个频率。
- (b) 两个包的一阶矩相同，但平均标签的方差不同（$`\tfrac18`$ 对 $`0`$）。部署值 $`p^2`$ 读取更新后学生的方差，所以 $`\mathbb E(p^+)^2`$ 相差 $`\eta^2/8`$；换成 $`1-p^2`$ 符号翻转。

*例示.* 对数据的静态评估（teacher 有多好、每个学生在数据的状态上看起来有多好、平均梯度）可以完全相同，而学习读取的是另一个统计量：(a) 中是动作频率，(b) 中是二阶矩。哪个统计量重要，由更新与部署目标的复合决定。

*用到* [Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)。*被用于* [Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)、[Proposition 7.3(a)](../proofs/07_second_moment_channel_ZH.md#s7-3)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/03_learning_consequence_ZH.md#s3-1) · [实验](../experiments/03_learning_consequence_ZH.md#c3-1) · 已证明

#### Theorem 3.2 · continuation response 刻画

**设定（[Definitions 1.3–1.4](../README_ZH.md#s1-3)）.** complete learner state $`z`$ 按 $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$ 演化，消息 $`Y_t\sim P_t(\cdot\mid Z_t)`$；积分后得到学习 kernel $`K_t`$，其中 $`K_t^0`$ 是参照，$`K_t^S`$ 是从同一初始分布出发的备选协议。参照 continuation 为 $`V_R^0=F_E`$、$`V_t^0=K_t^0V_{t+1}^0`$，单边界 response 为 $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$，$`L_E(S)=\mathbb EF_E(Z_R^S)`$，$`d_t^S`$ 是 $`Z_t^S`$ 的分布。

**(a)** 学习差距精确地 telescoping：

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big],
```

所以只要每轮差距不超过 $`\varepsilon_t`$，就有 $`\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t)`$。

**(b)** 两个消息分布在声明族的每个单边界替换上价值都相等，当且仅当对每个 $`g\in\mathcal G`$ 有 $`\mathbb E_Pg=\mathbb E_Qg`$。

**(c)** 若差距的界只在认证集合上成立，而备选过程在第 $`R`$ 轮之前离开它们的概率为 $`\rho`$，则 $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau_{\rm exit}}\varepsilon_t+\rho`$：代价是 $`\rho`$，不是 $`R\rho`$。

*推导要点.*

- (a) 把差写成 $`\sum_t(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S))`$。由 Markov 性质，每一项是参照 kernel 减备选 kernel 作用在 $`V_{t+1}^0`$ 上，取在备选过程实际访问的状态上。
- (b) 单边界替换抽一条消息、执行实际更新、再走参照尾段，所以它的值是 $`\mathbb E_Pg_{t,z}`$；值之差恰为 $`(P-Q)g`$，而能分开两个分布的那个替换本身就在族中。
- (c) 在首次离开时停止 telescoping。离开之前每一项都受控；离开之后，其余训练只贡献一个有界的回报差，至多为 1，发生在概率为 $`\rho`$ 的事件上。

*例示.* 各轮差距可以抵消：两个状态，参照 kernel 为恒等，备选为交换，两轮，$`F=(0,1)`$，最终差距为 0，而绝对漂移之和为 2。这就是 (b) 要对所有单边界替换取全称量词的原因：匹配单个最终均值不要求逐点匹配，但在每个替换上都可互换就要求了。

*用到* [Definitions 1.3–1.4](../README_ZH.md#s1-3)、[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)。*被用于* [Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)、[Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)、[Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)、[Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/03_learning_consequence_ZH.md#s3-2) · [实验](../experiments/03_learning_consequence_ZH.md#c3-3) · 已证明

#### Remark 3.3 · response 类是推导出来的，不是挑出来的

$`\mathcal G`$ 由声明的对象生成：从 $`F_E`$ 出发，反向作用参照 kernel，再经实际更新拉回。扩大声明的任务、学习器、后缀或快照族，只会扩大 $`\mathcal G`$，从而只会让 Theorem 3.2(b) 的等价关系更细。response 类由学习器与任务确定，不是为了方便而挑选的。

[证明](../proofs/03_learning_consequence_ZH.md#s3-3)
