# Proposition 8.4 · 在实际状态上测量平均增量

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-04_measuring_mean_increment_actual_states_ZH.md) · [证明](../proofs/08-04_measuring_mean_increment_actual_states_ZH.md) · [实验](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md) · [English](08-04_measuring_mean_increment_actual_states.md)

**本页证明什么.** 经验平均增量在 Hilbert 空间中的集中性（Jensen 加 McDiarmid），实际状态上的一阶 audit 界（在联合置信事件上用 Taylor 展开与 Cauchy–Schwarz），以及它沿抽样训练过程的整次训练版本。

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

沿用 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

**陈述（[Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4)）.**

**(a) Hilbert 空间中的集中性.** 设 $`X_1,\dots,X_n`$ 是实 Hilbert 空间中的独立同分布随机元，几乎必然 $`\lVert X\rVert\le b`$，均值为 $`\mu`$。则以至少 $`1-\delta`$ 的概率，

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

对大小为 $`n_P,n_Q`$ 的两个独立样本，以至少 $`1-\delta`$ 的概率，

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) 在一个实际状态上的一阶 audit.** 在产生被 audit 状态 $`z`$ 的历史条件下固定 $`z`$，并设 (B) 与 (S) 在该处成立。则在 audit 样本上以至少 $`1-\delta`$ 的概率，

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

其中

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) 沿实际训练的整次 audit.** 独立运行协议 $`Q`$ 共 $`J`$ 次。在每次运行的每一轮 $`t`$，在访问到的状态上以置信度 $`\delta_{\rm a}/(JR)`$ 应用 (b)，把 (b) 的右端截断到 $`1`$，记为 $`\hat\varepsilon_{j,t}`$。设 (B) 与 (S) 在每个访问到的状态上以所述常数成立，且抽取 audit 样本不改变训练过程。则以至少 $`1-\delta_{\rm a}-\delta_{\rm mc}`$ 的概率，

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**测了什么，没测什么.**

- 依赖数据的部分是 $`\bar X_P-\bar X_Q`$ 与 $`\hat m_2`$，它们只需要在访问到的状态上计算 consumed gradient。
- $`\Lambda`$ 与 $`M_2`$ 是 continuation value 的性质，与消息无关，样本不估计它们。它们必须解析地给界，或者用完整尾段的 audit 校准（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)），并把校准误差加进界里。

*Proof.*

**(a)** 令 $`f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert`$。

- *均值.* 由 Jensen 不等式与独立性，$`\mathbb Ef\le(\mathbb Ef^2)^{1/2}=(\mathbb E\lVert X-\mu\rVert^2/n)^{1/2}`$。交叉项 $`\mathbb E\langle X_i-\mu,X_l-\mu\rangle`$（$`i\ne l`$）由独立性为零，这里用到了内积。在 Hilbert 空间中 $`\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2`$，所以 $`\mathbb Ef\le b/\sqrt n`$。
- *偏离.* 把 $`X_i`$ 换成 $`X_i'`$，$`f`$ 的变化不超过 $`\lVert X_i-X_i'\rVert/n\le2b/n`$。对 $`c_i=2b/n`$ 用 McDiarmid 不等式 ([McDiarmid, 1989](../references_ZH.md#mcdiarmid1989method))，$`\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp(-2s^2/\sum_ic_i^2)=\exp(-ns^2/(2b^2))`$。令它等于 $`\delta`$，得 $`s=b\sqrt{2\log(1/\delta)/n}`$。
- *两个样本.* 对每个样本以 $`\delta/2`$ 用单样本的界，再用 $`\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert`$ 并取 union bound。

**(b)** 把 $`\delta/2`$ 分给 (a) 的两样本界，由 $`\log(2/(\delta/2))=\log(4/\delta)`$ 得到 $`e_1`$。把 $`\delta/4`$ 分别分给两个二阶矩：$`\lVert X\rVert^2\in[0,b^2]`$，单侧 Hoeffding 给出以至少 $`1-\delta/4`$ 的概率 $`m^P_2\le\hat m^P_2+b^2\sqrt{\log(4/\delta)/(2n_P)}`$，$`Q`$ 同理。在三个事件的交上，$`\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1`$，且 $`m^P_2+m^Q_2\le\hat m^P_2+\hat m^Q_2+e_2`$。

- 由 (S) 与带积分余项的 Taylor 定理，$`V(z+\Delta)=V(z)+\langle\nabla V(z),\Delta\rangle+\rho(\Delta)`$，$`\lvert\rho(\Delta)\rvert\le\tfrac{M_2}2\lVert\Delta\rVert^2`$。由于 $`(K^P_tV)(z)=\mathbb E_PV(z+\Delta)`$，这就是 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的单边界情形：

```math
\big\lvert(K^P_t-K^Q_t)V(z)\big\rvert\le\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert+\tfrac{M_2}{2}\big(m^P_2+m^Q_2\big).
```

- 由 Cauchy–Schwarz，$`\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert`$；代入上面两个不等式即可。样本是在状态固定之后抽取的，所以论证在给定历史的条件下成立。

**(c)** 令 $`\varepsilon_t(z)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\rbrace`$。

- *audit 事件.* 每次 audit $`(j,t)`$ 以至少 $`1-\delta_{\rm a}/(JR)`$ 的条件概率成功；条件失败概率之和至多 $`\delta_{\rm a}`$。在全部成功的事件上，利用 $`\varepsilon_t\le1`$，对所有 $`j,t`$ 有 $`\varepsilon_t(Z_{j,t})\le\hat\varepsilon_{j,t}`$。
- *Telescoping.* 以 $`P`$ 为参照应用 [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)，并在期望内用三角不等式，得 $`\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d^Q_t}\varepsilon_t(Z_t)`$。截断到 $`1`$ 是合法的，因为每一项都是两个取值于 $`[0,1]`$ 的期望之差。
- *Monte Carlo.* 每次运行的总和 $`S_j=\sum_t\varepsilon_t(Z_{j,t})`$ 独立同分布，取值于 $`[0,R]`$，均值为 $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t`$。单侧 Hoeffding 给出以至少 $`1-\delta_{\rm mc}`$ 的概率 $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\tfrac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$。在两个事件上 $`\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}`$，再取 union bound 即完成证明。$`\square`$

## 参考文献

- [McDiarmid, 1989](../references_ZH.md#mcdiarmid1989method). McDiarmid, C. (1989). On the method of bounded differences. Surveys in Combinatorics, LMS Lecture Note Series 141, 148–188.
