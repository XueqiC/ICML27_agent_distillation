# Theorem 8.2 · 带 transport 的冻结有限接口选择

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md) · [证明](../proofs/08-02_frozen_finite_interface_selection_transport_ZH.md) · [实验](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md) · [English](08-02_frozen_finite_interface_selection_transport.md)

**本页证明什么.** 被 audit 差距的同时集中性；在认证状态上借助 response 分解与 transport 界得到局部界；借助切换过程与停时形式的 telescoping 恒等式得到学习价值界；两个特例；特征漂移；以及一个说明 transport 项必要的见证。

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

**陈述（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)）.** 下面所有的界都可以截断到 $`1`$。

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

*Proof.*

**(a)** 每个 $`\phi_{u,\ell}\in[0,1]`$，每个经验均值背后的样本相互独立。由双侧 Hoeffding 不等式 ([Hoeffding, 1963](../references_ZH.md#hoeffding1963probability))，

```math
\Pr\big(\lvert\widehat\mu_{0,u,\ell}-\mu_{0,u,\ell}\rvert>e_0\big)\le2e^{-2n_0e_0^2}=\frac{\delta}{2\lvert\mathcal U\rvert d},\qquad\Pr\big(\lvert\widehat\mu_{j,u,\ell}-\mu_{j,u,\ell}\rvert>e_S\big)\le\frac{\delta}{2M\lvert\mathcal U\rvert d}.
```

- 对 $`\lvert\mathcal U\rvert d`$ 个参照均值与 $`M\lvert\mathcal U\rvert d`$ 个候选均值取 union bound，得到事件 $`\mathcal A`$，$`\Pr(\mathcal A^c)\le\delta`$。同一条消息上不同特征之间的相关性不影响 union bound。
- 在 $`\mathcal A`$ 上，对每个 $`j,u,\ell`$ 有 $`\big\lvert(\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell})-(\mu_{0,u,\ell}-\mu_{j,u,\ell})\big\rvert\le e_0+e_S`$。映射 $`w\mapsto\max_k\lvert w_k\rvert`$ 关于 sup 范数是 $`1`$-Lipschitz 的，所以对所有 $`j`$，$`\lvert\widehat D_j-D_j\rvert\le e_0+e_S`$。
- 取 $`j^\star\in\arg\min_jD_j`$。在 $`\mathcal A`$ 上，$`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}`$。在 audit 数据上做选择是合法的，因为 $`\mathcal A`$ 对冻结的候选集合同时成立。

**(b)** 由 (K)，$`(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}`$。把 (D) 逐项代入。

- 常数 $`c_{t,z}`$ 对两个概率律之差积分为零。
- 线性项：令 $`w=(P^0_{t,z}-P^j_{t,z})\phi_u\in\mathbb R^d`$，则 $`\lvert b^\top w\rvert\le\lVert b\rVert_1\lVert w\rVert_\infty`$；由 (T) 与三角不等式，$`\lVert w\rVert_\infty\le\lVert(P^0_u-P^j_u)\phi_u\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)`$。
- 残差：$`\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t`$。
- 由于 $`g\in[0,1]`$，左端也不超过 $`1`$。

**(c)** 在 $`\mathcal A`$ 上、固定一个 audit 结果来论证，于是 $`\widehat j`$ 固定。由新鲜训练，训练过程的条件分布就是由 $`\widehat j`$ 驱动的过程的分布。

- *切换过程.* 令 $`S'`$ 在第 $`t`$ 轮当 $`t<\sigma`$ 时用 $`K^{\widehat j}_t`$，否则用 $`K^0_t`$。由于 $`\lbrace\sigma\le t\rbrace`$ 由 $`Z_0,\dots,Z_t`$ 决定，$`S'`$ 是良定义的依赖历史的过程；$`V^0`$ 不依赖切换标记。用相同的随机性把 $`S'`$ 与 $`\widehat j`$ 过程耦合，两条路径在 $`Z_\sigma`$ 之前（含）一致，在 $`\lbrace\sigma\ge R\rbrace`$ 上直到 $`Z_R`$ 都一致。终值在 $`[0,1]`$ 中，所以 $`\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho`$。
- *Telescoping.* 由塔性质，$`\mathbb E[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t]=(K^{S'}_tV^0_{t+1})(Z_t)`$。对 $`t`$ 求和 $`\mathbb E_{S'}[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})]`$，利用相同的初始律与 $`V^0_R=F_E`$，得

```math
L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace(K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big],
```

第二个等号是因为 $`t\ge\sigma`$ 时两个 kernel 相同。这是 [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的停时形式。

- *退出前的占据分布.* 在 $`\sigma`$ 之前，$`S'`$ 与 $`\widehat j`$ 过程用相同的 kernel，所以对有界的 $`h`$ 有 $`\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]`$。在 $`\lbrace t<\sigma\rbrace`$ 上 $`Z_t\in\mathcal C_t`$，(b) 适用；在 $`\mathcal A`$ 上 $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S`$。这就把 $`\lvert L_E(0)-L_E(S')\rvert`$ 界为 (c) 中的和。
- 与切换过程的界合起来，用三角不等式即得 (c)。这个界对 $`\mathcal A`$ 中的每个 audit 结果都成立，所以 (T)、(D) 与 $`\rho`$ 的界必须对 $`\mathcal A`$ 上每个可能被选中的候选都成立。

**(d)** $`a\equiv0`$ 时，(c) 的每一项至多为 $`B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t`$，因为 $`1\lbrace t<\sigma\rbrace\le1`$ 且括号非负。在 IID 仿射假设下，候选 $`j`$ 的每一轮训练都从一个固定的律 $`P^j`$ 中抽一个新的 block，与状态无关。取单个 cell，特征为单项式 $`Z^\alpha`$（$`1\le\lvert\alpha\rvert\le N`$），它们取值于 $`[0,1]`$。于是 $`D_j=\delta_N(P^0,P^j)`$；[Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates_ZH.md#s4-5) 给出 $`\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)`$；在 $`\mathcal A`$ 上 $`\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S`$。这里不需要 transport，因为被 audit 的律就是每个状态上实际使用的律；训练 block 必须是从这个律中新抽的。

**(e)** 写 $`\phi_z=\phi_u+(\phi_z-\phi_u)`$。第一部分如 (b) 至多贡献 $`D_j+a_{t,j}(z)`$。对每个坐标 $`\ell`$，$`\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_u)_\ell\rvert\le2\alpha_{zu}`$。用对 $`\phi_z`$ 写出的 (D) 重复 (b)–(c) 即得。只在抽到的消息上成立的界，不是关于 $`\sup_y`$ 的界。

**(f)** 取 $`\mathcal Y=\lbrace0,1\rbrace`$，$`\phi(y)=y`$，$`R=1`$；更新把最终策略参数设为 $`p=y`$，$`F_E(p)=p`$。于是在每个状态上 $`g_{0,z}(y)=y`$，(D) 以 $`c=0`$、$`b=1`$、$`r=0`$ 成立。

- 在被 audit 的状态 $`z_a`$ 上，两个律都发出 $`0`$，所以 $`D_j=0`$。
- 训练从 $`z_b`$ 开始，参照发出 $`0`$，候选发出 $`1`$，所以 $`\lvert L_E(0)-L_E(j)\rvert=1`$。
- 若把 $`z_b`$ 认证为 cell $`u(z_b)=z_a`$ 而不加 transport 项，得到的表达式为 $`B(\widehat D+e_0+e_S)=e_0+e_S`$。取 $`n_0=n_S=10^4`$、$`\delta=0.05`$、$`\lvert\mathcal U\rvert=M=d=1`$，它等于 $`2\sqrt{\log80/20000}\approx0.0296<1`$。
- 在 (c) 中，要认证 $`z_b`$ 就必须 $`a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1`$，界至少为 $`1`$；或者 $`z_b\notin\mathcal C_0`$，于是 $`\sigma=0`$、$`\rho=1`$，界为 $`1`$。

两种情形下，截断后的 (c) 都等于差距，所以 transport 项既不能去掉，也不能按约定设为零。同样的构造中，若一个被覆盖的状态以概率 $`r`$ 到达一个未被覆盖的状态，则说明：每个访问到的退出前状态都必须贡献自己的项，或者首次访问未覆盖状态必须计入 $`\rho\ge r`$；否则界会差 $`r`$。$`\square`$

## 参考文献

- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability). Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58, no. 301, 13–30.
