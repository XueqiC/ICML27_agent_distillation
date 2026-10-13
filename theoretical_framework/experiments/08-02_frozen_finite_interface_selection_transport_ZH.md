# Theorem 8.2 · 带 transport 的冻结有限接口选择

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md) · [证明](../proofs/08-02_frozen_finite_interface_selection_transport_ZH.md) · [实验](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md) · [English](08-02_frozen_finite_interface_selection_transport.md)

**本页检验什么.**

- **[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)（带 transport 的冻结选择）成立，且 transport 项是必要的.**
  - 在 2,572 个精确的有限训练过程上没有一次违反，最坏残差为 0；界可以取到（比值 1）。
  - 在一个未被覆盖的训练状态上，被 audit 的差异为 0，而学习差距为 1。
  - 在受控网络中，带实测 transport 的局部界在 40/40 个到达状态上都包住了真实对比；假设 transport 为零的预测在 36/40 个状态上失准。

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

**被检验的陈述（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)）.** 下面所有的界都可以截断到 $`1`$。

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

<a id="c8-4"></a>

## Check 8.4 · 精确的有限训练过程（Theorem 8.2）

**结论.** 在 2,572 个精确的有限马尔可夫训练过程上，transport 界从未被违反（最坏残差 0），并在一个尖锐见证上取等。界中的每一项都由一个单独的见证表明不可去掉。

**动机.** 学习价值界 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(c) 把停止形式的 telescoping 恒等式、局部 transport 界 (b) 和 audit 事件 (a) 组合在一起。其中任何一环出错，都会产生精确学习差距超过界的实例。这里在随机和对抗实例上，用有理数算术精确计算两边。

**设置.**

- 有限马尔可夫训练过程：2–4 个被认证状态外加一个退出状态，消息字母表大小 2–4，$`R`$ 轮。
- 随机的参考与候选 kernel、随机的实际更新、随机的有界终端值。
- 显式的 response 分解 $`c+b^\top\phi+r`$、精确的 transport 误差 $`a_{t,j}(z)`$、特征漂移 $`\alpha_{zx}`$（Theorem 8.2(e)），以及退出概率 $`\rho`$。
- 种子 `SeedSequence([150150, 1])`；所有数值都是 `Fraction`。
- 做法：
  1. 用后向值计算精确差距 $`\lvert L_E(0)-L_E(j)\rvert`$；
  2. 独立地前向计算停止后的占据测度；
  3. 计算 Theorem 8.2(c) 的界，并在 2,000 个随机实例上断言差距 ≤ 界；
  4. 对候选 kernel 做爬山式对抗搜索；
  5. 加入下面的固定见证，以及 IID 仿射特例（Theorem 8.2(d)）的 120 个随机实例。

**结果**（`theory/v015_checks/results_v015_independent.json`，组 `v015:V1`）：

| 量 | 值 | 键 |
|---|---:|---|
| status | PASS | `status` |
| 实例 / 断言 | 2,572 / 28,182 | `n_instances`、`n_assertions` |
| 最坏残差（差距 ≤ 界 的违反量） | 0 | `worst_residual` |
| 全部实例上最大的差距/界 比值 | 1.0（尖锐见证） | `best_ratio`、`data.sharp_transport_witness` |
| 2,000 个随机实例上的最大比值 | 0.736（差距 0.294，界 0.400） | `data.worst_random` |
| 最佳对抗比值 | 0.198（差距 0.220，界 1.114，$`\rho=74/75`$） | `data.adversarial.best` |
| IID 仿射（Theorem 8.2(d)）：差距 ≤ 界 的实例 | 120 / 120；最大比值 1.0；对抗 0.565 | `data.affine_40`、`data.affine_adversarial` |
| 精确的 audit 事件概率（Theorem 8.2(a)），$`n=32`$，$`\delta=0.3`$ | 0.99948；最坏选择超出量 −0.605（为负表示 (a) 成立） | `data.exact_finite_audit_37_38.best` |

见证：每个见证说明界中的某一项不能去掉。

| 见证 | 精确值 | 必须计入的项 |
|---|---|---|
| 未被覆盖的训练状态（`data.regression`） | 差距 $`=1`$；计入 $`a=1`$ 时界为 $`1.0296`$（截断为 $`1`$）；计入退出 $`\rho=1`$ 时为 $`1`$；删掉 transport 项后为 $`0.0296`$ | transport $`a`$ 或退出 $`\rho`$ |
| 尖锐 transport（`data.sharp_transport_witness`） | $`D=0`$，$`a=1`$，差距 $`=`$ 界 $`=1`$ | 界是紧的 |
| 遗漏一个可达状态（`data.omission_witness`） | 到达概率 $`8/9`$；差距 $`8/9`$；把该状态的项置零后界为 $`0`$；有效的退出界为 $`8/9`$ | 每个退出前访问到的状态，或 $`\rho`$ |
| 完整状态（`data.optimizer_state_witness`） | 两个状态的 $`\theta=-1`$ 相同；$`(m,v)=(\tfrac12,\tfrac12)`$ 对 $`(1,2)`$；下一步一阶矩为 $`-1/8`$ 对 $`+1/8`$ | cell 必须定义在 complete learner state 上 |
| 系数大小（`data.coefficient_stability`） | 特征尺度 $`10^{-4}`$，$`B=10^4`$，$`D=10^{-4}`$，残差 0，差距 1 | $`B_t`$ 与 $`D`$ 相乘（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)） |

**为什么支持该陈述.** 只要假设成立，界就成立；界可以取到；而且每个假设都不可少：

- 可达状态要么连同实测的 transport 被覆盖，要么计入退出概率；
- cell 必须是 complete learner state，而不是参数检查点；
- 系数以乘法方式进入。

<a id="c8-5"></a>

## Check 8.5 · audit 集中性与覆盖见证（Theorem 8.2(a)、(f)）

**结论.** 在四组 Monte Carlo 设定中，同时成立的 audit 事件出现频率都不低于名义值，选择不等式在其上成立。在覆盖探针上，被 audit 的差异为 0，而学习差距为 1。检验套件按设计把这一检验记为 FAIL：失败的那条断言是在探针上计算一个不含 transport 的界，而这恰好是 Theorem 8.2(f) 预言的违反。

**动机.** 检验两件事：

- 用 Monte Carlo 检验 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(a) 的同时 Hoeffding 事件（[Hoeffding, 1963](#ref-hoeffding1963probability)）与选择不等式；
- 一个精确的、依赖状态的 provider：它在被 audit 的 cell 上与参考一致，在一个被访问的状态上却不一致。这是必要性陈述 (f) 背后最小的实例。

**设置.**

- 四组 audit 设定 $`(M,L,d,n_0,n_S,\delta)`$，每组 3,000 次独立试验。
- 一个有限学习实例：每个 response 恰为 $`c+b^\top\phi_x`$（$`\tau=0`$），transport 精确。
- 覆盖探针：被 audit 的 cell 0，被认证状态 $`\lbrace 0,1\rbrace`$，初始状态 1，更新 $`U(z,y)=y`$，终端值 $`(0,1)`$。
- 做法：抽取 audit 样本，记录事件是否发生，计算 $`\widehat j`$ 以及选择不等式 (a) 和精确 transport 证书 (d) 的超出量。对探针，计算精确差距，以及把状态 1 当作已覆盖、不计 transport 项时得到的表达式。

**结果**（`theory/independent_checks/results_independent_v014.json`，条目 `v014:C15`，`details`）：

| $`M,L,d`$ | $`n_0,n_S`$ | $`\delta`$ | 事件频率 | 最坏超出量，(a) | 最坏超出量，(d) |
|---|---|---:|---:|---:|---:|
| 2,2,2 | 40, 50 | 0.10 | 1.000 | −0.873 | −0.238 |
| 4,3,3 | 100, 80 | 0.05 | 1.000 | −0.573 | −0.500 |
| 3,1,1 | 1000, 1200 | 0.20 | 0.99967（SE 0.00033） | −0.160 | −0.061 |
| 2,2,2 | 8, 8 | 0.40 | 1.000 | −1.626 | −0.477 |

超出量为负，表示不等式以该余量成立。

覆盖探针（`details.coverage_probe`）：被 audit 的经验差异为 $`0`$，audit 事件概率为 $`1`$，真实学习差距为 $`1`$；删掉 transport 项得到的表达式是 $`0.0296`$。套件状态为 FAIL，原因是有一条断言在探针上计算了不含 transport 的表达式（`worst_residual` 为 0.970，即差距减去该表达式）。(a) 与 (d) 的所有 Monte Carlo 断言都通过，见上表。

**为什么支持该陈述.**

- 在每组设定中，audit 事件出现的频率都不低于 $`1-\delta`$，(a) 和精确 transport 证书 (d) 在其上成立。
- 探针是被 audit 的状态与被训练到的状态不同的最小情形。只有像 Theorem 8.2(c) 那样计入 transport 或退出的界，在这里才成立；这就是陈述 (f)。

<a id="c8-6"></a>

## Check 8.6 · 正确的特征、遗漏的特征，以及带退出惩罚的覆盖（Theorem 8.2）

**结论.** 特征和覆盖都正确时证书成立；遗漏一个相关特征会藏住 $`1/2`$ 的差距；被访问但未被覆盖的状态必须通过 $`\rho`$ 计入。

**动机.** 三个有限对照把证书的三个组成部分分开：正确的特征覆盖、与决策相关的特征、状态覆盖。

**设置.** 一个 response 已知的精确有限学习过程，分三种对照：(i) 正确的特征集；(ii) 去掉一个相关特征；(iii) 训练到达被 audit 集合之外的状态。

**结果**（条目 `v014:C20`，`details`；PASS，11 条断言，最坏残差 0）：

| 对照 | 数值 |
|---|---|
| (i) 正确特征，$`n=120`$ 的 audit，$`\delta=0.05`$ | $`\widehat D=1/12`$，$`e_0+e_S=0.270`$。精确 transport 证书（Theorem 8.2(d)）：$`0.354`$ ≥ 实际三步更新差距 $`1/12`$。平方回报任务上 $`C_J=2`$ 的仿射证书：$`0.707`$ ≥ 实际差距 $`23/216=0.106`$ |
| (ii) 遗漏特征 | 两个 provider 在每个被 audit 的特征上都一致，学习价值却相差 $`1/2`$ |
| (iii) 覆盖 | 差距 $`1`$；计入 $`\rho=1`$ 时界为 $`1`$；不计 $`\rho`$ 时，不含覆盖的值为 $`0`$ |

**为什么支持该陈述.**

- 特征与覆盖都正确时，证书成立。
- 缺一个特征留下的差距，任何对其余特征的 audit 都看不见；这就是 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(a) 的精确充分性条件。
- 被访问但未被覆盖的状态必须通过 $`\rho`$ 计入。

<a id="c8-7"></a>

## Check 8.7 · 受控网络中到达状态上的 transport（Theorem 8.2(b)、(e)）

**结论.** 在一个用 Adam 训练的小网络实际到达的 40 个状态上，transport 不为零（中位数 0.0195）。带实测 transport 的局部界在全部 40 个状态上都包住了 gold 全尾段对比，而假设 transport 为零的锚点预测在 36 个状态上失准。

**动机.** 在神经网络学习器中，训练会到达新的状态。Theorem 8.2(b) 的局部界（连同 (e) 的特征漂移）在这些状态上需要 transport 项。这里测量它的大小，并检验带实测 transport 与特征漂移的局部界能否包住实际对比。

**设置.**

- 网络：one-hot 上下文，4 个 tanh 单元，sigmoid 输出，平均交叉熵，float64，用 Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）训练（即各实验页通用的受控网络）。
- 锚点：三个实际参考状态，分别在 0、2、5 次更新之后，被 audit 的差异为 $`D_x=0.200, 0.169, 0.188`$。
- 到达状态：40 个新状态，位于第 1 到第 13 次更新之间。每个状态上的消息都是同样的四个标量标签。
- provider 规则 $`Q_z=(1-\lambda)Q_3+\lambda\delta_1`$，$`\lambda=0.1+0.6\,p(\theta_z)`$，所以候选律依赖于状态。
- 全尾段：6 次实际更新，精确枚举。种子 `SeedSequence([150150, 104])`。
- 在每个到达状态上：
  1. 测量到其锚点的 transport $`a_{t,j}(z)`$ 与特征漂移 $`\alpha_{zx}`$；
  2. 拟合 response 分解，记录 $`B`$ 与 $`\tau`$；
  3. 计算 gold 全尾段对比；
  4. 与 (a) 假设 transport 为零的锚点预测，以及 (b) 局部界 $`B(D_x+a+2\alpha)+2\tau`$ 比较。

**结果**（组 `v015:E4`，`data`；分位数依次为最小 / 中位数 / 90% / 最大）：

| 量 | 值 |
|---|---|
| 40 个状态上的 transport $`a_{t,j}(z)`$ | 最小 0.0014，中位数 0.0195，最大 0.0727（`observations[*].a`） |
| 特征漂移 $`\alpha_{zx}`$ | 最小 0.021，中位数 0.139，最大 0.283（`observations[*].alpha`） |
| gold 对比落在 $`B(D_x+a+2\alpha)+2\tau`$ 之内的状态 | **40 / 40**（断言） |
| 锚点预测误差 | 0.00048 / 0.0063 / 0.0328 / 0.0681（`absolute_error_quantiles`） |
| 误差 $`>10^{-3}`$ 的锚点预测 | 36 / 40（`failures`） |
| response 残差 $`\lvert r\rvert`$ | 0.0003 / 0.047 / 0.146 / 0.540（`response_residual_quantiles`） |

**为什么支持该陈述.**

- 在实际到达的状态上 transport 不为零：中位数 0.02，而被 audit 的差异约为 0.2。证书必须测量它。
- 测量之后，局部界处处包住真实对比；而用零 transport 外推锚点 audit，在 90% 的状态上是错的。
- 局部界的中位数为 0.34，gold 对比的中位数为 0.023。这个界成立但偏松；在神经网络学习器里，有用的信息在于测出 $`a`$，而不在于它的最坏情形大小。

<a id="c8-8"></a>

## Check 8.8 · 理论包检验 `pkg:T01` 与 `pkg:T02`（Theorem 8.2(e)、(f)）

**结论.** 一个独立实现在 1,200 个实例上确认了未覆盖状态的见证，以及带漂移修正的局部界。

**动机.** 这些是修订包里的独立实现。

**结果**（`docs/direction/v015_revision/results_v015.json`，`checks`）：

| 检验 | 结果 |
|---|---|
| `pkg:T01`（未覆盖状态的见证） | PASS。差距 $`1.0`$；transport 界（截断）$`1`$；退出界 $`1`$；删掉 transport 项后为 $`0.0296`$ |
| `pkg:T02`（局部 transport 与特征漂移，Theorem 8.2(b)、(e)） | 在 1,200 个实例上 PASS；局部 差距/界 的最大值为 $`0.503`$ |

**为什么支持该陈述.** 必要性见证 (f) 和局部界的特征漂移形式 (e)，由一份与 [Check 8.4](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md#c8-4)–[8.7](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md#c8-7) 不共享任何代码的实现复现。

![Figure 8.2：带 transport 的冻结 audit](../figures/fig8_2_transport.png)

*Figure 8.2.* 2,000 个随机有限训练过程上的精确差距对考虑 transport 的界（[Check 8.4](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md#c8-4)）：没有一个点落在对角线上方。在受控网络中（[Check 8.7](../experiments/08-02_frozen_finite_interface_selection_transport_ZH.md#c8-7)），带实测 transport 的局部界在全部 40 个到达状态上包住 gold 全尾段对比，而零 transport 的锚点预测失准 36 个。最后一幅显示到达状态上的 transport 与特征漂移，以及未覆盖状态的见证：删掉 transport 项得到的 0.0296 不成立，实际差距为 1。[PDF](../figures/fig8_2_transport.pdf)

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C15,C20 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V1,E4 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T01, T02
```

这些脚本使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产出文件

- `theory/independent_checks/results_independent_v014.json`：条目 `v014:C15`、`v014:C20`。
- `theory/v015_checks/results_v015_independent.json`：组 `v015:V1`、`v015:E4`；规范 `theory/v015_checks/SPEC.md`。
- `docs/direction/v015_revision/results_v015.json`：`checks.T01_*`、`checks.T02_*`。
- 图：`../figures/fig8_2_transport`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
