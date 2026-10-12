# Corollary 8.9 · 一阶版本、它的 margin 判据与上限

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-09_first_order_version_margin_criterion_ZH.md) · [证明](../proofs/08-09_first_order_version_margin_criterion_ZH.md) · [实验](../experiments/08-09_first_order_version_margin_criterion_ZH.md) · [English](08-09_first_order_version_margin_criterion.md)

**本页检验什么.**

- **[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)（一阶版本及其上限）.**
  - 用 oracle 延续值梯度时，一阶分数在 margin 超过余项界的 56/396 行上没有判错，与 (a) 一致。
  - 实测的一阶证书 (b) 在 1,584 个光滑候选对上认证 0 对。仅二阶余项一项就在 69% 的候选对上超过实测一阶分数、在 73% 上超过真实差距；按上限 (c)，这些候选对在一阶意义下永远无法被认证。

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

**被检验的陈述（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)）.** 沿用 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 的设定，令 $`\mu^j=\mathbb E\Delta^j`$，$`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$。

- **(a) margin 判据.** 设 continuation $`V`$ 沿到达的线段二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$。若 $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$，则 $`(K^j_t-K^k_t)V(z)`$ 与 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ 同号。整次训练层面，令 $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ 与 $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ 为 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 中以 $`P_j`$ 作 $`P`$ 时的一阶项与余项界；这里 $`d^k_t`$ 是由 $`P_k`$ 驱动的过程在第 $`t`$ 轮的状态分布。若 $`\lvert T_{jk}\rvert>B_{jk}`$，则 $`L(P_j)-L(P_k)`$ 与 $`T_{jk}`$ 同号。
- **(b) 测得的一阶证书.** 设 $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ 对到达的增量 $`u`$ 沿 $`\lbrace su:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$；令 $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$。对步长 $`h\in(0,1]`$，定义 $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$，其平均为 $`\hat S^{jk}(h)`$；设 (I)、(A) 成立，(B) 以区间长度 $`B_h`$ 成立。记 $`e^{jk}_{\rm stat}(h)`$ 为把 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的半径中 $`B`$ 换成 $`B_h`$ 得到的量，并令

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

则以至少 $`1-\delta`$ 的概率，对所有有序对同时有

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) 上限.** 在 (b) 的事件上，一个候选对能被 (b) 认证，仅当 $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$。一阶项不超过二阶余项界的候选对，无论真实差距多大，在一阶意义下都不会被认证。特别地，若两个候选的平均增量沿 $`\nabla\tilde V(0)`$ 方向一致，它们只在增量的二阶及更高阶矩上不同，这样的候选对永远不会被一阶证书认证；[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 则可以认证它们。按总体量来说，(b) 认证的是下界 $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$，而右端正是 Theorem 8.8 直接估计的量。
- **(d) oracle 分数不是证书.** 用任务的精确 continuation 梯度算出的分数 $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$（例如受控实验中对整个参照尾段求导得到的），在 margin 成立的地方满足 (a)。但它不是可测量的证书：$`\nabla V(z)`$ 与 $`M_2`$ 不能从反馈样本中得到，而检查 margin 两者都需要。一个便宜的分数要按 (a) 排对，需要：(i) 分数之差等于用实际学习器的 optimizer 变换后平均增量（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）在实际状态上算出的一阶项；(ii) margin 成立。用原始平均梯度构造的分数，或在不同状态与任务之间共用系数的分数，违反 (i)；在 margin 不成立处使用的任何分数，违反 (ii)。

<a id="c8-15"></a>

## Check 8.15 · 用 oracle 延续值梯度时的一阶 margin（Corollary 8.9）

**结论.** 在 396 个光滑行中一阶 margin 超过余项界的 56 行上，用 **oracle** 延续值梯度计算的一阶分数没有一次判错符号；在同样这些行上，每个静态 proxy 都有 35.7% 判错。在 margin 之外，oracle 分数在 gold 非零的行上判错 34.1%。

**动机.** [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)(a) 说，只要一阶 margin 超过二阶余项，一阶分数就能正确排出两个反馈律。Corollary 8.9(d) 补充：一个廉价分数只有满足 (i) 它把实际学习器经 optimizer 变换后的平均增量与实际状态上的延续值梯度配对，并且 (ii) 只在 margin 成立的地方使用，才能排对。这里在冻结 held-out 面板的同一批行上检验这两个条件，不重新拟合任何东西。它用的是每一行参考尾段的精确延续值梯度，真实的 audit 拿不到这个量；[Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14) 与 [8.16](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-16) 是对应的可测量版本。

**设置.**

- [§9 实验](../statements/09-03_derivation_design_rules_ZH.md)中的 held-out 面板与冻结的 proxy 模型，原样读入；前后 SHA-256 哈希一致（`data.frozen_sha256_before`、`_after`）。
- 每一行是某个学习器状态上的一次候选更新，之后是声明的七步参考更新，并有精确的全尾段 gold 对比。
- oracle 分数 $`\langle\nabla V^0(z),\mu^{\rm cand}-\mu^{\rm ref}\rangle`$，$`\nabla V^0`$ 是该行参考尾段的精确延续值梯度。
- 余项界用抽样得到、未加放大的分段曲率，所以"在 margin 内"是经验上的分类，不是被认证的。
- threshold 任务的 36 行回报不连续，违反光滑性假设，不在适用范围内。
- 种子 `SeedSequence([150150, 112])`。
- 按 |分数| / 余项界 的比值分组，报告 oracle、每个冻结 proxy、实测短尾段与随机选择的判错率。预测为零不算判错，只有 gold 非零的行计入比率。

**结果：总体**（`theory/v015_checks/results_v015_independent.json`，组 `v015:E12`，`data.summary[split=overall]`）：

| 量 | 值 |
|---|---:|
| 行数 / 适用（光滑）行数 | 432 / 396 |
| margin $`>`$ 余项界 的行 | 56（适用行的 14.1%） |
| margin 内 oracle 的判错率 | **0** |
| margin 外 oracle 的判错率 | 0.341（296 个 gold 非零行中的 101 个） |
| 全部适用且 gold 非零的行上 oracle 的判错率 | 0.287 |
| margin 内的静态 proxy（输入矩、平均梯度、符号、$`\Delta\theta`$、带 $`m,v`$ 的 $`\Delta\theta`$） | 各 0.357 |
| margin 内实测 1 步 / 4 步短尾段 | 0 / 0 |
| margin 外实测 1 步 / 4 步短尾段 | 0.203 / 0.030 |
| margin 内随机选择 | 0.536 |
| 全尾段 gold | 0 |

**结果：按 |oracle 分数| / 余项界 分组**（`data.bins`）：

| 比值 | 行数 | oracle | 平均梯度 | 带 $`m,v`$ 的 $`\Delta\theta`$ | 1 步短尾段 | 4 步短尾段 | 随机 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $`[0,0.25)`$ | 316 | 0.371 | 0.371 | 0.485 | 0.221 | 0.033 | 0.533 |
| $`[0.25,0.5)`$ | 12 | 0 | 0.250 | 0.083 | 0 | 0 | 0.333 |
| $`[0.5,1)`$ | 12 | 0 | 0 | 0 | 0 | 0 | 0.667 |
| $`[1,2)`$ | 8 | 0 | 0 | 0 | 0 | 0 | 0.625 |
| $`[2,\infty)`$ | 48 | 0 | 0.417 | 0.417 | 0 | 0 | 0.521 |

**结果：按 split**（`data.summary`）：

| split | 适用行 | margin 行 | oracle 判错率，margin 内 / 总体 |
|---|---:|---:|---|
| held-out 状态 | 72 | 16 | 0 / 0.234 |
| held-out $`\epsilon`$ | 72 | 16 | 0 / 0.281 |
| held-out 学习率 | 72 | 0 | — / 0.313 |
| held-out $`\epsilon`$ 与学习率 | 72 | 0 | — / 0.313 |
| fresh → warm | 72 | 16 | 0 / 0.234 |
| warm → fresh | 36 | 8 | 0 / 0.406 |
| held-out threshold 任务 | 0（36 行不适用） | — | — |

**为什么支持该陈述.**

- **margin 判据成立.** 只要一阶 margin 超过余项界，oracle 分数就从不排错，在含 margin 行的四个 split 中都是如此。比值超过 0.25 后它就再没错过，说明经验余项界是保守的。
- **在 margin 内，静态 proxy 失败是因为缺少延续值梯度.** 在比值 ≥ 2、由一阶决定的 48 行上，静态 proxy 判错 41.7%，oracle 为 0%。它们用的是同样的更新统计量，却没有依赖状态与任务的延续值梯度，因而违反 Corollary 8.9(d) 的条件 (i)。
- **在 margin 外，由尾段决定.** 在比值低于 0.25 的 316 行上，即使 oracle 也判错 37.1%，而实测四步尾段只判错 3.3%；那里条件 (ii) 不成立。
- **margin 很少见.** 只有 14% 的适用行满足它；held-out 学习率的 split 里一行也没有，那里的步长是 pilot 配方的十倍（Adam $`\eta=0.1`$，SGD $`\eta=1`$；`theory/v015_checks/proxy_split_freeze.json`，`recipe_shift_axes`）。这与 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的余项标度一致。
- **证据的边界.** 分数用的是 oracle 梯度，所以这一检验展示的是一阶分数最多能做到什么；如 Corollary 8.9(d) 所说，它不是可测量的证书。

![Figure 8.4：一阶测量与一阶 margin](../figures/fig8_4_first_order_audit.png)

*Figure 8.4.* 左：从 1 到 100,000 的每个维数上，平均增量的经验偏差都低于界的 0.62（[Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md#c8-13)）。中：在冻结的 held-out 面板上，用 **oracle** 延续值梯度计算的一阶分数，在 margin 超过余项界的四分之一后不再判错，而静态 proxy 在大 margin 处仍有 42% 判错（[Check 8.15](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-15)）。右：margin 内由一阶决定；margin 外只有实测尾段可靠。受控小网络。[PDF](../figures/fig8_4_first_order_audit.pdf)

<a id="c8-16"></a>

## Check 8.16 · 实测的一阶证书及其上限（Corollary 8.9）

**结论.** 不用任何 oracle 梯度测量时，Corollary 8.9(b) 的一阶证书在 1,584 个光滑候选对上认证 0 对，对每个步长 $`h`$、两个 surrogate 都是如此；因此它也没有认证错误，但这是空洞的。原因正是 Corollary 8.9(c) 与 Proposition 8.10(b) 预言的两个：

- 仅二阶余项 $`e_2`$ 一项，就在 69% 的候选对上超过实测一阶分数（在 73% 上超过真实差距）；
- 只有八个组时，统计半径约为典型差距的六倍。

测量的代价还是全尾段配对 audit 的 8.3 倍，其中 95% 花在曲率搜索上。

**动机.** Corollary 8.9(b) 说，一阶证书必须在统计半径之外再加上有限差分项 $`e_{\rm fd}`$ 与二阶余项 $`e_2`$。按上限 (c)，一个候选对只有在其一阶项超过 $`e_2+\varepsilon_{\rm sur}`$ 时才能被认证，无论真实差距多大。在与 [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14) 相同的面板上测出每一项，就能看到这个上限给面板留下了多少。

**设置.**

- 面板、gold 与校准 cell 同 [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14)。
- 每个 provider 只有一个或两个支撑点，所以平均增量可以精确积分；$`G=8`$ 个组，每组抽一条新的七步参考尾段，所有候选共用（CRN）。
- 组分数 $`Z_g=[\tilde F(T_\omega(z+h\mu_j))-\tilde F(T_\omega(z+h\mu_k))]/h`$，$`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$；每个候选对选任务单位半径最小的 $`h`$，选择时不看 gold。
- 半径为 $`e_{\rm stat}+e_{\rm fd}+\varepsilon_{\rm sur}+e_2/\alpha`$：
  - $`e_{\rm stat}`$ 用 empirical Bernstein，$`\log(4M^2/\delta)=8.5409`$，区间取八个组分数的观测区间（这个观测区间不是被证明的支撑界）；
  - $`e_{\rm fd}`$ 来自 CRN 延续沿射线实测的曲率（五点模板加 Richardson 外推）；
  - $`e_2=\tfrac12M_2(m_2^j+m_2^k)`$ 来自沿实际增量射线的实测曲率搜索。
- surrogate 同 [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14)。short surrogate 的校准：每个 $`h`$ 下都是 $`\alpha=0.902`$、$`\varepsilon_{\rm sur}=0.0078`$。
- 144 个 threshold 候选对不适用（不光滑）。

**结果：覆盖**（`theory/v016_checks/results_v016.json`，检验 `v016:C6`，`data[1..2]`）：

| surrogate | 适用候选对 | 认证数（每个 $`h`$） | 认证错误 | 漏掉的大差距候选对 | 认证的参考行（共 396） |
|---|---:|---:|---:|---:|---:|
| aligned | 1,584 | 0 | 0 | 281 / 281 | 0 |
| short | 1,584 | 0 | 0 | 281 / 281 | 0 |

aligned surrogate 选中的 $`h`$：1,501 对为 $`10^{-3}`$，83 对为 $`10^{-2}`$。

**结果：半径各项**（aligned surrogate、选中的 $`h`$，1,584 个适用候选对上的中位数，由 raw companion 计算）：

| 量 | 中位数 | 说明 |
|---|---:|---|
| $`\lvert{\rm gold}\rvert`$ | 0.0035 | 候选对的真实差距 |
| $`\lvert\text{一阶分数}\rvert`$ | 0.0035 | |
| $`e_{\rm stat}`$ | 0.0195 | 在 94.8% 的候选对上超过分数 |
| 其中区间项 $`7B\log(4M^2/\delta)/(3(G-1))`$ | 0.0164 | $`G=8`$ 时 $`=2.85\times`$ 观测区间 $`B`$ |
| $`e_{\rm fd}`$ | $`1.5\times10^{-5}`$ | 但在 16.7% 的候选对上大于 1（曲率爆炸） |
| $`e_2`$ | 0.0045 | 在 68.9% 的候选对上超过实测分数，在 73.0% 上超过真实差距 |

**结果：代价**（检验 `v016:C6`，`data[0].costs`）：

| 项 | 轨迹数 |
|---|---:|
| held-out 分数与有限差分端点（每个 surrogate） | 43,200 |
| 余项曲率搜索 | 2,995,200 |
| 校准（short surrogate） | 18,880 |
| 实际执行的全部尾段轨迹 | 3,151,680 |
| 曲率搜索所占比例 | 95.0%（按尾段更新步数为 96.4%） |
| 测量 / 全尾段配对 audit（365,760） | 8.31（aligned），8.36（short） |

**为什么支持该陈述.**

- **上限起作用.** 按 Corollary 8.9(c)，只有一阶项超过 $`e_2`$ 的候选对才可能在一阶意义下被认证。实测分数超过 $`e_2`$ 的只有 31.1%，超过 $`e_2+e_{\rm fd}`$ 的只有 28.0%；其余的候选对在做任何统计之前就已被排除，副本再多也没用。
- **剩下的候选对败在统计项上.** $`G=8`$ 时 empirical Bernstein 的区间项是观测区间的 2.85 倍，$`e_{\rm stat}`$ 在 95% 的候选对上超过分数。这正是 Proposition 8.10(b) 的小样本区间：$`M=8`$、$`\delta=0.05`$ 时，要让区间项低于 $`0.2B`$ 需要 $`r\ge101`$ 个副本，而不是 8 个组。
- **曲率项既贵又不稳定.** 曲率搜索占尾段成本的 95%，并在六分之一的候选对上给出极大的 $`e_{\rm fd}`$。
- **两个失败原因互相独立.** Theorem 8.8 完全去掉了二阶的原因；[Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14) 表明，剩下的统计原因归结为区间 $`B`$ 的选取。
- **没有错误的证书**，但覆盖率为零，所以这一点是空洞的；PASS 说明的是实现正确，而不是有用。

![Figure 8.5：受控面板上的可测量证书](../figures/fig8_5_certificate.png)

*Figure 8.5.* 一阶证书（[Check 8.16](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-16)）：半径各项 $`e_{\rm stat}`$、$`e_{\rm fd}`$、$`e_2`$ 的中位数对真实差距的中位数；仅余项一项就与差距同量级，所以什么都认证不了。有限 CRN 配对替换（[Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14)）：先验区间 $`B=2`$（严格）与观测区间（仅诊断）下的认证比例及认证错误，两种 surrogate、$`r\in\lbrace64,256\rbrace`$。其余各幅是 CRN 方差比的分布，以及相对全尾段配对 audit 的轨迹成本。受控小网络。[PDF](../figures/fig8_5_certificate.pdf)

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E12 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C6 --output theory/v016_checks/subset.json
```

## 产出文件

- `theory/v015_checks/results_v015_independent.json`：组 `v015:E12`；冻结的面板输入 `theory/v015_checks/proxy_split_freeze.json` 与 `proxy_models_frozen.json`。
- `theory/v016_checks/results_v016.json`：检验 `v016:C6`；raw companion `results_v016_raw.json.gz`；代码 `c5c6.py`。
- 图：`../figures/fig8_4_first_order_audit`、`fig8_5_certificate`。
