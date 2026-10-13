# Corollary 7.2 · 平稳情形的幅度定律

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-02_stationary_amplitude_law_ZH.md) · [证明](../proofs/07-02_stationary_amplitude_law_ZH.md) · [实验](07-02_stationary_amplitude_law_ZH.md) · [English](07-02_stationary_amplitude_law.md)

**本页检验什么.** 推论的三个部分，在带随机梯度的精确一维 Adam 上，与用 common random numbers 测得的有限 response 对比。

- **(a) 精确的 mean-field response.** 在全部 213 个网格点上，公式与实测有限 response 之差不超过 response 尺度的 2.7%（**通过**）。26 组有变负点的设定中，25 组预测变负点与实测相差不超过 3.9%；其余 7 组两者都不变负。紧挨变负判据的一组设定（$`\beta_2=0.99`$、$`\rho=0.245`$、$`T=50`$）没有达到 10% 的容差：预测 68.6，实测 $`77.0\pm4.0`$，相对误差 10.9%（**FAIL**，如实报告）。
- **(b) 二次律.** 在适用范围内 response 误差不超过 1.5%，$`2\lambda^\star`$ 与实测变负点相差不超过 0.6%（**通过**）。超出适用范围时，$`2\lambda^\star`$ 最多把变负点低估 5.7 倍，与推论中的警告一致。
- **(c) 大幅度判据.** 精确极限的符号在 33/33 组设定上正确区分了变负与不变负（**通过**）。

**本页用到的设定与记号.**

- 不做 bias correction 的 Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）的单个坐标，$`\epsilon=0`$，矩 $`m,v`$，$`\beta_1,\beta_2`$，学习率 $`\eta`$；最终得分 $`F`$ 关于参数仿射。
- 尾段梯度独立同分布，均值 $`\bar g`$，平方的均值 $`\bar v`$；信噪比 $`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$；$`\alpha=-(\nabla F)\,\mathrm{sign}(\bar g)`$ 在沿 $`-\bar g`$ 移动能提高得分时为正。
- 在 warm 状态注入消息 $`g=\lambda\sqrt{v}`$（幅度 $`\lambda`$ 以 running RMS 为单位），之后走 $`T`$ 次尾段更新。**实测有限 response** 是相对零梯度消息的 $`F`$ 变化在各副本上的平均；**公式**是 (a) 的 mean-field response $`\Delta F(\lambda)`$。
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$，$`c_j=(1-\beta_2)\beta_2^{\,j}`$，$`\omega_T=1-\beta_2^{\,T+1}`$，$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$，$`\kappa_{\rm iso}(T)=\sum_jb_j/\sqrt{c_j}`$，$`\Lambda(T)=\sum_j(1-b_j)/\sqrt{1-c_j}`$。
- $`\mathrm{Val}`$ 是 [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的一阶价值，用随机尾段的精确 costate 计算。

**被检验的陈述（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)）.** 在单个坐标的 warm、平稳、线性 response 区间里（尾段梯度与 $`\theta`$ 无关，$`\epsilon=0`$，不做 bias correction，无 decay）：

(a) 对任意 $`\lambda`$，mean-field response 为

```math
\Delta F(\lambda)=\alpha\eta\sum_{j=0}^{T}\Big[\frac{\rho+b_j(\lambda-\rho)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho(1-b_j)}{\sqrt{1-c_j}}\Big];
```

(b) 增量小时，它在相对误差 $`11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$ 内服从凹二次函数 $`Q(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho\,\omega_T\lambda^2`$，最优幅度为 $`\lambda^\star`$，在 $`2\lambda^\star`$ 附近变号；阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用；

(c) 当 $`\lambda\to\infty`$ 时 $`\Delta F\to\alpha\eta[\kappa_{\rm iso}(T)-\rho\Lambda(T)]`$，所以对 $`\alpha>0`$：$`\rho>\kappa_{\rm iso}(T)/\Lambda(T)`$ 时很大的同向消息劣于零消息（此时 $`\Delta F`$ 有正零点），$`\rho<\kappa_{\rm iso}(T)/\Lambda(T)`$ 时则更好。

| 检验 | 检验对象 | 代码 ID | 结果 |
|---|---|---|---|
| [Check 7.2](#c7-2) | (a) 与 (c) | `v016:C4.amplitude` | FAIL（26 个变负点中 1 个） |
| [Check 7.3](#c7-3) | (b) | `v016:C4.amplitude` | 通过 |

[Check 7.2](#c7-2) 与 [Check 7.3](#c7-3) 共用一个代码 ID：其中唯一 FAIL 的断言是 [Check 7.2](#c7-2) 的变负点比较；[Check 7.3](#c7-3) 的断言全部通过。

<a id="c7-2"></a>

## Check 7.2 · 精确 mean-field response 与大幅度判据（Corollary 7.2(a),(c)）

**结论.**

- (a) 的公式在整个幅度网格上与实测有限 response 一致：最大误差为 response 尺度的 2.74%（容差 5%）。
- 有变负点的 26 组设定中，25 组预测变负点与实测相差不超过 3.9%。紧挨判据边界的一组相差 10.9%：**FAIL**。
- 精确大幅度极限的符号在 33/33 组设定上正确预测了变负与不变负。

**动机.**

- (a) 在 mean-field 替换 $`m_k/\sqrt{v_k}\to\mathbb Em_k/\sqrt{\mathbb Ev_k}`$ 下是精确的。本检验在真实的随机 Adam 轨迹上，逐个幅度测量这一替换的代价，包括二次律已经失效的大幅度。
- (c) 只用一个数 $`\kappa_{\rm iso}(T)-\rho\Lambda(T)`$，就预言很大的同向消息最终是否劣于零梯度消息。

**设置.**

- 精确的一维 Adam，不做 bias correction：$`\beta_1=0.9`$，$`\epsilon=0`$，$`\eta=\alpha=1`$。最终得分 $`F=-\theta`$，所以 response 关于 $`\theta`$ 恰好线性。
- 尾段梯度独立同分布于 $`\mathcal N(\rho,1-\rho^2)`$，于是 $`\bar g=\rho`$，$`\bar v=1`$。
- 从零开始 burn-in $`\lceil12/(1-\beta_2)\rceil`$ 步。消息 $`g=\lambda\sqrt{v}`$ 注入每个副本实际 warm 后的状态，不把带噪声的状态换成均值。
- 4,096 个副本。每条消息和零梯度对照共用该副本此后的梯度（common random numbers）。
- 设定：
  - **核心**，24 组设定：$`\beta_2\in\lbrace0.99,0.999\rbrace`$，$`\rho\in\lbrace0.3,0.7,0.95\rbrace`$，$`T\in\lbrace50,200,1000,5000\rbrace`$，幅度 $`\lambda\in\lbrace-2,0.5,1,2,4,8,16\rbrace`$。
  - **跨判据**，6 组事先固定的设定：$`\beta_2=0.99`$，$`T\in\lbrace20,50,100\rbrace`$，$`\rho`$ 取 $`\kappa_{\rm iso}(T)/(T+1)`$ 的 0.8 或 1.2 倍。它们落在精确临界比 $`\kappa_{\rm iso}(T)/\Lambda(T)`$ 的两侧。
  - **边界**，3 组设定：$`\beta_2=0.99999`$，$`\rho=0.05`$，$`\lambda=20`$，$`T\in\lbrace0,1,5\rbrace`$，128 个副本（burn-in 120 万步）。
- 变负点从 $`2\lambda^\star`$ 起倍增搜索、再用 Brent 法求根；实测 rollout 与公式分别求根，搜索上界相同。
  - 核心设定的根用前 512 个副本；跨判据与边界设定用全部路径。
  - 根的标准误由隐函数求导得到。
- 断言与容差在运行前固定：
  - 网格误差不超过该设定最大 response 的 5%；
  - 实测变负点被括住时，变负点误差不超过 10%；
  - 没被括住时，公式在相同上界内也不得变负；
  - 精确极限的符号必须与两者的变负状态一致；
  - 有限公式在 $`\lambda=10^8/\sqrt{c_T}`$ 处与极限之差不超过 $`10^{-5}`$。

**结果.**

| 量 | 设定数 | 最大值 | 容差 | 状态 |
|---|---:|---:|---:|---|
| (a) 的网格误差（占 response 尺度） | 33（213 个点） | 2.74% | 5% | 通过 |
| (a) 的变负点误差（实测变负点被括住） | 26 | 10.93% | 10% | FAIL（1 组） |
| 公式与实测在相同上界内都不变负 | 7 | — | — | 通过 |
| 有限公式与 (c) 的极限 | 33 | $`1.0\times10^{-7}`$ | $`10^{-5}`$ | 通过 |
| 极限符号预测实测是否变负 | 33 | 0 个错误 | 0 | 通过 |
| 极限符号预测公式是否变负 | 33 | 0 个错误 | 0 | 通过 |
| 极限符号与随机分支极限一致 | 33 | 0 个错误 | 0 | 通过 |

- **分组的网格误差.** 核心设定最多 1.69%；跨判据设定最多 2.74%；边界设定最多 $`2.5\times10^{-5}`$。
- **变负点.** 24 组核心设定中，最大的变负点误差为 3.85%（$`\beta_2=0.999`$、$`\rho=0.7`$、$`T=50`$：公式 312.6，实测 $`325.1\pm19.0`$）。其余核心设定的误差都在 3% 以下，多数在 0.5% 以下。
- **FAIL 的那一组.** 它是跨判据设定 $`\beta_2=0.99`$、$`T=50`$、$`\rho=0.245`$，即精确临界比 0.2074 的 1.18 倍。公式给出的变负点为 68.62，实测为 $`77.03\pm3.98`$（标准误）。10.9% 的相对误差相当于 2.1 个标准误。
  - 另外两组会变负的跨判据设定，误差分别为 3.2%（$`T=20`$）和 3.1%（$`T=100`$）。
  - 容差没有调整。
  - 一个可能的原因（未单独检验）：在判据边界附近，任何模型误差都会让变负点大幅移动，而 mean-field 替换忽略了 $`v`$ 围绕 $`\bar v`$ 的波动。$`\beta_2=0.99`$ 的平均窗口短，这种波动最大。
- **精确判据与近似判据.** $`\beta_2=0.99`$、$`T=50`$ 时，精确临界比 $`\kappa_{\rm iso}/\Lambda`$ 为 0.2074，近似值 $`\kappa_{\rm iso}/(T+1)`$ 为 0.2042。两者在 $`T=20`$ 时相差 4.0%，$`T=50`$ 时相差 1.6%，$`T\ge1000`$ 时相差不到 0.1%。
  - 7 组不变负的设定恰好是 $`\rho`$ 低于精确临界比的那些：$`\beta_2=0.999`$、$`\rho=0.3`$、$`T=50`$；三组低 $`\rho`$ 的跨判据设定；以及临界比超过 1 的三组边界设定。

![Figure 7.1：幅度定律，实测与公式](../figures/fig7_1_amplitude_law.png)

*Figure 7.1.* 几组有代表性的 $`(\beta_2,\rho,T)`$ 下，实测有限 response $`\Delta F(\lambda)`$（点与 95% CRN 区间）与 [Corollary 7.2(a)](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 的精确 mean-field 公式、(b) 的二次律的对比。二次律只在小幅度时跟得上 response，精确公式在所有幅度上都跟得上。第二个子图把所有有变负点的设定的预测变负点与实测变负点画在 $`y=x`$ 上，并标出 ±10% 带。唯一落在带外的设定（$`\beta_2=0.99`$、$`\rho=0.245`$、$`T=50`$）已标出。[PDF](../figures/fig7_1_amplitude_law.pdf)

**为什么支持该陈述.**

- 精确公式是关于大消息的唯一一个不依赖近似展开的陈述。在 response 已经掉头的幅度上，它仍然复现了实测 response。
- 那一组 FAIL 显示了 mean-field 替换的边界：在判据边界附近，变负点的位置对公式忽略的波动很敏感。即便在那里，(c) 对"是否变负"的*分类*仍然正确。

**局限.** 单个坐标、外生高斯梯度、$`\epsilon=0`$、不做 bias correction。这些正是推论的假设，所以本检验检验的是公式本身，而不是它能否迁移到网络上；迁移问题在 [Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5) 的实验页检验（[Check 7.5](../experiments/07-05_where_first_order_rankers_fail_ZH.md#c7-5)）。

<a id="c7-3"></a>

## Check 7.3 · 二次律及其变号阈值（Corollary 7.2(b)）

**结论.** 通过。在 (b) 的适用范围内，一阶价值与有限 response 偏离二次律分别不超过 0.31% 与 1.53%。在 $`(1-\beta_2)(2\lambda^\star)^2\le0.1`$ 的设定上，$`2\lambda^\star`$ 与两种实测变负点都相差不超过 0.6%。超出这一范围，$`2\lambda^\star`$ 最多错 5.7 倍，与 (b) 的警告一致。

**动机.**

- (b) 断言一个二次律，误差界随 $`(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$ 缩小；并且说阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用。
- 本检验在这条边界的两侧都做测量，对象有两个：一阶价值 $`\mathrm{Val}`$（随机尾段的精确 costate 加上精确的消息增量），以及有限 rollout。

**设置.** 与 [Check 7.2](#c7-2) 用同一批运行。

- 只在 $`(1-\beta_2)\lambda^2\le0.02`$ 且 $`\lvert\lambda\rvert\le4`$ 的点上断言二次律（相对容差 5%），共 96 个点。
- 只在 $`(1-\beta_2)(2\lambda^\star)^2\le0.1`$ 的设定上把变负点与 $`2\lambda^\star`$ 比较（容差 10%），共 10 组设定。
- 其余的点与设定只报告，不断言。

**结果.**

- 适用的点上：$`\mathrm{Val}`$ 的最大相对误差 0.31%，rollout 1.53%。
- 适用的设定上：实测 $`\mathrm{Val}`$ 变负点与 $`2\lambda^\star`$ 相差不超过 0.24%，实测 rollout 变负点不超过 0.54%。
- 超出适用范围。在 $`(1-\beta_2)(2\lambda^\star)^2`$ 为 0.5–18、且有变负点的 9 组设定上，实测 rollout 变负点与 $`2\lambda^\star`$ 之比如下：

| $`\beta_2`$ | $`\rho`$ | $`T`$ | $`(1-\beta_2)(2\lambda^\star)^2`$ | $`2\lambda^\star`$ | 实测变负点 | 比值 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.7 | 50 | 0.50 | 7.09 | 7.86 | 1.11 |
| 0.99 | 0.3 | 200 | 0.59 | 7.69 | 7.70 | 1.00 |
| 0.999 | 0.3 | 200 | 1.34 | 36.6 | 49.9 | 1.36 |
| 0.999 | 0.95 | 50 | 1.77 | 42.1 | 76.6 | 1.82 |
| 0.99 | 0.3 | 50 | 2.74 | 16.5 | 29.9 | 1.81 |
| 0.99 | 0.526 | 20 | 3.17 | 17.8 | 75.4 | 4.23 |
| 0.999 | 0.7 | 50 | 3.27 | 57.2 | 325.1 | 5.69 |
| 0.99 | 0.245 | 50 | 4.11 | 20.3 | 77.0 | 3.80 |
| 0.99 | 0.124 | 100 | 6.35 | 25.2 | 78.1 | 3.10 |

- **超出适用范围后，一阶价值仍跟着二次律走.** $`\beta_2=0.999`$、$`\rho=0.7`$、$`T=50`$ 时，实测 $`\mathrm{Val}`$ 的变负点是 55.0，接近 $`2\lambda^\star=57.2`$，而有限 response 要到 325.1 才变负。一阶价值早了五倍多就变负：大幅度时，线性化高估了二阶矩通道的伤害。
- **边界设定**（$`\beta_2=0.99999`$、$`\rho=0.05`$；$`(1-\beta_2)(2\lambda^\star)^2\sim10^6`$）。二次律变负点的误差为 28%（$`T=1`$）和 33%（$`T=5`$），只报告；公式与 rollout 都不变负。

**为什么支持该陈述.**

- 推论说二次律及其阈值在哪里成立，它们就在哪里成立，误差远小于证明中那个保守的常数。
- 推论说它们可能在哪里失效，它们就在哪里失效。失效的程度（最多 5.7 倍）正是推论把大幅度交给 (a) 和 (c) 处理的原因。
- 一阶价值提前变号，是 [Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5) 背后机制的直接观测。

**局限.** 与 [Check 7.2](#c7-2) 相同：推论自身的单坐标、外生梯度设定。

## 复现

在仓库根目录下运行（仅 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- 幅度这一组（`v016:C4.amplitude`）在 rai 上约 29 秒。
- 随机种子：`numpy.SeedSequence([160160, group, subcase, ...])`，PCG64，float64，单线程 BLAS。

本检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C4.amplitude`，含断言、唯一的失败记录与逐设定的表。
- `theory/v016_checks/results_v016_raw.json.gz`：原始数组，通过 JSON pointer 引用。未提交到 git；SHA-256 为 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`。
- `theory/v016_checks/SPEC.md`（`v016:C4` 一节）、`check_v016.py` 及其测试。
- `figures/fig7_1_amplitude_law.{png,pdf}` 及其在 `figures/specs/` 中的说明。
