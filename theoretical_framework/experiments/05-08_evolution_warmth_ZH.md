# Proposition 5.8 · warmth 的演化

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-08_evolution_warmth_ZH.md) · [证明](../proofs/05-08_evolution_warmth_ZH.md) · [实验](../experiments/05-08_evolution_warmth_ZH.md) · [English](05-08_evolution_warmth.md)

**本页检验什么.** [Check 5.11](../experiments/05-08_evolution_warmth_ZH.md#c5-11) 用精确递推与 Monte Carlo 检验精确规律、两侧界、平台和累积界。[Check 5.12](../experiments/05-08_evolution_warmth_ZH.md#c5-12) 在实际训练轨迹上测量 warmth，这些轨迹只覆盖早期阶段。[Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) 追踪 cold 窗口在整个训练过程中的影响。

**结论.**

- **Proposition 5.8（warmth 的演化）.**
  - $`s`$ 直到 $`10^6`$ 时，精确递推与闭式之差不超过 $`7.1\times10^{-13}`$。
  - 两侧界在所有采样点上都成立，上界最多比 $`\bar\kappa(s)`$ 大 1.33 倍。warmth 会到达平台 $`(1-\beta_2)/\beta_2`$，并且不再下降。
  - 在平稳的 Gaussian、重尾和稀疏梯度下，75 个 Monte Carlo 区间全部包含 $`\sigma^2(1-\beta_2^{\,s})`$。累积界在 9,000 个格点上成立。
  - 真实轨迹只覆盖早期阶段：那里 warmth 中位数的 log–log 斜率为 $`-1.06`$，lr = 0 的 warm-up 与真实步的作用相同。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

**Adam 与 AdamW.** Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

*Warm 状态的记号（[Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate_ZH.md#s5-4)）.*

考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

**状态标签.**

- `PASS` 表示实现的一致性断言全部通过。
- `INFO` 表示经验诊断：每个单元格单独标 PASS、FAIL 或 UNAVAILABLE，整组不作通过与否的判断。

除非另有说明，网络都是 `v014:C23` 的受控小网络：one-hot 上下文、4 个 tanh 单元、sigmoid 输出、平均 cross-entropy、float64。这些都不是真实模型。

**被检验的陈述（[Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)）.** 设二阶矩从 $`v_0=0`$ 开始，按 $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$ 逐坐标吸收梯度计算 $`G_1,G_2,\dots`$。记 $`\gamma=1-\beta_2`$。

**(a) 平稳二阶矩：精确的 mean-field warmth.** 设对所有 $`k`$ 有 $`\mathbb E\,G_{k,i}^2=\sigma_i^2>0`$。定义 mean-field warmth $`\bar\kappa_i(s)=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E\,v_{s,i})`$。则

```math
\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2^{\,s}),\qquad \bar\kappa_i(s)=\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})},\qquad \frac{\max\lbrace 1/s,\;1-\beta_2\rbrace}{\beta_2}\le\bar\kappa_i(s)\le\frac{1/s+(1-\beta_2)}{\beta_2}.
```

- $`\bar\kappa_i(s)`$ 关于 $`s`$ 递减，收敛到平台 $`\kappa_\infty=(1-\beta_2)/\beta_2>0`$，不会趋于零。
- 早期它的行为像 $`1/(\beta_2s)`$：确切地说，$`1\le\beta_2s\,\bar\kappa_i(s)\le1+s(1-\beta_2)`$，所以近似 $`\bar\kappa_i(s)\approx1/(\beta_2s)`$ 的相对误差至多为 $`s(1-\beta_2)`$，只有在 $`s(1-\beta_2)\ll1`$ 时才准确。
- 若梯度恒为常数 $`G_k=g`$，上式对 warmth $`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 本身成立，不只对它的 mean-field 版本成立。

**(b) bias correction 不改变 warmth.** 用 bias-corrected 分母 $`d_i=\sqrt{v_i^+/(1-\beta_2^{\,t})}`$ 与 $`a_i=\sqrt{\beta_2v_i/(1-\beta_2^{\,t})}`$ 时，比值 $`d_i/a_i=\sqrt{1+\kappa_i}`$ 与不带 bias correction 时相同。所以 warmth 以及 Proposition 5.5(b) 的界都不受 bias correction 影响。

**(c) 沿训练过程的累积.** 经由 [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的 telescoping 恒等式，比较参考协议 $`0`$ 与替代协议 $`S`$ 在 $`W`$ 次 warm-up 计算之后的 $`R`$ 次真实更新 $`s=1,\dots,R`$。令 $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ 为这样的更新：在每个访问到的状态上，Proposition 5.6 的假设以下标为 $`s`$ 的常数成立，两个分布的平均 consumed gradient 相同，并且几乎必然地对某个常数 $`c>0`$ 有下式。

```math
\max_i\kappa_i\le c\Big(\frac1{W+s}+(1-\beta_2)\Big).
```

则

```math
\lvert L(0)-L(S)\rvert\le\big\lvert\lbrace s\le R:s\notin\mathcal S_{\rm warm}\rbrace\big\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
```

其中 $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$，$`\bar u_{i,s}`$ 是第 $`s`$ 次更新时 $`u_i`$ 的上界，$`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$。当 $`W\ge1`$ 时，

```math
\sum_{s=1}^{R}\min\Big\lbrace1,\;\frac{c}{W+s}+c(1-\beta_2)\Big\rbrace\le c\log\Big(1+\frac RW\Big)+c(1-\beta_2)R .
```

- 所以由 warmth 控制的部分每次更新都有一个量级为 $`c(1-\beta_2)`$ 的下限，不随训练变长而消失。只有当 $`(1-\beta_2)R\lesssim\log(1+R/W)`$ 时对数项才占主导。
- $`W=0`$ 时，第一次真实更新从 $`v=0`$ 开始，不满足 (W)：它是 Proposition 5.3 的 cold 步，记为至多 $`1`$；由 Proposition 5.7，它的贡献可以是 $`\eta`$ 的一阶量。若 $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$，由 warmth 控制的和至多为 $`c\log R+c(1-\beta_2)(R-1)`$。

**(d) Warm-up，以及 (c) 的假设何时成立.** 长度为 $`W`$ 的*矩 warm-up*，用声明数据的梯度做 $`W`$ 次学习率恰为 $`0`$ 的 optimizer 步。它不改变 $`\theta`$，并按与真实步相同的递推推进 $`m`$、$`v`$ 与计数器；被 $`v`$ 吸收的每个梯度，无论来自 warm-up 还是真实更新，都算一次计算。所以第 $`s`$ 次真实更新面对的二阶矩已吸收 $`W+s-1`$ 次计算。若把实际的 $`v`$ 换成其均值（mean-field 近似），并且梯度满足 $`G_{k,i}^2\le\bar\gamma\,\sigma_i^2`$，则当 $`W+s\ge2`$ 时 (c) 的假设以 $`c=2\bar\gamma/\beta_2`$ 成立。把 $`v`$ 换成均值是一个近似；它需要指数滑动平均的集中性和平稳的梯度尺度。

| 检验 | 检验对象 | 代码编号 | 结果 |
|---|---|---|---|
| [Check 5.11](../experiments/05-08_evolution_warmth_ZH.md#c5-11) | [Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)(a)–(c) | `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation` | PASS |
| [Check 5.12](../experiments/05-08_evolution_warmth_ZH.md#c5-12) | 真实轨迹上的 [Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)(a)（早期阶段）与 (d)（warm-up） | `v015:E8`（warmth 轨迹部分） | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth_ZH.md#c5-13) | 整段训练中的 [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)–[5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) | `v015:E8`（`data.gaps`） | PASS |

<a id="c5-11"></a>

## Check 5.11 · warmth 的精确规律：两侧界、平台与累积（Proposition 5.8(a)–(c)）

**结论.**

- **精确递推.** 梯度为常数时，二阶矩的原始递推与闭式 $`\bar\kappa(s)=(1-\beta_2)/(\beta_2(1-\beta_2^{\,s}))`$ 的相对误差不超过 $`7.1\times10^{-13}`$，覆盖五个 $`\beta_2`$、$`s`$ 直到 $`10^6`$。
- **两侧界与平台.** (a) 的两侧界在全部 10,900 个采样点上成立，上界最多比 $`\bar\kappa`$ 大 1.33 倍。warmth 到达平台 $`(1-\beta_2)/\beta_2`$ 后停在那里。
- **bias correction.** 它精确地不改变 warmth 比值，即 (b)。
- **平稳随机梯度.** 75 个 Monte Carlo 99.9% 区间全部包含 $`\sigma^2(1-\beta_2^{\,s})`$。
- **累积.** (c) 的界在全部 9,000 个格点上成立，其中的线性项不可缺少：一旦 $`s(1-\beta_2)`$ 达到 1 的量级，任何纯倒数形式的包络 $`c/(W+s)`$ 都不再成立。

**动机.**

- (a) 给出 mean-field warmth 的精确规律和两侧界。核心内容是一个非零的平台：只有在 $`s(1-\beta_2)\ll1`$ 时 warmth 才按 $`1/(\beta_2s)`$ 下降；$`\beta_2`$ 固定时，它不会趋于零。
- (c) 把这一点变成带线性项 $`c(1-\beta_2)R`$ 的累积界。
- 指数或常数只要错一处，就会在某个时刻出现违反的界，或出现落在区间之外的 Monte Carlo 均值。

**设置**（`theory/v016_checks`；CPU；固定的 `SeedSequence` 种子）.

- `v016:C1.exact`：
  - $`\beta_2\in\lbrace 0.9,0.99,0.995,0.999,0.9999\rbrace`$，常数梯度 1.7，原始 $`v_0=0`$；
  - 每个 $`\beta_2`$ 用因果 IIR 滤波执行一百万步更新；
  - 采样点为 $`s=1,\dots,2000`$ 的每一步，加上直到 $`10^6`$ 的 180 个对数等距点（每个 $`\beta_2`$ 共 2,180 个）；
  - 闭式用数值稳定的 `expm1` 计算；
  - bias correction 在 $`t=1,7,50`$ 处用精确有理数比较。
- `v016:C1.moments`：
  - 16,000 个独立副本，检查点 $`s\in\lbrace 1,10,100,1000,2000\rbrace`$；
  - 梯度：单位 Gaussian；缩放到单位二阶矩的 Student $`t_3`$；Bernoulli(0.05) × Gaussian$`/\sqrt{0.05}`$；
  - 区间：Gaussian 与稀疏分布用 Studentized 99.9% 区间。$`t_3`$ 梯度的平方方差无穷，因此把 $`g^2`$ 截断在 100，用有界变量的 Bernstein 区间，再在区间中心加上解析已知的尾部期望。
- `v016:C1.accumulation`：
  - $`W\in\lbrace 1,4,32\rbrace`$，$`c\in\lbrace 1,2,4\rbrace`$，$`R=1,\dots,100`$ 的每一步，加上直到 $`10^5`$ 的对数网格；
  - 直接对截断后的包络求和，它对所有容许的 warmth 序列都是上界；
  - 对 $`c\in\lbrace 1,1/\beta_2,2\rbrace`$，记录 $`\bar\kappa\le c/(W+s)`$ 首次不成立的时刻。
- 一个非平稳诊断（在假设之外，不作断言）：二阶矩按 $`e^{-s/400}`$ 衰减，$`\beta_2=0.999`$。

**结果**（`theory/v016_checks/results_v016.json`；claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`）.

| Claim id | 状态 | 实例 | 断言 | 最差残差 |
|---|---|---:|---:|---|
| `v016:C1.exact` | PASS | 6 | 32,715 | 递推相对误差 $`7.1\times10^{-13}`$（容差 $`10^{-10}`$）；下界最多违反 $`2.2\times10^{-16}`$，即舍入误差（容差 $`2\times10^{-13}`$）；bias correction 精确为 0 |
| `v016:C1.moments` | PASS | 75 | 75 | 没有目标值落在 99.9% 区间之外 |
| `v016:C1.accumulation` | PASS | 135 | 9,045 | lhs − rhs 最大为 $`-4.7\times10^{-4}`$ |

按 $`\beta_2`$ 列出的精确规律（`v016:C1.exact`）：

| $`\beta_2`$ | 平台 $`(1-\beta_2)/\beta_2`$ | 上界 / $`\bar\kappa`$ 的最大值 |
|---|---:|---:|
| 0.9 | 0.1111 | 1.324 |
| 0.99 | 0.01010 | 1.301 |
| 0.995 | 0.005025 | 1.300 |
| 0.999 | 0.001001 | 1.299 |
| 0.9999 | 0.0001000 | 1.298 |

上界在过渡区最松，位置在 $`s(1-\beta_2)\approx1.8`$（$`\beta_2=0.999`$ 时为 $`s=1791`$）。$`\beta_2\to1`$ 时这个比值趋于 $`\max_y(1+y)(1-e^{-y})/y\approx1.298`$。

$`\beta_2=0.999`$ 时的取值（来自原始数组）：

| $`s`$ | 1 | 10 | 100 | 1,000 | $`10^6`$ |
|---|---:|---:|---:|---:|---:|
| $`\bar\kappa(s)`$ | 1.001 | 0.1006 | 0.01051 | 0.001583 | 0.001001 |
| $`1/(\beta_2s)`$ | 1.001 | 0.1001 | 0.01001 | 0.001001 | $`1.0\times10^{-6}`$ |

在 $`s=10^6`$ 处，倒数规律小了一千倍，而 $`\bar\kappa`$ 就等于平台值。

平稳随机梯度（`v016:C1.moments`；每个分布族 25 个单元格）：

| 分布族 | 区间方法 | 均值的最大相对偏差 | 落在 99.9% 区间内的目标 |
|---|---|---:|---:|
| Gaussian | Studentized | 0.76% | 25/25 |
| Student $`t_3`$（单位二阶矩） | 截断 $`g^2`$ 的 Bernstein 区间 + 解析尾部 | 5.6%（$`s=1`$，$`\beta_2=0.9`$） | 25/25 |
| 稀疏 Gaussian（$`p=0.05`$） | Studentized | 2.4% | 25/25 |

累积（`v016:C1.accumulation`）：

- 在每个格点上，界 $`c\log(1+R/W)+c(1-\beta_2)R`$ 都不小于包络之和。$`R=10^5`$ 时 lhs/rhs 在 0.965 与 0.999998 之间，界几乎是紧的。
- **线性项不可缺少.** 取倒数包络 $`\bar\kappa\le c/(W+s)`$，$`c=2\ge1/\beta_2`$，从实际的原始时刻起算：
  - 它首次不成立的原始时刻是 14（$`\beta_2=0.9`$）、158（0.99）、317（0.995）、1,592（0.999）、15,935（0.9999），即 $`s(1-\beta_2)\approx1.6`$ 附近；
  - $`c<2`$ 时，它在 burn-in 之后的第一步就不成立。

非平稳诊断（$`\beta_2=0.999`$，二阶矩为 $`e^{-s/400}`$）：

- 在 $`s=2991`$，精确的 $`\mathbb E v_s`$ 为 0.0331；把当前的二阶矩代入平稳公式，得到的是 0.00054。
- 也就是说，$`v`$ 跟不上梯度变小的速度，warmth 比平稳规律预言的小约 60 倍。
- 这在 (a) 的假设之外，不作断言。

![Figure 5.3：optimizer warmth](../figures/fig5_3_warmth.png)

*Figure 5.3.*

- (A) 在短视野（$`R=1,4`$）下，lr = 0 的 moment warm-up 消除了矩相同的两个来源之间的差距。$`R=40`$ 时它不再能把差距压到零；在残差最大的地方，SGD 对照也不为零。
- (B) 四个 $`\beta_2`$ 下的精确 mean-field warmth $`\bar\kappa(s)`$，带两侧界（阴影）与平台 $`(1-\beta_2)/\beta_2`$（虚线）。只有在早期阶段 $`s(1-\beta_2)\ll1`$，warmth 才按 $`1/(\beta_2s)`$ 下降；之后趋于平稳，不会消失。网络轨迹上实测的 warmth 中位数（[Check 5.12](../experiments/05-08_evolution_warmth_ZH.md#c5-12)）落在早期阶段。
- (C) warm Adam 仿射界的残差从不超过界 $`r`$，并逼近到 0.2% 以内。

[PDF](../figures/fig5_3_warmth.pdf)

**为什么支持 Proposition 5.8.**

- (a) 精确成立，包括平台。常数梯度是 mean-field 公式取等号的情形。
- 两侧界在过渡区相差不超过 1.33 倍，在两端的极限下都是精确的。
- 对重尾和稀疏的平稳梯度，mean-field 规律同样成立，说明它只依赖二阶矩。
- (c) 成立，其线性项 $`c(1-\beta_2)R`$ 不能去掉。因此在 $`\beta_2`$ 固定时，warm Adam 相对冻结分母描述，每一步都有 $`c(1-\beta_2)`$ 量级的非仿射下限。
- 证据的局限：梯度非平稳时，平稳公式不适用；梯度变小时，warmth 下降得比它预言的更快。

<a id="c5-12"></a>

## Check 5.12 · 真实轨迹上的 warmth：早期阶段，以及 warm-up 与真实步作用相同（Proposition 5.8(a)、(d)）

**结论.**

- 网络轨迹在 $`\beta_2=0.999`$ 下最多吸收 129 次梯度评估，即 $`s(1-\beta_2)\le0.13`$，所以只覆盖 Proposition 5.8(a) 的早期阶段。
- 在这一阶段，$`\eta=0.01`$、$`W=0`$ 时，9 条拟合的任务/来源序列上 warmth 中位数的 log–log 斜率中位数为 $`-1.06`$（$`R^2`$ 中位数 0.999）。这正是 (a) 在 $`s(1-\beta_2)\ll1`$ 时预言的 $`1/(\beta_2s)`$ 行为。
- warmth 只取决于吸收的梯度评估总数，与这些评估来自 lr = 0 的 warm-up 还是真实更新无关，正如 (d) 所述。

**动机.**

- [Check 5.11](../experiments/05-08_evolution_warmth_ZH.md#c5-11) 用合成梯度检验 mean-field 规律。这里要问的是：在实际 Adam 轨迹的逐坐标 warmth 上，能否看到早期阶段的行为？这些轨迹的梯度既不平稳，在各坐标间也不同分布。
- lr = 0 的 moment warm-up 是让状态变 warm 的低成本办法；只有当 $`v`$ 在两种情形下按同样方式推进时，它才能与真实步互换。

**设置**（`v015:E8`，warmth 轨迹部分）.

- 网络上的实际 Adam 轨迹：moment 与 parity 任务，$`\eta\in\lbrace 0.01,0.1\rbrace`$，128 个副本，$`\beta_2=0.999`$。
- warm-up：在参考分布上做 $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$ 步 lr = 0 的更新；断言每一步 warm-up 都不改变参数的字节。
- 记录：每一步逐坐标 warmth 的分位数。
- 拟合：对 $`s=1,\dots,79`$ 步之前的更新，拟合中位数 $`=c/(W+s)`$，log–log 斜率不固定。

**结果**（`v015:E8`，warmth 轨迹部分）.

| 设定 | 斜率中位数 | $`R^2`$ 中位数 |
|---|---|---|
| $`\eta=0.01`$，$`W=0`$ | $`-1.06`$ | 0.999 |
| $`\eta=0.01`$，$`W\in\lbrace 5,10,20,50\rbrace`$ | $`-1.07`$ 至 $`-1.14`$ | 0.95–0.99 |
| $`\eta=0.1`$，$`W=0`$ | $`-1.18`$ | 0.94 |
| $`\eta=0.1`$，$`W\in\lbrace 10,20,50\rbrace`$ | $`-1.55`$ 至 $`-1.93`$ | 0.55–0.68 |

可互换性（moment $`N=3`$，$`\eta=0.01`$，来源 $`P`$）：先前评估次数 $`W+s`$ 相同时的 $`\kappa`$ 中位数。

| $`W+s`$ | $`W=0`$ | $`W=5`$ | $`W=20`$ | $`W=50`$ | mean-field $`\bar\kappa(W+s)`$ |
|---|---|---|---|---|---|
| 51 | 0.00661 | 0.00642 | 0.00619 | 0.00619 | 0.0201 |
| 79 | 0.00405 | 0.00395 | 0.00413 | 0.00379 | 0.0132 |

**为什么支持 Proposition 5.8(a)、(d).**

- 在早期阶段，实测 warmth 中位数的斜率与 (a) 的预言一致。
- 中位数比 $`\bar\kappa`$ 低一个大致恒定的因子（$`W+s=51`$ 时为 0.33，79 时为 0.31）。这是预期之中的：$`\bar\kappa`$ 是均值之比，而平方梯度与其滑动平均之比的逐坐标中位数更小。
- $`\eta=0.1`$ 时，consumed gradient 随训练收敛而变小，warmth 下降得比倒数规律更快，与 [Check 5.11](../experiments/05-08_evolution_warmth_ZH.md#c5-11) 的非平稳诊断一致。
- 对 warmth 而言，warm-up 评估与真实评估可以互换，这就是 (d)。
- 证据的局限：
  - 轨迹太短，到不了平台 $`(1-\beta_2)/\beta_2=0.001`$，平台大约从 $`s\approx1/(1-\beta_2)=1000`$ 开始。平台由 [Check 5.11](../experiments/05-08_evolution_warmth_ZH.md#c5-11) 检验。
  - 分位数描述的是大部分坐标的情况。Proposition 5.6 与 5.8(c) 的假设针对的是最大值或敏感度加权和（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)），最大值由 [Check 5.9](../experiments/05-06_warm_adam_first_order_affine_ZH.md#c5-9) 核对。

<a id="c5-13"></a>

## Check 5.13 · 整段训练中的 cold 窗口（Propositions 5.6–5.8）

**结论.**

- 在短视野下，矩相同的两个来源之间的差距是 cold 窗口的效应。$`W=10`$ 的 lr = 0 moment warm-up 消除了这一差距的中位数 99%（$`R=1`$）与 97.5%（$`R=4`$）。moment $`N=3`$、$`\eta=0.1`$ 时，差距在 $`R=1`$ 从 0.0769 降到 0.0004，在 $`R=4`$ 从 0.1196 降到 0.0016。
- $`R=40`$ 时，$`W\ge10`$ 仍留下至多 0.049 的差距；在残差最大的地方，SGD 对照也不为零（至多 0.011）。

**动机.** Propositions 5.6–5.8 对整段训练有两个预言：

- 受 warmth 控制的那部分差距集中在最初几步，warm-up 能把它消除；
- 曲率部分不依赖 warmth，会随视野累积。

**设置**（`v015:E8`，`data.gaps`）.

- 不加改动地复用 [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient_ZH.md#c5-7) 的分支（实际 Adam、冻结 Jacobian 的 Adam、切线仿射 SGD）。
- 网格：5 个任务（moment $`N=2,3,4`$；parity $`N=2,3`$）× $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$，共 210 个单元格。
- 不同 $`W`$ 与不同分支使用完全相同的历史。历史不超过 4,096 条时精确枚举，否则每个来源独立运行 2,000 次。
- warm-up 使用参考分布上的一段声明好的 IID 前缀。

**结果**（`v015:E8`，`data.gaps`）.

| 视野 | $`W=0`$ 时的最大差距 | $`W\ge10`$ 时的最大差距 | $`\lvert\text{gap}(W{=}10)\rvert/\lvert\text{gap}(W{=}0)\rvert`$ | SGD 对照的最大值 |
|---|---|---|---|---|
| $`R=1`$ | 0.0769 | 0.0006 | 中位数 0.009，最大 0.034（差距 $`>10^{-3}`$ 的 5 个单元格） | 0.0034（$`R\le4`$） |
| $`R=4`$ | 0.1196 | 0.0064 | 中位数 0.025，最大 0.226（7 个单元格） | （同上） |
| $`R=40`$ | 0.0436 | 0.0487 | 没有压到零 | 0.0110 |

| 单元格 | $`W=0`$ | $`W=1`$ | $`W=2`$ | $`W=5`$ | $`W=10`$ | $`W=20`$ | $`W=50`$ | 切线 SGD |
|---|---|---|---|---|---|---|---|---|
| moment $`N=3`$，$`\eta=0.1`$，$`R=1`$ | 0.0769 | 0.0460 | 0.0311 | 0.0032 | 0.0004 | 0.0004 | 0.0002 | 0.0000 |
| moment $`N=3`$，$`\eta=0.1`$，$`R=4`$ | 0.1196 | 0.1016 | 0.0881 | 0.0206 | 0.0016 | 0.0037 | 0.0033 | 0.0000 |
| parity $`N=2`$，$`\eta=0.1`$，$`R=40`$ | 0.0436 | 0.0477 | 0.0441 | 0.0488 | 0.0487 | 0.0391 | 0.0391 | 0.0110 |

- $`W\in\lbrace 1,2\rbrace`$ 时 warmth 仍是 1 的量级，Proposition 5.6 给不出控制，差距可以朝任一方向变化。例如 moment $`N=2`$、$`\eta=0.1`$、$`R=1`$，在 $`W=1`$ 时从 0.0175 变为 $`-0.0194`$。
- $`W\ge10`$ 时，短视野下的差距一致地很小。

**为什么支持 Propositions 5.6–5.8.**

- 在短视野下，矩相同的两个来源之间的差距就是 Proposition 5.7 中的 cold 步分离；把状态预热就能消除它，与 Propositions 5.6、5.8 的预言一致。
- 在长视野下，warm-up 不再能把差距压到零。最大的残差出现在 $`\eta=0.1`$ 的 parity $`N=2`$ 与 moment $`N=2`$，对每个 $`W\ge10`$ 分别停在 0.039–0.049 与 0.005–0.036；这些单元格里 SGD 对照也不为零（0.011 与 0.0067）。
  - 这部分残差是在 40 步更新上累积的曲率与轨迹项，SGD 也有，不是 cold-start 效应。
- 在最初几步里，warmth 决定了反馈来源必须匹配哪个统计量；视野变长之后，剩下的是轨迹本身的非线性。
- warm-up 是否改变低成本 proxy 的预测价值，是方法层面的问题，见 [§9 实验](09-03_derivation_design_rules_ZH.md#c9-5)。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

这些检查使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。`v015:V2` 与 `v015:V8` 中的线性规划由 `scipy.optimize.linprog` 调用 HiGHS solver（[Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)）求解。

## 产物

- `theory/v015_checks/results_v015_independent.json`：
  - `v015:V8`，以及 `v015:E10`（逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，cold Adam 单元格在 `data.C23_prediction`，设计在 `data.design`）；
  - `v015:V2`、`v015:V3`、`v015:V4`、`v015:E0`、`v015:E1`、`v015:E2`、`v015:E3`；
  - `v015:E8`：冻结分母、warm 仿射、cold 分离与 warmth 轨迹四个部分（最后一部分含 `per_step` 与 `fits`），以及 210 个 `data.gaps` 单元格；
  - `v015:E11`（`data.clipped_mean`、`data.always_active`、`data.warm_Adam`、`data.Lion`、`data.LoRA`）。
- `theory/v015_checks/c23_bundle.json`：精确分布、网络、optimizer 代码，以及每条历史的原始更新。
- `theory/v015_checks/raw/E8_trajectories.npz`：逐副本的 warmth 轨迹。它被 gitignore，由上面第一条命令重新生成；SHA-256 记录在 `v015:E8` warmth 轨迹部分的 `raw_sha256` 中。
- `theory/v016_checks/results_v016.json`：claim 为 `v016:C1.exact`、`v016:C1.moments`、`v016:C1.accumulation`。完整数组在 `theory/v016_checks/results_v016_raw.json.gz` 中，该文件被 gitignore，SHA-256 记录在 `raw_artifact`。
- `theory/independent_checks/results_independent_v014.json`：`controlled_neural`。
- `docs/direction/v015_revision/results_v015.json`：package checks `pkg:T03`–`pkg:T08`。
- 代码：
  - `theory/v015_checks/otlr.py`（`check_V8`、`check_E10`、`check_E11`）；
  - `theory/v015_checks/warmth.py`（`check_E8`）；
  - `theory/v016_checks/check_v016.py`（`v016:C1`）；
  - 说明在 `theory/v015_checks/README.md` 与 `theory/v016_checks/SPEC.md`。
- 图：`../figures/fig5_1_linear_response`、`fig5_2_cold_adam`、`fig5_3_warmth`、`fig5_4_learner_statistics`，各有 `.pdf` 与 `.png`；说明在 `../figures/specs/fig5_1.md` 至 `fig5_4.md`。
