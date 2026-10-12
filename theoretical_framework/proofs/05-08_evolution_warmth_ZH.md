# Proposition 5.8 · warmth 的演化

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-08_evolution_warmth_ZH.md) · [证明](../proofs/05-08_evolution_warmth_ZH.md) · [实验](../experiments/05-08_evolution_warmth_ZH.md) · [English](05-08_evolution_warmth.md)

**本页证明什么.** (a) 精确的 mean-field warmth 及其两侧界；(b) 在 bias correction 下不变；(c) 借助 telescoping 恒等式的累积界；(d) warm-up 的等价性以及 (c) 的充分条件。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

**Adam 与 AdamW.** Adam 带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

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

**陈述（[Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)）.** 设二阶矩从 $`v_0=0`$ 开始，按 $`v_s=\beta_2v_{s-1}+(1-\beta_2)G_s^{\odot2}`$ 逐坐标吸收梯度计算 $`G_1,G_2,\dots`$。记 $`\gamma=1-\beta_2`$。

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

*(a) 的证明.* 展开递推，$`v_{s,i}=(1-\beta_2)\sum_{r=1}^s\beta_2^{\,s-r}G_{r,i}^2`$，所以 $`\mathbb E\,v_{s,i}=\sigma_i^2(1-\beta_2)\sum_{j<s}\beta_2^{\,j}=\sigma_i^2(1-\beta_2^{\,s})`$。代入定义得 $`\bar\kappa_i(s)`$。关于界：

- 由 Bernoulli 不等式，$`1-(1-\gamma)^s\le s\gamma`$，所以 $`\bar\kappa\ge1/(\beta_2s)`$；又 $`1-(1-\gamma)^s\le1`$，所以 $`\bar\kappa\ge\gamma/\beta_2`$。
- 上界等价于 $`s\gamma\le(1+s\gamma)\big(1-(1-\gamma)^s\big)`$，即 $`(1-\gamma)^s(1+s\gamma)\le1`$。这由 $`(1-\gamma)^s\le e^{-s\gamma}`$ 以及对 $`y\ge0`$ 有 $`e^{-y}(1+y)\le1`$ 得到。
- 单调性是因为 $`1-\beta_2^{\,s}`$ 关于 $`s`$ 递增，极限是直接的。两侧界乘以 $`\beta_2s`$ 得 $`1\le\beta_2s\bar\kappa\le1+s\gamma`$。
- 梯度为常数时，$`v_{s,i}=g_i^2(1-\beta_2^{\,s})`$ 是确定的，$`\kappa_i=(1-\beta_2)g_i^2/(\beta_2v_{s,i})`$ 就等于该公式。$`\square`$

*(b) 的证明.* $`d_i^2/a_i^2=v_i^+/(\beta_2v_i)=1+\kappa_i`$：因子 $`1/(1-\beta_2^{\,t})`$ 同时除在分子与分母上。$`\square`$

*(c) 的证明.* [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 给出

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

并且在每个访问到的状态上，$`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$，是两个取值于 $`[0,1]`$ 的期望之差。

- 对 $`s\notin\mathcal S_{\rm warm}`$，其绝对值以 $`1`$ 为界。
- 对 $`s\in\mathcal S_{\rm warm}`$，均值相同时 Proposition 5.6 把它控制在 $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$ 以内。用 $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$，以及无论 $`\kappa\le1`$ 还是 $`\kappa>1`$ 都成立的 $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$。由 $`\kappa_{\max}`$ 的假设，$`r_s`$ 的前两项至多为 $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)+c(1-\beta_2)\rbrace`$，曲率项至多为 $`\tfrac12\bar B_s`$。对两个分布求和得到因子 $`2`$。

对 $`s`$ 求和即得第一式。对第二式：当 $`a,b\ge0`$ 时 $`\min\lbrace1,a+b\rbrace\le\min\lbrace1,a\rbrace+b`$，所以和至多为 $`\sum_{s=1}^R c/(W+s)+c(1-\beta_2)R`$；又 $`c/(W+s)\le\int_{s-1}^{s}c\,dz/(W+z)`$，所以 $`W\ge1`$ 时 $`\sum_{s=1}^Rc/(W+s)\le c\log(1+R/W)`$。$`W=0`$、$`s`$ 从 $`2`$ 开始时，同理 $`\sum_{s=2}^Rc/s\le c\log R`$。$`\square`$

*(d) 的证明.* 学习率为 $`0`$ 时参数更新为 $`\theta^+=(1-0\cdot\lambda_{\rm wd})\theta-0=\theta`$，而 $`m`$、$`v`$ 与计数器按与真实步相同的递推推进。所以经过 $`W`$ 次 warm-up 计算与 $`s-1`$ 次真实更新后，$`v`$ 已吸收 $`W+s-1`$ 次计算。在 mean-field 替换下，$`v_i=\sigma_i^2(1-\beta_2^{\,k})`$，$`k=W+s-1\ge1`$，所以由 (a)，$`\kappa_i=(G_i^2/\sigma_i^2)\,\bar\kappa_i(k)\le\bar\gamma\,(1/k+1-\beta_2)/\beta_2`$。当 $`W+s\ge2`$ 时 $`k\ge(W+s)/2`$，即 $`1/k\le2/(W+s)`$，因此 $`\kappa_i\le(2\bar\gamma/\beta_2)\big(1/(W+s)+(1-\beta_2)\big)`$。$`\square`$

**关于 Proposition 5.5–5.8 的讨论.**

- *切换.* 反馈来源必须匹配的统计量，随 optimizer state 而变。
  - cold Adam 状态下，要匹配 consumed gradient 的联合符号分布，以及保留在 $`m`$、$`v`$ 中的幅度（Proposition 5.3）。
  - warm Adam 状态下，精确统计量是 Adam 增量的均值；在 warmth 的一阶意义下，它是经过 preconditioner 的平均 consumed gradient，加上 $`v`$ 块中的 $`\mathbb E\,G^{\odot2}`$（Proposition 5.5–5.6）。
  - SGD 下，要匹配平均 consumed gradient，误差 $`O(\eta^2)`$（[Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7) 的局部 SGD 展开）。
- *warmth 的下限.* $`\beta_2`$ 固定时，典型梯度的 warmth 在此后的训练中一直停在 $`(1-\beta_2)/\beta_2`$ 附近（Proposition 5.8(a)）。所以 frozen-denominator 描述在每一步都只准确到这个相对阶，累积的非仿射贡献有一个关于 $`R`$ 线性的项（Proposition 5.8(c)）。
- *曲率项的阶.* 对光滑的尾段，$`F`$ 通过 $`\tilde\theta^+`$（被 $`\eta`$ 缩放）与 $`m^+`$（在之后的更新中被 $`\eta`$ 缩放）依赖 $`g`$。所以可以预期 $`H_G=O(\eta^2)`$。这是对尾段的预期，不是 (H) 的一部分。
- *warmth 消除不了的部分.* 曲率项 $`\tfrac12H_G\lVert G\rVert_\infty^2`$ 不依赖 $`\kappa`$，随 $`R`$ 线性累积。在长 horizon 上，均值相同的两个来源之间剩下的差别由曲率决定，而 SGD 同样有这一项。warm-up 只控制 cold 窗口。
- *消息与梯度.* Proposition 5.6 是关于 *consumed* gradient 的仿射性。当 $`G(y)`$ 本身关于 $`y`$ 非线性时，response 关于消息不是仿射的。例如序列级 teacher forcing（前缀会改变 Jacobian），以及处于激活状态的全局范数裁剪。
- *局部性.* Proposition 5.6 是单边界的结论，常数是 continuation value 的性质。要沿训练过程给出证书，还需要 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的 transport 与 coverage 项。
