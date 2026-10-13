# Corollary 7.2 · 平稳情形的幅度定律

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-02_stationary_amplitude_law_ZH.md) · [证明](../proofs/07-02_stationary_amplitude_law_ZH.md) · [实验](../experiments/07-02_stationary_amplitude_law_ZH.md) · [English](07-02_stationary_amplitude_law.md)

**一段话概括.** 在普通 SGD 下，有用方向上的梯度越大越好。Adam 下并非如此：大消息还会抬高 running 二阶矩 $`v`$，从而压低此后的步长。本推论在一切都能写成闭式的最简单区间里计算 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的状态通道分解：单个坐标、warm 的平稳 optimizer state、尾段梯度与参数无关。它给出任意幅度消息的精确 response、最优幅度、变号点，以及幅度很大时的判据。它与 Theorem 7.1 合称框架中的 **Theorem B**。

**本页用到的设定与记号.**

- *不做 bias correction 的 Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)），单个坐标 $`i`$.* 矩 $`m,v`$，$`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，学习率 $`\eta`$，$`\epsilon=0`$，无 weight decay。consumed gradient $`g`$ 使 $`m'=\beta_1m+(1-\beta_1)g`$、$`v'=\beta_2v+(1-\beta_2)g^2`$，参数移动 $`-\eta m'/\sqrt{v'}`$。
- *消息与尾段.* 第 $`t`$ 轮施加一条 consumed gradient 为 $`g_i`$ 的消息，之后有 $`T=R-1-t`$ 次尾段更新。最终得分为 $`F`$，$`(\nabla F)_i`$ 是它在坐标 $`i`$ 上的导数。
- *平稳线性 response 区间.* 尾段梯度 $`g_{k,i}`$ 与 $`\theta`$ 无关，平稳，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i>0`$；$`F`$ 关于 $`\theta_i`$ 仿射。
- *warm 状态.* 消息到达时 $`m_{t,i}=\bar g_i`$，$`v_{t,i}=\bar v_i`$。
- *归一化的量.*
  - $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$：沿 $`-\bar g_i`$ 方向移动对结果有利时 $`\alpha_i>0`$；
  - $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$：消息以 running RMS $`\sqrt{\bar v_i}`$ 为单位的同向幅度；
  - $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$：该坐标的信噪比；
  - $`b_j=(1-\beta_1)\beta_1^{\,j}`$，$`c_j=(1-\beta_2)\beta_2^{\,j}`$，$`j=0,\dots,T`$（$`j=0`$ 是消息那次更新）；$`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$。
- *mean-field response.* $`\Delta F_i(\lambda)`$：用幅度为 $`\lambda`$ 的消息替换一条零梯度消息时 $`F`$ 的变化，其中每一步的 $`m_k/\sqrt{v_k}`$ 换成 $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$，期望对尾段梯度取。
- *一阶价值.* $`\mathrm{Val}_i(\lambda)`$ 是消息在坐标 $`i`$ 上的一阶价值（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)）。
- $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ 是 [Proposition 6.11(c)](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 中的孤立访问常数。

<a id="s7-2"></a>

**Corollary 7.2（平稳情形的幅度定律）.** 对坐标 $`i`$ 假设：

- 尾段梯度 $`g_{k,i}`$ 与 $`\theta`$ 无关（线性 response 区间，$`J_k=0`$），平稳，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i>0`$；$`F`$ 关于 $`\theta_i`$ 是仿射的；
- $`\epsilon=0`$，不做 bias correction，$`\lambda_{\rm wd}=0`$；
- 消息到达时状态 warm：$`m_{t,i}=\bar g_i`$，$`v_{t,i}=\bar v_i`$；消息那次更新之后还有 $`T=R-1-t`$ 次尾段更新。

$`\alpha_i,\lambda,\rho_i,b_j,c_j,\omega_T`$ 与 mean-field response $`\Delta F_i(\lambda)`$ 按上面的设定定义。则：

**(a) 精确的 mean-field response（任意 $`\lambda`$）.**

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big].
```

**(b) 小增量：二次律.** 令 $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$，$`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$。若 $`\beta_2\ge\tfrac12`$，则对所有 $`\lambda`$，

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda).
```

mean-field 一阶价值之差 $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ 满足同样的界。由此：

- $`Q_i`$ 是凹的，在 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$ 处取最大值。
- 令 $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$，并设 $`\alpha_i>0`$。若 $`u\le1/8`$，则 $`\Delta F_i`$ 在 $`[\lambda^\star,\,2\lambda^\star(1-4u)]`$ 上为正，在 $`[2\lambda^\star(1+4u),\,3\lambda^\star]`$ 上为负。所以变号点离 $`2\lambda^\star`$ 的相对距离不超过 $`4u`$，略高于它的同向消息劣于零梯度消息。
- 因此阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用。否则变号点（如果存在）要用 (a) 和 (c) 计算。
- 剩余步数短（$`T(1-\beta_2)\ll1`$）. 此时 $`\omega_T\approx(T+1)(1-\beta_2)`$，所以 $`\lambda^\star`$ 很大，且 $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_1^{\,T+1})^2/\big(\rho_i^2(T+1)^2(1-\beta_2)\big)`$。除非这个量很小，$`2\lambda^\star`$ 落在 (b) 的适用范围之外。
- 剩余步数长（$`T(1-\beta_2)\gg1`$）. 此时 $`\omega_T\to1`$，$`\lambda^\star\to(1-\beta_1^{\,T+1})/\rho_i\approx1/\rho_i`$，$`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_2)/\rho_i^2`$。除非该坐标噪声很大，这个量都很小。

**(c) 大幅度.** 当 $`\lambda\to\infty`$ 时，

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}} .
```

- 设 $`\alpha_i>0`$。若 $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$，则所有足够大的同向消息都劣于零梯度消息，且 $`\Delta F_i`$ 有正零点。若 $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$，则所有足够大的同向消息都更好。精确的临界信噪比是 $`\kappa_{\rm iso}(T)/\Lambda(T)`$。
- 由于 $`\Lambda(T)=(T+1)\big(1+O(\tfrac1{T+1}+(1-\beta_2))\big)`$，判据近似为 $`\rho_i(T+1)>\kappa_{\rm iso}(T)`$。
- $`\kappa_{\rm iso}(T)`$ 是 $`\kappa_{\rm iso}`$ 的有限 horizon 版本；当 $`\beta_1<\sqrt{\beta_2}`$ 时，$`T\to\infty`$ 下收敛到它。
- 含义：幅度很大的消息自己决定了分母，所以它的直接效应饱和在 $`\eta\kappa_{\rm iso}(T)`$ 附近。与此同时，它压低了此后每一步：滞后 $`j`$ 时，对照的步长约为 $`\rho_i`$，消息一侧的步长约为 $`\rho_i/\sqrt{1+c_j\lambda^2}`$。只要 $`c_j\lambda^2\gtrsim1`$，这种压低就持续，即约 $`\log((1-\beta_2)\lambda^2)/(1-\beta_2)`$ 步，最多 $`T+1`$ 步。

**(d) 与 SGD 对比.** 同一区间下，SGD 的 $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$ 是线性的：同向消息越大越好。

**说了什么.**

- warm Adam 下，最终结果对一条消息的 response 关于消息幅度 $`\lambda`$ 有精确闭式；在 mean-field 替换下对所有 $`\lambda`$ 成立。
- 增量小时，response 是带显式误差界的凹二次函数：在 $`\lambda^\star`$ 处最大，在 $`2\lambda^\star`$ 附近变号。
- 幅度很大时 response 有显式极限；极限的符号决定了巨大的消息是否有害。

**直觉.**

- 大梯度抬高 $`v`$，从而压低此后很多步。超过某个大小后，压低造成的损失超过大步长带来的收益。
- 信噪比 $`\rho_i`$ 高的坐标效应最强，因为它们此后的步长最有价值。

**在框架中的作用.**

- 这是框架中第一个定量说明 Adam 下消息价值关于幅度不单调的陈述。
- 它要求方法相对于 running 二阶矩控制消息幅度（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 4 步）；[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 正是在这一设定下证明两类影响分数会排错。

**推导要点.**

- 在 mean-field 替换下，消息造成的偏差按几何速度衰减：$`\mathbb Em_{t+j}=\bar g+b_j(g-\bar g)`$，$`\mathbb Ev_{t+j}=\bar v(1+c_j(g^2/\bar v-1))`$。对有消息、无消息两种情形，把 $`T+1`$ 步的 $`\mathbb Em/\sqrt{\mathbb Ev}`$ 相加再相减，就得到 (a)。
- (b) 展开 $`(1+c_j(\lambda^2-1))^{-1/2}`$，对一阶、二阶余项用中值定理给出显式界；交叉项 $`\sum_jb_jc_j`$ 并入常数。
- (c) 逐项取极限，第一个分式趋于 $`b_j/\sqrt{c_j}`$。极限为负、又有 $`\Delta F'(0)>0`$，就必然存在正零点。

**示例（由公式解析算出）.** 取 $`\beta_1=0.9`$、$`\beta_2=0.99`$、$`\rho=0.5`$。

- 剩余 $`T=1000`$ 次更新时，$`\lambda^\star=2`$。精确公式在 $`\lambda\approx1.96`$ 处最大，在 $`\lambda\approx3.98`$ 处变号，紧挨着 $`2\lambda^\star=4`$。幅度为 running RMS 两倍的消息最好；幅度为 RMS 四倍以上的消息还不如不发。
- 剩余 $`T=20`$ 次更新时，$`2\lambda^\star\approx18.7`$，但精确的变号点在 $`\lambda\approx112`$，那里二次律早已不适用。大幅度极限为负，但只是略负（$`\rho\Lambda\approx10.1`$，$`\kappa_{\rm iso}\approx9.2`$）。

**用到.**

- [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)：三通道分解及其线性 response 下的 costate。
- [Proposition 6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)：孤立访问常数 $`\kappa_{\rm iso}`$。

**被用于.**

- [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)：只看参数的分数与余弦分数在哪里排错。
- [Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：裁剪作为 $`\lambda`$ 的上限。
- [Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#s7-5)：一阶排序器失效的饱和区域。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：幅度控制（第 4 步）。

## Related work

- Adam 在单个坐标上的步长由该坐标的信噪比决定；把 Adam 看作"符号步乘以方差自适应因子"的解读正基于此（[Balles & Hennig, 2018](#ref-balles2018dissecting)）。这里同一个比值 $`\rho_i`$ 决定大消息对此后各步是有利还是有害。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
