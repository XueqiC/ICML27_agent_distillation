# Proposition 8.14 · 锚点复用与配对漂移证书

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](08-14_anchor_reuse_drift_certificate_ZH.md) · [证明](../proofs/08-14_anchor_reuse_drift_certificate_ZH.md) · [实验](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md) · [English](08-14_anchor_reuse_drift_certificate.md)

**一段话概括.** 假设在某个锚点 checkpoint 上已经估计了一个 response 接口。能否在后面的 checkpoint 上直接复用，而不必重新估计整个算子？这个命题给出逐 checkpoint 的回答。锚点接口在新 checkpoint 上的残差，至多是锚点第一个被丢弃的奇异值，加上锚点的估计误差，再加上两个 checkpoint 之间的算子漂移；不需要谱间隙。漂移可以用一个配对实验来界，这个实验只需要随机消息，不需要整个目录：配对的有界输出之差的均方，界住了算子漂移的平方。再加一个 Hoeffding 半径，就成为严格证书。在真实模型上测得的评估噪声下，它成立但为空：几十对以内半径高于平凡界，要降到 0.03 大约需要五百万对。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 共同消息空间上的参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$；有限的声明测试集 $`\mathcal A`$，共 $`p`$ 个测试，权重 $`\omega_a>0`$，在两个 checkpoint 上相同；$`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$。
- 两个 checkpoint：*锚点* $`z_0`$ 与*新 checkpoint* $`z`$，response operator 分别为 $`\mathcal R_0`$、$`\mathcal R_z`$（[Definition 4.8](04-08_response_operator_ZH.md#s4-8)），response 向量为 $`g_0(y)=(g_{0,a}(y))_a`$、$`g_z(y)=(g_{z,a}(y))_a\in[0,1]^p`$。
- *锚点估计.* $`\hat{\mathcal R}_0`$ 满足 $`\lVert\hat{\mathcal R}_0-\mathcal R_0\rVert_{\rm op}\le\epsilon_0`$（例如来自 [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)），奇异值为 $`\hat\sigma_k(z_0)`$，$`\hat\Pi_r`$ 投影到它的前 $`r`$ 个右奇异向量上。
- *配对漂移实验.* 从 $`\rho`$ 独立抽取 $`y_1,\dots,y_n`$。对每个 $`y_i`$，在消息 $`y_i`$ 之后分别从 $`z`$ 和从 $`z_0`$ 各跑一次声明的测试，各自用正确的边缘协议，记录有界输出向量 $`X^{(i)}_z,X^{(i)}_0\in[0,1]^p`$。同一对内的两次运行可以共享随机数；不同对之间独立。于是 $`\mathbb E[X^{(i)}_z\mid y_i]=g_z(y_i)`$，$`\mathbb E[X^{(i)}_0\mid y_i]=g_0(y_i)`$。记 $`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$，$`W_i=\lVert D^{(i)}\rVert_\omega^2\in[0,1]`$。
- 在跑任何一对之前冻结 $`M`$ 个 checkpoint 对；$`\bar W_z`$ 是 checkpoint 对 $`z`$ 的 $`W_i`$ 的平均。

<a id="s8-14"></a>

**Proposition 8.14（锚点复用与配对漂移证书）.** 在上述设定下：

**(a) 无需间隙的锚点复用.** 若 $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$，则

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z .
```

因此，对 $`z`$ 处满足 $`\lVert h\rVert_\rho\le B_z`$ 且 $`\hat\Pi_rh=0`$（保留的特征均值相同）的反馈差 $`h`$，有 $`\lVert\mathcal R_zh\rVert_\omega\le B_z\big(\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z\big)`$；对单个测试 $`a`$，分数差至多为它除以 $`\sqrt{\omega_a}`$。

**(b) 配对二阶矩界住漂移.**

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2\le\mathbb E_{y\sim\rho}\lVert g_z(y)-g_0(y)\rVert_\omega^2\le\mathbb E\,W_i .
```

**(c) 严格的漂移证书.** 以至少 $`1-\delta`$ 的概率，对 $`M`$ 个冻结的 checkpoint 对同时有

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\Delta_z=\sqrt{\min\Big\lbrace1,\ \bar W_z+\sqrt{\frac{\log(M/\delta)}{2n}}\Big\rbrace},
```

所以 $`\hat\Delta_z`$ 可以作为 (a) 中的 $`\Delta_z`$。每一对在每个不同的续训协议下需要两个续训分支。

**(d) 估计量的地板及其去偏版本.** $`\mathbb EW_i=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2+\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)`$，其中 $`\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)=\mathbb E[\lVert D-\mathbb E[D\mid y]\rVert_\omega^2\mid y]`$。$`n`$ 增大时 $`\hat\Delta_z`$ 收敛到 $`(\mathbb EW_i)^{1/2}`$，它永远不低于过程噪声水平 $`(\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y))^{1/2}`$。如果每条消息跑两次，得到条件独立的配对差 $`D^{(i,1)},D^{(i,2)}`$，那么 $`U_i=\langle D^{(i,1)},D^{(i,2)}\rangle_\omega\in[-1,1]`$ 满足 $`\mathbb E[U_i\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2`$，并且以至少 $`1-\delta`$ 的概率，对全部 $`M`$ 对，

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\tilde\Delta_z=\sqrt{\min\Big\lbrace1,\ \max\Big\lbrace0,\ \bar U_z+\sqrt{\frac{2\log(M/\delta)}{n}}\Big\rbrace\Big\rbrace}.
```

它以两倍成本去掉了地板，但漂移为零时半径仍只按 $`n^{-1/4}`$ 下降。

**说了什么.**

- 在一个 checkpoint 上估计的接口可以在另一个 checkpoint 上复用，代价是测得的算子漂移；奇异向量可以旋转或交叉而不受影响，因为只用到算子范数。
- 漂移可以用随机消息上的配对实验测量，并有严格的有限样本证书。
- 这个证书成立，但在实测的评估噪声下为空：以真实模型噪声参照的噪声水平，10 对时半径为 0.718，40 对 0.507，100 对 0.404，1,000 对 0.227（$`M=10`$，$`\delta=0.05`$），而残差本身的平凡界是 $`\tfrac12`$；半径要降到 0.03 约需 $`4.9\times10^6`$ 对（[Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md#c8-21)）。

**直觉.** 复用就是一个三角不等式：新算子在锚点保留方向之外做的事，至多是锚点在那里做的事，加上锚点估计错了多少，再加上算子移动了多远。漂移本身是分数*之差*的 response operator，所以配对输出差的二阶矩能控制它；但这个二阶矩里也含有运行之间的噪声，而有界取值的集中不等式要为整个取值范围付费。

**在框架中的作用.**

- 它是在持续训练中维护 response 接口时逐 checkpoint 的复用步骤，补充 [Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13) 的平均界。
- 它为 [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 提供 $`\Delta_{t,z}`$。
- 它在实测量级下为空，这是 [Remark 8.17](08-17_open_problems_ZH.md#s8-17) 的一个未解决问题：要一个在负担得起的配对数下、半径随效应大小缩放的漂移证书。

**关于陈述的说明.**

- **噪声数字的来源.** 真实模型的噪声参照在两台配置相同的服务器上，用 300 个 dev 任务评估同一个 checkpoint：分别成功 241 和 242 个，15 个任务翻转（一个方向 7 个，另一个方向 8 个）。如果翻转是对称的，单次分数差的标准差约为 $`\sqrt{15}/300\approx0.013`$，单个测试上 $`\bar W\approx1.7\times10^{-4}`$。这只包括推理服务的噪声；两次续训之间的训练随机性还会再加上去。一对服务器、一个 checkpoint 不足以说明评估噪声是不可约且独立同分布的。
- **为什么为空.** Hoeffding 项 $`\sqrt{\log(M/\delta)/(2n)}`$ 不随 $`\bar W`$ 缩小。方差自适应的界能去掉一部分，但范围项仍在；除非用去偏乘积，(d) 的地板也仍在。
- **多个锚点.** 可以为每个 checkpoint 选认证界最小的锚点；如果这个选择用了同一批审计数据，所有候选锚点的事件都必须同时成立。
- **逐测试使用.** 对单个测试 $`a`$ 的保证要付 $`1/\sqrt{\omega_a}`$，或者直接审计那一行；RMS 界不是逐测试的界。

**推导要点.**

- (a)：$`\mathcal R_z=\hat{\mathcal R}_0+(\mathcal R_0-\hat{\mathcal R}_0)+(\mathcal R_z-\mathcal R_0)`$，且 $`\lVert I-\hat\Pi_r\rVert_{\rm op}\le1`$。
- (b)：$`\mathcal R_z-\mathcal R_0`$ 是分数差 $`g_z-g_0`$ 的 response operator；它的 Hilbert–Schmidt 范数是加权方差，不超过二阶矩；条件 Jensen 把条件均值换成配对输出。
- (c)：对 $`W_i\in[0,1]`$ 用单侧 Hoeffding 不等式，再对 $`M`$ 对取 union bound。
- (d)：方差分解；两个配对差的条件独立性；在 $`[-1,1]`$ 上用 Hoeffding。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8) — response operator
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) — 锚点估计及其误差

**被用于.**

- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) — 组合保证中的漂移项
- [Remark 8.17](08-17_open_problems_ZH.md#s8-17) — 半径随效应缩放的证书

## 相关工作

- 漂移界是 Eckart–Young 定理（[Eckart & Young, 1936](#ref-eckart1936approximation)）意义下的扰动论证；证书用的是 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）。方差自适应的替代是 empirical Bernstein 界（[Maurer & Pontil, 2009](#ref-maurer2009empirical)），它仍要付 $`1/n`$ 量级的范围项。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
