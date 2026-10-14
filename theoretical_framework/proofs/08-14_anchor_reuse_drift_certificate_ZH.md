# Proposition 8.14 · 锚点复用与配对漂移证书

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-14_anchor_reuse_drift_certificate_ZH.md) · [证明](08-14_anchor_reuse_drift_certificate_ZH.md) · [实验](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md) · [English](08-14_anchor_reuse_drift_certificate.md)

**本页证明什么.** (a) 是三项三角不等式。(b) 把漂移识别为分数差的 response operator，用二阶矩界住它，再借条件 Jensen 换成配对输出。(c) 是单侧 Hoeffding 界加 union bound。(d) 是条件方差分解，以及两个条件独立的差之乘积的无偏性。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$；$`p`$ 个测试，权重 $`\omega_a>0`$，$`\sum_a\omega_a=1`$；$`\lVert u\rVert_\omega^2=\sum_a\omega_au_a^2`$，$`\langle u,u'\rangle_\omega=\sum_a\omega_au_au'_a`$。
- 锚点 $`z_0`$ 与新 checkpoint $`z`$，response operator 为 $`\mathcal R_0`$、$`\mathcal R_z`$，response 向量 $`g_0(y),g_z(y)\in[0,1]^p`$，所以 $`(\mathcal R_zh)(a)=\langle g_{z,a}-\mathbb E_\rho g_{z,a},h\rangle_\rho`$。
- $`\hat{\mathcal R}_0`$ 满足 $`\lVert\hat{\mathcal R}_0-\mathcal R_0\rVert_{\rm op}\le\epsilon_0`$，奇异值 $`\hat\sigma_k(z_0)`$，$`\hat\Pi_r`$ 投影到它的前 $`r`$ 个右奇异向量上。
- 配对实验：$`y_i\sim\rho`$ 独立；$`X^{(i)}_z,X^{(i)}_0\in[0,1]^p`$ 满足 $`\mathbb E[X^{(i)}_z\mid y_i]=g_z(y_i)`$、$`\mathbb E[X^{(i)}_0\mid y_i]=g_0(y_i)`$；不同对独立；$`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$，$`W_i=\lVert D^{(i)}\rVert_\omega^2`$；冻结 $`M`$ 个 checkpoint 对。

**陈述（[Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)）.** (a) 若 $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$，则 $`\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z`$；因此当 $`\lVert h\rVert_\rho\le B_z`$ 且 $`\hat\Pi_rh=0`$ 时 $`\lVert\mathcal R_zh\rVert_\omega\le B_z(\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z)`$，单个测试要付 $`1/\sqrt{\omega_a}`$。(b) $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$。(c) 以至少 $`1-\delta`$ 的概率，对全部 $`M`$ 对，$`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\sqrt{\min\lbrace1,\bar W_z+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$。(d) $`\mathbb EW_i=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2+\mathbb E\,\mathrm{tr}_\omega\mathrm{Cov}(D\mid y)`$；每条消息有两个条件独立的配对差时，$`U_i=\langle D^{(i,1)},D^{(i,2)}\rangle_\omega\in[-1,1]`$ 满足 $`\mathbb E[U_i\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2`$，并且以至少 $`1-\delta`$ 的概率，对全部 $`M`$ 对，$`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\sqrt{\min\lbrace1,\max\lbrace0,\bar U_z+\sqrt{2\log(M/\delta)/n}\rbrace\rbrace}`$。

*证明.*

**(a)** 写成 $`\mathcal R_z=\hat{\mathcal R}_0+(\mathcal R_0-\hat{\mathcal R}_0)+(\mathcal R_z-\mathcal R_0)`$。右乘算子范数至多为 1 的 $`I-\hat\Pi_r`$，再用三角不等式：

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\lVert\hat{\mathcal R}_0(I-\hat\Pi_r)\rVert_{\rm op}+\lVert\mathcal R_0-\hat{\mathcal R}_0\rVert_{\rm op}+\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z .
```

由 $`\hat{\mathcal R}_0`$ 的奇异值分解，第一项等于 $`\hat\sigma_{r+1}(z_0)`$。这里不用间隙，也不需要识别单个奇异向量，所以从 $`z_0`$ 到 $`z`$ 奇异向量可以旋转或交叉。若 $`\hat\Pi_rh=0`$，则 $`\mathcal R_zh=\mathcal R_z(I-\hat\Pi_r)h`$，算子界乘以 $`\lVert h\rVert_\rho\le B_z`$ 就是 response 的界。对单个测试，$`\omega_a\lvert(\mathcal R_zh)(a)\rvert^2\le\lVert\mathcal R_zh\rVert_\omega^2`$。

**(b)** *第 1 步.* response operator 对分数函数是线性的：$`((\mathcal R_z-\mathcal R_0)h)(a)=\langle\tilde d_a,h\rangle_\rho`$，其中 $`d_a=g_{z,a}-g_{0,a}`$，$`\tilde d_a=d_a-\mathbb E_\rho d_a`$。在 $`\mathcal H`$ 的一组正交基 $`(e_k)`$ 下用 Parseval 计算 Hilbert–Schmidt 范数：

```math
\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm HS}^2=\sum_a\omega_a\sum_k\langle\tilde d_a,e_k\rangle_\rho^2=\sum_a\omega_a\lVert\tilde d_a\rVert_\rho^2=\sum_a\omega_a\mathrm{Var}_\rho d_a\le\sum_a\omega_a\mathbb E_\rho d_a^2=\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2 .
```

（常数与 $`\mathcal H`$ 正交，而 $`\tilde d_a\in\mathcal H`$。）算子范数不超过 Hilbert–Schmidt 范数。

*第 2 步.* 每个输出的条件均值都正确，所以无论对内如何耦合，$`\mathbb E[D^{(i)}\mid y_i]=g_z(y_i)-g_0(y_i)`$。映射 $`u\mapsto\lVert u\rVert_\omega^2`$ 是凸的，条件 Jensen 给出 $`\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2\le\mathbb E[\lVert D^{(i)}\rVert_\omega^2\mid y_i]`$。对 $`y_i\sim\rho`$ 取期望，得 $`\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$。

**(c)** $`D^{(i)}`$ 的每个坐标都在 $`[-1,1]`$ 中，且 $`\sum_a\omega_a=1`$，所以 $`W_i\in[0,1]`$；对一个 checkpoint 对，$`W_i`$ 独立同分布。单侧 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）给出 $`\Pr\big(\mathbb EW_i>\bar W_z+s\big)\le e^{-2ns^2}`$。取 $`s=\sqrt{\log(M/\delta)/(2n)}`$，右边为 $`\delta/M`$；对 $`M`$ 个冻结的对取 union bound，以至少 $`1-\delta`$ 的概率对所有对都有 $`\mathbb EW_i\le\bar W_z+s`$。再结合 (b) 与 $`\mathbb EW_i\le1`$。这些对是在运行之前冻结的；看过漂移数据之后再挑的 checkpoint 不在覆盖范围内。

**(d)** *分解.* 在给定 $`y`$ 的条件下，$`\mathbb E[\lVert D\rVert_\omega^2\mid y]=\lVert\mathbb E[D\mid y]\rVert_\omega^2+\mathbb E[\lVert D-\mathbb E[D\mid y]\rVert_\omega^2\mid y]`$；取期望即可。$`n\to\infty`$ 时，由大数定律 $`\bar W_z\to\mathbb EW_i`$ 几乎必然成立，Hoeffding 项趋于零，所以 $`\hat\Delta_z\to(\mathbb EW_i)^{1/2}`$，它不小于噪声项的平方根。

*去偏乘积.* 令 $`D^{(i,1)},D^{(i,2)}`$ 为同一条消息 $`y_i`$ 下两次配对的差，给定 $`y_i`$ 条件独立，条件均值都是 $`g_z(y_i)-g_0(y_i)`$。则

```math
\mathbb E[U_i\mid y_i]=\sum_a\omega_a\,\mathbb E[D^{(i,1)}_a\mid y_i]\,\mathbb E[D^{(i,2)}_a\mid y_i]=\lVert g_z(y_i)-g_0(y_i)\rVert_\omega^2 .
```

由 Cauchy–Schwarz，$`\lvert U_i\rvert\le\lVert D^{(i,1)}\rVert_\omega\lVert D^{(i,2)}\rVert_\omega\le1`$。对取值区间长度为 2 的变量用单侧 Hoeffding 不等式，$`\Pr(\mathbb EU_i>\bar U_z+s)\le e^{-ns^2/2}`$；取 $`s=\sqrt{2\log(M/\delta)/n}`$ 并取 union bound，以至少 $`1-\delta`$ 的概率对所有 $`M`$ 对都有 $`\mathbb EU_i\le\bar U_z+s`$。由 (b)，$`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2=\mathbb EU_i`$；平方非负，而且 $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\lVert\mathcal R_z\rVert_{\rm HS}+\lVert\mathcal R_0\rVert_{\rm HS}\le1`$，这说明截断是合理的。漂移为零时界为 $`(2\log(M/\delta)/n)^{1/4}`$。$`\square`$

## 相关工作

- 证书用的是 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）；复用一步是 Eckart–Young 类型的扰动论证（[Eckart & Young, 1936](#ref-eckart1936approximation)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
