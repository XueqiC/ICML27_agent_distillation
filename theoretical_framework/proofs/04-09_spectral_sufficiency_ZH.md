# Theorem 4.9 · 谱充分性与线性接口复杂度

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-09_spectral_sufficiency_ZH.md) · [证明](04-09_spectral_sufficiency_ZH.md) · [实验](../experiments/04-09_spectral_sufficiency_ZH.md) · [English](04-09_spectral_sufficiency.md)

**本页证明什么.** (a)–(b) 是奇异基上的 Parseval 与 Bessel。(c) 是允许任意解码器的 Eckart–Young 论证：$`r`$ 个线性统计量总有一个 $`r+1`$ 维空间中的方向看不见。(d) 是对单个测试 response 的截断部分用 Cauchy–Schwarz。(e) 是一个计数界，外加一个达到它的显式例子。(f) 把 (c) 中看不见的方向变成两个概率分布。

**本页用到的设定.**

- $`H=L_0^2(\rho)`$，response operator $`\mathcal R:H\to L^2(\omega)`$，$`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$，$`\lVert\mathcal R\rVert_{\rm HS}^2\le\tfrac14`$，奇异值分解 $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)）。
- $`h=p-q`$，$`p,q`$ 是 $`L^2(\rho)`$ 中的密度，$`\lVert h\rVert_\rho\le B`$，$`\alpha_j=\langle v_j,h\rangle_\rho`$。
- 秩 $`r`$ 线性接口 $`h\mapsto D(s(h))`$，$`s(h)=(\langle w_i,h\rangle_\rho)_{i\le r}`$，$`w_i\in H`$，$`D`$ 为任意可测解码器。
- $`b_j(a)=\langle\tilde g_a,v_j\rangle_\rho`$，$`e_r(a)=\lVert\tilde g_a-\sum_{j\le r}b_j(a)v_j\rVert_\rho`$；$`\Pi_r`$ 是 $`H`$ 到 $`\mathrm{span}(v_1,\dots,v_r)`$ 的正交投影。

**陈述（[Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9)）.** (a) $`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\alpha_j^2`$。(b) 对 $`j\le r`$ 有 $`\lvert\alpha_j\rvert\le\delta_j`$ 时 $`\lVert\mathcal Rh\rVert_\omega^2\le\sum_{j\le r}\sigma_j^2\delta_j^2+B^2\sigma_{r+1}^2`$。(c) 秩 $`r`$ 线性接口在 $`\lbrace h\in H:\lVert h\rVert_\rho\le B\rbrace`$ 上的极小极大误差为 $`B\sigma_{r+1}`$，由前几个模式达到。(d) $`\lvert\mathbb E_Pg_a-\mathbb E_Qg_a\rvert\le\lvert\sum_{j\le r}b_j(a)\alpha_j\rvert+Be_r(a)`$，且 $`\mathbb E_\omega e_r^2=\sum_{j>r}\sigma_j^2`$。(e) $`(r+1)\sigma_{r+1}^2\le\tfrac14`$；在 $`J`$ 候选混合测度下界 $`B^2=2J`$ 能达到，且 $`\varepsilon\le1/\sqrt2`$ 时得到的维数不小于平凡维数。(f) 在有限消息集上，(4.17) 中的分布是统计量相等、response 差至少为 $`a\sigma_{r+1}`$ 的概率分布。

**证明.**

**预备.** $`\mathcal R`$ 是 Hilbert–Schmidt 算子，因而是紧的，且 $`H`$ 与 $`L^2(\omega)`$ 可分。紧算子的奇异值分解给出张成 $`(\ker\mathcal R)^\perp`$ 的正交归一 $`(v_j)`$、正交归一 $`(u_j)`$，以及对每个 $`h\in H`$ 成立的 $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$，级数在 $`L^2(\omega)`$ 中收敛。若 $`\mathcal R`$ 的秩是有限的 $`k`$，对 $`j>k`$ 令 $`\sigma_j=0`$，并用 $`\ker\mathcal R`$ 的一组正交归一基把 $`(v_j)`$ 补全。

**(a)** 由 $`u_j`$ 的正交归一性（Parseval），$`\lVert\mathcal Rh\rVert_\omega^2=\sum_j\sigma_j^2\langle v_j,h\rangle_\rho^2=\sum_j\sigma_j^2\alpha_j^2`$。因为 $`v_j\in H`$ 的 $`\rho`$ 均值为零，$`\alpha_j=\int v_j(p-q)\,d\rho=\mathbb E_Pv_j-\mathbb E_Qv_j`$。

**(b)** 把 (a) 在 $`r`$ 处拆开。前一部分至多为 $`\sum_{j\le r}\sigma_j^2\delta_j^2`$。尾部中 $`j>r`$ 时 $`\sigma_j\le\sigma_{r+1}`$，Bessel 不等式给出 $`\sum_{j>r}\alpha_j^2\le\lVert h\rVert_\rho^2\le B^2`$。

**(c)** *上界.* 取 $`w_j=v_j`$、$`D(s)=\sum_{j\le r}\sigma_js_ju_j`$，误差为 $`\mathcal Rh-\mathcal R\Pi_rh=\sum_{j>r}\sigma_j\alpha_ju_j`$，由 (b) 的尾部估计，其范数至多为 $`\sigma_{r+1}\lVert h\rVert_\rho\le B\sigma_{r+1}`$。

*下界.* 任意固定 $`w_1,\dots,w_r\in H`$ 与解码器 $`D`$。若 $`\sigma_{r+1}=0`$，无需证明。否则 $`V=\mathrm{span}(v_1,\dots,v_{r+1})`$ 是 $`r+1`$ 维的，线性映射 $`V\to\mathbb R^r`$，$`x\mapsto s(x)`$ 有非平凡的核。在核中取 $`v\in V`$ 且 $`\lVert v\rVert_\rho=1`$。由 (a)，$`\lVert\mathcal Rv\rVert_\omega^2=\sum_{j\le r+1}\sigma_j^2\langle v_j,v\rangle_\rho^2\ge\sigma_{r+1}^2`$。输入 $`h_\pm=\pm Bv`$ 都在球内且 $`s(h_\pm)=0`$，所以接口对两者返回同一个 $`f=D(0)`$。由三角不等式，

```math
\lVert\mathcal Rh_+-f\rVert_\omega+\lVert\mathcal Rh_--f\rVert_\omega\ge\lVert\mathcal Rh_+-\mathcal Rh_-\rVert_\omega=2B\lVert\mathcal Rv\rVert_\omega\ge2B\sigma_{r+1},
```

因此两个误差中至少一个不小于 $`B\sigma_{r+1}`$。解码器是任意的，所以 (4.14) 中的下确界至少为 $`B\sigma_{r+1}`$；上界说明它能达到。

**(d)** 因为 $`\mathbb E_\rho h=0`$，$`\mathbb E_Pg_a-\mathbb E_Qg_a=\langle\tilde g_a,h\rangle_\rho`$。写 $`\tilde g_a=\sum_{j\le r}b_j(a)v_j+\eta_a`$，其中 $`\eta_a=(I-\Pi_r)\tilde g_a`$，$`\lVert\eta_a\rVert_\rho=e_r(a)`$。于是

```math
\langle\tilde g_a,h\rangle_\rho=\sum_{j\le r}b_j(a)\alpha_j+\langle\eta_a,h\rangle_\rho,\qquad\lvert\langle\eta_a,h\rangle_\rho\rvert\le e_r(a)\,B
```

（Cauchy–Schwarz）。关于平均：$`(\mathcal R(I-\Pi_r)h)(a)=\langle\tilde g_a,(I-\Pi_r)h\rangle_\rho=\langle\eta_a,h\rangle_\rho`$，所以 $`\eta_a`$ 是算子 $`\mathcal R(I-\Pi_r)`$ 的核，$`\mathbb E_\omega e_r(a)^2=\lVert\mathcal R(I-\Pi_r)\rVert_{\rm HS}^2=\sum_{j>r}\sigma_j^2`$。最后，对有限的 $`\mathcal A`$，$`\omega_a(\mathcal Rh)(a)^2\le\sum_{a'}\omega_{a'}(\mathcal Rh)(a')^2=\lVert\mathcal Rh\rVert_\omega^2`$。

**(e)** 奇异值单调不增，所以由 (4.11)，$`(r+1)\sigma_{r+1}^2\le\sum_{j\le r+1}\sigma_j^2\le\lVert\mathcal R\rVert_{\rm HS}^2\le\tfrac14`$。因此 $`B\sigma_{r+1}\le B/(2\sqrt{r+1})`$，当 $`r+1\ge B^2/(4\varepsilon^2)`$ 时它至多为 $`\varepsilon`$。

*$`B^2=2J`$ 能达到.* 对不同的消息 $`y_1,\dots,y_J`$ 取 $`P^j=\delta_{y_j}`$。此时 $`\rho`$ 是 $`\lbrace y_1,\dots,y_J\rbrace`$ 上的均匀分布，$`P^j`$ 的密度为 $`J\mathbf 1\lbrace y=y_j\rbrace`$，对 $`h=p_1-p_2`$ 有 $`\lVert h\rVert_\rho^2=\tfrac1J(J^2+J^2)=2J`$。

*没有压缩内容.* 取 $`B^2=2J`$ 时，(4.16) 要求 $`r+1\ge J/(2\varepsilon^2)`$；$`\varepsilon\le1/\sqrt2`$ 时右边不小于 $`J`$，即 $`r\ge J-1`$。两个混合 $`\sum_j\lambda_jP^j`$ 与 $`\sum_j\lambda_j'P^j`$ 之差的密度为 $`\sum_j(\lambda_j-\lambda_j')p_j`$，且 $`\sum_j(\lambda_j-\lambda_j')=0`$，所以它落在 $`W=\mathrm{span}\lbrace p_j-p_1:j\le J\rbrace`$ 中，维数至多 $`J-1`$。这些 $`p_j-p_1`$ 的 $`\rho`$ 均值为零，所以 $`W\subseteq H`$。取张成 $`W`$ 的 $`w_1,\dots,w_{J-1}`$（维数更小时补零）。在 $`W`$ 上映射 $`h\mapsto s(h)`$ 是单射，所以线性解码器能精确恢复 $`h`$，进而恢复 $`\mathcal Rh`$。一个要求至少 $`J-1`$ 个统计量的界，保证不了平凡接口保证不了的东西。

**(f)** 这里 $`H=\lbrace h:\sum_y\rho_yh_y=0\rbrace`$ 的维数为 $`m-1\ge r+1`$，所以 (c) 的构造适用（秩低于 $`r+1`$ 时在 $`H`$ 内补全 $`v_j`$；此时 $`\sigma_{r+1}=0`$，结论平凡）。令 $`v`$ 为 (c) 中使 $`r`$ 个统计量为零的单位向量，按 (4.17) 定义 $`P,Q`$。

- *是概率分布.* 因为 $`a\lvert v_y\rvert\le a\lVert v\rVert_\infty\le2`$，所以 $`P_y,Q_y\ge0`$。总质量为 $`1\pm\tfrac a2\sum_y\rho_yv_y=1`$，因为 $`v\in H`$。
- *统计量相等.* $`p-q=av`$，所以 $`s(p-q)=a\,s(v)=0`$；对特征 $`\phi_i`$ 即 $`\mathbb E_P\phi_i=\mathbb E_Q\phi_i`$。
- *差距.* $`\lVert p-q\rVert_\rho=a`$，$`\lVert\mathcal R(p-q)\rVert_\omega=a\lVert\mathcal Rv\rVert_\omega\ge a\sigma_{r+1}`$。
- *$`a`$ 的统一取法.* 对每个 $`y`$，$`1=\lVert v\rVert_\rho^2\ge\rho_yv_y^2`$，所以 $`\lVert v\rVert_\infty\le1/\sqrt{\rho_{\min}}`$，$`a=2\sqrt{\rho_{\min}}\le2/\lVert v\rVert_\infty`$。
- *概率分布对上的最坏情形.* $`B\le2\sqrt{\rho_{\min}}`$ 时取 $`a=B`$。分布对 $`(P,Q)`$ 与 $`(Q,P)`$ 都给出统计量 $`0`$，差分别为 $`\pm Bv`$；(c) 的论证给出其中之一的误差至少为 $`B\sigma_{r+1}`$，而 (c) 的上界对每个满足 $`\lVert p-q\rVert_\rho\le B`$ 的分布对都成立。$`\square`$

## 相关工作

- (c) 的下界就是 Eckart–Young 的论证 ([Eckart & Young, 1936](#ref-eckart1936approximation))，只是把线性重构换成任意解码器；编码器是线性的，所以解码器帮不上忙。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).
