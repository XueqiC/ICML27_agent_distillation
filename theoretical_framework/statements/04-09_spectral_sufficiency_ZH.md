# Theorem 4.9 · 谱充分性与线性接口复杂度

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-09_spectral_sufficiency_ZH.md) · [证明](../proofs/04-09_spectral_sufficiency_ZH.md) · [实验](../experiments/04-09_spectral_sufficiency_ZH.md) · [English](04-09_spectral_sufficiency.md)

**一段话.** 在固定检查点上，[Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator 把两个 feedback law 之间的差变成每个声明测试上的得分差。它的奇异值说明 feedback 差异的每个方向能让测试动多少。这个定理把这一点说精确：匹配前几个输入模式后，剩下的误差至多是下一个奇异值乘以差的大小；任何由 $`r`$ 个线性统计量构成的接口，即使配上非线性解码器，最坏情形也做不到更好；对每个单独的测试都有逐测试的界；而且同样的下界能被真正的概率分布达到。定理也写明了它不给出什么：仅由有界性推出的一般维数界，在自然的混合测度下没有任何压缩内容，所以有用的压缩是一个要测量的性质，不是设定本身的推论。

**本页用到的设定.**

- 检查点 $`(t,z)`$，声明的测试 $`\mathcal A`$ 及权重 $`\omega`$，参考消息测度 $`\rho`$，response $`g_a\in[0,1]`$，以及 response operator $`\mathcal R:H=L_0^2(\rho)\to L^2(\omega)`$，其奇异值分解为 $`\mathcal Rh=\sum_j\sigma_j\langle v_j,h\rangle_\rho u_j`$（[Definition 4.8](04-08_response_operator_ZH.md#s4-8)）。
- 两个 feedback law $`P,Q`$，密度 $`p,q\in L^2(\rho)`$，差 $`h=p-q\in H`$，界 $`\lVert h\rVert_\rho\le B`$。*模式差*为 $`\alpha_j=\langle v_j,h\rangle_\rho=\mathbb E_Pv_j-\mathbb E_Qv_j`$。
- *秩 $`r`$ 线性接口*读取 $`r`$ 个线性统计量 $`s(h)=(\langle w_1,h\rangle_\rho,\dots,\langle w_r,h\rangle_\rho)`$，其中 $`w_i\in H`$（对特征 $`\phi_i\in L^2(\rho)`$，这就是特征均值之差 $`\mathbb E_P\phi_i-\mathbb E_Q\phi_i`$，取 $`w_i=\phi_i-\mathbb E_\rho\phi_i`$），并对任意可测解码器 $`D:\mathbb R^r\to L^2(\omega)`$ 输出 $`D(s(h))`$。
- 对测试 $`a`$：$`b_j(a)=\langle\tilde g_a,v_j\rangle_\rho`$，$`e_r(a)=\lVert\tilde g_a-\sum_{j\le r}b_j(a)v_j\rVert_\rho`$，即测试 $`a`$ 的中心化 response 落在前 $`r`$ 个输入模式之外的部分。

<a id="s4-9"></a>

**Theorem 4.9（谱充分性与线性接口复杂度）.** 在上述设定下：

**(a) 模式恒等式.**

```math
\lVert\mathcal Rh\rVert_\omega^2=\sum_{j\ge1}\sigma_j^2\alpha_j^2 . \qquad(4.12)
```

**(b) 匹配前几个模式.** 若对 $`j\le r`$ 有 $`\lvert\alpha_j\rvert\le\delta_j`$，则

```math
\lVert\mathcal Rh\rVert_\omega^2\le\sum_{j\le r}\sigma_j^2\delta_j^2+B^2\sigma_{r+1}^2 ; \qquad(4.13)
```

特别地，$`\alpha_1=\dots=\alpha_r=0`$ 时 $`\lVert\mathcal Rh\rVert_\omega\le B\sigma_{r+1}`$。

**(c) 最优线性接口.** 对所有秩 $`r`$ 线性接口与所有解码器，

```math
\inf_{w_1,\dots,w_r,\,D}\ \sup_{h\in H,\ \lVert h\rVert_\rho\le B}\ \big\lVert\mathcal Rh-D(s(h))\big\rVert_\omega=B\sigma_{r+1}, \qquad(4.14)
```

由 $`w_j=v_j`$ 与线性解码器 $`D(s)=\sum_{j\le r}\sigma_js_ju_j`$ 达到。*线性接口复杂度*定义为 $`r_\varepsilon(B)=\min\lbrace r:B\sigma_{r+1}\le\varepsilon\rbrace`$。

**(d) 每个测试.** 对每个测试 $`a`$，

```math
\lvert\mathbb E_Pg_a-\mathbb E_Qg_a\rvert\le\Big\lvert\sum_{j\le r}b_j(a)\alpha_j\Big\rvert+B\,e_r(a),\qquad \mathbb E_{a\sim\omega}e_r(a)^2=\sum_{j>r}\sigma_j^2 . \qquad(4.15)
```

若 $`\mathcal A`$ 有限且 $`\omega_a>0`$，还有 $`\lvert(\mathcal Rh)(a)\rvert\le\lVert\mathcal Rh\rVert_\omega/\sqrt{\omega_a}`$。

**(e) 一般维数界，以及它在哪里是空的.** 不对谱作任何假设，有 $`(r+1)\sigma_{r+1}^2\le\tfrac14`$，所以 $`B\sigma_{r+1}\le B/(2\sqrt{r+1})`$，并且

```math
r+1\ge\frac{B^2}{4\varepsilon^2}\quad\Longrightarrow\quad B\sigma_{r+1}\le\varepsilon . \qquad(4.16)
```

这个维数是充分的，不是必要的。在 $`J`$ 个候选分布的混合测度 $`\rho=J^{-1}\sum_{j\le J}P^j`$ 下，[Definition 4.8](04-08_response_operator_ZH.md#s4-8) 中的最坏界 $`B^2=2J`$ 在全部 $`J`$ 个候选都是落在不同消息上的点质量时被达到（此时任意两个候选都有 $`\lVert h\rVert_\rho^2=2J`$）。用这个 $`B`$，(4.16) 要求 $`r+1\ge J/(2\varepsilon^2)`$，对每个 $`\varepsilon\le1/\sqrt2`$ 即 $`r\ge J-1`$。可是 $`J`$ 个候选的所有混合之差都落在一个至多 $`J-1`$ 维的空间里，在那里 $`J-1`$ 个线性统计量本来就精确。因此在这个设定下，一般界没有任何压缩内容。

**(f) 下界能被概率分布达到.** 设 $`\mathcal Y`$ 有限，含 $`m`$ 个消息，对每个 $`y`$ 有 $`\rho_y>0`$，$`\rho_{\min}=\min_y\rho_y`$，且 $`0\le r<m-1`$。对任意 $`r`$ 个线性统计量，存在单位向量 $`v\in\mathrm{span}(v_1,\dots,v_{r+1})`$ 使它们全部为零；对每个 $`0<a\le2/\lVert v\rVert_\infty`$，分布

```math
P_y=\rho_y\big(1+\tfrac a2v_y\big),\qquad Q_y=\rho_y\big(1-\tfrac a2v_y\big) \qquad(4.17)
```

是统计量相等的概率分布，$`\lVert p-q\rVert_\rho=a`$，且 $`\lVert\mathcal R(p-q)\rVert_\omega\ge a\sigma_{r+1}`$。取 $`a=2\sqrt{\rho_{\min}}`$ 总是可行。因此当 $`B\le2\sqrt{\rho_{\min}}`$ 时，(4.14) 的值 $`B\sigma_{r+1}`$ 也是概率分布对上的最坏误差。

**说了什么.**

- response operator 的奇异值是 feedback law 可能相差的各个方向的增益。匹配增益最大的 $`r`$ 个方向后，剩下的误差至多是下一个增益乘以差的大小。
- 由 $`r`$ 个线性统计量构成的任何东西，即使配上非线性解码器，最坏情形也赢不了这个；差足够小时，真实的概率分布已经能达到最坏情形。
- 对测试的加权平均的界不是对稀有测试的界；(d) 给出逐测试的版本及其代价。
- 仅有有界性推不出有用的压缩：它保证的维数在混合测度下不比平凡维数小。

**直觉.** 两个 feedback law 可以在很多方向上不同，但 learner 和声明的测试把其中大多数方向变成很小的得分变化。response operator 精确地度量小到什么程度，奇异值按后果给这些方向排序，而不是按消息看起来差多少排序。

**在框架中的作用.**

- 它是 [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) 在明确度量下的对应物：那里最坏的隐藏差是一致逼近误差的两倍，不需要密度条件；这里度量由 $`(\rho,\omega)`$ 固定，最优接口可由奇异值分解算出。
- 它规定了在一个检查点上，压缩后的 feedback 接口必须保留什么。训练继续进行时，保留的方向是否仍有意义，是 [Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11) 的闭合问题；用有限次运行估计这些方向，是 [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 的内容。

**推导要点.**

- (a) 是输出模式上的 Parseval 恒等式；(b) 在 $`r`$ 处把和拆开，尾部用 Bessel 不等式。
- (c) 上界：投影到前 $`r`$ 个输入模式。下界：任意 $`r`$ 个线性统计量都会在 $`v_1,\dots,v_{r+1}`$ 张成的 $`r+1`$ 维空间中的某个单位向量 $`v`$ 上为零；输入 $`\pm Bv`$ 给出相同的统计量，所以解码器的输出与其中之一的距离至少为 $`\lVert\mathcal R(Bv)\rVert_\omega\ge B\sigma_{r+1}`$。
- (d) 是对 $`\tilde g_a`$ 在前几个模式之外的部分用 Cauchy–Schwarz；它的 $`\omega`$ 平均是截断算子的 Hilbert–Schmidt 范数。
- (e) 奇异值单调不增，由 (4.11) 得 $`(r+1)\sigma_{r+1}^2\le\sum_j\sigma_j^2\le\tfrac14`$。
- (f) 沿 (c) 中的 $`v`$ 扰动 $`\rho`$；只要 $`a\lvert v_y\rvert\le2`$ 就保持非负，而 $`\lVert v\rVert_\rho=1`$ 推出 $`\lvert v_y\rvert\le1/\sqrt{\rho_{\min}}`$。

**例示（解析）.** 取 $`J`$ 个落在不同消息上的点质量候选。此时 $`\rho`$ 是这 $`J`$ 个消息上的均匀分布，两个候选的密度差满足 $`\lVert h\rVert_\rho^2=2J`$。取 $`\varepsilon=0.3`$、$`J=20`$，(4.16) 要求 $`r+1\ge112`$ 个模式，而 19 个统计量（候选权重）本来就精确。有意义的问题是测得的谱，而不是一般界。

**说明.**

- (c) 的最优性针对的是线性统计量加任意解码器。它不是说每个压缩算法都需要 $`r`$ 个数：非线性编码器、有限精度编码和比特复杂度都不在其中。
- (c) 的有符号球比概率分布之差构成的集合大；(f) 表明 $`B`$ 小时两者的差距消失，代价是支撑覆盖因子 $`2\sqrt{\rho_{\min}}`$。
- 低秩是 $`(\rho,\omega)`$、声明的测试与检查点的性质。只有一个部署得分和一个 continuation 时，秩至多为 1，这没有科学内容。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8)：response operator、界 $`\sum_j\sigma_j^2\le\tfrac14`$ 与混合测度.
- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)：一致范数版本.

**被用于.**

- [Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)：算子经过下一步 predictive state 的分解.
- [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)：保留的模式定义信息–编译半径所用的 response 接口.
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)：从有限目录估计谱.
- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)：组合保证中的截断项.

## 相关工作

- (c) 是关于最佳低秩逼近的 Eckart–Young 定理 ([Eckart & Young, 1936](#ref-eckart1936approximation)) 的算子形式，应用在 response operator 上；它是经典结论的实例化，不是新的逼近结果。
- 按 $`\sigma_j`$ 排序是按后果增益排序，不是按频率。response 核 $`k(y,y')=\mathbb E_{a\sim\omega}\tilde g_a(y)\tilde g_a(y')`$ 定义 $`\mathcal R^*\mathcal R`$，可以有负的元素，所以它不是图的邻接矩阵。要有频率的解释，需要声明一个关于 $`\rho`$ 可逆的扰动核 $`N`$，取 $`L=I-N`$，其 Dirichlet 型为 $`\tfrac12\mathbb E_{Y\sim\rho,Y'\sim N(Y)}(\phi(Y)-\phi(Y'))^2`$；只有当 $`L`$ 与 $`\mathcal R^*\mathcal R`$ 可交换时，两者才有公共特征基。对 datamodels 的 Fourier 与 noise-stability 分析用的正是这类 Boolean 结构 ([Saunshi et al., 2023](#ref-saunshi2023understanding); [O'Donnell, 2014](#ref-odonnell2014analysis))；这里的增益谱是另一个对象。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-odonnell2014analysis"></a>

**Ryan O'Donnell (2014).** Analysis of Boolean Functions. Cambridge University Press. DOI [10.1017/CBO9781139814782](https://doi.org/10.1017/CBO9781139814782).

<a id="ref-saunshi2023understanding"></a>

**Nikunj Saunshi, Arushi Gupta, Mark Braverman, Sanjeev Arora (2023).** Understanding influence functions and datamodels via harmonic analysis. *11th International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=cxCEOSF99f](https://openreview.net/forum?id=cxCEOSF99f); arXiv [2210.01072](https://arxiv.org/abs/2210.01072).
