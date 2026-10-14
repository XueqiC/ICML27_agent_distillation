# Proposition 8.13 · 跨 checkpoint 共享的消息特征

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-13_shared_features_across_checkpoints_ZH.md) · [证明](08-13_shared_features_across_checkpoints_ZH.md) · [实验](../experiments/08-13_shared_features_across_checkpoints_ZH.md) · [English](08-13_shared_features_across_checkpoints.md)

**本页证明什么.** (a) 把逐 checkpoint 的 Hilbert–Schmidt 界积分，并把结果识别为堆叠谱的尾部。(b) 是堆叠算子上的 Ky Fan 极大值原理。(c) 是一个用有界分数给出的显式构造。(d) 是测度变换。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 共同消息空间上的一个参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$；测试 $`a\in\mathcal A`$ 与权重 $`\omega`$，在每个 checkpoint 上含义相同；checkpoint 权重 $`\nu`$。
- $`\mathcal R_z:\mathcal H\to L^2(\omega)`$ 是 checkpoint $`z`$ 处的 response operator（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)）；$`h_z\in\mathcal H`$ 是 $`z`$ 处的实际反馈差。
- $`(\mathcal Sh)(z,a)=(\mathcal R_zh)(a)`$ 是映到 $`L^2(\nu\otimes\omega)`$ 的堆叠算子，奇异值为 $`\sigma_j(\mathcal S)`$，$`\Pi_r`$ 投影到它的前 $`r`$ 个右奇异向量上。假设映射 $`(z,h)\mapsto\mathcal R_zh`$ 联合可测。

**陈述（[Proposition 8.13](../statements/08-13_shared_features_across_checkpoints_ZH.md#s8-13)）.** (a) 若 $`\nu`$-几乎处处 $`\lVert h_z\rVert_\rho\le B`$，则 $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$。(b) $`\Pi_r`$ 在秩 $`r`$ 投影中使 $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2`$ 最小，最小值为 $`\sum_{j>r}\sigma_j(\mathcal S)^2`$。(c) 存在两个等权 checkpoint、有界分数与满足 $`\lVert h_z\rVert_\rho=B`$ 的输入，使得取 $`r=0`$ 时 (a) 的左边为 $`B^2c^2`$，$`B^2\sigma_1(\mathcal S)^2=B^2c^2/2`$，尾部之和为 $`B^2c^2`$。(d) 若实际访问的分布 $`d`$ 满足 $`dd/d\nu\le\kappa`$，则在 $`d`$ 下左边至多为 $`\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$。

*证明.*

**第 1 步：堆叠的 Hilbert–Schmidt 范数.** 设 $`A`$ 是 $`\mathcal H`$ 上的有界算子，$`(e_k)`$ 是 $`\mathcal H`$ 的一组正交基（标准可测空间上 $`\mathcal H`$ 可分）。由 Hilbert–Schmidt 范数的定义与 Tonelli 定理，

```math
\lVert\mathcal SA\rVert_{\rm HS}^2=\sum_k\lVert\mathcal SAe_k\rVert_{L^2(\nu\otimes\omega)}^2=\sum_k\mathbb E_\nu\lVert\mathcal R_zAe_k\rVert_\omega^2=\mathbb E_\nu\lVert\mathcal R_zA\rVert_{\rm HS}^2 .
```

取 $`A=I`$ 得 $`\lVert\mathcal S\rVert_{\rm HS}^2=\mathbb E_\nu\lVert\mathcal R_z\rVert_{\rm HS}^2\le\tfrac14`$，所以 $`\mathcal S`$ 是 Hilbert–Schmidt 算子，有奇异值分解 $`\mathcal S=\sum_j\sigma_j(\mathcal S)\,u_j\langle v_j,\cdot\rangle_\rho`$。

**(a)** 对每个 $`z`$，$`\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega\le\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm op}\lVert h_z\rVert_\rho\le\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm HS}\,B`$。平方后对 $`\nu`$ 积分，再对 $`A=I-\Pi_r`$ 用第 1 步：

```math
\mathbb E_\nu\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2\le B^2\,\lVert\mathcal S(I-\Pi_r)\rVert_{\rm HS}^2=B^2\sum_{j>r}\sigma_j(\mathcal S)^2 ,
```

最后的等号是因为 $`\mathcal S(I-\Pi_r)=\sum_{j>r}\sigma_j(\mathcal S)u_j\langle v_j,\cdot\rangle_\rho`$。

**(b)** 对秩 $`r`$ 的正交投影 $`\Pi`$，第 1 步给出 $`\mathbb E_\nu\lVert\mathcal R_z(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2`$。由于 $`\Pi`$ 与 $`I-\Pi`$ 是互补的正交投影，

```math
\lVert\mathcal S(I-\Pi)\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\lVert\mathcal S\Pi\rVert_{\rm HS}^2=\lVert\mathcal S\rVert_{\rm HS}^2-\mathrm{tr}\big(\Pi\,\mathcal S^*\mathcal S\,\Pi\big).
```

$`\mathcal S^*\mathcal S`$ 是紧、自伴、正的算子，特征值为 $`\sigma_j(\mathcal S)^2`$。由 Ky Fan 极大值原理，迹在秩 $`r`$ 投影上的最大值为 $`\sum_{j\le r}\sigma_j(\mathcal S)^2`$，在 $`\Pi_r`$ 处取到。所以最小值是 $`\lVert\mathcal S\rVert_{\rm HS}^2-\sum_{j\le r}\sigma_j(\mathcal S)^2=\sum_{j>r}\sigma_j(\mathcal S)^2`$。目标是 $`\nu`$-平均；关于最差的 checkpoint 不作任何断言。

**(c)** 取三条消息、$`\rho`$ 均匀，于是 $`\mathcal H`$ 是二维的；取 $`\mathcal H`$ 的正交基 $`e_1,e_2`$，令 $`\lVert e\rVert_\infty=\max_k\max_y\lvert e_k(y)\rvert`$。两个 checkpoint $`z_1,z_2`$ 各占权重 $`\tfrac12`$，只有一个测试、$`\omega=1`$，分数为 $`g_{z_k}(y)=\tfrac12+c\,e_k(y)`$；当 $`0<c\le1/(2\lVert e\rVert_\infty)`$ 时它们落在 $`[0,1]`$ 中。于是 $`\mathcal R_{z_k}h=\langle g_{z_k}-\tfrac12,h\rangle_\rho=c\langle e_k,h\rangle_\rho`$。

- 输入：当 $`B\le2/\lVert e\rVert_\infty`$ 时，$`P_k=\rho(1+\tfrac B2e_k)`$、$`Q_k=\rho(1-\tfrac B2e_k)`$ 是概率分布（非负，且 $`e_k`$ 在 $`\rho`$ 下均值为零），且 $`h_{z_k}=Be_k`$，$`\lVert h_{z_k}\rVert_\rho=B`$。
- 取 $`r=0`$，$`\Pi_0=0`$，(a) 的左边为 $`\tfrac12(c^2B^2+c^2B^2)=B^2c^2`$。
- 在坐标 $`(e_1,e_2)`$ 与 $`L^2(\nu\otimes\omega)\cong\mathbb R^2`$（吸收权重 $`\tfrac12`$）下，$`\mathcal S`$ 是矩阵 $`\tfrac{c}{\sqrt2}I_2`$，所以 $`\sigma_1(\mathcal S)^2=\sigma_2(\mathcal S)^2=c^2/2`$：第一个被丢弃的奇异值给出 $`B^2c^2/2`$，尾部之和给出 $`B^2c^2`$，后者被取到。
- 在允许的范围 $`0<c\le1/(2\lVert e\rVert_\infty)`$、$`B\le2/\lVert e\rVert_\infty`$ 内，两个界之比恒为 2，与 $`c`$、$`B`$ 的取值无关。
- 用 $`K`$ 个 checkpoint 和 $`K+1`$ 条消息做同样的构造，得到 $`\sigma_1(\mathcal S)^2=c^2/K`$，而实际值是 $`c^2`$：错误的界与正确的界相差 $`K`$ 倍。

**(d)** 对任意非负可测 $`f`$，$`\mathbb E_df=\mathbb E_\nu[f\,dd/d\nu]\le\kappa\,\mathbb E_\nu f`$。令 $`f(z)=\lVert\mathcal R_z(I-\Pi_r)h_z\rVert_\omega^2`$、$`g(z)=B^2\lVert\mathcal R_z(I-\Pi_r)\rVert_{\rm HS}^2`$。如 (a) 所示，$`\nu`$-几乎处处 $`f\le g`$；因为 $`d\ll\nu`$，$`d`$-几乎处处也成立。于是由第 1 步，$`\mathbb E_df\le\mathbb E_dg\le\kappa\,\mathbb E_\nu g=\kappa B^2\sum_{j>r}\sigma_j(\mathcal S)^2`$。$`\square`$

## 相关工作

- (b) 是 Eckart–Young 定理（[Eckart & Young, 1936](#ref-eckart1936approximation)）用于堆叠的 Hilbert–Schmidt 算子；Ky Fan 原理是它的迹形式。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).
