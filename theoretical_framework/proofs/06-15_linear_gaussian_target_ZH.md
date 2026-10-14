# Proposition 6.15 · 线性高斯目标：精确风险、白化 response 与单次查询收益

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-15_linear_gaussian_target_ZH.md) · [证明](06-15_linear_gaussian_target_ZH.md) · [实验](../experiments/06-15_linear_gaussian_target_ZH.md) · [English](06-15_linear_gaussian_target.md)

**本页证明什么.** 精确计算广义最小二乘的风险；用一族高斯先验说明在最坏情形下没有估计量能做得更好；特征值恒等式与单次查询公式都是矩阵恒等式。

**本页用到的设定与记号.**

- $`y=D\vartheta+\epsilon`$，$`\epsilon\sim\mathcal N(0,\Sigma)`$，$`\Sigma\succ0`$，$`D`$ 已知，$`\vartheta\in\mathbb R^d`$ 未知且不受限制。
- 目标 $`f(\vartheta)=C\vartheta\in\mathbb R^k`$，损失 $`\lVert x\rVert_W^2=x^\top Wx`$，$`W\succeq0`$。
- $`I=D^\top\Sigma^{-1}D`$，$`B=C^\top WC`$，$`I^+`$ 为 Moore–Penrose 伪逆。
- 一次查询加入独立观测 $`a^\top\vartheta+\varepsilon`$，$`\varepsilon\sim\mathcal N(0,1)`$，因而给 $`I`$ 加上 $`aa^\top`$。

**陈述（[Proposition 6.15](../statements/06-15_linear_gaussian_target_ZH.md#s6-15)）.** (a) 若 $`I\succ0`$，$`\hat f=CI^{-1}D^\top\Sigma^{-1}y`$ 具有常数风险 $`\mathrm{tr}(BI^{-1})=\sum_j\lambda_j`$（$`Bv_j=\lambda_jIv_j`$），这是 $`\mathbb R^d`$ 上的 minimax 风险。(b) 若 $`W^{1/2}C`$ 在 $`\ker I`$ 上为零，$`\hat f=CI^+D^\top\Sigma^{-1}y`$ 具有常数风险 $`\mathrm{tr}(BI^+)`$，这是 minimax 风险，且当 $`C\ker I=\lbrace0\rbrace`$ 时无偏；若某个 $`v\in\ker I`$ 使 $`W^{1/2}Cv\ne0`$，minimax 风险为 $`+\infty`$，而在包含沿这样的 $`v`$ 相差的两点的有界集合上有正下限。(c) 对 $`I\succ0`$，$`\mathcal R_I=W^{1/2}CI^{-1/2}`$ 的奇异值平方就是 $`\lambda_j`$。(d) 对 $`I\succ0`$，一次查询使 minimax 风险下降 $`\Delta(a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$。(e) 当 $`C=c^\top`$、$`W=1`$ 时，风险为 $`c^\top I^{-1}c`$。

**证明用到的结果.**

- [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)，只在 (e) 的比较中用到。

**用到的经典工具.** 高斯共轭；Bayes 风险是 minimax 风险的下界；Sherman–Morrison 公式；Le Cam 的两点论证（[Le Cam, 1973](#ref-lecam1973convergence)）。

**证明.**

**第 1 步（广义最小二乘的风险）.** 设 $`I\succ0`$，$`\hat\vartheta=I^{-1}D^\top\Sigma^{-1}y`$。则 $`\hat\vartheta-\vartheta=I^{-1}D^\top\Sigma^{-1}\epsilon`$ 均值为零，协方差为 $`I^{-1}D^\top\Sigma^{-1}\Sigma\Sigma^{-1}DI^{-1}=I^{-1}`$。所以对每个 $`\vartheta`$，

```math
\mathbb E\lVert C(\hat\vartheta-\vartheta)\rVert_W^2=\mathrm{tr}\big(WC\,I^{-1}C^\top\big)=\mathrm{tr}\big(C^\top WC\,I^{-1}\big)=\mathrm{tr}(BI^{-1}).
```

**第 2 步（下界）.** 取先验 $`\vartheta\sim\mathcal N(0,\tau^2I_d)`$。给定 $`y`$ 时 $`\vartheta`$ 的后验是协方差为 $`(I+\tau^{-2}I_d)^{-1}`$ 的高斯分布；对 $`C\vartheta`$ 上的损失 $`\lVert\cdot\rVert_W^2`$，后验均值 $`C\,\mathbb E[\vartheta\mid y]`$ 是 Bayes 估计，Bayes 风险为 $`\mathrm{tr}\big(B(I+\tau^{-2}I_d)^{-1}\big)`$。任何估计量的最坏风险都不小于它在任何先验下的平均风险，因而不小于每一个 Bayes 风险：

```math
\inf_{\tilde f}\sup_\vartheta\mathbb E\lVert\tilde f-C\vartheta\rVert_W^2\ \ge\ \sup_{\tau>0}\mathrm{tr}\big(B(I+\tau^{-2}I_d)^{-1}\big).
```

写 $`I=U\,\mathrm{diag}(\iota_1,\dots,\iota_d)\,U^\top`$。则 $`\mathrm{tr}(B(I+\varepsilon I_d)^{-1})=\sum_iu_i^\top Bu_i/(\iota_i+\varepsilon)`$，其中 $`u_i^\top Bu_i=\lVert W^{1/2}Cu_i\rVert^2\ge0`$。当 $`\varepsilon=\tau^{-2}\downarrow0`$ 时每一项都单调增加。若 $`I\succ0`$，极限为 $`\sum_iu_i^\top Bu_i/\iota_i=\mathrm{tr}(BI^{-1})`$；结合第 1 步，(a) 中的风险是 minimax 的。

**第 3 步（特征值恒等式）.** 对 $`I\succ0`$，$`\mathrm{tr}(BI^{-1})=\mathrm{tr}(I^{-1/2}BI^{-1/2})`$ 是对称矩阵 $`I^{-1/2}BI^{-1/2}`$ 的特征值之和。而 $`I^{-1/2}BI^{-1/2}w=\lambda w`$ 当且仅当 $`Bv=\lambda Iv`$，其中 $`v=I^{-1/2}w`$，所以这些特征值就是 $`(B,I)`$ 的广义特征值 $`\lambda_j`$。(a) 证毕。对 (c)，$`\mathcal R_I^\top\mathcal R_I=I^{-1/2}C^\top WCI^{-1/2}=I^{-1/2}BI^{-1/2}`$，它的特征值是 $`\mathcal R_I`$ 的奇异值平方，$`\lVert\mathcal R_I\rVert_{\mathrm{HS}}^2`$ 是它们之和。

**第 4 步（奇异信息）.** 设 $`P_0`$ 是到 $`\ker I`$ 的正交投影。由于 $`D^\top\Sigma^{-1}Dv=0`$ 蕴含 $`\lVert\Sigma^{-1/2}Dv\rVert^2=0`$，每个 $`v\in\ker I`$ 都有 $`Dv=0`$，所以 $`y`$ 的分布不依赖 $`P_0\vartheta`$。

- 设 $`W^{1/2}C`$ 在 $`\ker I`$ 上为零。对 $`\hat f=CI^+D^\top\Sigma^{-1}y`$，$`\mathbb E\hat f=CI^+I\vartheta=C(I_d-P_0)\vartheta`$，所以偏差 $`-CP_0\vartheta`$ 满足 $`\lVert CP_0\vartheta\rVert_W=\lVert W^{1/2}CP_0\vartheta\rVert=0`$；当 $`C\ker I=\lbrace0\rbrace`$ 时偏差本身为零。波动的协方差为 $`CI^+D^\top\Sigma^{-1}DI^+C^\top=CI^+II^+C^\top=CI^+C^\top`$。因此对每个 $`\vartheta`$，风险都是 $`\mathrm{tr}(WCI^+C^\top)=\mathrm{tr}(BI^+)`$。在第 2 步的和中，每个特征向量 $`u_i\in\ker I`$ 都有 $`u_i^\top Bu_i=\lVert W^{1/2}Cu_i\rVert^2=0`$，其余各项单调增加到 $`\sum_{\iota_i>0}u_i^\top Bu_i/\iota_i=\mathrm{tr}(BI^+)`$。所以这个风险是 minimax 的。
- 设某个单位向量 $`v\in\ker I`$ 使 $`W^{1/2}Cv\ne0`$。则 $`v^\top Bv>0`$；由于 $`(I+\varepsilon I_d)^{-1}`$ 在 $`v`$ 上的特征值为 $`1/\varepsilon`$，$`\mathrm{tr}(B(I+\varepsilon I_d)^{-1})\ge v^\top Bv/\varepsilon\to\infty`$。$`\mathbb R^d`$ 上的 minimax 风险为 $`+\infty`$。
- 在包含 $`\vartheta_0`$ 与 $`\vartheta_1=\vartheta_0+sv`$（$`v`$ 如上，$`s\ne0`$）的有界参数集上，这两点给出相同的 $`y`$ 分布。对任意估计量 $`\tilde f`$，三角不等式逐点给出 $`\lVert\tilde f-C\vartheta_0\rVert_W+\lVert\tilde f-C\vartheta_1\rVert_W\ge s\lVert W^{1/2}Cv\rVert`$，所以两个风险中较大者至少为 $`s^2\lVert W^{1/2}Cv\rVert^2/4`$：这是一个正下限。这就是总变差距离为零时的 Le Cam 两点论证。

**第 5 步（一次查询）.** 在 $`I\succ0`$ 时，Sherman–Morrison 公式给出

```math
(I+aa^\top)^{-1}=I^{-1}-\frac{I^{-1}aa^\top I^{-1}}{1+a^\top I^{-1}a},
```

所以 $`\mathrm{tr}(BI^{-1})-\mathrm{tr}\big(B(I+aa^\top)^{-1}\big)=\mathrm{tr}(BI^{-1}aa^\top I^{-1})/(1+a^\top I^{-1}a)=a^\top I^{-1}BI^{-1}a/(1+a^\top I^{-1}a)`$。对扩大后的实验再用一次 (a)，两个迹都是 minimax 风险。

**第 6 步（标量情形）.** 当 $`C=c^\top`$、$`W=1`$ 时，$`B=cc^\top`$，$`\mathrm{tr}(BI^{-1})=c^\top I^{-1}c`$。取 $`I=A(\xi)`$，这正是 [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4) 中对 design 求最小的准则。$`\square`$

**附注.**

- 第 4 步的下限用到 $`\max(R_0,R_1)\ge(R_0+R_1)/2`$ 与 $`X_0^2+X_1^2\ge(X_0+X_1)^2/2`$，其中 $`X_j=\lVert\tilde f-C\vartheta_j\rVert_W`$，两个期望都在 $`y`$ 的共同分布下计算；常数 $`1/4`$ 没有优化。

## 相关工作

- 迹准则、它的 A-、L- 与 c-optimal 特例，以及对奇异信息矩阵的伪逆处理，都是最优设计中的标准内容（[Pukelsheim, 1993](#ref-pukelsheim1993optimal)；[Elfving, 1952](#ref-elfving1952optimum)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).
