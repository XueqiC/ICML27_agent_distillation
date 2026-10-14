# Proposition 8.15 · 双保真 response 估计及其成本条件

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-15_two_fidelity_response_estimation_ZH.md) · [证明](08-15_two_fidelity_response_estimation_ZH.md) · [实验](../experiments/08-15_two_fidelity_response_estimation_ZH.md) · [English](08-15_two_fidelity_response_estimation.md)

**本页证明什么.** (a) 来自线性与独立性。(b) 是 Cauchy–Schwarz 分配。(c) 比较两个成本并解出相关系数门槛。(d) 计算嵌套估计量的方差并求最优。(e) 计数每格最少的完整运行次数。(f) 与 (g) 是 Hoeffding 界。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 一格（checkpoint、候选、$`p`$ 个测试与权重 $`\omega`$）；$`X\in[0,1]^p`$ 为完整续训输出，$`f=\mathbb EX`$；$`Y\in[0,1]^p`$ 为冻结的便宜估计器的输出。
- 第 0 组：$`Y_1,\dots,Y_{n_0}`$ 独立同分布；第 1 组：$`(X'_1,Y'_1),\dots,(X'_{n_1},Y'_{n_1})`$ 为独立同分布的耦合对，边缘分别与 $`X`$、$`Y`$ 相同；两组相互独立。
- $`v_0`$、$`v_1`$、$`v_H`$ 分别是 $`Y`$、耦合下的 $`X-Y`$、$`X`$ 的加权总方差；$`c_0`$、$`c_1`$、$`c_H`$ 分别是便宜样本、配对样本、完整样本的成本。

**陈述（[Proposition 8.15](../statements/08-15_two_fidelity_response_estimation_ZH.md#s8-15)）.** 对 $`\hat f=\bar Y+\overline{X'-Y'}`$：(a) $`\mathbb E\hat f=f`$，$`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$；(b) 均方误差 $`\varepsilon^2`$ 下的最小成本为 $`S^2/\varepsilon^2`$，$`S=\sqrt{v_0c_0}+\sqrt{v_1c_1}`$，在 $`n_\ell=(S/\varepsilon^2)\sqrt{v_\ell/c_\ell}`$ 处取到，取整至多多花 $`c_0+c_1`$；(c) 它比只用完整样本便宜当且仅当 $`\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H}`$，对 $`v_0=v_H`$、$`c_0=c_H/10`$ 的标量测试门槛为 $`\varrho>0.766`$（$`c_1=c_H`$）与 $`\varrho>0.787`$（$`c_1=1.1c_H`$）；(d) 取最优系数与最优分配的嵌套估计量相对成本为 $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$，小于 1 当且仅当 $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$；(e) 相对于 $`n_H`$ 次完整复现，节省至多 $`n_H`$ 倍；(f) 以至少 $`1-\delta_v`$ 的概率 $`\mathrm{Var}\,Z\le\hat v+\tfrac{L^2}2\sqrt{\log(1/\delta_v)/(2k)}`$；(g) 对 $`N`$ 个冻结的标量 response，以至少 $`1-\delta`$ 的概率同时有 $`\lvert\hat f-f\rvert\le\sqrt{\log(4N/\delta)/(2n_0)}+2\sqrt{\log(4N/\delta)/(2n_1)}`$。

*证明.*

**(a)** $`\mathbb E\bar Y=\mathbb EY`$；因为每个配对的边缘与 $`X`$、$`Y`$ 相同，$`\mathbb E\,\overline{X'-Y'}=\mathbb EX-\mathbb EY`$；两者之和为 $`f`$。写成 $`\hat f-f=A_0+A_1`$，$`A_0=\bar Y-\mathbb EY`$，$`A_1=\overline{X'-Y'}-\mathbb E(X-Y)`$。两组独立且两项均值为零，所以 $`\mathbb E\langle A_0,A_1\rangle_\omega=\langle\mathbb EA_0,\mathbb EA_1\rangle_\omega=0`$。每组内各项独立同分布，所以 $`\mathbb E\lVert A_0\rVert_\omega^2=v_0/n_0`$，$`\mathbb E\lVert A_1\rVert_\omega^2=v_1/n_1`$。各坐标之间可以任意相关；只用到加权总方差。

**(b)** 对正的 $`n_0,n_1`$，Cauchy–Schwarz 给出

```math
\Big(\sum_\ell c_\ell n_\ell\Big)\Big(\sum_\ell\frac{v_\ell}{n_\ell}\Big)\ge\Big(\sum_\ell\sqrt{c_\ell n_\ell}\sqrt{\frac{v_\ell}{n_\ell}}\Big)^2=S^2 .
```

若误差约束 $`\sum_\ell v_\ell/n_\ell\le\varepsilon^2`$ 成立，成本至少为 $`S^2/\varepsilon^2`$。在 $`n_\ell=(S/\varepsilon^2)\sqrt{v_\ell/c_\ell}`$ 处，$`\sum_\ell v_\ell/n_\ell=(\varepsilon^2/S)\sum_\ell\sqrt{v_\ell c_\ell}=\varepsilon^2`$，$`\sum_\ell c_\ell n_\ell=(S/\varepsilon^2)\sum_\ell\sqrt{v_\ell c_\ell}=S^2/\varepsilon^2`$。每个 $`n_\ell`$ 上取整不到一个样本，多花不到 $`c_\ell`$，误差只会更小。若 $`v_\ell=0`$，这一组仍需一个样本，因为它的均值未知。

**(c)** 在同样的连续分配口径下，只用完整样本达到均方误差 $`\varepsilon^2`$ 的成本为 $`v_Hc_H/\varepsilon^2`$。与 (b) 比较，双保真成本更小当且仅当 $`S^2<v_Hc_H`$，即当且仅当 $`\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H}`$。对 $`v_0=v_H`$、相关系数为 $`\varrho`$ 的标量测试，$`v_1=\mathrm{Var}(X-Y)=2v_H(1-\varrho)`$。两边除以 $`\sqrt{v_Hc_H}`$ 并取 $`c_0/c_H=0.1`$，条件为

```math
\sqrt{0.1}+\sqrt{2(1-\varrho)\,c_1/c_H}<1\iff\varrho>1-\frac{(1-\sqrt{0.1})^2}{2\,c_1/c_H} ,
```

$`c_1=c_H`$ 时为 $`0.7662`$，$`c_1=1.1c_H`$ 时为 $`0.7875`$。

**(d)** 令 $`\sigma_H^2=v_H`$、$`\sigma_L^2=v_0`$；$`\bar Y_{n_H}`$ 为与完整输出配对的前 $`n_H`$ 个便宜输出的均值，$`\bar Y_{n_L}`$ 为全部 $`n_L`$ 个的均值。因为 $`\mathbb E(\bar Y_{n_L}-\bar Y_{n_H})=0`$，对任意 $`\alpha`$ 估计量都无偏。把 $`\bar Y_{n_L}-\bar Y_{n_H}`$ 写成 $`\frac{n_L-n_H}{n_L}(\bar Y_{\rm rest}-\bar Y_{n_H})`$，其中 $`\bar Y_{\rm rest}`$ 是未配对输出的均值，得

```math
\mathrm{Var}(\bar Y_{n_L}-\bar Y_{n_H})=\sigma_L^2\Big(\frac1{n_H}-\frac1{n_L}\Big),\qquad\mathrm{Cov}(\bar X_{n_H},\bar Y_{n_L}-\bar Y_{n_H})=\varrho\sigma_H\sigma_L\Big(\frac1{n_L}-\frac1{n_H}\Big),
```

所以 $`\mathrm{Var}\hat f_\alpha=\sigma_H^2/n_H+(\alpha^2\sigma_L^2-2\alpha\varrho\sigma_H\sigma_L)(1/n_H-1/n_L)`$。它在 $`\alpha=\varrho\sigma_H/\sigma_L`$ 处最小，此时

```math
\mathrm{Var}\hat f_\alpha=\sigma_H^2\Big(\frac{1-\varrho^2}{n_H}+\frac{\varrho^2}{n_L}\Big).
```

成本为 $`c_Hn_H+c_Ln_L`$。与 (b) 同样用 Cauchy–Schwarz，成本乘方差至少为 $`\sigma_H^2\big(\sqrt{c_H(1-\varrho^2)}+\sqrt{c_L\varrho^2}\big)^2`$，在 $`n_L/n_H=\sqrt{c_H\varrho^2/(c_L(1-\varrho^2))}`$ 处取等号。这个比例不小于 1（嵌套所要求的）当且仅当 $`\varrho^2\ge c/(1+c)`$。除以只用完整样本时的值 $`\sigma_H^2c_H`$，得到相对成本 $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$。它小于 1 当且仅当 $`\sqrt{1-\varrho^2}<1-\lvert\varrho\rvert\sqrt c`$，即（两边为正）$`1-\varrho^2<1-2\lvert\varrho\rvert\sqrt c+\varrho^2c`$，即 $`\lvert\varrho\rvert\big((1+c)\lvert\varrho\rvert-2\sqrt c\big)>0`$，即 $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$。由于 $`c\le3`$ 时 $`(2\sqrt c/(1+c))^2=4c/(1+c)^2\ge c/(1+c)`$，只要有节省，最优比例就是可行的。$`c=0.1`$ 时门槛为 $`2\sqrt{0.1}/1.1=0.5750`$。若 $`\alpha`$ 由同一批样本估计，$`\hat f_{\hat\alpha}`$ 不再精确无偏；用独立的 pilot 冻结 $`\alpha`$ 可以恢复无偏性。

**(e)** 在 (a) 中，约束 $`n_1\ge1`$ 使成本至少为 $`c_1\ge c_H`$；在 (d) 中，$`n_H\ge1`$ 使成本至少为 $`c_H`$。每格 $`n_H`$ 次完整复现时，只用完整样本的成本是 $`n_Hc_H`$，所以比值至少为 $`1/n_H`$。

**(f)** 对独立副本，$`\mathbb E(Z_{2i}-Z_{2i-1})^2/2=\mathrm{Var}\,Z`$，且每项在 $`[0,L^2/2]`$ 中。对 $`k`$ 个取值范围为 $`L^2/2`$ 的独立同分布项用 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)），$`\Pr\big(\mathrm{Var}\,Z>\hat v+s\big)\le\exp(-2ks^2/(L^2/2)^2)`$；令右边等于 $`\delta_v`$ 即得。对向量的加权平方范数用同样的论证，就界住加权总方差。

**(g)** 对一个 response，$`\bar Y`$ 是 $`n_0`$ 个 $`[0,1]`$ 变量的平均，$`\overline{X'-Y'}`$ 是 $`n_1`$ 个 $`[-1,1]`$ 变量的平均。在水平 $`\delta'`$ 下的双侧 Hoeffding 界给出半径 $`\sqrt{\log(2/\delta')/(2n_0)}`$ 与 $`2\sqrt{\log(2/\delta')/(2n_1)}`$。这样的事件共 $`2N`$ 个；取 $`\delta'=\delta/(2N)`$，$`\log(2/\delta')=\log(4N/\delta)`$，union bound 给出它们同时成立的概率至少为 $`1-\delta`$。在该事件上 $`\lvert\hat f-f\rvert\le\lvert\bar Y-\mathbb EY\rvert+\lvert\overline{X'-Y'}-\mathbb E(X-Y)\rvert\le\eta`$。(a)、(f)、(g) 都要求代理、系数与分配在抽样之前固定。$`\square`$

## 相关工作

- (d) 是最优多保真 Monte Carlo（[Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)）的双模型情形；(a)–(c) 是多层 Monte Carlo（[Giles, 2015](#ref-giles2015multilevel)）的两层形式。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).
