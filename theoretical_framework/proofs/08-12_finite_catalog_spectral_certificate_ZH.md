# Proposition 8.12 · 有限目录的谱证书

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-12_finite_catalog_spectral_certificate_ZH.md) · [证明](08-12_finite_catalog_spectral_certificate_ZH.md) · [实验](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md) · [English](08-12_finite_catalog_spectral_certificate.md)

**本页证明什么.** (a) 是逐格 Hoeffding 界加 union bound。(b) 利用误差矩阵的加权 Frobenius 范数不超过最大单格误差，再用 Weyl 不等式。(c) 是三项拆分。(d) 把估计的行展开到估计模式上，并为格的估计误差付费。(e) 是均方误差的正交分解。(f) 是计数。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 一个 checkpoint $`(t,z)`$；$`p`$ 个测试，权重 $`w_i>0`$ 且和为 1；$`m`$ 个冻结的候选 feedback law，权重 $`q_j>0`$ 且和为 1；测试中有 $`p_{\rm train}`$ 个不同的续训协议。
- $`G_{ij}\in[0,1]`$ 是用候选 $`j`$ 走一轮、再接测试 $`i`$ 的续训之后，测试 $`i`$ 的期望最终分数；$`\hat G_{ij}`$ 是 $`n`$ 次复现的平均，复现之间独立。
- $`M=D_{\sqrt w}GD_{\sqrt q}\Pi_q`$，$`\Pi_q=I-\sqrt q\sqrt q^{\top}`$；$`\hat M`$ 由 $`\hat G`$ 同样构造；奇异值 $`\sigma_k`$、$`\hat\sigma_k`$；$`\hat\Pi_r`$ 投影到 $`\hat M`$ 的前 $`r`$ 个右奇异向量 $`\hat v_1,\dots,\hat v_r`$ 上。
- 对候选上的分布 $`P,Q`$：$`h_j=(P_j-Q_j)/q_j`$，$`x=D_{\sqrt q}h`$，$`a_i=\sum_j(P_j-Q_j)G_{ij}`$，$`\lVert a\rVert_\omega^2=\sum_iw_ia_i^2`$。
- $`\hat g_i`$ 是 $`\hat G`$ 的第 $`i`$ 行，在 $`q`$ 下中心化；$`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$；$`\hat b_k(i)=\langle\hat g_i,\hat\phi_k\rangle_q`$；$`\hat e_r(i)=\lVert\hat g_i-\sum_{k\le r}\hat b_k(i)\hat\phi_k\rVert_q`$；$`\alpha_k=\sum_j(P_j-Q_j)\hat\phi_k(j)`$。

**陈述（[Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）.** 令 $`\varepsilon_n=\sqrt{\log(2pm/\delta)/(2n)}`$：(a) 以至少 $`1-\delta`$ 的概率所有格满足 $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$；在该事件上 (b) $`\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n`$ 且 $`\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n`$；(c) $`\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}+\varepsilon_n\le\sigma_{r+1}+2\varepsilon_n`$；(d) 若 $`\lVert h\rVert_q\le B`$，则 $`\lvert a_i\rvert\le\lvert\sum_{k\le r}\hat b_k(i)\alpha_k\rvert+B\hat e_r(i)+2\varepsilon_n`$，当 $`\alpha_k`$ 换成误差至多 $`\eta_k`$ 的估计时再加 $`\sum_k\lvert\hat b_k(i)\rvert\eta_k`$；(e) 设 $`M`$ 精确的奇异三元组 $`(\sigma_k,u_k,v_k)`$ 已知，$`\phi_k=D_{\sqrt q}^{-1}v_k`$，$`\hat\alpha_k`$ 为分别从 $`P`$、$`Q`$ 独立抽取 $`n'`$ 次的样本均值之差，$`f=Mx`$，$`\hat f_r=\sum_{k\le r}\sigma_k\hat\alpha_ku_k`$，则

```math
\mathbb E\,\lVert\hat f_r-f\rVert_2^2=\sum_{k\le r}\frac{\sigma_k^2}{n'}\big(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k\big)+\sum_{k>r}\sigma_k^2\alpha_k^2 ;
```

若改用一个独立冻结的估计 $`\hat M`$ 的三元组，方差项加帽，偏差项为 $`\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$；(f) 完整目录需要 $`m\,n\,p_{\rm train}`$ 次续训。

*证明.*

**第 0 步：$`M`$ 表示 response operator.** 对满足 $`\sum_jq_jh_j=0`$ 的 $`x=D_{\sqrt q}h`$，有 $`\sqrt q^{\top}x=0`$，所以 $`\Pi_qx=x`$，并且

```math
(Mx)_i=\sqrt{w_i}\sum_jG_{ij}\sqrt{q_j}\,x_j=\sqrt{w_i}\sum_jG_{ij}(P_j-Q_j)=\sqrt{w_i}\,a_i .
```

因此 $`\lVert Mx\rVert_2=\lVert a\rVert_\omega`$，$`\lVert x\rVert_2^2=\sum_jq_jh_j^2=\lVert h\rVert_q^2`$。又因每行取值于 $`[0,1]`$，$`\lVert M\rVert_{\rm HS}^2=\sum_iw_i\,\mathrm{Var}_q(G_{i\cdot})\le\tfrac14`$。

**(a)** 对固定的一格，$`n`$ 次复现的结果独立、取值于 $`[0,1]`$，均值为 $`G_{ij}`$（候选之间或同一次运行读出的测试之间的相关性，不影响单个格）。Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）给出 $`\Pr(\lvert\hat G_{ij}-G_{ij}\rvert>s)\le2e^{-2ns^2}`$。取 $`s=\varepsilon_n`$，右边为 $`\delta/(pm)`$；对 $`pm`$ 个格取 union bound 即得 (a)。

**(b)** 令 $`E=\hat G-G`$，则 $`\hat M-M=D_{\sqrt w}ED_{\sqrt q}\Pi_q`$。在 (a) 的事件上，

```math
\lVert D_{\sqrt w}ED_{\sqrt q}\rVert_{\rm F}^2=\sum_{i,j}w_iq_jE_{ij}^2\le\varepsilon_n^2\sum_iw_i\sum_jq_j=\varepsilon_n^2 .
```

算子范数不超过 Frobenius 范数，且 $`\lVert\Pi_q\rVert_{\rm op}=1`$，所以 $`\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n`$。奇异值的 Weyl 不等式 $`\lvert\sigma_k(A+E')-\sigma_k(A)\rvert\le\lVert E'\rVert_{\rm op}`$ 给出第二个结论。

**(c)** 写成 $`M(I-\hat\Pi_r)=\hat M(I-\hat\Pi_r)+(M-\hat M)(I-\hat\Pi_r)`$。由 $`\hat M`$ 的奇异值分解，$`\lVert\hat M(I-\hat\Pi_r)\rVert_{\rm op}=\hat\sigma_{r+1}`$。第二项的范数至多为 $`\lVert M-\hat M\rVert_{\rm op}\lVert I-\hat\Pi_r\rVert_{\rm op}\le\varepsilon_n`$。最后由 (b)，$`\hat\sigma_{r+1}\le\sigma_{r+1}+\varepsilon_n`$。整个过程没有用到 $`\sigma_r`$ 与 $`\sigma_{r+1}`$ 之间的间隙。

**(d)** 令 $`\hat a_i=\sum_j(P_j-Q_j)\hat G_{ij}`$。在 (a) 的事件上，

```math
\lvert a_i-\hat a_i\rvert\le\sum_j\lvert P_j-Q_j\rvert\,\lvert G_{ij}-\hat G_{ij}\rvert\le2\varepsilon_n ,
```

因为两个概率向量满足 $`\sum_j\lvert P_j-Q_j\rvert\le2`$。由于 $`\sum_jq_jh_j=0`$，对行中心化不改变内积：$`\hat a_i=\sum_jq_jh_j\hat G_{ij}=\langle\hat g_i,h\rangle_q`$。分解 $`\hat g_i=\sum_{k\le r}\hat b_k(i)\hat\phi_k+\hat\varrho_i`$，其中 $`\lVert\hat\varrho_i\rVert_q=\hat e_r(i)`$。因为 $`\langle\hat\phi_k,h\rangle_q=\sum_jq_j\hat\phi_k(j)h_j=\sum_j(P_j-Q_j)\hat\phi_k(j)=\alpha_k`$，

```math
\hat a_i=\sum_{k\le r}\hat b_k(i)\alpha_k+\langle\hat\varrho_i,h\rangle_q,\qquad\lvert\langle\hat\varrho_i,h\rangle_q\rvert\le\hat e_r(i)\,B
```

（Cauchy–Schwarz）。与第一个式子合起来就是所述的界。若把 $`\alpha_k`$ 换成满足 $`\lvert\hat\alpha_k-\alpha_k\rvert\le\eta_k`$ 的 $`\hat\alpha_k`$，则 $`\lvert\sum_k\hat b_k(i)\alpha_k\rvert\le\lvert\sum_k\hat b_k(i)\hat\alpha_k\rvert+\sum_k\lvert\hat b_k(i)\rvert\eta_k`$。

**(e)** 由第 0 步与奇异值分解，$`f=Mx=\sum_k\sigma_k\langle v_k,x\rangle u_k`$，且 $`\langle v_k,x\rangle=\sum_jv_{kj}\sqrt{q_j}h_j=\sum_j(P_j-Q_j)\phi_k(j)=\alpha_k`$。于是

```math
\hat f_r-f=\sum_{k\le r}\sigma_k(\hat\alpha_k-\alpha_k)u_k-\sum_{k>r}\sigma_k\alpha_ku_k .
```

$`u_k`$ 两两正交且为单位向量，所以平方范数等于各系数平方之和；无论 $`\hat\alpha_k`$ 之间如何相关，交叉项都消失。每个 $`\hat\alpha_k`$ 是 $`\phi_k`$ 的两个独立样本均值之差，无偏，方差为 $`(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k)/n'`$。取期望即得公式。模式与这些抽样无关，这正是 $`\hat\alpha_k`$ 对固定的 $`\alpha_k`$ 无偏的原因。

*冻结的估计三元组.* 设 $`(\hat\sigma_k,\hat u_k,\hat v_k)`$ 是一个与抽样独立、已固定的估计 $`\hat M`$ 的奇异三元组，$`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$，$`\hat\alpha_k`$ 为 $`\hat\phi_k`$ 的样本均值之差，$`\hat f_r=\sum_{k\le r}\hat\sigma_k\hat\alpha_k\hat u_k`$。同上，$`\mathbb E\hat\alpha_k=\langle\hat v_k,x\rangle`$，所以 $`\mathbb E\hat f_r=\hat M\hat\Pi_rx`$；而 $`\hat f_r-\mathbb E\hat f_r=\sum_{k\le r}\hat\sigma_k(\hat\alpha_k-\mathbb E\hat\alpha_k)\hat u_k`$ 均值为零，由 $`\hat u_k`$ 的正交性，其平方范数的期望为 $`\sum_{k\le r}\hat\sigma_k^2(\mathrm{Var}_P\hat\phi_k+\mathrm{Var}_Q\hat\phi_k)/n'`$。与确定性偏差的交叉项期望为零，所以均方误差等于这个方差加上 $`\lVert\hat M\hat\Pi_rx-Mx\rVert_2^2=\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$。只有 $`\hat M=M`$ 时它才等于第一个公式；一般情形下偏差不是 $`\sum_{k>r}\sigma_k^2\alpha_k^2`$。

**(f)** 候选 $`j`$ 在一个续训协议下的一次运行，给出从该协议读出的全部测试，所以第 $`j`$ 列的一次复现需要 $`p_{\rm train}`$ 次运行；$`m`$ 列、$`n`$ 次复现共需 $`m\,n\,p_{\rm train}`$ 次。$`\square`$

## 相关工作

- (b)–(c) 是 Eckart–Young 定理（[Eckart & Young, 1936](#ref-eckart1936approximation)）的标准扰动推论；逐格半径来自 Hoeffding（[Hoeffding, 1963](#ref-hoeffding1963probability)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
