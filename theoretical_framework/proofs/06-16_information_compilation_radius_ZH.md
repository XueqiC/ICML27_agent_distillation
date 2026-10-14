# Proposition 6.16 · 信息—编译半径

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-16_information_compilation_radius_ZH.md) · [证明](06-16_information_compilation_radius_ZH.md) · [实验](../experiments/06-16_information_compilation_radius_ZH.md) · [English](06-16_information_compilation_radius.md)

**本页证明什么.** 鲁棒半径的存在性与凸性、它的两个下界与半径为零的刻画、半径最优编译的保证，以及关于内层上确界的计算性陈述。(e) 的三种补救是 (b)、(c) 的直接解读，在最后说明。

**本页用到的设定与记号.**

**接口.** 有限个声明的测试 $`a`$，权重 $`\omega_a>0`$；$`g_a(y)=k_a+c_a^\top\phi(y)+e_a(y)`$，$`\phi(y)\in\mathbb R^r`$，$`\phi\in L^2(\rho)`$，$`C\in\mathbb R^{p\times r}`$ 的各行为 $`c_a^\top`$；$`\mu_P=\mathbb E_P\phi`$；$`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$。

**算子.** $`L^2_0(\rho)`$ 上的 $`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$ 与 $`(\mathcal Eh)(a)=\langle\tilde e_a,h\rangle_\rho`$，其中 $`\tilde g_a=g_a-\mathbb E_\rho g_a`$，$`\tilde e_a=e_a-\mathbb E_\rho e_a`$，$`\lVert\mathcal E\rVert_{\rm op}\le\epsilon_{\rm tail}`$（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)）。

**不确定集合与合法集合.** $`\mathcal M\subset\mathbb R^r`$ 非空紧，$`\Pr(\mu_P\in\mathcal M)\ge1-\delta`$；$`\mathcal K=\lbrace C\mu_Q:Q\in\mathcal Q_{\rm legal}\rbrace`$ 非空紧；$`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$，$`\mathfrak r=\min_{q\in\mathcal K}\Gamma(q)`$。

**陈述（[Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16)）.** (a) 最小值存在；$`\Gamma`$ 是凸的 1-Lipschitz 函数。(b) $`\mathfrak r\ge\max\lbrace\tfrac12\mathrm{diam}_\omega(C\mathcal M),\sup_{\mu\in\mathcal M}\mathrm{dist}_\omega(C\mu,\mathcal K)\rbrace`$。(c) $`\mathfrak r=0`$ 当且仅当 $`C\mathcal M`$ 是 $`\mathcal K`$ 中的单点。(d) 对最小点 $`q^*=C\mu_{Q^*}`$ 与 $`\lVert dP/d\rho-dQ^*/d\rho\rVert_\rho\le B`$，以至少 $`1-\delta`$ 的概率 $`\lVert\mathcal R(dP/d\rho-dQ^*/d\rho)\rVert_\omega\le\mathfrak r+B\epsilon_{\rm tail}`$；优化误差 $`\eta_{\rm opt}`$ 再加 $`\eta_{\rm opt}`$。(e) 三种补救。(f) 内层上确界在 $`\mathcal M`$ 的极点处取到；有限情景、单椭球与小盒子都易于计算，一般盒子是困难的。

**证明用到的结果.**

- [Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)：response operator 就是精确的得分差。

**用到的经典工具.** 紧性与连续性；三角不等式；凸函数在多胞形上的最大化；S-lemma（[Pólik & Terlaky, 2007](#ref-polik2007survey)）。

**证明.**

**(a)** 固定 $`\mu`$，$`q\mapsto\lVert C\mu-q\rVert_\omega`$ 是凸的，并且关于 $`\lVert\cdot\rVert_\omega`$ 1-Lipschitz。凸 1-Lipschitz 函数的逐点上确界在有限处仍是凸的 1-Lipschitz 函数；由于 $`\mathcal M`$ 紧，$`\Gamma(q)\le\sup_{\mu\in\mathcal M}\lVert C\mu\rVert_\omega+\lVert q\rVert_\omega<\infty`$。连续函数在非空紧集 $`\mathcal K`$ 上取到最小值。若 $`\mathcal K`$ 是凸集，在它上面最小化凸函数 $`\Gamma`$ 就是凸规划。

**(b)** 固定 $`q\in\mathcal K`$。对 $`\mu,\mu'\in\mathcal M`$，$`\lVert C\mu-C\mu'\rVert_\omega\le\lVert C\mu-q\rVert_\omega+\lVert C\mu'-q\rVert_\omega\le2\Gamma(q)`$；对这一对取上确界，得 $`\Gamma(q)\ge\tfrac12\mathrm{diam}_\omega(C\mathcal M)`$。又因 $`q\in\mathcal K`$，对每个 $`\mu`$ 有 $`\lVert C\mu-q\rVert_\omega\ge\mathrm{dist}_\omega(C\mu,\mathcal K)`$，所以 $`\Gamma(q)\ge\sup_\mu\mathrm{dist}_\omega(C\mu,\mathcal K)`$。两个界对每个 $`q\in\mathcal K`$ 都成立，因而对最小值也成立。

**(c)** 若 $`C\mathcal M=\lbrace x\rbrace`$ 且 $`x\in\mathcal K`$，则 $`\Gamma(x)=0`$。反之，若 $`\mathfrak r=0`$，最小点 $`q^*\in\mathcal K`$ 对每个 $`\mu\in\mathcal M`$ 都满足 $`\lVert C\mu-q^*\rVert_\omega=0`$；由于每个测试都有 $`\omega_a>0`$，$`\lVert\cdot\rVert_\omega`$ 是范数，所以对每个 $`\mu`$ 都有 $`C\mu=q^*`$。

**(d)** 令 $`h=dP/d\rho-dQ^*/d\rho\in L^2_0(\rho)`$。由于 $`\mathbb E_\rho h=0`$，常数在与 $`h`$ 的每个内积中都消失，所以

```math
(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho=c_a^\top\langle\phi,h\rangle_\rho+\langle\tilde e_a,h\rangle_\rho=c_a^\top(\mu_P-\mu_{Q^*})+(\mathcal Eh)(a),
```

这里用到 $`\langle\phi,h\rangle_\rho=\mathbb E_P\phi-\mathbb E_{Q^*}\phi`$。于是 $`\mathcal Rh=(C\mu_P-q^*)+\mathcal Eh`$。在概率至少为 $`1-\delta`$ 的事件 $`\mu_P\in\mathcal M`$ 上，$`\lVert C\mu_P-q^*\rVert_\omega\le\Gamma(q^*)=\mathfrak r`$。而 $`\lVert\mathcal Eh\rVert_\omega\le\epsilon_{\rm tail}\lVert h\rVert_\rho\le\epsilon_{\rm tail}B`$ 总是成立。三角不等式给出所述的界。若只有近似最小点 $`\hat q`$，用 $`\Gamma(\hat q)\le\mathfrak r+\eta_{\rm opt}`$ 代替 $`\mathfrak r`$。

**(e)** 若 $`\mathcal M`$ 只沿 $`\ker C`$ 变化，$`C\mathcal M`$ 是一个点，由 (b)–(c) 半径就是这个点到 $`\mathcal K`$ 的距离；沿 $`\ker C`$ 缩小 $`\mathcal M`$ 不改变 $`C\mathcal M`$，因而 $`\Gamma`$ 与 $`\mathfrak r`$ 都不变。若 $`C\mathcal M`$ 的直径为 $`D`$，(b) 给出对每个合法集合都有 $`\mathfrak r\ge D/2`$，所以只有更小的 $`C\mathcal M`$，即能区分后果方向的证据，才能降低这一部分。若 $`C\mathcal M=\lbrace x\rbrace`$ 而 $`x\notin\mathcal K`$，由 $`\mathcal K`$ 的紧性，$`\mathfrak r=\mathrm{dist}_\omega(x,\mathcal K)>0`$；更多证据无法让一个点再缩小，只有更大的 $`\mathcal K`$ 能降低半径。

**(f)** 固定 $`q`$，$`\mu\mapsto\lVert C\mu-q\rVert_\omega`$ 是凸函数，而紧凸集上的凸函数在极点处取到最大值，所以对 $`\mathcal M`$ 的上确界等于对其凸包极点的上确界。

- *有限情景* $`\mathcal M=\lbrace\mu_1,\dots,\mu_s\rbrace`$：当 $`\mathcal K`$ 是多胞形时，在约束 $`\lVert C\mu_j-q\rVert_\omega\le t`$（对所有 $`j`$）与 $`q\in\mathcal K`$ 下最小化 $`t`$ 是二阶锥规划。
- *单个椭球* $`\mathcal M=\lbrace\mu:(\mu-m_0)^\top S^{-1}(\mu-m_0)\le1\rbrace`$：内层问题是在一个二次约束下最大化二次函数。由 S-lemma，$`\Gamma(q)^2\le t`$ 当且仅当存在 $`\lambda\ge0`$，使得对每个 $`\mu`$ 都有 $`t-\lVert C\mu-q\rVert_\omega^2-\lambda\big(1-(\mu-m_0)^\top S^{-1}(\mu-m_0)\big)\ge0`$；对 $`q`$ 做 Schur 补之后，这是关于 $`(t,\lambda)`$ 的线性矩阵不等式，外层问题成为半定规划。
- *盒子*：最大值在它的 $`2^r`$ 个顶点之一取到，$`r`$ 较小时可以枚举。一般情形是困难的：对 Laplacian 为 $`L\succeq0`$ 的图，maximum cut 等于 $`\max_{x\in\lbrace-1,1\rbrace^n}x^\top Lx/4`$，而它就是凸二次函数 $`x^\top Lx/4`$ 在盒子 $`[-1,1]^n`$ 上的最大值，因为这个最大值在顶点取到。取 $`r=n`$、满足 $`C^\top\mathrm{diag}(\omega)C=L/4`$ 的 $`C`$、$`\mathcal K=\lbrace0\rbrace`$ 与 $`\mathcal M=[-1,1]^n`$，$`\Gamma(0)^2`$ 恰好就是 maximum cut 的值。$`\square`$

**附注.**

- (d) 中的界是逐 checkpoint、加权范数下的；除以 $`\sqrt{\omega_a}`$ 就化为单个测试 $`a`$ 的界，代价是权重小的测试得到很大的因子。
- 当 $`C`$ 本身是估计出来的时候，同样的论证适用于 $`(C,\mu)`$ 的联合不确定集合，但这时 $`\Gamma`$ 不再只是 $`C\mu`$ 的函数，(a) 中的凸性必须重新检查。

## 相关工作

- 在可行集上最小化最坏情形距离是鲁棒逼近问题，也就是 Chebyshev 中心问题；S-lemma 及其在单个二次约束下的用法，综述见 [Pólik & Terlaky, 2007](#ref-polik2007survey)。
- (b) 中的直径界是 [Le Cam, 1973](#ref-lecam1973convergence) 两点法的确定性核心：证据无法排除的两个目标，迫使其中之一上的误差至少为两者距离的一半。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-polik2007survey"></a>

**Imre Pólik, Tamás Terlaky (2007).** A survey of the S-lemma. *SIAM Review* 49, no. 3, 371–418. DOI [10.1137/S003614450444614X](https://doi.org/10.1137/S003614450444614X).
