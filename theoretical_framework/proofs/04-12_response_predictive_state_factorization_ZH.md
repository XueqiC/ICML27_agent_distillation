# Proposition 4.12 · response 经过下一步 predictive state 分解

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-12_response_predictive_state_factorization_ZH.md) · [证明](04-12_response_predictive_state_factorization_ZH.md) · [实验](../experiments/04-12_response_predictive_state_factorization_ZH.md) · [English](04-12_response_predictive_state_factorization.md)

**本页证明什么.** (a) 是以消息为条件，加上 $`h`$ 均值为零的性质。(b) 是 $`C`$ 的单射性。(c) 用不变的 kernel 传递相等的 predictive state。(d) 是对残差部分用 Cauchy–Schwarz。

**本页用到的设定.**

- $`H=L_0^2(\rho)`$；测试 $`a=1,\dots,p`$，权重 $`\omega_a>0`$；$`(\mathcal R_zh)(a)=\langle\tilde g_a,h\rangle_\rho`$（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)）。
- $`g_a(y)=\mathbb E[V_a(U_t(z,y,\xi_t))\mid z,y]`$；有界特征 $`\psi`$，$`V_a=k_a+c_a^\top\psi`$（(d) 中再加 $`e_a`$）；$`C`$ 的行为 $`c_a^\top`$；$`\Gamma_z(y)=\mathbb E[\psi(U_t(z,y,\xi_t))\mid z,y]`$；$`\mathcal T_zh=\mathbb E_\rho[\Gamma_zh]`$。

**陈述（[Proposition 4.12](../statements/04-12_response_predictive_state_factorization_ZH.md#s4-12)）.** (a) $`\mathcal R_z=C\mathcal T_z`$。(b) 若 $`\mathrm{rank}\,C=d`$，则 $`\ker\mathcal R_z=\ker\mathcal T_z`$，因而声明 response 相等等价于 $`\mathbb E_P\Gamma_z=\mathbb E_Q\Gamma_z`$。(c) 在声明 continuation 的 kernel（或观察算子）下不变时，$`\mathbb E\Gamma_z`$ 相等推出此后声明的期望相等。(d) 有残差时 $`\mathcal R_z=C\mathcal T_z+\mathcal E_z`$，且 $`\lVert\mathcal E_zh\rVert_\omega\le\epsilon_{\rm pred}\lVert h\rVert_\rho`$。

**证明.**

**(a)** 由条件期望的塔性质与线性性，$`g_a(y)=\mathbb E[k_a+c_a^\top\psi(U_t(z,y,\xi_t))\mid z,y]=k_a+c_a^\top\Gamma_z(y)`$。对 $`h\in H`$ 有 $`\mathbb E_\rho h=0`$，所以常数与 $`h`$ 正交，

```math
(\mathcal R_zh)(a)=\langle\tilde g_a,h\rangle_\rho=\mathbb E_\rho[g_ah]=k_a\mathbb E_\rho h+c_a^\top\mathbb E_\rho[\Gamma_zh]=c_a^\top\mathcal T_zh .
```

$`\psi`$ 有界，所以 $`\Gamma_z`$ 有界，$`\mathcal T_zh`$ 对 $`h\in L^2(\rho)`$ 有定义。

**(b)** 若 $`\mathcal T_zh=0`$，则 $`\mathcal R_zh=C\mathcal T_zh=0`$。反过来，若 $`\mathcal R_zh=0`$，则 $`C(\mathcal T_zh)=0`$；列满秩意味着 $`Cx=0\Rightarrow x=0`$，所以 $`\mathcal T_zh=0`$。对密度 $`p,q\in L^2(\rho)`$ 的分布，取 $`h=p-q`$：$`(\mathcal R_zh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)），且 $`\mathcal T_zh=\int\Gamma_z\,dP-\int\Gamma_z\,dQ`$。

**(c)** 令 $`\mu_P`$ 为消息从 $`P`$ 抽取时本轮之后的状态分布；则 $`\mathbb E_{\mu_P}\psi=\mathbb E_P\Gamma_z`$，$`Q`$ 同理。记 $`\psi'=(\mathbf 1,\psi)`$，不变性给出矩阵 $`A_b`$，使对每个声明控制 $`b`$ 有 $`K_b\psi'=A_b\psi'`$；由 [Theorem 4.10(a)](../statements/04-10_hankel_rank_predictive_realization_ZH.md#s4-10)，对每个控制词 $`w`$ 有 $`\mathbb E_{\mu K_w}\psi'=A_{b_\ell}\cdots A_{b_1}\mathbb E_\mu\psi'`$。因此 $`\mathbb E_{\mu_P}\psi'`$ 与 $`\mathbb E_{\mu_Q}\psi'`$ 相等，推出每个声明控制词之后 $`\mathbb E\psi'`$ 都相等，而每个声明测试的值都是 $`\psi`$ 的仿射函数。对观察自适应的 continuation，沿每个观察分支用条件化更新 (4.20)：对分支归纳，两个分布的未归一化向量 $`M_{b_\ell,o_\ell}\cdots M_{b_1,o_1}\mathbb E_\mu\psi'`$ 相同，所以每个分支的概率相同，后验 predictive state 也相同。

**(d)** 与 (a) 一样，$`g_a=k_a+c_a^\top\Gamma_z+\bar e_a`$，所以 $`(\mathcal R_zh)(a)=c_a^\top\mathcal T_zh+\mathbb E_\rho[\bar e_ah]`$；又因为 $`h`$ 均值为零，$`\mathbb E_\rho[\bar e_ah]=\langle\bar e_a-\mathbb E_\rho\bar e_a,h\rangle_\rho`$。由 Cauchy–Schwarz，$`\lvert(\mathcal E_zh)(a)\rvert\le\lVert\bar e_a-\mathbb E_\rho\bar e_a\rVert_\rho\lVert h\rVert_\rho`$；平方、乘以 $`\omega_a`$ 再求和即得 (4.24)，并且 $`\mathrm{Var}_\rho\bar e_a\le\mathbb E_\rho\bar e_a^2`$。$`\square`$

## 相关工作

- 这个分解是用测试的预测描述受控过程的单步形式 ([Singh, James & Rudary, 2004](#ref-singh2004predictive))；其推论 (c) 就是充分 predictive state 的定义性质。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
