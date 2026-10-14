# Proposition 4.11 · 静态充分不等于动态充分；有限时域的闭合误差

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-11_static_not_dynamic_closure_error_ZH.md) · [证明](04-11_static_not_dynamic_closure_error_ZH.md) · [实验](../experiments/04-11_static_not_dynamic_closure_error_ZH.md) · [English](04-11_static_not_dynamic_closure_error.md)

**本页证明什么.** (a) 是四个状态上的直接计算。(b) 是由精确的一步恒等式 $`q_{t+1}=A_tq_t+\mathbb Er_t(Z_t)`$ 驱动的线性预测器的误差递推，再用 Hölder 不等式。

**本页用到的设定.**

- 有界特征 $`\psi:\mathcal Z\to\mathbb R^d`$，固定控制序列的 kernel $`K_t`$，矩阵 $`A_t`$，残差 $`r_t=K_t\psi-A_t\psi`$，$`F=c^\top\psi+e_F`$，$`q_t=\mathbb E\psi(Z_t)`$，$`\hat q_{t+1}=A_t\hat q_t`$，$`A_{s:u}=A_s\cdots A_u`$（空乘积 $`=I`$）（[Theorem 4.10](../statements/04-10_hankel_rank_predictive_realization_ZH.md#s4-10)）。

**陈述（[Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error_ZH.md#s4-11)）.** (a) 取 $`z=(s,b)\in\lbrace0,1\rbrace^2`$、$`F=s`$、$`U(s,b)=(b,b)`$，特征 $`(\mathbf 1,s)`$ 静态充分但不闭合；$`(0,0)`$ 与 $`(0,1)`$ 的 $`q`$ 与当前得分相同，下一步得分为 $`0`$ 与 $`1`$；$`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$ 闭合。(b) (4.21) 精确成立，(4.22) 对每个范数及其对偶成立。

**证明.**

**(a)**

- *静态充分.* $`F=s=(0,1)\cdot(1,s)^\top`$。
- *不闭合.* $`(KF)(s,b)=F(b,b)=b`$。设在四个状态上 $`b=\alpha+\beta s`$。在 $`(0,0)`$ 与 $`(0,1)`$ 处分别得 $`0=\alpha`$ 与 $`1=\alpha`$，矛盾。所以 $`KF\notin\mathrm{span}\lbrace\mathbf 1,s\rbrace`$，不存在矩阵 $`A`$ 使 $`K\psi=A\psi`$。
- *predictive state 相同，未来不同.* 对 $`(0,0)`$ 与 $`(0,1)`$ 上的点质量，两者都有 $`q=(1,0)`$ 且 $`F=0`$。一次更新后状态分别为 $`(0,0)`$ 与 $`(1,1)`$，得分为 $`0`$ 与 $`1`$。没有任何 $`q`$ 的函数能预测下一步得分。
- *加入 $`b`$ 后闭合.* $`K\mathbf 1=\mathbf 1`$，$`Ks=b`$，$`Kb=b`$，所以 $`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$ 是 $`K`$-不变的。

**(b)** *一步.* 由 Markov 性质 $`q_{t+1}=\mathbb E\psi(Z_{t+1})=\mathbb E(K_t\psi)(Z_t)`$，再由 $`r_t`$ 的定义，

```math
q_{t+1}=\mathbb E\big[A_t\psi(Z_t)+r_t(Z_t)\big]=A_tq_t+\mathbb Er_t(Z_t).
```

*递推.* 记 $`\Delta_t=q_t-\hat q_t`$，由 $`\hat q_{t+1}=A_t\hat q_t`$ 相减得 $`\Delta_{t+1}=A_t\Delta_t+\mathbb Er_t(Z_t)`$。对 $`H`$ 归纳：$`H=0`$ 平凡；若 (4.21) 对 $`H`$ 成立，则

```math
\Delta_{H+1}=A_H\Delta_H+\mathbb Er_H(Z_H)=A_{H:0}\Delta_0+\sum_{t<H}A_{H:t+1}\mathbb Er_t(Z_t)+A_{H:H+1}\mathbb Er_H(Z_H),
```

因为 $`A_HA_{H-1:t+1}=A_{H:t+1}`$ 且 $`A_{H:H+1}=I`$。这就是 $`H+1`$ 时的 (4.21)。

*得分误差.* $`\mathbb EF(Z_H)=c^\top q_H+\mathbb Ee_F(Z_H)`$，所以 $`\mathbb EF(Z_H)-c^\top\hat q_H=\mathbb Ee_F(Z_H)+c^\top\Delta_H`$。代入 (4.21)，逐项用 Hölder 不等式 $`\lvert c^\top Mx\rvert=\lvert(M^\top c)^\top x\rvert\le\lVert M^\top c\rVert_*\lVert x\rVert`$，再用 $`\lVert\mathbb Er_t(Z_t)\rVert\le\mathbb E\lVert r_t(Z_t)\rVert`$（Jensen，范数是凸的）与 $`\lvert\mathbb Ee_F\rvert\le\mathbb E\lvert e_F\rvert`$，即得 (4.22)。$`\square`$

**关于证明的说明.** 恒等式 (4.21) 对任意矩阵 $`A_t`$ 成立，包括最小二乘拟合出来的；它们的拟合误差按定义就是 $`r_t`$ 的一部分。没有任何地方要求 $`\lVert A_t\rVert\le1`$。

## 相关工作

- 这个误差递推，就是近似 predictive state 与近似 value-directed compression 中所用的有限时域模型误差传播 ([Poupart & Boutilier, 2002](#ref-poupart2002value))。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).
