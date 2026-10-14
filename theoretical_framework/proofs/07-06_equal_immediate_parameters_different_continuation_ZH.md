# Proposition 7.6 · 即时参数相同并不决定后续

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [证明](07-06_equal_immediate_parameters_different_continuation_ZH.md) · [实验](../experiments/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [English](07-06_equal_immediate_parameters_different_continuation.md)

**本页证明什么.** (a) 对参照 kernel 做反向归纳。(b) 由步长大小导数的符号与 Cauchy–Schwarz 不等式得到。(c) 用精确的有理数运算。(d) 用隐函数定理。(e) 对矩的递推直接做代数运算。

**本页用到的设定与记号.**

- $`z=(\theta,o)`$ 是 complete learner state（[Definition 1.3](../README_ZH.md#s1-3)）；$`V^0_{t+1}`$ 是参照 continuation value，$`g_{t,z}(y)=\mathbb E_{\xi_t}V^0_{t+1}(U_t(z,y,\xi_t))`$ 是单边界 response（[Definition 1.4](../README_ZH.md#s1-4)）。
- 学习器称为*无记忆的*，如果参照 kernel $`K^0_k`$（$`k>t`$）把 $`(\theta,k)`$ 的函数映为 $`(\theta,k)`$ 的函数，且 $`F_E(z)=F_E(\theta)`$。
- Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）的单个坐标，做 bias correction，无 weight decay：步数计数为 $`s`$ 时，$`m'=\beta_1m+(1-\beta_1)g`$，$`v'=\beta_2v+(1-\beta_2)g^2`$，$`\theta'=\theta-\eta q_{1,s}m'/(\sqrt{q_{2,s}v'}+\epsilon)`$，其中 $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$，$`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$。
- 对 $`\epsilon=0`$ 与固定状态：$`a=\beta_1m`$，$`b=1-\beta_1`$，$`c=\beta_2v`$，$`d=1-\beta_2`$，$`S(g)=(a+bg)/\sqrt{c+dg^2}`$，于是 $`\theta'=\theta-\eta q_{1,s}q_{2,s}^{-1/2}S(g)`$。
- consumed gradient 由平方损失 $`\tfrac12(\theta-y)^2`$ 实现：在当前参数 $`\theta`$ 处，标签 $`y=\theta-g`$ 给出梯度 $`g`$，Hessian 为 $`1`$。

**陈述（[Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）.** (a) 对无记忆的学习器，只要更新后下一参数的分布相同，两条消息对每个声明的任务、每个无记忆的后续训练，单边界 response 都相等。(b) 对 $`\epsilon=0`$、$`v>0`$、$`m>0`$ 的 Adam，步长大小 $`S`$ 在 $`(-a/b,g^\star]`$ 上严格递增、在 $`[g^\star,\infty)`$ 上严格递减，$`g^\star=bc/(ad)`$，$`S(g^\star)=\sqrt{a^2/c+b^2/d}`$，极限为 $`b/\sqrt d`$；严格介于 $`b/\sqrt d`$ 与 $`S(g^\star)`$ 之间的每个值恰由两个梯度取到，它们给出相同的下一参数与不同的一阶矩。(c) 取 $`\beta_1=0.9`$、$`\beta_2=0.999`$、$`\eta=1`$、$`\epsilon=0`$ 并做 bias correction，在 $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$ 处，梯度 $`\tfrac12`$ 与 $`\tfrac{2103}{790}`$ 给出相同的第二步参数；经过同一个第三步损失后，第三步参数不同。(d) 对每个足够小的 $`\epsilon>0`$，用一个关于 $`\epsilon`$ 连续的梯度 $`g_B(\epsilon)`$，结论同样成立。(e) $`(v'-v)/v=(1-\beta_2)(g^2/v-1)`$；若 $`v`$ 固定且尾段梯度不依赖 $`\theta`$，扰动 $`\delta g`$ 在滞后 $`k`$ 时造成的位移为 $`-\eta(1-\beta_1)\beta_1^{\,k}\delta g/\sqrt v`$。

**(a) 的证明.**

1. $`V^0_R=F_E`$ 只是 $`\theta`$ 的函数。
2. 设对某个 $`k>t`$ 有 $`V^0_{k+1}(z)=\phi_{k+1}(\theta)`$。由于 $`K^0_k`$ 把 $`(\theta,k)`$ 的函数映为 $`(\theta,k)`$ 的函数，$`V^0_k=K^0_kV^0_{k+1}`$ 仍只是 $`\theta`$ 的函数。归纳到 $`k=t+1`$，得 $`V^0_{t+1}(z)=\phi_{t+1}(\theta)`$。
3. 因此 $`g_{t,z}(y)=\mathbb E\,\phi_{t+1}(\theta_{t+1})`$，其中 $`\theta_{t+1}`$ 是 $`U_t(z,y,\xi_t)`$ 的参数分量。它只通过 $`\theta_{t+1}`$ 的分布依赖于 $`y`$。上述论证对每个声明的任务、每个无记忆的后续训练分别成立。$`\square`$

**(b) 的证明.**

**1.** 当 $`g>-a/b`$ 时分子 $`a+bg`$ 为正，所以 $`S(g)>0`$，且

```math
S'(g)=\frac{b(c+dg^2)-dg(a+bg)}{(c+dg^2)^{3/2}}=\frac{bc-adg}{(c+dg^2)^{3/2}}.
```

**2.** 由于 $`a,b,c,d>0`$，当 $`g<g^\star=bc/(ad)`$ 时 $`S'>0`$，当 $`g>g^\star`$ 时 $`S'<0`$。这给出两段严格单调性。

**3.** *峰值.* 把 $`a+bg`$ 写成 $`(a/\sqrt c)\sqrt c+(b/\sqrt d)(\sqrt d\,g)`$。由 Cauchy–Schwarz，$`a+bg\le\sqrt{a^2/c+b^2/d}\,\sqrt{c+dg^2}`$，等号恰在 $`(\sqrt c,\sqrt d\,g)`$ 是 $`(a/\sqrt c,b/\sqrt d)`$ 的正倍数时成立，也就是 $`g=bc/(ad)=g^\star`$。所以 $`S(g^\star)=\sqrt{a^2/c+b^2/d}`$。

**4.** *取值范围的两端.* 当 $`g\downarrow-a/b`$ 时 $`S(g)\to0`$；当 $`g\to\infty`$ 时 $`S(g)\to b/\sqrt d`$。在 $`[g^\star,\infty)`$ 上 $`S`$ 严格递减地趋于极限，所以那里 $`S(g)>b/\sqrt d`$。

**5.** 由连续性与严格单调性，每个满足 $`b/\sqrt d<u<S(g^\star)`$ 的 $`u`$ 在 $`(-a/b,g^\star)`$ 中恰有一个原像 $`g_A`$（$`S`$ 在那里从 $`0`$ 增到 $`S(g^\star)`$），在 $`(g^\star,\infty)`$ 中也恰有一个原像 $`g_B`$（$`S`$ 在那里从 $`S(g^\star)`$ 减向 $`b/\sqrt d`$，但取不到它）。

**6.** 在同一状态 $`(\theta,s)`$ 下，$`S`$ 相等意味着 $`\theta'=\theta-\eta q_{1,s}q_{2,s}^{-1/2}S(g)`$ 相等。一阶矩 $`m'=a+bg`$ 关于 $`g`$ 严格递增，所以 $`g_A<g_B`$ 给出 $`m'_A<m'_B`$。$`\square`$

**(c) 的证明.**

**1.** *第一步*，$`s=1`$，从 $`(0,0,0)`$ 出发，梯度为 $`1`$：$`m_1=\tfrac1{10}`$，$`v_1=\tfrac1{1000}`$，$`q_{1,1}=10`$，$`q_{2,1}=1000`$，所以 $`\theta_1=-10\cdot\tfrac1{10}/\sqrt{1000\cdot\tfrac1{1000}}=-1`$。这就是 $`z_1`$。

**2.** *第二步*，$`s=2`$：$`q_{1,2}=\tfrac{100}{19}`$，$`q_{2,2}=\tfrac{10^6}{1999}`$，且

```math
m_2(g)=\frac{0.9+g}{10},\qquad v_2(g)=\frac{0.999+g^2}{1000},\qquad \frac{m_2(g)^2}{v_2(g)}=10\,\frac{(0.9+g)^2}{0.999+g^2}.
```

当 $`g>-0.9`$ 时，步长为 $`\eta q_{1,2}q_{2,2}^{-1/2}\sqrt{m_2^2/v_2}`$，所以两个这样的梯度给出相同的 $`\theta_2`$，当且仅当它们给出相同的 $`m_2^2/v_2`$。

**3.** *两个梯度.* 对 $`g_A=\tfrac12`$：$`(0.9+g_A)^2=\tfrac{49}{25}`$，$`0.999+g_A^2=\tfrac{1249}{1000}`$，比值为 $`\tfrac{1960}{1249}`$。对 $`g_B=\tfrac{2103}{790}`$：$`0.9+g_B=\tfrac{1407}{395}`$，$`(0.9+g_B)^2=\tfrac{1979649}{156025}`$，$`0.999+g_B^2=\tfrac{50460849}{6241000}`$。由于 $`6241000=40\cdot156025`$、$`50460849=1249\cdot40401`$、$`1979649\cdot40=79185960=1960\cdot40401`$，比值仍为 $`\tfrac{1960}{1249}`$。所以两边都有 $`m_2^2/v_2=\tfrac{19600}{1249}`$，$`\theta_2`$ 相同。

**4.** *二次方程.* 在 $`(0.9+g)^2/(0.999+g^2)=\tfrac{1960}{1249}`$ 中去分母，得 $`1249(g^2+1.8g+0.81)=1960(g^2+0.999)`$，即 $`711g^2-2248.2\,g+946.35=0`$。两根之和为 $`\tfrac{2248.2}{711}=\tfrac{1249}{395}=\tfrac12+\tfrac{2103}{790}`$，之积为 $`\tfrac{946.35}{711}=\tfrac{2103}{1580}=\tfrac12\cdot\tfrac{2103}{790}`$，所以两根恰为 $`g_A`$ 与 $`g_B`$。用 (b) 的记号，$`a=0.09`$、$`b=0.1`$、$`c=0.000999`$、$`d=0.001`$，$`g^\star=1.11`$ 位于两根之间。

**5.** *第二步之后的矩.* $`m_2^A=\tfrac7{50}`$，$`v_2^A=\tfrac{1249}{10^6}`$；$`m_2^B=\tfrac{1407}{3950}`$，$`v_2^B=\tfrac{50460849}{6241\cdot10^6}`$。两者不同。

**6.** *第三步*，$`s=3`$，两边用同一个损失，它在共同的 $`\theta_2`$ 处梯度为 $`1`$：$`m_3=0.9m_2+0.1`$，$`v_3=0.999v_2+0.001`$，都为正，$`\theta_3=\theta_2-\eta q_{1,3}q_{2,3}^{-1/2}\sqrt{m_3^2/v_3}`$。精确地，

```math
\Big(\frac{m_3^2}{v_3}\Big)^A=\frac{0.226^2}{0.002247751}=\frac{51076000}{2247751},\qquad \Big(\frac{m_3^2}{v_3}\Big)^B=\frac{1103967076000}{56651388151}.
```

这两个有理数不同（前者约为 $`22.72`$，后者约为 $`19.49`$），所以 $`\theta_3^A\ne\theta_3^B`$，且 $`\theta_3^A<\theta_3^B`$。用浮点数计算，$`\theta_3^A-\theta_3^B\approx-0.0712027`$，sigmoid 得分之差 $`\approx-0.00364994`$（[Check 7.8](../experiments/07-06_equal_immediate_parameters_different_continuation_ZH.md#c7-8)）。

**7.** *一般的 $`\eta`$.* 矩不依赖 $`\eta`$，每一步参数变化都是 $`\eta`$ 乘以一个与 $`\eta`$ 无关的量。用相同的 consumed gradient（在当前参数处取标签 $`y=\theta-g`$），第二步参数仍然相同，第三步参数之差是上面数值的 $`\eta`$ 倍。$`\square`$

**(d) 的证明.**

1. 让 (c) 的整个实例（包括第一步）都用同一个 $`\epsilon`$ 运行，记第二步梯度为 $`g`$ 时第二步与第三步的参数为 $`\theta_2(g,\epsilon)`$、$`\theta_3(g,\epsilon)`$。$`\lvert\epsilon\rvert`$ 足够小时，每个分母 $`\sqrt{q_{2,s}v'}+\epsilon`$ 都为正（因为所有 $`v'`$ 都为正），所以两个映射在包含 $`(g_B,0)`$ 的一个开集上属于 $`C^1`$。
2. 令 $`\Phi(g,\epsilon)=\theta_2(g,\epsilon)-\theta_2(\tfrac12,\epsilon)`$。由 (c)，$`\Phi(g_B,0)=0`$。
3. 在 $`\epsilon=0`$ 处，$`\partial_g\theta_2=-\eta q_{1,2}q_{2,2}^{-1/2}S'(g)`$；由于 $`g_B>g^\star=1.11`$，由 (b) 得 $`S'(g_B)<0`$。所以 $`\partial_g\Phi(g_B,0)\ne0`$。
4. 由隐函数定理，存在 $`\epsilon_0>0`$ 与 $`(-\epsilon_0,\epsilon_0)`$ 上的 $`C^1`$ 函数 $`g_B(\epsilon)`$，使 $`g_B(0)=g_B`$ 且 $`\Phi(g_B(\epsilon),\epsilon)=0`$：对每个这样的 $`\epsilon`$，即时参数都相同。
5. $`\Psi(\epsilon)=\theta_3(\tfrac12,\epsilon)-\theta_3(g_B(\epsilon),\epsilon)`$ 连续，且由 (c) 有 $`\Psi(0)\ne0`$。所以对所有足够小的 $`\epsilon`$，$`\Psi(\epsilon)\ne0`$。$`\epsilon=10^{-8}`$ 时，区间求根给出 $`g_B(\epsilon)\approx2.662025475`$，$`\Psi(\epsilon)\approx-0.0712027`$（[Check 7.8](../experiments/07-06_equal_immediate_parameters_different_continuation_ZH.md#c7-8)）。$`\square`$

**(e) 的证明.**

1. $`v'-v=(\beta_2-1)v+(1-\beta_2)g^2=(1-\beta_2)(g^2-v)`$；两边除以 $`v>0`$。
2. 比较两次只在消息的 consumed gradient 上相差 $`\delta g`$ 的运行。尾段梯度不依赖 $`\theta`$，所以两次运行消耗相同的尾段梯度；由线性递推 $`m_{k+1}=\beta_1m_k+(1-\beta_1)g_{k+1}`$，滞后 $`k`$ 时的动量差为 $`\delta m_k=(1-\beta_1)\beta_1^{\,k}\delta g`$（$`k=0`$ 是消息自己的更新）。
3. 在 $`v`$ 保持不变、不做 bias correction、$`\epsilon=0`$ 时，参数步之差为 $`-\eta\,\delta m_k/\sqrt v=-\eta(1-\beta_1)\beta_1^{\,k}\delta g/\sqrt v`$。
4. 求和得 $`\sum_{k<K}(1-\beta_1)\beta_1^{\,k}=1-\beta_1^{\,K}`$，它趋于 $`1`$；总一阶位移为 $`-\eta\,\delta g/\sqrt v`$，其中消息自己的更新承担 $`1-\beta_1`$。$`\square`$

## 相关工作

- 用 [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的 costate 来说，(c) 中两条消息的参数通道增量相同，所以它们一阶价值之差为 $`\langle\chi^m_{t+1},(1-\beta_1)(g_A-g_B)\rangle+\langle\chi^v_{t+1},(1-\beta_2)(g_A^2-g_B^2)\rangle`$。这个构造是 Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）中只有记忆通道才能区分两条消息的一个实例。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
