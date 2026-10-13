# Theorem 2.10 · 少样本信息下限与硬预算下的最优查询

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-10_few_shot_information_floor_ZH.md) · [证明](02-10_few_shot_information_floor_ZH.md) · [实验](../experiments/02-10_few_shot_information_floor_ZH.md) · [English](02-10_few_shot_information_floor.md)

**本页证明什么.** (a) 把"未揭示比特仍是均匀硬币"这一引理与揭示集合的独立性结合起来。(b) 对剩余预算归纳：按值分支没有收益，贪心选择最优；Jensen 不等式与一个小例子说明正部的位置。

**本页用到的设定.**

- 未知比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j`$ 出现（$`\sum_jw_j=1`$），其正确决策为 $`\Theta_j`$；学生能学任何"上下文到动作"的表，所以没有表示误差。
- $`K`$ 条示范中的每一条揭示一个随机的比特集合 $`S_k`$ 及其值；这些集合独立同分布，与比特独立，$`q_j=\Pr(j\in S)`$ 是一条示范揭示比特 $`j`$ 的概率。
- 示范之后，硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特；选择可以依赖于此前看到的一切。
- $`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。$`R_{\rm info}`$ 表示可达到的最小期望 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 中的信息项）。
- 完整定义：[Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8)。

**陈述（[Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10)）.** 在 [Definition 2.8](../statements/02-08_hidden_bit_model_ZH.md#s2-8) 的模型中：

**(a) 信息下限.** 不查询时，$`K`$ 条轨迹之后可达到的最小期望 regret 为

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

约定 $`(1-q)^0=1`$。再多一条轨迹的边际价值为

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) 最优查询.** 把未揭示比特的权重排序为 $`w_{(1)}\ge\dots\ge w_{(M_K)}`$，则

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

并且查询权重最大的 $`n`$ 个未揭示比特是最优的；根据揭示出的值调整查询没有帮助。在均匀情形（$`w_j=1/d`$，每条轨迹揭示一个均匀抽取的上下文）下，

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

一般有 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$，且不等号可以严格。

**证明.**

**(a)**

- *下界.* 固定任意学习器。对 $`j\in U_K`$，由 [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9)，对任何（可能随机的）动作 $`A_j`$ 有 $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$；对 $`j\notin U_K`$，成功概率至多为 1。所以条件 regret 至少为 $`\tfrac12\sum_{j\in U_K}w_j`$。
- *可达性.* 照抄每个已揭示的标签、其余随机猜的表达到这个下界，且它属于学生类。
- *取期望.* $`S_k`$ 独立同分布，所以 $`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$，从而 $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$，即 (2.7)。
- *边际价值.* 由 $`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$ 得 (2.8)。

**(b)** 以示范 transcript 为条件，令 $`V(U,b)`$ 为未揭示集合为 $`U`$、剩余预算为 $`b`$ 时的最优条件 regret。对 $`b`$ 归纳证明

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

其中 $`\mathrm{top}_b(U)`$ 是 $`U`$ 中最大的 $`\min(b,\lvert U\rvert)`$ 个权重之和。

- *基础情形.* $`b=0`$ 时就是条件形式的 (a)。
- *归纳步.* $`b\ge1`$ 时，学习器可以停止、查询一个已知比特，或查询某个 $`j\in U`$。
  - 停止得到 $`V(U,0)`$，查询已知比特得到 $`V(U,b-1)`$。由归纳假设，两者都不小于右端，因为 $`\mathrm{top}_b`$ 关于 $`b`$ 单调。
  - 查询 $`j\in U`$ 揭示 $`\Theta_j`$。由 [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits_ZH.md#s2-9)，剩余的未揭示比特仍然独立同分布、均匀，且与揭示出的值独立，所以无论观察到什么值，后续问题都是 $`V(U\setminus\lbrace j\rbrace ,b-1)`$；按值分支没有收益。

因此

```math
V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
```

最大值等于 $`\mathrm{top}_b(U)`$，取权重最大的 $`j`$ 即可达到。归纳完成，并且贪心规则最优。对示范取 $`V(U_K,n)`$ 的平均即得 (2.9)。

- *均匀情形.* 权重为 $`1/d`$ 时 $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$，即 (2.10)；$`n=0`$ 时由 (a) 取 $`q_j=1/d`$ 得 $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$。
- *Jensen.* $`x\mapsto(x-n)_+`$ 是凸函数，由 Jensen 不等式 ([Jensen, 1906](#ref-jensen1906fonctions))，$`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$。
- *严格性.* 取 $`d=3`$、$`K=3`$、$`n=1`$。则 $`\Pr(M_3=0)=6/27`$，$`\Pr(M_3=1)=18/27`$，$`\Pr(M_3=2)=3/27`$，$`\mathbb EM_3=8/9<1`$。右端为 $`0`$，而 $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$。$`\square`$

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-jensen1906fonctions"></a>

**J. L. W. V. Jensen (1906).** Sur les fonctions convexes et les inégalités entre les valeurs moyennes. *Acta Mathematica* 30, 175–193. DOI [10.1007/BF02418571](https://doi.org/10.1007/BF02418571).
