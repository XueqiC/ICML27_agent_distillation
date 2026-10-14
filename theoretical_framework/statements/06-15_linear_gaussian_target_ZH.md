# Proposition 6.15 · 线性高斯目标：精确风险、白化 response 与单次查询收益

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](06-15_linear_gaussian_target_ZH.md) · [证明](../proofs/06-15_linear_gaussian_target_ZH.md) · [实验](../experiments/06-15_linear_gaussian_target_ZH.md) · [English](06-15_linear_gaussian_target.md)

**一段话概括.** [Theorem 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3) 与 [Lemma 6.4](06-04_design_minimax_equality_ZH.md#s6-4) 在主阶意义下给出单个学习器相对泛函的信息几何。当证据是线性高斯时，同样的几何对整个学习后果向量都是精确的：估计这些后果的 minimax 风险是一个迹，它的特征值衡量在已有证据下每个后果方向有多难辨识，再多一次查询带来的风险下降也有闭式解。由此得到一条可以计算的查询规则，并把两个容易混淆的量分开：一个后果有多大，和现有证据把它辨识得有多差。

**本页用到的设定与记号.**

- **线性高斯证据.** 未知的环境参数为 $`\vartheta\in\mathbb R^d`$。全部证据（示范与查询）合成一个高斯线性观测 $`y=D\vartheta+\epsilon`$，$`\epsilon\sim\mathcal N(0,\Sigma)`$，$`D`$ 与 $`\Sigma\succ0`$ 已知。参数在 $`\mathbb R^d`$ 中不受限制。
- **学习后果目标.** 已知的线性映射 $`f(\vartheta)=C\vartheta\in\mathbb R^k`$，例如 oracle 反馈在 $`k`$ 个声明测试上的一阶效应，相当于 [Definition 6.2](06-02_learner_relative_functional_information_norm_ZH.md#s6-2) 带多个任务方向的情形。不要求恢复 $`\vartheta`$ 本身。
- **损失.** $`\lVert x\rVert_W^2=x^\top Wx`$，$`k`$ 个输出上的权重 $`W\succeq0`$ 已知。
- **信息矩阵与目标矩阵.** $`I=D^\top\Sigma^{-1}D`$（$`y`$ 的 Fisher 信息），$`B=C^\top WC`$。$`I^+`$ 为 Moore–Penrose 伪逆。
- **一次查询.** 再加一个单位噪声、与 $`y`$ 独立的观测 $`a^\top\vartheta+\varepsilon`$；它给 $`I`$ 加上 $`aa^\top`$。

<a id="s6-15"></a>

**Proposition 6.15（线性高斯目标：精确风险、白化 response 与单次查询收益）.** 在上述设定下：

**(a) 精确的 minimax 风险.** 若 $`I\succ0`$，广义最小二乘估计 $`\hat f=C\hat\vartheta`$，$`\hat\vartheta=I^{-1}D^\top\Sigma^{-1}y`$，具有常数风险

```math
\mathbb E\lVert\hat f-f\rVert_W^2=\mathrm{tr}(BI^{-1})=\sum_j\lambda_j,\qquad Bv_j=\lambda_jIv_j ,
```

这就是 $`\vartheta\in\mathbb R^d`$ 上的 minimax 风险。

**(b) 奇异信息.** 设对每个 $`v\in\ker I`$ 都有 $`W^{1/2}Cv=0`$（例如 $`C\ker I=\lbrace0\rbrace`$）。则 $`\hat f=CI^+D^\top\Sigma^{-1}y`$ 具有常数风险 $`\mathrm{tr}(BI^+)`$，这是 minimax 风险；当 $`C\ker I=\lbrace0\rbrace`$ 时它还是无偏的。反之，若某个 $`v\in\ker I`$ 使 $`W^{1/2}Cv\ne0`$，$`\mathbb R^d`$ 上的 minimax 风险为 $`+\infty`$。若参数集有界、但包含沿这样的 $`v`$ 相差的两点，minimax 风险不是无穷，却有一个任何估计都去不掉的正下限，因为证据在这两点上的分布相同。

**(c) 信息白化的 response.** 对 $`I\succ0`$，令 $`\mathcal R_I=W^{1/2}CI^{-1/2}`$。它的奇异值平方就是 (a) 中的 $`\lambda_j`$，所以 $`\mathrm{tr}(BI^{-1})=\lVert\mathcal R_I\rVert_{\mathrm{HS}}^2`$。

**(d) 精确的单次查询收益.** 对 $`I\succ0`$，加入查询 $`a`$ 使 minimax 风险恰好下降

```math
\Delta(a)=\frac{a^\top I^{-1}BI^{-1}a}{1+a^\top I^{-1}a}.
```

**(e) 标量情形.** 当 $`k=1`$、$`C=c^\top`$、$`W=1`$ 时，$`B=cc^\top`$，风险为 $`c^\top I^{-1}c`$，即 [Lemma 6.4](06-04_design_minimax_equality_ZH.md#s6-4) 中 c-optimal design 的准则。多个目标给出加权的 A- 或 L-optimal 准则 $`\mathrm{tr}(BI^{-1})`$，而不是标量的 c-optimal 准则。

**说了什么.**

- 在线性高斯模型中，估计学习后果向量的风险是精确已知的，由广义最小二乘达到，并分布在 $`I^{-1/2}BI^{-1/2}`$ 的特征方向上。
- 两个谱必须分开。[Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator 衡量反馈差异造成的后果有多大；白化算子 $`\mathcal R_I`$ 衡量现有证据把每个后果方向辨识得有多差。一个已经被很好辨识的大后果，并不值得再去查询。
- 多一次查询使风险下降一个显式的量，所以候选查询可以按 $`\Delta(a)`$ 或单位成本的 $`\Delta(a)`$ 排序。

**直觉.**

- 广义最小二乘的误差协方差是 $`I^{-1}`$；目标通过 $`C`$ 看到这个误差，再按 $`W`$ 加权，迹说的就只是这件事。
- 一次查询的价值，是它在目标关心的方向上从误差协方差里拿掉了多少；Sherman–Morrison 公式恰好算出这个量。

**在框架中的作用.**

- 它是 §6 中查询设计完整证明的特例：[Theorem 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3) 与 [Theorem 6.5](06-05_achievability_ZH.md#s6-5) 只在主阶意义下给出一般的学习器相对界，这里每个量都是精确的。
- 它为 [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16) 中"获取能区分后果方向的证据"这一步提供可计算的实例。
- 回到真实学习系统时，这个特例**不**允许以下推论：
  - 按 $`\Delta(a)/\mathrm{cost}(a)`$ 贪心选择查询，只是一条单步规则；不声称它对整个预算最优。
  - 在一般的正则参数模型中，$`I`$ 要换成局部信息，$`C`$ 要换成**完整**后果函数的导数；局部近似、可辨识性与估计量的性质都需要另外证明。
  - 若 $`f(\vartheta)=\int g_\vartheta(y)P_\vartheta(dy)`$，环境的变化除了改变 feedback law，还可能改变之后的训练、访问分布与测试结果，所以在微分与积分可交换时 $`\partial f=\int g_\vartheta\,\partial P_\vartheta+\int(\partial g_\vartheta)\,dP_\vartheta`$。只有在 response 机制固定时，这个导数才化为 response operator 作用在 $`\partial P_\vartheta`$ 上。

**推导要点.**

- (a)：$`\hat\vartheta`$ 的误差是协方差为 $`I^{-1}`$ 的高斯变量，所以风险为 $`\mathrm{tr}(WCI^{-1}C^\top)`$。高斯先验 $`\mathcal N(0,\tau^2I_d)`$ 的 Bayes 风险为 $`\mathrm{tr}(B(I+\tau^{-2}I_d)^{-1})`$，当 $`\tau\to\infty`$ 时单调增加到同一个迹；没有估计量能在最坏情形下胜过每一个 Bayes 风险。
- (b)：当 $`W^{1/2}C`$ 在 $`\ker I`$ 上为零时，$`CI^+D^\top\Sigma^{-1}y`$ 的偏差在损失中看不见，同样的计算在可辨识的方向上照样成立；否则 Bayes 风险无界增长，而在有界集合上 Le Cam 的两点论证（[Le Cam, 1973](#ref-lecam1973convergence)）给出下限。
- (c)：$`\mathcal R_I^\top\mathcal R_I=I^{-1/2}BI^{-1/2}`$，它的特征值就是 $`(B,I)`$ 的广义特征值。
- (d)：把 $`(I+aa^\top)^{-1}`$ 的 Sherman–Morrison 公式代入迹。

**用到.**

- [Definition 6.2](06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：学习器相对泛函，这里带多个任务方向。
- [Lemma 6.4](06-04_design_minimax_equality_ZH.md#s6-4)：标量准则与伪逆约定。

**被用于.**

- [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)：可计算的获取一步。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：线性高斯情形下的查询选择。

**状态.** 已证明。

## 相关工作

- 广义最小二乘、它在高斯噪声下的 minimax 最优性以及迹准则都是经典内容；A-、L- 与 c-optimality 准则以及信息矩阵的作用见 [Pukelsheim, 1993](#ref-pukelsheim1993optimal)，可追溯到 [Elfving, 1952](#ref-elfving1952optimum)。
- 这里特有的是目标 $`C`$：它是声明的学习器与测试上的学习后果向量，而不是环境参数本身。正因如此，决定哪些查询值得做的是白化算子，而不是原始参数的协方差。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).
