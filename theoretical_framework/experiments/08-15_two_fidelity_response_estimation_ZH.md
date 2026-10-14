# Proposition 8.15 · 双保真 response 估计及其成本条件

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-15_two_fidelity_response_estimation_ZH.md) · [证明](../proofs/08-15_two_fidelity_response_estimation_ZH.md) · [实验](08-15_two_fidelity_response_estimation_ZH.md) · [English](08-15_two_fidelity_response_estimation.md)

**本页检验什么.**

- **(a)–(b)，精确.** 在一个有限联合分布上，双保真估计量无偏，最优分配以成本 $`S^2/\varepsilon^2`$ 恰好达到误差目标（**PASS**）。
- **(c)–(d)，门槛.** 成本比为十分之一时，固定系数的门槛为 0.7662（便宜部分被复用）与 0.7875（不复用）。每个便宜输出都计费时，最优系数的嵌套估计量门槛为 0.5750；对系数与分配做数值最小化，在五个相关系数上重现了它的成本比公式（**PASS**）。若每个配对样本中的便宜输出被复用，同样的最小化给出 0.3163，与 $`\sqrt c=0.3162`$ 一致。同口径比较：不复用时 0.7875 对 0.5750，复用时 0.7662 对 0.3163。
- **量级.** 合成算例的成本比 0.0402 假设配对相关系数约为 0.995。真实模型上还没有测过短续训与完整续训的相关性，所以不主张任何真实的节省。每格仍至少需要一次完整续训：每格 4 次完整复现时，节省至多 4 倍。

**本页用到的设定与记号.**

- 一格：完整续训输出 $`X\in[0,1]^p`$，目标 $`f=\mathbb EX`$；冻结估计器的便宜输出 $`Y\in[0,1]^p`$。
- 第 0 组：$`n_0`$ 个便宜输出；第 1 组：$`n_1`$ 个耦合对 $`(X',Y')`$；两组独立。$`v_0`$、$`v_1`$、$`v_H`$ 是 $`Y`$、$`X-Y`$、$`X`$ 的加权总方差；$`c_0`$、$`c_1`$、$`c_H`$ 是便宜样本、配对样本、完整样本的成本；$`\varrho`$ 是标量测试下 $`X`$ 与 $`Y`$ 的相关系数；嵌套估计量中 $`c=c_L/c_H`$。

**被检验的陈述（[Proposition 8.15](../statements/08-15_two_fidelity_response_estimation_ZH.md#s8-15)）.** $`\hat f=\bar Y+\overline{X'-Y'}`$ 无偏，$`\mathbb E\lVert\hat f-f\rVert_\omega^2=v_0/n_0+v_1/n_1`$；均方误差 $`\varepsilon^2`$ 下的最小成本为 $`(\sqrt{v_0c_0}+\sqrt{v_1c_1})^2/\varepsilon^2`$；它省计算当且仅当

```math
\sqrt{v_0c_0}+\sqrt{v_1c_1}<\sqrt{v_Hc_H} ;
```

最优系数的嵌套估计量相对成本为 $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$，小于 1 当且仅当 $`\lvert\varrho\rvert>2\sqrt c/(1+c)`$；每格仍至少需要一次完整续训。

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 8.22](#c8-22) | (a)–(e) | `v017:C21.fidelity` | PASS |

<a id="c8-22"></a>

## Check 8.22 · 无偏性、分配与成本门槛（Proposition 8.15）

**结论.**

- 在有限分布上，无偏性与分配恒等式精确成立：**PASS**。
- 门槛：系数为 1 时 $`\varrho>0.7662`$（$`c_1=c_H`$）、$`\varrho>0.7875`$（$`c_1=1.1c_H`$）；最优系数的嵌套估计量在 $`c=0.1`$、每个便宜输出都计费时 $`\lvert\varrho\rvert>0.5750`$，每个配对样本的便宜输出被复用时 $`\lvert\varrho\rvert>0.3163`$。
- 在 $`\varrho\in\lbrace0.5,0.6,0.8,0.9,0.99\rbrace`$ 上，公式 $`(\sqrt{1-\varrho^2}+\lvert\varrho\rvert\sqrt c)^2`$ 与直接数值最小化的差小于 $`10^{-6}`$。

**动机.** 这个命题用便宜预测器加校正代替昂贵的完整续训。它是否有价值取决于两件事，检验把它们摆清楚：无偏性与分配代数是否精确，以及真实的便宜预测器需要多大的相关性。嵌套最优系数公式取自多保真文献（[Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)）；暴力最小化在这里的参数化下独立地核对它。

**设置.**

- *有限分布.* $`(X,Y)\in\lbrace0,1\rbrace^2`$，$`\Pr(X=Y=0)=\Pr(X=Y=1)=0.5(1-0.0025)`$，$`\Pr(X\ne Y)=0.0025`$，平均分给两种不一致。于是 $`v_H=v_0=0.25`$，$`v_1=0.0025`$，相关系数 $`0.995`$。成本 $`c_0=0.01`$、$`c_1=1.01`$、$`c_H=1`$；目标均方误差 $`10^{-4}`$。
- *固定系数门槛.* 标量测试，$`v_0=v_H`$，$`c_0=c_H/10`$；用 Brent 方法（SciPy，[Virtanen et al., 2020](#ref-virtanen2020scipy)）求 $`\sqrt{0.1}+\sqrt{2(1-\varrho)c_1/c_H}=1`$ 在 $`c_1/c_H\in\lbrace1,1.1\rbrace`$ 时的根。
- *嵌套估计量.* 相对的"成本乘方差"为 $`(1+c\,\kappa)\big(1+(\alpha^2-2\alpha\varrho)(1-1/\kappa)\big)`$，$`\kappa=n_L/n_H\ge1`$；用 Nelder–Mead 在 $`(\alpha,\log(\kappa-1))`$ 上最小化，并以 1（只用完整样本）为上限；与同样以 1 为上限的闭式比较。

**结果**（`theory/v017_checks/results_v017.json`，条目 `v017:C21.fidelity`）：

| 量 | 值 |
|---|---:|
| $`v_H`$、$`v_0`$、$`v_1`$ | 0.25、0.25、0.0025 |
| 最优 $`n_0`$、$`n_1`$（连续） | 5,012.5、49.9 |
| 双保真成本 / 只用完整样本的成本 | 0.0402 |
| 固定系数门槛，$`c_1=c_H`$ | 0.7662 |
| 固定系数门槛，$`c_1=1.1c_H`$ | 0.7875 |
| 嵌套最优系数门槛，$`c=0.1`$，每个便宜输出都计费 | 0.5750 |
| 嵌套最优系数门槛，$`c=0.1`$，每对中的便宜输出被复用 | 0.3163（$`\sqrt c=0.3162`$） |
| 每格 4 次完整复现时的最大节省 | 4 倍 |

| $`\varrho`$ | 0.50 | 0.60 | 0.80 | 0.90 | 0.99 |
|---|---:|---:|---:|---:|---:|
| 相对成本，公式 | 1.000 | 0.980 | 0.728 | 0.519 | 0.206 |
| 相对成本，数值最小值 | 1.000 | 0.980 | 0.728 | 0.519 | 0.206 |

**为什么支持该陈述.**

- (a)–(b)：方差与成本恒等式在机器精度内成立；0.0402 的比值是公式在便宜预测器几乎完全相关时给出的结果。
- (c)–(d)：门槛就是成本条件翻转的地方；公式与数值最小化一致，确认了嵌套成本比。$`\varrho=0.5`$ 时没有节省。
- (e)：每格一次完整续训不可避免，所以每格 4 次完整复现时最多省 4 倍，格子数不变。
- 真实模型上的前提，即同一格中短续训与完整续训的相关系数超过这些门槛，还没有测过；在测之前，这个命题给出的是一条决策规则，不是加速。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验程序使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 产物

- `theory/v017_checks/results_v017.json`：条目 `v017:C21.fidelity`；代码 `theory/v017_checks/check_v017.py`，函数 `c_fidelity`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
