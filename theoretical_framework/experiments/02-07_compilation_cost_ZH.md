# Proposition 2.7 · 编译代价

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-07_compilation_cost_ZH.md) · [证明](../proofs/02-07_compilation_cost_ZH.md) · [实验](02-07_compilation_cost_ZH.md) · [English](02-07_compilation_cost.md)

**本页检验什么.** [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability_ZH.md#c2-3) 的编译部分对五个 $`\delta_L`$ 精确计算频率学习器在 $`m=\lceil32\log(2/\delta_L)\rceil`$ 时的失败概率：它比 $`\delta_L`$ 低两到四个数量级，所需标签数只依赖于 $`\delta_L`$（通过）。

**本页用到的设定.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一；差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$，两者都以概率 1 成功，所以 teacher 在两个世界中都最优。
- 学生是混叠的（看不到 $`s`$），以概率 $`p`$ 选 $`A`$。另外两个状态–动作对的成功概率为 $`\tfrac12\mp2v\Delta`$，所以学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最好值为 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- *被动接口：* 学习器看到 teacher 示范（可能附带 $`s`$）及对它们的后处理。*查询接口：* 它还可以自适应地 reset 到 $`s=1`$、执行 $`A`$，观察 Bernoulli $`(\tfrac12+2v\Delta)`$ 结果，查询次数是停时 $`N`$。
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ 是二元 KL 散度。
- 完整定义：[Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5)。
- $`\widehat v`$ 是估计出的符号；频率学习器把 $`\widehat p`$ 设为 $`m`$ 个 Bernoulli $`(q)`$ 标签的经验频率。

**被检验的陈述（[Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7)）.** 在 [Definition 2.5](../statements/02-05_two_world_aliased_family_ZH.md#s2-5) 的族中，符号估计为 $`\widehat v`$ 之后：

- 若 $`\widehat v=+1`$，取目标 $`q=7/8`$，否则取 $`q=1/8`$；
- 生成 $`m`$ 个 Bernoulli $`(q)`$ 目标标签，令学生为它们的经验频率 $`\widehat p`$。

若 $`m\ge\lceil32\log(2/\delta_L)\rceil`$，则以至少 $`1-\delta_L`$ 的概率有 $`\lvert\widehat p-q\rvert\le1/8`$。若 $`\widehat v`$ 以至少 $`1-\delta`$ 的概率正确（[Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)），则编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率 regret 不超过 $`\Delta/2`$。

### [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability_ZH.md#c2-3) · 两世界族中的查询复杂度与编译

- *对本命题检验什么.* 对两个目标 $`q\in\lbrace\tfrac18,\tfrac78\rbrace`$，$`m=\lceil32\log(2/\delta_L)\rceil`$ 个编译标签以至少 $`1-\delta_L`$ 的概率给出 $`\lvert\widehat p-q\rvert\le\tfrac18`$，且每个落在容差内的结果 regret 至多为 $`\Delta/2`$。
- *设置.* 对 $`\delta_L\in\lbrace0.01,0.05,0.1,0.25,0.4\rbrace`$ 计算精确的有理二项尾概率（整数分子求和，不用正态近似）。
- *结果.* 所需标签数 $`m=170,119,96,67,52`$；精确失败概率 $`4.2\times10^{-6}`$、$`1.2\times10^{-4}`$、$`2.4\times10^{-4}`$、$`3.0\times10^{-3}`$、$`3.9\times10^{-3}`$，都远低于 $`\delta_L`$，这是 $`m=\lceil32\log(2/\delta_L)\rceil`$ 所依据的 Hoeffding 界 ([Hoeffding, 1963](#ref-hoeffding1963probability)) 的余量。标签数与 $`\Delta`$ 无关，而同一检查中识别的代价随 $`\Delta^{-2}`$ 增长。
- *结论.* 通过。

## 复现

脚本用到 NumPy ([Harris et al., 2020](#ref-harris2020array))、SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) 与 SymPy ([Meurer et al., 2017](#ref-meurer2017sympy))。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C2 --output theory/independent_checks/subset.json
```

## 产物

- `theory/independent_checks/results_independent_v014.json`：`v014:C2` 的条目；代码 `theory/independent_checks/check_v014_independent.py`（`check_C2`、`binomial_tail`）。键 `details.compilation[*].m`、`details.compilation[*].error`。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-meurer2017sympy"></a>

**Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, Anthony Scopatz (2017).** SymPy: Symbolic computing in Python. *PeerJ Computer Science* 3, e103. DOI [10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
