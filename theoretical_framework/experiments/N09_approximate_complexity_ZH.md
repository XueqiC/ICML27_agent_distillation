# N09 · 精确阶数不等于近似复杂度 — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N09_approximate_complexity_ZH.md) · [English](N09_approximate_complexity.md)

**结论。** N09 成立；所有检验都通过。

- **(a)** **展开与截断误差（C12 与包内检验）：**
  - Chebyshev 恒等式 [N09(a)](../proofs/N09_approximate_complexity_ZH.md) 对 $`N=1,\dots,16`$ 精确成立。
  - 对每个 $`q<N`$，一致截断误差都精确等于二项尾概率 $`\Pr\lbrace \lvert B-N\rvert>q\rbrace`$，并且低于 $`2e^{-q^2/N}`$；尾概率与界之比最大为 $`0.430`$。
- **(a)** **所需阶数（包内检验）：** 独立检验覆盖 $`N\in\lbrace 4,8,16,32\rbrace`$（64 个情形）。要使一致误差 $`\le10^{-3}`$，所需阶数只有 $`q=4,6,9,13`$，增长大致按 $`\sqrt N`$。
- **(b)** **Parity（C12）：** 在 $`N=1,\dots,5`$ 上，用遍历全部顶点的 minimax LP 求得 parity 的最佳次数 $`<N`$ 一致逼近误差，精确为 $`1/2`$。
- **(b)** **同次数、不同复杂度（C19）：** 三个 response 族最高次数都是 3。标量族和结构化族用 3 个特征就把误差降到零；parity 族在 7 个非常数方向全部到位之前，误差始终 $`\ge0.0625`$（差异 $`\ge0.125`$）。
![精确次数相同，近似信息阶数却大不相同](../figures/N09_approximation.png)

[PDF](../figures/N09_approximation.pdf)

### C12 — Chebyshev 恒等式、精确尾概率与 parity minimax

**为什么做。** [N09(a)](../proofs/N09_approximate_complexity_ZH.md) 有三条可证伪的断言：

- [N09(a)](../proofs/N09_approximate_complexity_ZH.md) 是恒等式；
- 截断误差*等于*二项尾概率，而不只是以它为上界；
- 这个尾概率 $`\le2e^{-q^2/N}`$。

parity 命题断言最佳低次逼近误差恰为 $`1/2`$，只要找到严格更小的值就能将其证伪。

**设置。**

- 标量任务：$`N=1,\dots,16`$，以及所有 $`q<N`$。
- parity：$`N=1,\dots,5`$，在全部 $`2^N`$ 个立方体顶点上。
- 每个 $`N`$ 另取 10 个随机有理点，检验常数界 $`\lvert J_{\rm even}-\tfrac12\rvert\le\tfrac12`$。
- 种子：`SeedSequence([20261010, 12])`。

**做法。**

1. 用 SymPy 符号验证 [N09(a)](../proofs/N09_approximate_complexity_ZH.md)：$`\mathrm{expand}(\text{RHS}-p^N)=0`$。
2. 对每个 $`(N,q)`$：
   - 在含 $`p=1`$ 的 3,001 点网格上计算 $`\max\lvert p^N-h_q\rvert`$，断言它等于系数尾和；
   - 用精确有理数断言系数尾和等于二项尾概率；
   - 断言尾概率 $`\le2e^{-q^2/N}`$。
3. parity：用所有次数 $`<N`$ 的非常数单项式求解 minimax LP $`\min_{h\in\mathcal P_{N-1}}\max_{\rm vertices}\lvert J_{\rm even}-h\rvert`$，断言最优值为 $`0.5`$。

**结果。**

| 量 | 值 | 来源 |
|---|---:|---|
| 实例数 / 断言数 | 136 / 484 | `claims[id=C12].n_instances`, `.n_assertions` |
| [N09(a)](../proofs/N09_approximate_complexity_ZH.md) 与尾概率检验的 $`N`$ 范围 | 1–16 | `claims[id=C12].details.N_range` |
| 最大的 尾概率 $`/\,2e^{-q^2/N}`$ | 0.430 | `claims[id=C12].details.exhaustive_tail_best_ratio` |
| parity 最佳次数 $`<N`$ 逼近误差 | 恰为 0.5（$`N=1,\dots,5`$） | `claims[id=C12].details.parity_minimax_error` |
| 最大残差 | $`1.0\times10^{-17}`$ | `claims[id=C12].worst_residual` |
| 状态 | PASS | `claims[id=C12].status` |

**说明了什么。**

- $`p^N`$ 的截断误差恰好是二项尾概率，按 $`e^{-q^2/N}`$ 衰减；在检验范围内，所给的界至少保守 2.3 倍。
- parity 在任何更低的次数下，都无法比平凡的常数逼近得更好。

### 包内检验 — N 最大到 32 的 Chebyshev 截断（独立实现）

**为什么做。** 这是一个独立实现，把范围扩展到 $`N=32`$，并记录达到给定精度实际所需的阶数。

**做法。** 对 $`N\in\lbrace 4,8,16,32\rbrace`$、$`q=0,\dots,N`$：

- 在 8,001 点网格上计算 $`\lvert x^N-h_q\rvert`$；
- 与系数尾和比较；
- 与 $`\min(1,2e^{-q^2/N})`$ 比较。

**结果。** 全部 64 行都满足 网格误差 = 尾和（$`\max\lvert\text{grid}-\text{tail}\rvert=8.9\times10^{-16}`$），且尾和 $`\le`$ 界。部分结果如下：

| $`N`$ | $`q`$ | 截断误差 | $`2e^{-q^2/N}`$ | parity 最佳 $`q`$ 次误差 |
|---:|---:|---:|---:|---:|
| 16 | 4 | 0.1102 | 0.7358 | 0.5 |
| 16 | 8 | 0.00210 | 0.0366 | 0.5 |
| 16 | 12 | $`2.56\times10^{-6}`$ | $`2.47\times10^{-4}`$ | 0.5 |
| 32 | 8 | 0.0328 | 0.2707 | 0.5 |
| 32 | 16 | $`2.44\times10^{-5}`$ | $`6.71\times10^{-4}`$ | 0.5 |
| 32 | 20 | $`1.01\times10^{-7}`$ | $`7.45\times10^{-6}`$ | 0.5 |

截断误差 $`\le10^{-3}`$ 的最小 $`q`$：$`N=4,8,16,32`$ 时分别为 $`q=4,6,9,13`$。来源：`theory/v014_repro/results_v014_rerun.json → scalar_approximation`（字段 `Chebyshev_truncation_error`、`grid_check_error`、`Hoeffding_upper`、`parity_best_degree_q_error`），该文件与包内自带结果一致。

**说明了什么。** 对标量任务，达到实用精度所需的阶数按 $`\sqrt N`$ 增长：$`N`$ 翻倍，只多出几阶。对 parity，任何低于 $`N`$ 的阶都留下 $`1/2`$ 的误差。

### C19 — 最高次数相同，response 空间的复杂度不同

**为什么做。** response 族的次数并不决定它的近似复杂度，起作用的是逼近宽度 $`d_\varepsilon`$（[N06(b)](../proofs/N06_sufficiency_duality_ZH.md)）。C19 比较三个最高次数*相同*的族，预期结构化的族代价低，parity 代价高。

**设置。**

- 网格 $`\lbrace 0,\tfrac14,\dots,1\rbrace ^3`$（125 条消息），每个族的最高次数都是 3。
- *标量仿射幂* $`(a+bx_1)^3`$（5 个 response）。
- *多元 parity 拉回*（9 个 response）。
- *结构化求和幂* $`(a+b\bar x)^3`$（5 个 response）。
- 种子：`SeedSequence([20261010, 19])`。

**做法。**

1. 在拟合之前固定一个有限字典：各 response 向量、全部多重线性单项式，以及标量/结构化的 Chebyshev 基。
2. 对每个预算 $`d=1,\dots,7`$，贪心选出一个子空间，并计算两个量：
   - 一致误差 $`\max_g\mathrm{dist}_\infty(g,H)`$，在全网格上用 LP 求；
   - 该子空间的最坏分布对差异，用独立的原问题 LP 求。
3. 断言 差异 $`=2\times`$ 误差。由于找到的子空间未必最优，正的误差值是最优宽度的上界；零误差则是精确值。

**结果（一致误差；差异恰为其两倍）。**

| 特征预算 $`d`$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 标量仿射幂 | 0.0569 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| 结构化求和幂 | 0.0582 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| 多元 parity | 0.25 | 0.125 | 0.125 | 0.125 | 0.0625 | 0.0625 | **0** |

来源：`claims[id=C19].details.families[*].curve[*].uniform_error`（以及 `.discrepancy`、`.optimality`）。含常数的精确 span 维数为 4、4、8（`span_dimension_including_constant`）。在同族的留出 response 上，所选的标量和结构化子空间在 $`d=1`$ 时误差至多 $`0.017`$，从 $`d=3`$ 起为 $`0`$（`heldout_uniform_error`）。状态 PASS，共 1,445 条断言。

**说明了什么。** 在最高次数相同的情况下：

- 标量族和结构化族用 3 个统计量就能精确刻画；
- parity 需要全部 $`2^3-1=7`$ 个；
- 同样只给 3 个统计量时，parity 族仍留下 $`0.25`$ 的最坏分布对差异。

决定接口需要携带多高信息阶数的，是任务结构，而不是次数或决策步数。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C12,C19 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

## 产物

- 结果：
  - `theory/independent_checks/results_independent_v014.json`（C12、C19）
  - `theory/v014_repro/results_v014_rerun.json`（`scalar_approximation`）
- 已有图：`theory/independent_checks/figs/C19_response_complexity.png`
- 代码：`theory/independent_checks/check_v014_independent.py`（`check_C12`、`check_C19`）
- 图规格：`theoretical_framework/figures/specs/N09.md`
