# N18 · 经优化器变换的线性 response — 实验

[← 框架总图](../README_ZH.md) · [证明](../proofs/N18_optimizer_transformed_response_ZH.md)

**结论.** N18 成立。检验 V8 在 3,712 个实例上用精确算术确认了[N18(a)](../proofs/N18_optimizer_transformed_response_ZH.md)、[N18(c)](../proofs/N18_optimizer_transformed_response_ZH.md)，残差为零：

- **(a)** 在全部 140 个随机多项式延续值上，取 $`k=\deg V`$ 的展开都等于精确差距；
- **(a)** 每个截断余项都在其界之内（余项/界的最大值 0.275）；
- **(a)** 一个光滑的 logistic 延续值的误差不超过其认证界的 0.183；
- **(c)** 在 160 个面板上，一阶统计量个数 $`d_0^{(1)}`$ 从不超过 $`\min\lbrace\dim U,m-1\rbrace`$；少保留一个统计量时，总能找到一对特征一致、但一阶项不同的分布（158 个 LP 见证）。

检验 E10 在一个真实的受控网络上，对 11 种优化器配置确认了[N18(b)](../proofs/N18_optimizer_transformed_response_ZH.md)：

- **(b)** **界.** 在每个 $`S_1`$ 成立的精确历史单元格里，差距减去一阶项都落在余项界之内；每个光滑优化器的最大比值都不超过 0.9986。
- **(b)** **解释 [C23](N11_adam_cold_start_ZH.md)（v0.14 受控网络实验，其中 Adam 偏离仿射对照） 中的 Adam 差距.** 完整状态上的一阶项预测了 C23 冷启动 Adam 的差距：预测 $`0.07568`$，实测 $`0.07691`$；而原始平均梯度项为 $`-8.9\times10^{-11}`$。
- **(d)** **阶数.** 在 $`R=1`$ 时对每种优化器、在 $`R=4`$ 时对 SGD、动量、裁剪 SGD 和 sign 优化器，余项都是 $`\eta`$ 的二阶量。两阶梯度矩都一致时，SGD 的余项是三阶的（斜率 3），这正是矩层级 [N18(d)](../proofs/N18_optimizer_transformed_response_ZH.md) 的预测。
- **(b)** **多轮 Adam.** 对 Adam 类优化器，$`R=4`$ 时余项是 $`\eta`$ 的一阶量（斜率约 1），这就是[N18，注记 3](../proofs/N18_optimizer_transformed_response_ZH.md#注记) 中指出的二阶矩块。

唯一一个超出界的精确单元格是多步裁剪 SGD（比值 1.245）：它的延续值含有裁剪 kink，$`S_1`$ 在那里不保证成立。$`R=40`$ 时有两个 Monte Carlo 单元格在四个标准误之后仍超出界，都是 sign 类优化器。这些单元格标出了光滑性假设的边界。
![各优化器的一阶表示](../figures/N18_linear_response.png)

[PDF](../figures/N18_linear_response.pdf)

## V8 · [N18(a)](../proofs/N18_optimizer_transformed_response_ZH.md) 与 [N18(c)](../proofs/N18_optimizer_transformed_response_ZH.md) 的精确有限检验

**动机.** [N18(a)](../proofs/N18_optimizer_transformed_response_ZH.md) 是对两个分布积分后的 Taylor 恒等式。若张量矩的记账、$`V(x)`$ 的抵消或余项常数出错，就会出现差距与截断展开之差超过界的实例。[N18(c)](../proofs/N18_optimizer_transformed_response_ZH.md) 声称投影后的平均增量就是精确的最小一阶统计量；若计数有误，要么会在声称的维数下出现一阶项非零的特征一致对，要么会出现多余的统计量。

**设置.**

- *多项式.* $`D=2,\dots,5`$，每个 $`D`$ 重复 35 次，次数在 $`1,\dots,4`$ 之间循环。系数取自 $`\lbrace -1,-2/3,\dots,1\rbrace`$，首项单项式系数为 1。每个实例有一个随机有理状态 $`x`$、六个有理增量，以及其上的两个随机分布 $`P,Q`$。全部使用精确 `Fraction` 算术。
- *余项常数.* $`M_{k+1}`$ 来自包含 $`x`$ 与 $`x+\Delta`$ 的盒子上的系数和优函数（`data.box_rule`）。这是一个有理上界，不是拟合常数。
- *Logistic.* 在 $`\mathbb R^3`$ 上取 $`V=\sigma(q)`$，$`q(z)=\tfrac12z^\top Az+b^\top z`$，配六个增量和随机分布。Hessian certificate 为 $`M_2=L^2/(6\sqrt3)+\lVert A\rVert_{\rm op}/4`$，$`L=\max_{\rm vertices}\lVert Az+b\rVert_2`$（`data.logistic.certificate`）。2,200 个随机实例，另加 12 次重启 × 100 步爬山搜索。
- *秩.* 160 个面板，字母表大小 $`m\in\lbrace 4,\dots,9\rbrace`$，增量维数 $`D\in\lbrace 2,\dots,5\rbrace`$，随机整数增量 $`D(y)`$，以及张成 $`U`$ 的随机梯度方向。秩是精确计算的（SymPy），见证用 LP 求得（HiGHS）。
- 种子 `SeedSequence([150150, 8])`。

**做法.**

1. 对每个多项式，计算精确差距以及 $`j`$ 阶项 $`\langle\nabla^jV(x),\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\rangle/j!`$。断言 $`k=\deg V`$ 时它们的和等于差距，并且对每个 $`k<\deg V`$，余项都在界之内。
2. 对 logistic 族，断言 $`\lvert\text{gap}-\text{first order}\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)`$，再用对抗搜索最大化这个比值。
3. 对每个秩面板：
   - 计算 $`d_0^{(1)}=\mathrm{rank}[\mathbf 1,\mathcal G^{(1)}]-1`$，并与中心化秩核对；
   - 断言 $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$；
   - 检查保留统计量零空间中的扰动 $`U`$-投影为零，且不改变任何一阶项；
   - 只保留 $`d_0^{(1)}-1`$ 个统计量时，用 LP 求一对特征一致的分布 $`P,Q`$，并断言某个一阶项不同。

**结果**（`theory/v015_checks/results_v015_independent.json`，`groups[id=V8]`）：

| 量 | 数值 | Key |
|---|---:|---|
| status / 实例 / 断言 / 最坏残差 | PASS / 3,712 / 4,978 / 0 | `status`、`n_instances`、`n_assertions`、`worst_residual` |
| $`\deg V`$ 阶展开精确的多项式 | 140 / 140 | `data.polynomials[*].gap`、`[*].expansion` |
| 在界之内的截断余项 | 344 / 344；余项/界最大 0.275 | `data.polynomials[*].bounds` |
| logistic：最大 $`\lvert`$gap − first order $`\rvert`$ / 界 | 0.183（差距 −0.0231，一阶项 −0.2307，界 1.137，$`M_2=0.823`$） | `data.logistic.best` |
| 满足 $`d_0^{(1)}\le\min\lbrace\dim U,m-1\rbrace`$ 的秩面板 | 160 / 160 | `data.ranks[*].d0`、`[*].dim_U` |
| 在 $`d_0^{(1)}-1`$ 个统计量下有 LP 见证的面板 | 158（另 2 个 $`d_0^{(1)}=0`$）；最小分离目标值 0.113 | `data.ranks[*].LP_witness` |

**说明了什么.** [N18(a)](../proofs/N18_optimizer_transformed_response_ZH.md) 的展开在有限算术中是精确的；对一个光滑的非多项式延续值，余项界成立且留有余量。[N18(c)](../proofs/N18_optimizer_transformed_response_ZH.md) 的计数是精确的：$`d_0^{(1)}`$ 个统计量足够，少一个就总能找到一对在所有保留统计量上一致、却在一阶上不同的分布。

## E10 · 受控网络中的整次训练表示，跨优化器

**动机.** [N18(b)](../proofs/N18_optimizer_transformed_response_ZH.md) 声称，在整次训练上，两个反馈分布之间的差距等于完整状态上一阶项之和，加上一个由曲率常数控制的余项。在真实网络中有三处可能出错：一阶项可能漏掉差距，余项可能超出其界，或者余项在 $`\eta\to0`$ 时不缩小。E10 同时测这三点，并与使用原始平均梯度的朴素一阶项作比较。

**设置.**

- *网络.* C23 网络：one-hot 上下文、4 个 tanh 单元、sigmoid、平均交叉熵、float64。受控神经网络，不是 Gemma。
- *分布.* moment $`N=2,3,4`$ 与 parity $`N=2,3`$（C23 的分布，在 $`N`$ 阶以下的标签矩一致），外加在线性 logit 模型上实现的裁剪均值见证。
- *优化器（11 种）.* SGD；heavy-ball（$`\beta=0.9`$）；Nesterov（$`\beta=0.9`$）；冷启动 Adam（$`\beta=(0.9,0.999)`$、$`\epsilon=10^{-8}`$、带 bias correction）；经过 $`W=5`$ 和 $`W=20`$ 步 lr $`=0`$ moment warm-up 的 warm Adam；AdamW（$`\lambda=0.01`$）；裁剪 SGD；裁剪 Adam；signSGD；Lion（$`\beta_1=0.9`$、$`\beta_2=0.99`$）。
- *裁剪阈值.* 按任务选取最接近 50% 裁剪质量的阈值。梯度范数出现并列时无法达到 30–70% 区间（moment $`N=3`$、parity $`N=3`$：25%），记录在 `data.design[*].clip_active_probability`。
- *网格.* $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$，$`R\in\lbrace 1,4,40\rbrace`$，共 990 个单元格。
- 种子 `SeedSequence([150150, 110])`。

**做法.** 对每个单元格：

1. $`R=1,4`$ 时用完整历史枚举计算精确差距 $`L(P)-L(Q)`$；$`R=40`$ 时用独立训练并给出标准误。
2. 在完整状态 $`(\theta,m,v)`$ 上计算 [N18(b)](../proofs/N18_optimizer_transformed_response_ZH.md) 一阶项 $`\sum_t\mathbb E_{d_t^Q}\langle\nabla V^P_{t+1}(X_t),\mu^P_t-\mu^Q_t\rangle`$。$`\nabla V^P`$ 由精确枚举的 $`P`$-延续值经 Richardson 外推的中心差分得到，并记录求导误差。
3. 计算朴素项：把原始平均梯度当作 SGD 来用，并用当前的 Adam 分母做预条件。
4. 用采样和爬山搜索得到的线段曲率估计 $`M_2`$（观察到差距之后从不加垫），并计算余项界。
5. 记录比值 $`\lvert`$gap − first order $`\rvert`$/界，以及关于 $`\eta`$ 的 log–log 斜率。

**结果 1 — 精确单元格中的界**（`groups[id=E10].data.optimizer_summary`、`data.cells`）：

| 优化器 | 精确单元格 | 最大精确比值 | 超出界的精确单元格 |
|---|---:|---:|---:|
| SGD | 60 | 0.9986 | 0 |
| heavy-ball | 60 | 0.9984 | 0 |
| Nesterov | 60 | 0.9980 | 0 |
| 冷启动 Adam | 60 | 0.9981 | 0 |
| warm Adam，$`W=5`$ | 60 | 0.3339 | 0 |
| warm Adam，$`W=20`$ | 60 | 0.8432 | 0 |
| AdamW | 60 | 0.9817 | 0 |
| 裁剪 Adam | 55 个有限（5 个导数非有限） | 0.9982 | 0 |
| signSGD | 60 | 0.9981 | 0 |
| Lion | 60 | 0.9981 | 0 |
| 裁剪 SGD | 60 | 1.2453 | 1（多步 kink） |

**结果 2 — C23 中的 Adam 差距是优化器增量的一阶效应**（`data.C23_prediction`；moment $`N=3`$，$`\eta=0.1`$，$`R=1`$）：

| 量 | 数值 |
|---|---:|
| 实测差距 $`L(P)-L(Q)`$ | 0.076907 |
| [N18(b)](../proofs/N18_optimizer_transformed_response_ZH.md) 一阶项（完整状态增量） | 0.075683 |
| 朴素一阶项（原始平均梯度） | $`-8.87\times10^{-11}`$ |
| 余项 $`\lvert`$gap − first order $`\rvert`$ | 0.001225 |
| 余项界（$`\hat M_2=0.342`$） | 0.04447 |

两个标签分布的均值相同，所以任何平均梯度统计量都看不出它们的差别。冷启动 Adam 的增量 $`-\eta h_\epsilon(G)`$ 能看出来，它的一阶项解释了 98.4% 的差距。在同一个单元格上，signSGD 和 Lion 给出相同的差距和一阶项（0.07691 和 0.07568），因为它们的第一步是同一个 sign 映射。裁剪 SGD 的两个量都是 0.00303，而它的原始平均梯度项为 0（`data.cells`）。

**结果 3 — 余项的阶**（$`\lvert`$gap − first order $`\rvert`$ 关于 $`\eta`$ 的 log–log 斜率，按任务列出；`data.optimizer_summary[*].slopes`；余项落在数值下限的单元格没有斜率）：

| 优化器 | $`R=1`$ | $`R=4`$ |
|---|---|---|
| SGD、heavy-ball、Nesterov | $`N=2`$ 与裁剪见证上 1.98–2.0；moment/parity $`N=3`$ 上 3.0 | $`N=2`$ 与裁剪见证上 1.95–1.99；$`N=3`$ 上 2.91–2.99；Nesterov 在 moment $`N=4`$ 上 3.98 |
| 裁剪 SGD、signSGD、Lion | 2.0；$`N=3`$ 上 3.0 | 2.0–2.09；$`N=3`$ 上 2.71–3.01 |
| 冷启动 Adam | 2.0–3.0 | 0.99（moment $`N=3`$）、1.0（裁剪见证）；其余任务 1.96–2.98 |
| AdamW | 2.0–3.0 | 0.99（moment $`N=3`$）、1.31（moment $`N=4`$）、1.0（裁剪见证）；其余任务 1.98–2.98 |
| 裁剪 Adam | 2.0–3.0 | 0.91（moment $`N=3`$）、1.04（裁剪见证）；其余任务 2.0–2.93 |
| warm Adam，$`W=5`$ | 1.99–2.98 | 1.11–1.17（moment $`N=2,3,4`$）、1.0（裁剪见证）；2.03 与 2.92（parity） |
| warm Adam，$`W=20`$ | 2.0–3.0 | 1.0（moment $`N=3`$）、1.16（moment $`N=4`$）、1.0（裁剪见证）；3.04（moment $`N=2`$）、1.99 与 3.0（parity） |

**结果 4 — 多轮 Adam 差距中有多少是一阶的**（moment $`N=3`$，$`R=4`$；`data.cells`）：

| 优化器 | $`\eta=10^{-3}`$：差距 / 一阶项 | $`\eta=10^{-1}`$：差距 / 一阶项 | $`\eta=10^{-3}`$ 时的绝对余项 |
|---|---|---|---:|
| 冷启动 Adam | $`1.90\times10^{-3}`$ / $`1.24\times10^{-3}`$ | 0.1196 / 0.0810 | $`6.6\times10^{-4}`$ |
| warm Adam，$`W=5`$ | $`4.38\times10^{-4}`$ / $`2.22\times10^{-4}`$ | 0.0663 / 0.0365 | $`2.2\times10^{-4}`$ |
| warm Adam，$`W=20`$ | $`4.44\times10^{-5}`$ / $`2.01\times10^{-5}`$ | 0.0044 / 0.0020 | $`2.4\times10^{-5}`$ |
| 裁剪 SGD | $`1.21\times10^{-4}`$ / $`1.21\times10^{-4}`$ | 0.0115 / 0.0115 | $`2.2\times10^{-11}`$ |

**结果 5 — 光滑性假设的边界.**

| 位置 | 发生了什么 | Key |
|---|---|---|
| 裁剪 SGD、裁剪见证、$`\eta=0.1`$、$`R=4`$ | 唯一超出界的精确单元格：比值 1.245（差距 0.04476，一阶项 0.04637，界 0.00129）。之后的裁剪决策使延续值不光滑，$`S_1`$ 不保证成立。 | `data.cells` |
| $`R=40`$（330 个 Monte Carlo 单元格） | 未计标准误时有 25 个单元格超出点界；四个标准误之后剩 2 个，都是 sign 类（signSGD moment $`N=4`$、$`\eta=0.01`$；Lion 裁剪见证、$`\eta=10^{-3}`$） | `[*].excess_over_bound_after_four_SE` |
| 裁剪 Adam、moment $`N=4`$ | 10 个单元格在冷启动零梯度坐标处中心差分非有限，记为 unavailable | `[*].bound_status` |

**说明了什么.**

- 在真实网络中，只要延续值光滑，两个反馈分布之间的差距就等于*优化器增量*的一阶项，加上一个落在其曲率界内的余项。
- 增量是正确的统计量。对 C23 的分布，原始平均梯度什么也预测不了，而增量预测了冷启动 Adam 差距的 98%。
- 对 SGD、动量、裁剪和 sign 优化器，余项是 $`O(\eta^2)`$；两阶增量矩都一致时是 $`O(\eta^3)`$，这就是矩层级 [N18(d)](../proofs/N18_optimizer_transformed_response_ZH.md)。
- 对多轮的 Adam 类优化器，二阶矩块使余项成为 $`\eta`$ 的一阶量。此时在 moment 任务上一阶项承载 45–68% 的差距；从冷状态到 $`W=20`$ 步 warm-up，绝对余项下降 27 倍。正是在这个区间里，节点 [N20](N20_first_order_audit_ZH.md) 要求先满足 margin 条件，才能信任一阶排序。
- 所有超出界的单元格都在定理的光滑性假设之外：多步裁剪 SGD，以及 sign 优化器的 Monte Carlo 单元格。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10 --output theory/v015_checks/quick/n18.json
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：`groups[id=V8]` 与 `groups[id=E10]`。E10 的逐单元格记录在 `data.cells`，汇总在 `data.optimizer_summary`，C23 单元格在 `data.C23_prediction`，设计在 `data.design`。
- 代码：`theory/v015_checks/otlr.py`（`check_V8`、`check_E10`）；说明见 `theory/v015_checks/README.md`。
- 套件图：`theory/v015_checks/figs/E10.png` 与 `.svg`。
- 图：`../figures/N18_linear_response.pdf` 与 `.png`（规格 `../figures/specs/N18.md`）。
