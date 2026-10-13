# Proposition 2.11 · 示范与查询共用一份信息预算

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/02-11_shared_information_budget_ZH.md) · [证明](../proofs/02-11_shared_information_budget_ZH.md) · [实验](02-11_shared_information_budget_ZH.md) · [English](02-11_shared_information_budget.md)

**本页检验什么.** [Check 2.5](../experiments/02-11_shared_information_budget_ZH.md#c2-5) 在 297 个自适应查询–停止过程上用精确 transcript 概率检验预算不等式 (2.11)：它从未被违反，单个有信息的查询使它取等（比值 1.0），所以常数不能改进（通过）。[Check 2.7](../experiments/02-10_few_shot_information_floor_ZH.md#c2-7) 中的包构造 `pkg:T11` 在基于得分的停止下也确认了它（通过）。

**本页用到的设定.**

- 两个世界 $`v\in\lbrace0,1\rbrace`$；$`K`$ 条独立同分布的示范，每条 KL 至多 $`\kappa_D^{(v)}`$；给定任何历史 $`h`$ 时，每次查询结果的 KL 至多 $`\kappa_X^{(v)}`$；与世界无关的查询–停止规则，$`N\le N_{\max}`$。
- $`\mathrm{kl}(a,b)`$ 是二元 KL 散度；$`H_{t-1}`$ 是第 $`t`$ 次查询之前的历史。

**被检验的陈述（[Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)）.** 考虑两世界的自适应获取。

- 世界为 $`v\in\lbrace 0,1\rbrace`$，两个世界要求不同的最终决策。
- $`K`$ 条独立同分布的示范中，每条满足 $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$。
- 一个共同的、依赖历史的规则选择查询，并在 $`N\le N_{\max}<\infty`$ 时停止。该规则的选择、停止与私有随机化 kernel 都不依赖于 $`v`$。
- 对每个允许的查询 $`x`$ 和相关支撑上的每段之前的历史 $`h`$，$`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$。
- 最终决策在每个世界中出错的概率至多为 $`\delta`$，$`0<\delta<1/2`$。

则

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

若 $`0<\kappa_X^{(v)}<\infty`$，

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

若 $`\kappa_X^{(v)}=0`$ 且 $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$，则任何查询预算都达不到所述误差。

<a id="c2-5"></a>

## Check 2.5 · 示范信息抵扣自适应查询预算（Proposition 2.11）

**结论.** 通过：297 个实例，2,969 条断言，最大残差 $`2.7\times10^{-15}`$。不等式 (2.11) 从未被违反，并且单个有信息的查询能使它取等（比值 1.0）。

**为什么做.** 与它所推广的 best-arm identification 中的 change of measure 不等式 ([Kaufmann et al., 2016](#ref-kaufmann2016complexity)) 一样，Proposition 2.11 必须对*每个*共同的、依赖历史的查询–停止规则成立。出现 $`\mathrm{kl}(1-\delta,\delta)>K\kappa_D+\kappa_X\mathbb E_vN`$ 的过程就能否定它；若比值处处远小于 1，说明界是松的。

**设置.**

- 两世界过程：$`K\in\lbrace 1,2\rbrace`$ 条对称的有信息示范，$`N_{\max}\le4`$，两个随机二元查询信道，两个世界共用的随机化、依赖历史的选择与停止规则。
- 精确的有理 transcript 概率。transcript KL 直接计算，并单独按自适应链式和计算；对数用 float64，容差 $`10^{-8}`$。
- 160 个随机过程，对抗爬山（8 次重启 × 16 步），以及两个构造情形：紧的情形（$`K=0`$，一次二元查询，信道 $`(\tfrac45,\tfrac15)`$，$`\delta=\tfrac15`$）与零信息情形（$`\kappa_D=\kappa_X=0`$）。
- 种子 `SeedSequence([150150, 6])`。

**结果.**

| 量 | 值 | JSON 键（条目 `v015:V6`） |
|---|---|---|
| 实例数 / 断言数 / 最大残差 | 297 / 2,969 / 2.7e-15 | `n_instances`、`n_assertions`、`worst_residual` |
| 满足前提 $`0<\delta<\tfrac12`$ 的随机过程 | 159 / 160 | `data.random_premise_count` |
| 随机过程中比值 $`\mathrm{kl}(1-\delta,\delta)/(K\kappa_D+\kappa_X\mathbb EN)`$ 的最大值 | 0.9973 | `data.worst_random.score` |
| 对抗搜索中的最大比值 | 0.9760 | `data.adversarial.best.score` |
| 构造的紧情形的比值 | **1.0**（KL $`=\mathrm{kl}(\tfrac45,\tfrac15)=0.8318`$，$`\mathbb EN=1`$） | `best_ratio`、`data.tight` |
| 零信息情形 | transcript KL $`=0`$；两个错误概率之和为 1 | `data.zero_information` |

**为什么支持这个陈述.**

- KL 链式恒等式（直接算的 transcript KL 等于累积的链式和）在自适应停止下精确成立，这是证明的核心。
- (2.11) 从未被违反，单个有信息的查询使它取等，所以它的常数不能改进。
- 示范与查询都不携带信息时，一致识别不可能，与 Proposition 2.11 的最后一句一致。

### [Check 2.7](../experiments/02-10_few_shot_information_floor_ZH.md#c2-7) · 独立的包构造

- *对本命题检验什么.* (2.11) 与 (2.12) 背后的双侧夹逼：二元 KL 下界 ≤ 精确 transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$，且 $`\mathbb EN`$ 不小于界 (2.12)。
- *设置.* 包构造 `pkg:T11`：两个世界，示范分布 $`(\tfrac3{10},\tfrac7{10})`$，两个查询信道，基于得分的停止，$`N_{\max}=4`$，$`K\le4`$；精确枚举。
- *结果.* 全部 10 个方向行都满足夹逼与 $`\mathbb EN`$ 的界；下侧与上侧之比为 0.37–0.56。
- *结论.* 通过。

## 复现

脚本用到 NumPy ([Harris et al., 2020](#ref-harris2020array))。

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V6 --output theory/v015_checks/quick/sec2.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## 产物

- `theory/v015_checks/results_v015_independent.json`：`v015:V6` 的条目；代码 `theory/v015_checks/mathematics.py`（`check_V6`）。
- `docs/direction/v015_revision/results_v015.json`：`checks.T11_adaptive_query_KL_enumeration.rows`；代码 `docs/direction/v015_revision/verify_v015_extensions.py`。
