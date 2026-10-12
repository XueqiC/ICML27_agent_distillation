# §9 Mechanism and method — Experiments

[← Framework](../README.md) · [Proofs](../proofs/09_mechanism_and_method.md) · [中文](09_mechanism_and_method_ZH.md)

**Verdict.**

- **[Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1) (coupling width) holds.** On 320 random packet-response tables the width is exactly zero for additive tables and positive for non-additive ones, the bound (b) holds with median slack 0.231, and the restricted-pairing and strict-null statements hold exactly (worst residual $`9.9\times10^{-16}`$). An independent implementation agrees.
- **The strict null and its limit appear in a trained network.** In a controlled network, one common-state SGD step is exactly regrouping-invariant (parameter difference 0). Sequential updates on the same labels, grouped differently, change the final return by $`9.6\times10^{-6}`$ to $`0.75`$, growing with step size and horizon under SGD and with the opposite sign under Adam.
- **Selection by measured consequence ([Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3), Rules 2–3).** On frozen held-out panels of the controlled network, the five static proxies tested did not reliably select: on held-out learner states their wrong-sign rate is 0.36–0.55, against 0.55 for a random score, and their Spearman correlation with the gold contrast is at most 0.10. The measured 4-update tail has wrong-sign rate at most 3.1% and zero selection regret on every state, $`\epsilon`$, learning-rate and warmth shift. On the held-out threshold task both short tails select worse than random, and only the full tail is exact on every split.
- **Warm-up does not rescue the static proxies (Rule 3).** An lr = 0 moment warm-up compresses the differences between candidates (random-selection regret falls in 4 of 5 splits), but none of the 10 combinations of the five static proxies with $`W\in\lbrace 5,20\rbrace`$ improves regret, wrong-sign rate and Spearman together.
- **Scope.** Exact finite computations and a controlled small network. No real-model or benchmark result is reported; the planned real-model tests are listed in [Remark 9.4](../proofs/09_mechanism_and_method.md#s9-4). The first-order margin check on the same panels (code ID `v015:E12`), the ranker comparison (`v016:C5`, `v016:C5b`) and the paired-replacement certificate (`v016:C6`, `v016:C6b`) are reported with the statements they test, on the experiment pages of [§8](08_verification.md) and [§7](07_second_moment_channel.md).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 9.1](#c9-1) | [Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1)(a)–(d) | `v014:C17` | PASS |
| [Check 9.2](#c9-2) | [Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1)(a), (b), (d) | `pkg:coupling_and_additive_null` | PASS |
| [Check 9.3](#c9-3) | [Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1)(d) in a trained network | `v014:C23` (regrouping) | exact null; sequential effects present |
| [Check 9.4](#c9-4) | [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3), Rules 2 and 5; [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2), step 5 | `v015:E6` | controlled finding |
| [Check 9.5](#c9-5) | [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3), Rule 3 | `v015:E9` | controlled finding |

![Figure 9.1: coupling width — exact null and sequential regrouping effects](../figures/fig9_1_coupling.png)

*Figure 9.1.* (A) In a controlled network, grouping the same four labels as $`\lbrace(0,0),(1,1)\rbrace`$ versus $`\lbrace(0,1),(0,1)\rbrace`$ changes the final return. The gap is exactly zero for one common-state step and grows with step size and horizon under SGD; under Adam it has the opposite sign and different dynamics. (B) The range of the expected packet response over the coupling polytope: a single point for an additive table, positive width for a non-additive one. [PDF](../figures/fig9_1_coupling.pdf)

<a id="c9-1"></a>

## Check 9.1 · Coupling polytope, additive characterization and strict null (Proposition 9.1)

**Verdict.** PASS: 320 instances, 2,567 assertions, worst residual $`9.85\times10^{-16}`$ (LP feasibility). Each of the four parts holds on every instance.

**Why.** Proposition 9.1 makes four claims: the "if and only if" characterization of zero width, the bound (b) by the additive residual, the failure of that characterization under forbidden pairings, and the single-step null. Each would fail visibly on some random instance if it were wrong.

**Setup.**

- $`\lvert\mathcal Y_1\rvert,\lvert\mathcal Y_2\rvert\in[2,5]`$, with strictly positive marginals.
- Random additive tables $`H=a+b`$ and random non-additive tables.
- Seed `SeedSequence([20261010, 17])`.

**Procedure.**

1. Compute the width with two linear programs: the maximum and the minimum of $`\langle\Gamma,H\rangle`$ over the coupling polytope.
2. Compute $`\inf_{a,b}\lVert H-a-b\rVert_\infty`$ with a third linear program and check the bound (b).
3. Construct the forbidden-pairing example of part (c).
4. Check the null exactly in rational arithmetic: one SGD step of a linear model at a common parameter, with the same examples regrouped. Then run two sequential updates with regrouping.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C17`):

| Quantity | Value |
|---|---:|
| status | PASS |
| instances / assertions | 320 / 2,567 |
| worst residual (LP feasibility) | $`9.85\times10^{-16}`$ |
| slack of (b): median / max | 0.231 / 0.712 |
| forbidden-pairing table with zero feasible width | $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$ |
| one-step regrouping difference (linear model, exact) | 0 |
| two-step sequential regrouping difference (linear model, exact) | $`1/32`$ |

**Why it supports Proposition 9.1.**

- Zero width holds exactly for the additive tables, and positive width for the non-additive ones, as part (a) states.
- The bound (b) is valid but generally strict; the typical slack is about a quarter of the return range, consistent with the strictness example in the proof.
- The single-step null is exact in rational arithmetic, so any measured single-step effect would be an implementation artefact.
- Sequential updates create a genuine regrouping effect. It is the $`\eta^2H_{P_2}g_{P_1}`$ term in the proof of part (d).

<a id="c9-2"></a>

## Check 9.2 · Independent implementation (Proposition 9.1)

**Verdict.** PASS: the additive table has width 0, a non-additive quadratic pair has width 0.25, and the frozen-gradient null holds to $`2.8\times10^{-17}`$.

**Why.** An implementation written separately from Check 9.1 guards against a shared coding error.

**Setup.** Finite packet linear programs and frozen-gradient arithmetic; no neural optimizer.

**Result** (`theory/v014_repro/results_v014_rerun.json`, `coupling_and_additive_null`):

| Quantity | Value |
|---|---:|
| width of an additive table | 0.0 |
| quadratic pair: min / max / width | 0.25 / 0.50 / 0.25 |
| strict frozen-gradient additive null, max residual | $`2.8\times10^{-17}`$ |

**Why it supports Proposition 9.1.** The additive table has zero width and the non-additive pair positive width, as in part (a). The frozen-gradient null of part (d) is exact up to machine precision.

<a id="c9-3"></a>

## Check 9.3 · Regrouping in a controlled network (Proposition 9.1(d))

**Verdict.** The single common-state step is exactly regrouping-invariant (parameter difference 0.0). After sequential updates, regrouping changes the final return by up to 0.749 under SGD and by up to 0.073 in absolute value under Adam.

**Why.** The width is defined through actual learning consequences. A trained network fed the same labels, grouped differently, must show an exactly zero difference at one common-state step; after sequential updates it may show a real difference.

**Setup.**

- Network: one-hot context, 4 tanh units, sigmoid output, float64. This is a controlled small network, not Gemma and not an external benchmark.
- Source A packets are $`\lbrace (0,0),(1,1)\rbrace`$ and source B packets are $`\lbrace (0,1),(0,1)\rbrace`$. Both contain the same four labels, so the marginals are identical.
- Optimizers: SGD with $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$; Adam with $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$ and $`\eta\in\lbrace 0.01,0.1\rbrace`$.
- 1, 4 or 40 cycles, giving 2, 8 or 80 updates.
- Exact enumeration of all deterministic histories.

**Procedure.** Compute the final expected-return gap between the two groupings. Separately, compute the parameter difference after one common-state step with identical weights, and check the manual gradients against central differences.

**Result** (`theory/independent_checks/results_independent_v014.json`, `controlled_neural.details`):

- `additive_null_parameter_residual` = **0.0** for the single common-state step;
- `gradient_max_residual` = $`8.7\times10^{-11}`$ for the manual gradients against central differences.

Return gap A − B (`regrouping`):

| optimizer | $`\eta`$ | 2 updates | 8 updates | 80 updates |
|---|---:|---:|---:|---:|
| SGD | 0.01 | $`9.6\times10^{-6}`$ | $`3.8\times10^{-5}`$ | $`3.2\times10^{-4}`$ |
| SGD | 0.1 | $`9.2\times10^{-4}`$ | $`3.2\times10^{-3}`$ | $`7.9\times10^{-3}`$ |
| SGD | 1.0 | 0.040 | 0.062 | 0.076 |
| SGD | 3.0 | 0.658 | 0.747 | 0.749 |
| Adam | 0.01 | −0.0094 | −0.0136 | −0.0032 |
| Adam | 0.1 | −0.0729 | −0.0503 | −0.0029 |

**Why it supports Proposition 9.1.**

- The effect is exactly zero where part (d) says it must be: one step at a common parameter.
- In an actual nonlinear learner, the joint organization of identical-marginal feedback changes the learned student once updates are sequential. Under SGD the effect grows with step size and horizon; under Adam it has the opposite sign and other dynamics.
- So the coupling width is a real, optimizer-dependent quantity of the learning consequence. Packet organization belongs to the response that [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2) measures (step 3), not to a fixed property of the data.

![Figure 9.2: which measurements select feedback correctly](../figures/fig9_2_selection.png)

*Figure 9.2.* Controlled-learner evidence for the selection rules. (A) Wrong-sign rate of frozen proxies on held-out learner states and on warmth, $`\epsilon`$ and learning-rate shifts (`v015:E6`): the tested static proxies are at the level of a random score, while measured short-tail responses are near zero. (B) Selection regret: short tails select near-optimally on held-out states but worse than random on the held-out threshold task; only the full tail is exact everywhere. (C) Effect of an lr = 0 moment warm-up (`v015:E9`): the spread between candidates shrinks (random regret falls) while the static proxies do not improve. Exact finite-history computations in a controlled small network. [PDF](../figures/fig9_2_selection.pdf)

<a id="c9-4"></a>

## Check 9.4 · Held-out selection with frozen proxies (Remark 9.3, Rules 2 and 5)

**Verdict.** Controlled finding. On held-out learner states the five static proxies tested have wrong-sign rate 0.359–0.547 (random 0.547) and Spearman at most 0.10. The measured 4-update tail has wrong-sign rate at most 0.031 and zero regret on every state, $`\epsilon`$, learning-rate and warmth shift. On the held-out threshold task the 1- and 4-update tails have regret 0.044 and 0.032, against 0.019 for random. The full tail has zero regret and zero wrong signs on every split.

**Why.** Rule 2 says a candidate must be judged by its actual update-plus-tail response. This check asks whether cheaper statistics can stand in for that response when the learner state, the task, the recipe or the optimizer warmth changes. Each change is a separate held-out split, so a proxy that only fits the pilot cannot pass. The short-tail proxies are measured one-boundary responses with a truncated tail, that is, surrogates in the sense of [Definition 8.6](../proofs/08_certificate.md#s8-6); the threshold split tests whether such a surrogate keeps its ordering on a new task, which is the calibration-transfer question of [Remark 8.11](../proofs/08_certificate.md#s8-11).

**Setup** (`theory/v015_checks/proxy_split_freeze.json`, SHA-256 `ac8f4aee…`, written before any held-out measurement):

- **Learner.** The controlled network of `v014:C23` (one-hot context, 4 tanh units, sigmoid, mean cross-entropy, float64).
  - States: `fresh`, `warm2`, `warm5`, `warm9` (complete states, including Adam moments).
  - Recipes: 8, namely Adam with $`\eta\in\lbrace 0.01,0.1\rbrace`$ and SGD with $`\eta\in\lbrace 0.1,1\rbrace`$, each with $`\epsilon\in\lbrace 10^{-8},10^{-3}\rbrace`$.
- **Candidates.** 9 label laws on 10 two-label packets: `reference_even`, moment-matched `moment3_P`/`moment3_Q`, sign-matched `low_sign_A`/`B` and `high_sign_A`/`B`, `odd_parity` and `middle_mixture`.
- **Gold.** One candidate boundary followed by 7 IID reference updates (8 updates in total), enumerated exactly over all finite histories. Its value is the contrast with the reference candidate.
- **Proxies**, fitted by ridge regression on the pilot only and frozen in `proxy_models_frozen.json`:
  - static: input-moment error, mean gradient, sign only, first-step $`\Delta\theta`$, and $`\Delta\theta`$ with $`(m,v)`$;
  - measured short-tail responses after 1 and after 4 updates;
  - controls: random score, task identity, training step.
- **Pilot.** States `fresh` and `warm2`, tasks `moment` and `parity`, recipes 0 and 4; 72 rows.
- **Held-out splits.** States `warm5`/`warm9`; task `threshold`; $`\epsilon`$ only (recipes 1, 5); learning rate only (2, 6); both (3, 7); a fresh-only fit evaluated on warm states, and the reverse. 432 held-out rows in total; seed `SeedSequence([150150, 106])`.
- **Metrics.**
  - *Selection regret*: the gold value of the best candidate minus that of the proxy-selected candidate, averaged over cells.
  - *Wrong-sign rate*: the fraction of candidates with nonzero gold whose predicted contrast has the opposite sign.
  - *Spearman*: the rank correlation of predicted and gold contrasts.

**Procedure.** Write the split manifest and the frozen proxy coefficients to disk, then evaluate every held-out split once. No held-out consequence changes a proxy, its features, the ridge penalty or the abstention thresholds.

**Result** (`theory/v015_checks/results_v015_independent.json`, entry `v015:E6`, `data.reports.<split>.metrics.<proxy>`):

*Wrong-sign rate* (lower is better):

|split|input moment|mean grad|sign|Δθ|Δθ+m,v|short tail 1|short tail 4|random|full tail|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|0.359|0.547|0.406|0.516|0.516|0.047|0.016|0.547|0.000|
|`held_out_epsilon`|0.234|0.250|0.312|0.391|0.453|0.031|0.031|0.531|0.000|
|`held_out_lr`|0.438|0.344|0.297|0.312|0.344|0.391|0.031|0.469|0.000|
|`held_out_epsilon_lr`|0.438|0.344|0.297|0.312|0.344|0.391|0.031|0.562|0.000|
|`fresh_to_warm`|0.359|0.391|0.562|0.422|0.516|0.047|0.016|0.578|0.000|
|`warm_to_fresh`|0.125|0.125|0.188|0.250|0.438|0.062|0.031|0.469|0.000|
|`held_out_task`|0.250|0.250|0.083|0.250|0.083|0.000|0.042|0.458|0.000|

*Selection regret* (lower is better):

|split|input moment|mean grad|sign|Δθ|Δθ+m,v|short tail 1|short tail 4|random|full tail|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|0.0064|0.0043|0.0043|0.0043|0.0039|0.0000|0.0000|0.0056|0.0000|
|`held_out_epsilon`|0.0013|0.0013|0.0022|0.0013|0.0013|0.0000|0.0000|0.0067|0.0000|
|`held_out_lr`|0.0041|0.0041|0.0070|0.0041|0.0041|0.0007|0.0000|0.0105|0.0000|
|`held_out_epsilon_lr`|0.0040|0.0040|0.0070|0.0040|0.0040|0.0007|0.0000|0.0093|0.0000|
|`fresh_to_warm`|0.0064|0.0064|0.0054|0.0064|0.0064|0.0000|0.0000|0.0045|0.0000|
|`warm_to_fresh`|0.0000|0.0000|0.0012|0.0000|0.0000|0.0000|0.0000|0.0056|0.0000|
|`held_out_task`|0.0000|0.0000|0.0020|0.0000|0.0000|0.0439|0.0322|0.0186|0.0000|

*Spearman* (higher is better; — means undefined because the prediction is constant):

|split|input moment|mean grad|sign|Δθ|Δθ+m,v|short tail 1|short tail 4|random|full tail|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`held_out_states`|-0.00|0.05|0.10|0.07|0.07|0.95|0.99|-0.04|1.00|
|`held_out_epsilon`|0.45|0.40|0.45|0.42|0.41|0.96|0.99|-0.07|1.00|
|`held_out_lr`|0.38|0.40|0.46|0.57|0.58|0.34|0.95|0.03|1.00|
|`held_out_epsilon_lr`|0.38|0.40|0.46|0.58|0.58|0.33|0.95|0.04|1.00|
|`fresh_to_warm`|-0.00|-0.07|-0.07|-0.06|-0.09|0.95|0.99|-0.07|1.00|
|`warm_to_fresh`|0.79|0.58|0.73|0.72|0.59|0.90|0.97|0.10|1.00|
|`held_out_task`|0.63|0.58|0.73|0.71|0.84|—|0.54|-0.05|1.00|

**Why it supports Rules 2 and 5.**

1. *The tested static proxies are not responses.* On held-out learner states, and when moving from fresh to warm states, they are at the level of a random score: wrong-sign rates 0.36–0.56 against 0.55–0.58 for random, Spearman at most 0.10. The same update statistic maps to different learning consequences at different complete states, as [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1) and the optimizer-state results of [§5](../proofs/05_optimizer_state.md) predict. This is a finding about the proxies tested, not a proof about all static statistics; proved inconsistency is limited to the classes of [Proposition 7.3](../proofs/07_second_moment_channel.md#s7-3).
2. *Measured responses select.* The 4-update tail keeps wrong signs at or below 3.1% and regret at 0 on every state, $`\epsilon`$, learning-rate and warmth shift. The 1-update tail degrades under learning-rate shifts (wrong-sign rate 0.39) but still selects nearly optimally (regret 0.0007).
3. *Tail length is part of the estimand.* On the held-out `threshold` task, the short tails select worse than random (regret 0.044 and 0.032 against 0.019), because their ordering differs from that of the 8-update full tail. The full tail is the only measurement with zero regret and zero wrong signs on every split. In the terms of [Remark 8.11](../proofs/08_certificate.md#s8-11), a surrogate calibrated on one task family can lose its ordering on another, so calibration transfer has to be checked. The static proxies happen to reach zero regret on this split, but with 8–25% wrong signs, and they are at chance on held-out states, so they do not transfer across splits.
4. *Scale.* The regrets are of order $`10^{-3}`$–$`10^{-2}`$ in return units, and the panel has 8 cells per split (4 on the task and warm-to-fresh splits). The cell bootstraps (`regret_bootstrap95`) describe variation across this finite panel; they are not population confidence intervals.

<a id="c9-5"></a>

## Check 9.5 · Proxy transfer under lr = 0 moment warm-up (Remark 9.3, Rule 3)

**Verdict.** Controlled finding. Warm-up compresses the differences between candidates: the regret of a random choice falls in 4 of 5 held-out splits, on average by 0.0079 at $`W=5`$ and by 0.0100 at $`W=20`$. It does not make the static proxies predictive: none of the 10 combinations of the five static proxies with $`W\in\lbrace 5,20\rbrace`$ improves regret, wrong-sign rate and Spearman together, and their wrong-sign rate rises by 1–13 points.

**Why.** At a warm optimizer state the first-order statistic a candidate must match changes ([Propositions 5.6–5.8](../proofs/05_warm_adam.md#s5-6)). This check asks whether warming the moments makes the static statistics of Check 9.4 predictive.

**Setup** (`theory/v015_checks/proxy_split_freeze_E9.json`, SHA-256 `f895853d…`; models `proxy_models_frozen_E9.json`, SHA-256 `c96065f0…`; both written before any new held-out panel):

- **Arms.** `fresh`, `fresh_W5` and `fresh_W20`. The warm arms run $`W\in\lbrace 5,20\rbrace`$ Adam steps at learning rate exactly 0 ($`\beta=(0.9,0.999)`$) on one declared IID prefix from `reference_even`. The parameters are unchanged; only $`m`$, $`v`$ and the step counter advance.
- **Fits.** The candidate menu, features, gold, ridge fits and metrics of Check 9.4 are unchanged. Only the original frozen pilot is used for fitting, and the coefficients are verified identical. The state-only comparison uses the original warm2-only fit.
- **Splits.** Five paired cold/warm splits: held-out states, task (`threshold`), $`\epsilon`$, learning rate, and both; 918 held-out rows in total. Random controls are paired across arms.
- **Seed.** `SeedSequence([150150, 109])`.

**Procedure.** For each split, proxy and $`W`$, compute the warm-minus-cold difference of each metric and average with equal weight per split. An improvement means lower regret, lower wrong-sign rate and higher Spearman.

**Result** (`theory/v015_checks/results_v015_independent.json`, entry `v015:E9`, `data.summary` and `cheap_proxy_results`; mean warm − cold difference over 5 splits, with the number of splits improved in parentheses):

|proxy|W|Δ regret|Δ wrong-sign|Δ Spearman|all three improve|
|---|---:|---|---|---|---|
|input moment|5|+0.00000 (0/5)|+0.0375 (1/5)|−0.030 (0/5)|no|
|input moment|20|+0.00021 (0/5)|+0.0625 (0/5)|−0.207 (0/5)|no|
|mean gradient|5|+0.00000 (0/5)|+0.0125 (1/5)|+0.042 (4/5)|no|
|mean gradient|20|+0.00021 (0/5)|+0.0500 (0/5)|−0.150 (0/5)|no|
|Δθ|5|+0.00000 (0/5)|+0.0750 (0/5)|−0.082 (2/5)|no|
|Δθ|20|+0.00021 (0/5)|+0.0375 (0/5)|−0.200 (0/5)|no|
|Δθ+m,v|5|+0.00000 (0/5)|+0.0500 (1/5)|−0.072 (2/5)|no|
|Δθ+m,v|20|+0.00021 (0/5)|+0.0750 (0/5)|−0.191 (0/5)|no|
|sign only|5|−0.00062 (3/5)|+0.0625 (0/5)|−0.052 (0/5)|no|
|sign only|20|−0.00090 (4/5)|+0.1250 (0/5)|−0.270 (0/5)|no|
|short tail 1|5|−0.00625 (1/5)|−0.0187 (2/5)|+0.053 (4/5)|yes|
|short tail 1|20|−0.00662 (1/5)|+0.0063 (2/5)|−0.152 (2/5)|no|
|short tail 4|5|−0.00626 (3/5)|+0.0000 (2/5)|−0.028 (2/5)|no|
|short tail 4|20|−0.00662 (3/5)|+0.0000 (2/5)|+0.039 (5/5)|no|
|random|5|−0.00785 (4/5)|−0.0125 (2/5)|−0.024 (2/5)|no|
|random|20|−0.00996 (4/5)|−0.0875 (3/5)|+0.048 (3/5)|yes|

Random-selection regret by split, cold → $`W=5`$ / $`W=20`$ (`comparisons[proxy=random]`):

| split | cold | W=5 | W=20 |
|---|---:|---:|---:|
| held-out states | 0.0067 | 0.0042 | 0.0034 |
| held-out task (`threshold`) | 0.0430 | 0.0098 | 0.0039 |
| held-out $`\epsilon`$ | 0.0076 | 0.0042 | 0.0035 |
| held-out lr | 0.0089 | 0.0086 | 0.0054 |
| held-out $`\epsilon`$ + lr | 0.0047 | 0.0049 | 0.0049 |

**Why it supports Rule 3.**

1. *Warm-up compresses the differences between candidates.* The regret of a uniformly random choice measures the spread of gold contrasts across the menu. It falls in 4 of 5 splits, most on the threshold task (0.043 to 0.004): after warm-up there is less to choose between. This is consistent with [Proposition 5.6](../proofs/05_warm_adam.md#s5-6): at a warm state the response depends, up to the stated residual, on the mean preconditioned gradient, so candidates that differ mainly in their sign structure move closer together.
2. *Warm-up does not make the static proxies predictive.* For the four static proxies other than sign only, regret does not improve in any split, the wrong-sign rate rises by 1–8 points and Spearman falls by 0.15–0.21 at $`W=20`$. The sign-only proxy gains a little regret but loses 6–13 points of sign accuracy. No static proxy/$`W`$ pair improves all three metrics.
3. *Consequence for the method.* Warmth determines *which* statistic carries the information (Rule 3). The static proxies used here are not that statistic at either state, so responses must still be measured, at states whose warmth is recorded ([Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2), steps 1 and 5).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C17,C23 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # package check
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/s9.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entry `v014:C17`, `controlled_neural.details.regrouping`, `controlled_neural.details.additive_null_parameter_residual`.
- `theory/v014_repro/results_v014_rerun.json`: `coupling_and_additive_null`.
- `theory/v015_checks/results_v015_independent.json`: entry `v015:E6` (`reports`, `negative_results`, `sign_pair_audit`) and entry `v015:E9` (`summary`, `cheap_proxy_results`, `comparisons`, `freeze_events`).
- Frozen contracts: `theory/v015_checks/proxy_split_freeze.json`, `proxy_models_frozen.json`, `proxy_split_freeze_E9.json`, `proxy_models_frozen_E9.json`.
- Specifications: `theory/v015_checks/SPEC.md` (`v015:E6`) and `theory/v015_checks/README.md` (`v015:E9` methods).
- Figures: `../figures/fig9_1_coupling.{pdf,png}` and `../figures/fig9_2_selection.{pdf,png}` (specs `../figures/specs/fig9_1.md`, `fig9_2.md`).
