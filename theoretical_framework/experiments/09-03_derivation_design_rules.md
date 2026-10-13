# Remark 9.3 · Derivation of the design rules

*Experiments*

[← Framework](../README.md) · [Statement](../statements/09-03_derivation_design_rules.md) · [Proof](../proofs/09-03_derivation_design_rules.md) · [Experiments](09-03_derivation_design_rules.md) · [中文](09-03_derivation_design_rules_ZH.md)

**What is tested here.** The design rules are necessary conditions with proved derivations; what the checks below test is whether, in a controlled learner, following Rules 2, 3 and 5 actually matters for selection: can cheaper statistics stand in for a measured response, and does warming the optimizer change that?

- **Rules 2 and 5: measured responses select; the tested static proxies do not** ([Check 9.4](#c9-4)). On held-out learner states the five static proxies tested have wrong-sign rate 0.359–0.547 (random 0.547) and Spearman at most 0.10. The measured 4-update tail has wrong-sign rate at most 0.031 and zero regret on every state, $`\epsilon`$, learning-rate and warmth shift. On the held-out threshold task the 1- and 4-update tails have regret 0.044 and 0.032, against 0.019 for random; only the full tail has zero regret and zero wrong signs on every split.
- **Rule 3: warmth matters, but warming does not rescue static proxies** ([Check 9.5](#c9-5)). An lr = 0 moment warm-up lowers random-selection regret in 4 of 5 splits (by 0.0079 at $`W=5`$ and 0.0100 at $`W=20`$ on average), while none of the 10 static proxy/$`W`$ combinations improves regret, wrong-sign rate and Spearman together.
- **Scope.** Exact finite-history computations in a controlled small network; no real-model or benchmark result. These are controlled findings about the proxies tested, not proofs about all static statistics.

**Setting used on this page.**

- *Learner.* A controlled network: one-hot context, 4 tanh units, sigmoid output, mean cross-entropy, float64 (the network of `v014:C23`). Complete states, including Adam moments $`m,v`$ and the step counter: `fresh`, `warm2`, `warm5`, `warm9`. Recipes: 8, namely Adam with $`\eta\in\lbrace 0.01,0.1\rbrace`$ and SGD with $`\eta\in\lbrace 0.1,1\rbrace`$, each with $`\epsilon\in\lbrace 10^{-8},10^{-3}\rbrace`$.
- *Candidates.* Nine label laws on ten two-label packets; `reference_even` is the reference $`P^0`$.
- *Gold.* The exact learning value of a candidate: one candidate boundary followed by 7 IID reference updates (8 updates in total), enumerated over all finite histories, as a contrast with the reference candidate. This is the advantage $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$ of candidate $`j`$ at state $`z`$ ([Definition 1.4](../README.md#s1-4)).
- *Proxies.* A proxy is a cheap score meant to predict the gold contrast. Static proxies use only statistics of the candidate's first update (input moments, mean gradient, sign, $`\Delta\theta`$, $`\Delta\theta`$ with $`(m,v)`$). Measured short-tail proxies run the update and then 1 or 4 reference updates: surrogate responses with a truncated tail, in the sense of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6). Controls are a random score and the full tail itself.
- *Metrics.* Selection regret (gold value of the best candidate minus that of the proxy-selected candidate, averaged over cells); wrong-sign rate (fraction of candidates with nonzero gold whose predicted contrast has the opposite sign); Spearman rank correlation of predicted and gold contrasts.

**Statement under test ([Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3)).** [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) implements six rules, each a necessary condition for the corresponding guarantee: **Rule 1** acquire before compiling (facts the transcript does not determine can be supplied only by new permitted experiments; how much and where is set by the learner); **Rule 2** select by measured consequence (a selection rule is consistent with the learning-value ordering only if it is a function of the mean responses $`\lbrace\bar g_t(z,j)\rbrace_j`$, or of a statistic proved sufficient for them on the declared family); **Rule 3** match the learner's statistic, not the raw gradient (what must be matched to first order is the mean increment of the complete learner state in the directions of the continuation-value gradients, so responses are indexed by complete states and their warmth); **Rule 4** fit stable response models (minimize $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$); **Rule 5** account for coverage, certify by paired replacement, use first-order audits only where the margin dominates (a guarantee takes one of four forms: frozen-audit certificate with transport, actual-state full-tail lower-confidence-bound guarantee, CRN paired-replacement certificate with a calibrated surrogate, or a labelled empirical comparison); **Rule 6** freeze, then test transfer (everything frozen before the data that evaluates it; transfer only through a verified decomposition; method and policy transfer reported separately). The rules do not by themselves show that an instance of CED beats a given baseline.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 9.4](#c9-4) | Rules 2 and 5; [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2), step 5 | `v015:E6` | controlled finding |
| [Check 9.5](#c9-5) | Rule 3; Algorithm 9.2, steps 1 and 5 | `v015:E9` | controlled finding |

Related checks on other pages: the comparison of first-order rankers with the CRN paired-replacement score, which bears on Rule 5, is [Check 7.5](07-05_where_first_order_rankers_fail.md#c7-5).

![Figure 9.2: which measurements select feedback correctly](../figures/fig9_2_selection.png)

*Figure 9.2.* Controlled-learner evidence for the selection rules. (A) Wrong-sign rate of frozen proxies on held-out learner states and on warmth, $`\epsilon`$ and learning-rate shifts (`v015:E6`): the tested static proxies are at the level of a random score, while measured short-tail responses are near zero. (B) Selection regret: short tails select near-optimally on held-out states but worse than random on the held-out threshold task; only the full tail is exact everywhere. (C) Effect of an lr = 0 moment warm-up (`v015:E9`): the spread between candidates shrinks (random regret falls) while the static proxies do not improve. Exact finite-history computations in a controlled small network. [PDF](../figures/fig9_2_selection.pdf)

<a id="c9-4"></a>

## Check 9.4 · Held-out selection with frozen proxies (Remark 9.3, Rules 2 and 5)

**Verdict.** Controlled finding. On held-out learner states the five static proxies tested have wrong-sign rate 0.359–0.547 (random 0.547) and Spearman at most 0.10. The measured 4-update tail has wrong-sign rate at most 0.031 and zero regret on every state, $`\epsilon`$, learning-rate and warmth shift. On the held-out threshold task the 1- and 4-update tails have regret 0.044 and 0.032, against 0.019 for random. The full tail has zero regret and zero wrong signs on every split.

**Why.** Rule 2 says a candidate must be judged by its actual update-plus-tail response. This check asks whether cheaper statistics can stand in for that response when the learner state, the task, the recipe or the optimizer warmth changes. Each change is a separate held-out split, so a proxy that only fits the pilot cannot pass. The short-tail proxies are measured one-boundary responses with a truncated tail, that is, surrogates in the sense of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6); the threshold split tests whether such a surrogate keeps its ordering on a new task, which is the calibration-transfer question of [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11).

**Setup** (`theory/v015_checks/proxy_split_freeze.json`, SHA-256 `ac8f4aee…`, written before any held-out measurement):

- **Learner.** The controlled network of `v014:C23` (one-hot context, 4 tanh units, sigmoid, mean cross-entropy, float64).
  - States: `fresh`, `warm2`, `warm5`, `warm9` (complete states, including Adam moments).
  - Recipes: 8, namely Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\eta\in\lbrace 0.01,0.1\rbrace`$ and SGD with $`\eta\in\lbrace 0.1,1\rbrace`$, each with $`\epsilon\in\lbrace 10^{-8},10^{-3}\rbrace`$.
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

1. *The tested static proxies are not responses.* On held-out learner states, and when moving from fresh to warm states, they are at the level of a random score: wrong-sign rates 0.36–0.56 against 0.55–0.58 for random, Spearman at most 0.10. The same update statistic maps to different learning consequences at different complete states, as [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1) and the optimizer-state results of [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) predict. This is a finding about the proxies tested, not a proof about all static statistics; proved inconsistency is limited to the classes of [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3).
2. *Measured responses select.* The 4-update tail keeps wrong signs at or below 3.1% and regret at 0 on every state, $`\epsilon`$, learning-rate and warmth shift. The 1-update tail degrades under learning-rate shifts (wrong-sign rate 0.39) but still selects nearly optimally (regret 0.0007).
3. *Tail length is part of the estimand.* On the held-out `threshold` task, the short tails select worse than random (regret 0.044 and 0.032 against 0.019), because their ordering differs from that of the 8-update full tail. The full tail is the only measurement with zero regret and zero wrong signs on every split. In the terms of [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11), a surrogate calibrated on one task family can lose its ordering on another, so calibration transfer has to be checked. The static proxies happen to reach zero regret on this split, but with 8–25% wrong signs, and they are at chance on held-out states, so they do not transfer across splits.
4. *Scale.* The regrets are of order $`10^{-3}`$–$`10^{-2}`$ in return units, and the panel has 8 cells per split (4 on the task and warm-to-fresh splits). The cell bootstraps (`regret_bootstrap95`) describe variation across this finite panel; they are not population confidence intervals.

**Limits.** One controlled small network, nine candidates and a finite panel; the static proxies are five specific statistics, so the check cannot show that every static statistic fails.

<a id="c9-5"></a>

## Check 9.5 · Proxy transfer under lr = 0 moment warm-up (Remark 9.3, Rule 3)

**Verdict.** Controlled finding. Warm-up compresses the differences between candidates: the regret of a random choice falls in 4 of 5 held-out splits, on average by 0.0079 at $`W=5`$ and by 0.0100 at $`W=20`$. It does not make the static proxies predictive: none of the 10 combinations of the five static proxies with $`W\in\lbrace 5,20\rbrace`$ improves regret, wrong-sign rate and Spearman together, and their wrong-sign rate rises by 1–13 points.

**Why.** At a warm optimizer state the first-order statistic a candidate must match changes ([Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8)). This check asks whether warming the moments makes the static statistics of [Check 9.4](#c9-4) predictive.

**Setup** (`theory/v015_checks/proxy_split_freeze_E9.json`, SHA-256 `f895853d…`; models `proxy_models_frozen_E9.json`, SHA-256 `c96065f0…`; both written before any new held-out panel):

- **Arms.** `fresh`, `fresh_W5` and `fresh_W20`. The warm arms run $`W\in\lbrace 5,20\rbrace`$ Adam steps at learning rate exactly 0 ($`\beta=(0.9,0.999)`$) on one declared IID prefix from `reference_even`. The parameters are unchanged; only $`m`$, $`v`$ and the step counter advance.
- **Fits.** The candidate menu, features, gold, ridge fits and metrics of [Check 9.4](#c9-4) are unchanged. Only the original frozen pilot is used for fitting, and the coefficients are verified identical. The state-only comparison uses the original warm2-only fit.
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

1. *Warm-up compresses the differences between candidates.* The regret of a uniformly random choice measures the spread of gold contrasts across the menu. It falls in 4 of 5 splits, most on the threshold task (0.043 to 0.004): after warm-up there is less to choose between. This is consistent with [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6): at a warm state the response depends, up to the stated residual, on the mean preconditioned gradient, so candidates that differ mainly in their sign structure move closer together.
2. *Warm-up does not make the static proxies predictive.* For the four static proxies other than sign only, regret does not improve in any split, the wrong-sign rate rises by 1–8 points and Spearman falls by 0.15–0.21 at $`W=20`$. The sign-only proxy gains a little regret but loses 6–13 points of sign accuracy. No static proxy/$`W`$ pair improves all three metrics.
3. *Consequence for the method.* Warmth determines *which* statistic carries the information (Rule 3). The static proxies used here are not that statistic at either state, so responses must still be measured, at states whose warmth is recorded ([Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2), steps 1 and 5).

**Limits.** The warm-up is at learning rate 0, so it isolates the optimizer moments and leaves the parameters fixed; warmth reached during real training also moves the parameters. Same controlled network and panel as [Check 9.4](#c9-4).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/s9.json
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: entry `v015:E6` (`reports`, `negative_results`, `sign_pair_audit`) and entry `v015:E9` (`summary`, `cheap_proxy_results`, `comparisons`, `freeze_events`).
- Frozen contracts: `theory/v015_checks/proxy_split_freeze.json`, `proxy_models_frozen.json`, `proxy_split_freeze_E9.json`, `proxy_models_frozen_E9.json`.
- Specifications: `theory/v015_checks/SPEC.md` (`v015:E6`) and `theory/v015_checks/README.md` (`v015:E9` methods).
- Figure: `../figures/fig9_2_selection.{pdf,png}` (spec `../figures/specs/fig9_2.md`).
