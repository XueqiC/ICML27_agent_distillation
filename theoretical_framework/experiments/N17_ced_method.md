# N17 · Consequence-Calibrated Experience Distillation (CED) — Experiments

[← Framework map](../README.md) · [Derivation](../proofs/N17_ced_method.md) · [中文](N17_ced_method_ZH.md)

**Verdict.** The controlled-learner experiments establish the facts behind CED's design rules 2–4.

- **Static statistics fail (E6).** Static update statistics do not predict the sign of the full-tail response on held-out learner states. The input-moment error, mean gradient, sign pattern, $\Delta\theta$ and $\Delta\theta$ with $(m,v)$ give wrong signs at 36–55% of providers; random scores give 55%. Their Spearman correlation with the gold contrast is at most 0.10. Selection must therefore use measured responses (rule 2).
- **Measured short tails work, with one limit (E6).** The 4-update tail has 1.6–3.1% wrong signs and zero selection regret on every state, $\epsilon$, learning-rate and warmth shift. On the held-out threshold task, both short tails select worse than random: regret 0.044 (1 update) and 0.032 (4 updates), against 0.019. The full tail is the only measurement with zero regret and zero wrong signs on every split. Tail length is therefore part of the estimand.
- **Warm-up changes which statistic matters, not whether static statistics suffice (E9).** An lr = 0 moment warm-up compresses the differences between providers: random-selection regret falls in 4 of 5 held-out splits, on average by 0.0079 at W = 5 and 0.0100 at W = 20. It does not make static statistics predictive. None of the 8 static-proxy/W combinations improves regret, wrong-sign rate and Spearman together, and their wrong-sign rate rises by 1–8 points. This matches rule 3.

All results are exact finite-history computations in a controlled small network, with frozen pilot fits and pre-registered splits. They are not real-model or benchmark results.

![Which measurements select feedback correctly](../figures/N17_selection.png)

[PDF](../figures/N17_selection.pdf)

## E6 · Held-out provider selection with frozen proxies

**Why.** Rule 2 says a provider must be judged by its actual update-plus-tail response. E6 tests whether cheaper statistics can stand in for that response when the learner state, the task, the recipe or the optimizer warmth changes. Each change is a separate held-out split.

**Setup** (`theory/v015_checks/proxy_split_freeze.json`, SHA-256 `ac8f4aee…`, written before any held-out measurement):

- **Learner.** The C23 network (one-hot context, 4 tanh units, sigmoid, mean cross-entropy, float64).
  - States: `fresh`, `warm2`, `warm5`, `warm9` (complete states including Adam moments).
  - Recipes: 8, namely Adam with $\eta\in\lbrace 0.01,0.1\rbrace $ and SGD with $\eta\in\lbrace 0.1,1\rbrace $, each with $\epsilon\in\lbrace 10^{-8},10^{-3}\rbrace $.
- **Providers.** 9 label laws on 10 two-label packets: `reference_even`, moment-matched `moment3_P`/`moment3_Q`, sign-matched `low_sign_A`/`B` and `high_sign_A`/`B`, `odd_parity` and `middle_mixture`.
- **Gold.** One candidate boundary followed by 7 IID reference updates (8 updates in total), enumerated exactly over all finite histories. Its value is the contrast with the reference provider.
- **Proxies**, fitted by ridge regression on the pilot only and frozen in `proxy_models_frozen.json`:
  - static: input-moment error, mean gradient, sign-only, first-step $\Delta\theta$, and $\Delta\theta$ with $(m,v)$;
  - measured short-tail responses after 1 and after 4 updates;
  - controls: random score, task identity, training step.
- **Pilot.** States `fresh` and `warm2`, tasks `moment` and `parity`, recipes 0 and 4; 72 rows.
- **Held-out splits.**
  - states `warm5`/`warm9`;
  - task `threshold`;
  - $\epsilon$ only (recipes 1, 5);
  - learning rate only (2, 6);
  - both (3, 7);
  - fresh-only fit evaluated on warm states, and the reverse.
  - In total 432 held-out rows, with seed `SeedSequence([150150, 106])`.
- **Metrics.**
  - *Selection regret* is the gold value of the best provider minus that of the proxy-selected provider, averaged over cells.
  - *Wrong-sign rate* is the fraction of providers with nonzero gold whose predicted contrast has the opposite sign.
  - *Spearman* is the rank correlation of predicted and gold contrasts.

**Procedure.** Write the split manifest and the frozen proxy coefficients to disk. Then evaluate every held-out split once. No held-out consequence changes a proxy, its features, the ridge penalty or the abstention thresholds.

**Result** (`theory/v015_checks/results_v015_independent.json`, `groups[id=E6].data.reports.<split>.metrics.<proxy>`):

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

**What this shows.**

1. *Static statistics are not responses.* On held-out learner states, and when moving from fresh to warm states, the static proxies are at chance:
   - wrong-sign rates are 0.36–0.56, against 0.55–0.58 for random;
   - Spearman is at most 0.10.

   This is the controlled-learner form of N04 and N11–N12: the same update statistic maps to different learning consequences at different complete states.
2. *Measured responses are.* The 4-update short tail keeps wrong signs at or below 3.1% and regret at 0 on every state, $\epsilon$, learning-rate and warmth shift. The 1-update tail degrades under learning-rate shifts (wrong sign 0.39) but still selects nearly optimally (regret 0.0007).
3. *Tail length is part of the estimand.* On the held-out `threshold` task, the short tails select worse than random: regret 0.044 and 0.032, against 0.019. Their ordering differs from that of the 8-update full tail, which is the only measurement with zero regret and zero wrong signs on every split. Static proxies happen to reach zero regret on this split, but with 8–25% wrong signs and chance-level behaviour on held-out states, so they do not transfer across splits.
4. *Scale.* The regrets are of order $10^{-3}$–$10^{-2}$ in return units, and the panel has 8 cells per split. Cell bootstraps (`regret_bootstrap95`) describe variation across this finite panel. They are not population confidence intervals.

## E9 · Proxy transfer under lr = 0 moment warm-up

**Why.** N12 shows that warm optimizer states change the first-order statistic a provider must match. E9 asks whether warming the moments makes the static statistics, which failed in E6, predictive.

**Setup** (`theory/v015_checks/proxy_split_freeze_E9.json`, SHA-256 `f895853d…`; models `proxy_models_frozen_E9.json`, SHA-256 `c96065f0…`; both written before any new held-out panel):

- **Arms.** `fresh`, `fresh_W5` and `fresh_W20`. The warm arms run $W\in\lbrace 5,20\rbrace $ Adam steps at learning rate exactly 0 ($\beta=(0.9,0.999)$) on one declared IID prefix from `reference_even`. The parameters are unchanged; only $m$, $v$ and $t$ advance.
- **Fits.** The E6 provider menu, features, gold, ridge fits and metrics are unchanged. Only the original frozen E6 pilot is used for fitting, and the coefficients are verified identical. The state-only comparison uses the original warm2-only fit.
- **Splits.** Five paired cold/warm splits: held-out states, task (`threshold`), $\epsilon$, learning rate, and both; 918 held-out rows in total. Random controls are paired across arms.
- **Seed.** `SeedSequence([150150, 109])`.

**Procedure.** For each split, proxy and $W$, compute the warm-minus-cold difference of each metric. Average with equal weight per split. An improvement means lower regret, lower wrong-sign rate and higher Spearman.

**Result** (`groups[id=E9].data.summary` and `cheap_proxy_results`; the mean warm − cold difference over 5 splits is shown, with the number of splits improved in parentheses):

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

Random-selection regret by split, cold → W=5 / W=20 (`comparisons[proxy=random]`):

| split | cold | W=5 | W=20 |
|---|---:|---:|---:|
| held-out states | 0.0067 | 0.0042 | 0.0034 |
| held-out task (`threshold`) | 0.0430 | 0.0098 | 0.0039 |
| held-out $\epsilon$ | 0.0076 | 0.0042 | 0.0035 |
| held-out lr | 0.0089 | 0.0086 | 0.0054 |
| held-out $\epsilon$ + lr | 0.0048 | 0.0049 | 0.0049 |

**What this shows.**

1. *Warm-up compresses differences between providers.* The regret of a uniformly random choice measures the spread of gold contrasts across the menu, and it falls in 4 of 5 splits. The largest fall is on the threshold task, from 0.043 to 0.004. After warm-up there is less to choose between. This is consistent with W1–W3: at a warm state the response depends on the mean preconditioned gradient, so providers that differ mainly in their sign structure move closer together.
2. *Warm-up does not make static statistics predictive.* For the four static proxies, regret does not improve in any split, the wrong-sign rate rises by 1–8 points, and Spearman falls by 0.15–0.21 at $W=20$. The count is 0/8 for proxy/W pairs improving all three metrics. The sign-only feature gains a little regret but loses 6–13 points of sign accuracy.
3. *Consequence for the method.* Under rule 3, warmth determines *which* statistic carries the information. The static proxies used here are not that statistic at either state. Responses must still be measured, at states whose warmth is recorded.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/n17.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: `groups[id=E6]` (`reports`, `negative_results`, `sign_pair_audit`) and `groups[id=E9]` (`summary`, `cheap_proxy_results`, `comparisons`, `freeze_events`).
- Frozen contracts: `theory/v015_checks/proxy_split_freeze.json`, `proxy_models_frozen.json`, `proxy_split_freeze_E9.json` and `proxy_models_frozen_E9.json`.
- Specification: `theory/v015_checks/SPEC.md` (E6), and `theory/v015_checks/README.md` (E9 methods).
- Figure: `../figures/N17_selection.pdf` and `.png` (spec `../figures/specs/N17.md`).
