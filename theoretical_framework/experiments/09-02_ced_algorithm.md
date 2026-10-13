# Algorithm 9.2 · Consequence-Calibrated Experience Distillation, CED

*Experiments*

[← Framework](../README.md) · [Statement](../statements/09-02_ced_algorithm.md) · [Proof](../proofs/09-02_ced_algorithm.md) · [Experiments](09-02_ced_algorithm.md) · [中文](09-02_ced_algorithm_ZH.md)

**What is tested here.** The algorithm as a whole has not been run on a real model; the planned real-model tests R0–R3 are listed in [Remark 9.4](../statements/09-04_scope_of_evidence.md#s9-4). What is tested so far, in a controlled small network, is the choice made in steps 1 and 5: rank candidates by a measured update-plus-tail response at a recorded complete state, not by a static statistic.

- **Measured responses select; the tested static proxies do not.** On frozen held-out panels, the five static proxies tested have wrong-sign rate 0.359–0.547 on held-out learner states (random 0.547) and Spearman at most 0.10. The measured 4-update tail has wrong-sign rate at most 0.031 and zero regret on every state, $`\epsilon`$, learning-rate and warmth shift. On a held-out task the short tails select worse than random; only the full tail is exact on every split ([Check 9.4](09-03_derivation_design_rules.md#c9-4)).
- **Warmth must be recorded, but warming does not rescue static proxies.** An lr = 0 moment warm-up compresses the differences between candidates, yet none of the 10 static proxy/warm-up combinations improves regret, wrong-sign rate and Spearman together ([Check 9.5](09-03_derivation_design_rules.md#c9-5)).
- Both checks are declared on the experiments page of [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3), whose design rules they mainly test; below are summaries for this algorithm.

**Setting used on this page.**

- *Learner.* A controlled network: one-hot context, 4 tanh units, sigmoid output, mean cross-entropy, float64. Its complete states (parameters plus Adam moments $`m,v`$ and step counter) are `fresh`, `warm2`, `warm5`, `warm9`; recipes are Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\eta\in\lbrace0.01,0.1\rbrace`$ and SGD with $`\eta\in\lbrace0.1,1\rbrace`$, each with $`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$.
- *Candidates.* Nine label laws on ten two-label packets, one of which (`reference_even`) is the reference $`P^0`$.
- *Gold.* The exact learning value of a candidate: one candidate boundary followed by 7 reference updates (8 updates in total), enumerated over all finite histories, minus the same quantity for the reference. This is the advantage $`A_t(z,j)`$ of [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) at state $`z`$.
- *Proxies.* Static statistics (input-moment error, mean gradient, sign only, first-step $`\Delta\theta`$, $`\Delta\theta`$ with $`(m,v)`$); measured short-tail responses after 1 and after 4 updates (surrogates in the sense of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)); controls (random score, full tail). Each proxy is fitted by ridge regression on a pilot only and frozen.
- *Metrics.* Selection regret (gold value of the best candidate minus that of the proxy-selected one), wrong-sign rate (fraction of candidates whose predicted contrast has the opposite sign to the gold), Spearman correlation with the gold contrast.

**Statement under test ([Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2)).** Given a student with its declared recipe, $`K`$ trusted demonstrations and a budget of $`n`$ permitted experiments, candidate feedback laws $`P^1,\dots,P^M`$ and a reference $`P^0`$, CED, in step 1, records the complete learner state $`z_t=(\theta_t,o_t)`$ and its warmth at every checkpoint; in step 2 acquires queries by learner-relative water-filling before compiling; in step 3 compiles candidates as packet laws; in step 4 records each candidate's optimizer statistic and flags oversized messages; in step 5 measures each candidate's consequence at the actual state by CRN paired-replacement scores $`\hat S^{jk}`$ with radii $`e^{jk}_{\rm stat}`$, calibrated against a few full-tail audits, using first-order scores only as a screen where the margin dominates; in step 6 selects $`j_t\in\arg\max_j[\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}]`$, keeping the reference if no lower bound is positive, trains, re-audits, freezes, and reports method and policy transfer separately. Its guarantees are (G1) a per-checkpoint certificate, (G2) a whole-run guarantee $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$ with full-tail audits, (G3) no loss of information from post-processing, and (G4) otherwise an empirical claim.

| Check | Tests for this algorithm | Code ID | Result |
|---|---|---|---|
| [Check 9.4](09-03_derivation_design_rules.md#c9-4) | step 5: measured responses vs static proxies; tail length | `v015:E6` | controlled finding |
| [Check 9.5](09-03_derivation_design_rules.md#c9-5) | steps 1 and 5: warmth and proxy transfer | `v015:E9` | controlled finding |

Related checks on other pages: the comparison of first-order rankers with the CRN score ([Check 7.5](07-05_where_first_order_rankers_fail.md#c7-5)) and the paired-replacement certificate itself are reported with the statements they test ([Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5), [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8)).

### [Check 9.4](09-03_derivation_design_rules.md#c9-4) · Held-out selection with frozen proxies

**What it tests for this algorithm.** Whether step 5 could be replaced by a cheaper statistic: if a static proxy, fitted once and frozen, selected as well as a measured response on held-out learner states, tasks, recipes and warmth, the measurement of step 5 would be unnecessary.

**Setup.**

- Proxies fitted on a 72-row pilot (states `fresh`, `warm2`; tasks `moment`, `parity`; two recipes) and frozen on disk before any held-out measurement.
- Seven held-out splits (held-out states, $`\epsilon`$ only, learning rate only, both, fresh-to-warm, warm-to-fresh, held-out task `threshold`); 432 held-out rows; each split evaluated once.

**Key numbers** (`v015:E6`).

| split | static proxies, wrong-sign | short tail 4, wrong-sign | random, wrong-sign | short tail 1 / 4, regret | random, regret | full tail, regret |
|---|---:|---:|---:|---:|---:|---:|
| held-out states | 0.359–0.547 | 0.016 | 0.547 | 0.0000 / 0.0000 | 0.0056 | 0.0000 |
| fresh to warm | 0.359–0.562 | 0.016 | 0.578 | 0.0000 / 0.0000 | 0.0045 | 0.0000 |
| held-out task | 0.083–0.250 | 0.042 | 0.458 | 0.0439 / 0.0322 | 0.0186 | 0.0000 |

- On held-out states the static proxies have Spearman between −0.00 and 0.10; the 4-update tail has 0.99.

**Verdict.** Controlled finding in support of step 5. Measured responses select and the tested static proxies do not; but a short surrogate tail calibrated on one task family can lose its ordering on another, which is why step 5 calibrates the surrogate against full-tail audits and [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11) treats calibration transfer as something to check. This concerns the proxies tested, not all static statistics.

### [Check 9.5](09-03_derivation_design_rules.md#c9-5) · Proxy transfer under lr = 0 moment warm-up

**What it tests for this algorithm.** Whether recording the complete state and its warmth (step 1) could be avoided by warming the optimizer moments first, so that static statistics become predictive at the warm state.

**Setup.**

- Arms `fresh`, `fresh_W5`, `fresh_W20`: $`W\in\lbrace5,20\rbrace`$ Adam steps at learning rate exactly 0, so only $`m`$, $`v`$ and the step counter advance.
- Same candidates, gold, frozen fits and metrics as [Check 9.4](09-03_derivation_design_rules.md#c9-4); five paired cold/warm held-out splits, 918 rows.

**Key numbers** (`v015:E9`, mean warm − cold over 5 splits).

- Random-selection regret falls in 4 of 5 splits, on average by 0.0079 at $`W=5`$ and by 0.0100 at $`W=20`$ (on the threshold task from 0.043 to 0.004 at $`W=20`$).
- None of the 10 static proxy/$`W`$ combinations improves regret, wrong-sign rate and Spearman together; their wrong-sign rate rises by 1–13 points.

**Verdict.** Controlled finding in support of steps 1 and 5. Warmth changes which statistic carries the information, and the static proxies tested are not that statistic at either state, so responses must still be measured, at states whose warmth is recorded.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E6,E9 --output theory/v015_checks/quick/s9.json
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: entry `v015:E6` (`reports`, `negative_results`, `sign_pair_audit`) and entry `v015:E9` (`summary`, `cheap_proxy_results`, `comparisons`, `freeze_events`).
- Frozen contracts: `theory/v015_checks/proxy_split_freeze.json`, `proxy_models_frozen.json`, `proxy_split_freeze_E9.json`, `proxy_models_frozen_E9.json`.
