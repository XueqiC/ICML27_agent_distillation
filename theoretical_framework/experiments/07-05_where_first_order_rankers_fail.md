# Remark 7.5 · Where first-order rankers fail

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-05_where_first_order_rankers_fail.md) · [Experiments](07-05_where_first_order_rankers_fail.md) · [中文](07-05_where_first_order_rankers_fail_ZH.md)

**What is tested here.** Whether first-order rankers are least reliable where the second-moment channel dominates, on a trained controlled network with real optimizer states (descriptive; all assertions PASS).

- Every first-order or cosine ranker has a wrong-sign rate between 11.4% and 28.8% on 1,687 provider pairs.
- The full-state first-order score, which uses the exact continuation gradient and all three channels, is wrong 51%–67% of the time where the second-moment share is at least 0.5 (59.6% on the 161 such pairs).
- The finite paired-replacement score of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) is wrong 2.8% of the time overall and 4.3% at high share (7 of 161 pairs, against 96 for the full-state score).

**Setting used on this page.**

- *Candidates and gold.* Pairs of provider (feedback) laws applied at the same complete state of a small network trained by Adam or SGD; the *gold* difference of a pair is the exact difference of the eight-update outcomes (one candidate round followed by the reference tail).
- *Wrong sign.* A ranker is wrong on a pair when (score difference) × (gold difference) < 0; exact ties are counted separately and are not errors.
- *Channel contractions and second-moment share.* With the exact continuation gradient $`(\chi^\theta,\chi^m,\chi^v)`$ of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1), the $`\theta`$-, $`m`$- and $`v`$-channel contractions of a pair's increment contrast sum to its first-order difference; the second-moment share is $`\lvert v\text{-contraction}\rvert/(\lvert\theta\rvert+\lvert m\rvert+\lvert v\rvert\text{ contractions})`$.
- *Rankers.* First-order scores (parameter-only, full-state with the exact gradient, CRN-measured directional differences), LESS-style cosine scores ([Xia et al., 2024](#ref-xia2024less)), and the finite paired-replacement score $`\hat S`$ (one round replaced by each candidate, a common reference tail, $`r`$ CRN replicas; common random numbers as in [Glasserman, 2004](#ref-glasserman2004monte)).

**Statement under test ([Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5)).** A high second-moment share means the increment is large relative to $`v`$, the saturating regime of [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)(a),(c), where the linearization overstates the $`v`$-channel harm. So any first-order score, including the full-state score with the exact continuation gradient, is unreliable there, and ranking there should use the finite-replacement measurement of Theorem 8.8 or the nonlinear formula.

<a id="c7-5"></a>

## Check 7.5 · Rankers on a controlled network, by second-moment share (Proposition 7.3, Remark 7.5)

**Verdict.** Descriptive; all its assertions PASS.

- Every first-order or cosine ranker has a wrong-sign rate between 11.4% and 28.8% on 1,687 provider pairs.
- The full-state first-order score uses the exact continuation gradient and all three channels. It is wrong 51%–67% of the time where the second-moment share is at least 0.5.
- The finite paired-replacement score of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) is wrong 2.8% of the time overall and 4.3% at high share (7 of 161 pairs, against 96 for the full-state score).

**Why.**

- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5) predicts that first-order rankers are least reliable where the second-moment channel dominates, because that is where the message is large relative to $`v`$.
- [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3) predicts that parameter-only and cosine scores are not consistent rankers.
- The check tests both on a trained network with real optimizer states, where the assumptions of [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2) do not hold.

**Setup.**

- **Panel.** The frozen controlled-network provider panel of the provider-selection experiments ([Check 9.4](09-03_derivation_design_rules.md#c9-4)), read only and unchanged.
  - The network has a one-hot context, 4 tanh units, a sigmoid output and mean cross-entropy, in float64.
  - Complete states `fresh`, `warm2`, `warm5`, `warm9`. Eight recipes: Adam with $`\eta\in\lbrace0.01,0.1\rbrace`$ and SGD with $`\eta\in\lbrace0.1,1\rbrace`$, each with $`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$.
  - Nine provider laws on two-label packets, and the frozen held-out splits.
  - All 36 unordered provider pairs at every split and cell: 1,728 pairs, 1,687 with nonzero gold.
- **Gold.** The exact difference of the eight-update outcomes: one candidate boundary followed by all 128 seven-update reference histories.
- **Rankers** (pair score = difference of provider scores):
  - LESS-style cosine between the provider's mean parameter step and the task-objective gradient. For SGD the primary version uses the positive loss gradient, as specified; the update-oriented variant flips it to the update direction. This is a LESS-style proxy, not the official LESS implementation.
  - Parameter-only first-order score: the exact continuation gradient restricted to $`\theta`$, contracted with the mean $`\Delta\theta`$.
  - Full-state first-order score: the $`\theta`$, $`m`$ and $`v`$ contractions with the exact continuation gradient, the oracle score of the first-order audit ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)).
  - CRN-measured first-order scores, which use no gradient oracle: a common-random-number directional difference along the mean increment, with the task-aligned surrogate or with a short-tail surrogate.
  - Finite paired-replacement score $`\hat S`$ of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8), task-aligned surrogate, $`r=256`$ replicates.
- **Second-moment share** of a pair: $`\lvert v\text{-contraction}\rvert/(\lvert\theta\rvert+\lvert m\rvert+\lvert v\rvert\text{ contractions})`$ of the pair's increment contrast, computed with the exact continuation gradient.
  - Fixed bins $`[0,.25),[.25,.5),[.5,.75),[.75,1]`$.
  - Unavailable for the nonsmooth threshold task.
  - SGD pairs have share 0.
- **Conventions.** A wrong sign means score × gold < 0. Exact ties are counted separately and are not errors.
- **Assertions.**
  - The full-state oracle scores and margin rows reproduce the frozen records of the first-order margin audit ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)): 56 margin rows out of 396, and no wrong sign on them.
  - The finite-replacement scores reproduce the `v016:C6b` measurements exactly.
  - The frozen files are unchanged.

**Result.** Wrong-sign rate (wrong / pairs with nonzero gold; ties in parentheses where present):

| Ranker | all, 1,687 | share $`[0,.25)`$, 1,318 | $`[.25,.5)`$, 84 | $`[.5,.75)`$, 76 | $`[.75,1]`$, 85 | share $`\ge.5`$, 161 | threshold task, 124 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cosine (LESS-style) | 28.8% (345 ties) | 33.1% | 8.3% | 19.7% | 28.2% | 24.2% | 3.2% |
| cosine, update-oriented | 15.6% (345 ties) | 16.3% | 8.3% | 19.7% | 28.2% | 24.2% | 1.6% |
| parameter-only, exact gradient | 14.7% (120 ties) | 13.8% | 23.8% | 15.8% | 37.6% | 27.3% | 1.6% |
| full-state, exact gradient | 16.4% (118 ties) | 13.1% | 7.1% | **51.3%** | **67.1%** | **59.6%** | 1.6% |
| CRN first-order, aligned surrogate | 13.2% (323 ties) | 7.8% | 31.0% | 47.4% | 67.1% | 57.8% | 0.0% |
| CRN first-order, short surrogate | 11.4% (323 ties) | 6.4% | 26.2% | 38.2% | 65.9% | 52.8% | 0.0% |
| finite paired replacement, $`r=256`$ | **2.8%** (0 ties) | 2.9% | 2.4% | 7.9% | 1.2% | **4.3%** | 0.8% |

- **High share.** At share at least 0.5 the finite score is wrong on 7 of 161 pairs. The full-state score is wrong on 96, the CRN first-order scores on 93 and 85, and the parameter-only score on 44. The finite score improves on every first-order ranker there but does not eliminate errors.
- **Where errors concentrate.** The parameter-only score is wrong 14.4% of the time below share 0.5 and 27.3% above; its errors concentrate at high share. The primary cosine is wrong 31.6% below and 24.2% above; its errors do not concentrate.
- **By recipe** (wrong-sign rate):

| Recipe | pairs | cosine | param.-only | full-state | CRN 1st-order (aligned) | finite replacement |
|---|---:|---:|---:|---:|---:|---:|
| Adam, $`\eta=0.01`$, $`\epsilon=10^{-8}`$ | 395 | 5.6% | 4.1% | 11.4% | 13.9% | 2.3% |
| Adam, $`\eta=0.01`$, $`\epsilon=10^{-3}`$ | 144 | 19.4% | 9.0% | 18.8% | 20.8% | 2.8% |
| Adam, $`\eta=0.1`$, $`\epsilon=10^{-8}`$ | 144 | 13.9% | 27.1% | 13.2% | 29.2% | 2.8% |
| Adam, $`\eta=0.1`$, $`\epsilon=10^{-3}`$ | 144 | 16.0% | 9.0% | 12.5% | 13.2% | 0.7% |
| SGD, $`\eta=0.1`$, $`\epsilon=10^{-8}`$ | 428 | 44.9% | 12.9% | 12.9% | 2.1% | 1.6% |
| SGD, $`\eta=0.1`$, $`\epsilon=10^{-3}`$ | 144 | 49.3% | 18.1% | 18.1% | 6.2% | 2.1% |
| SGD, $`\eta=1`$, $`\epsilon=10^{-8}`$ | 144 | 45.1% | 29.9% | 29.9% | 21.5% | 7.6% |
| SGD, $`\eta=1`$, $`\epsilon=10^{-3}`$ | 144 | 45.1% | 29.9% | 29.9% | 18.8% | 6.2% |

- **By held-out split**, finite replacement against the full-state score:
  - held-out states: 0.0% vs 9.7%, 288 pairs;
  - held-out task: 1.0% vs 0.0%, 103 pairs, on which every first-order and cosine score ties;
  - held-out $`\epsilon`$: 2.4% vs 18.4%, 288 pairs;
  - held-out learning rate: 5.2% vs 21.5%, 288 pairs;
  - held-out $`\epsilon`$ and learning rate: 3.5% vs 21.2%, 288 pairs;
  - fresh to warm: 0.0% vs 9.7%, 288 pairs;
  - warm to fresh: 10.4% vs 30.6%, 144 pairs.
- **Reproduction.** The full-state scores match the frozen records to $`2.8\times10^{-17}`$ (tolerance $`2\times10^{-6}`$). The margin rows are 56/396, recomputed and frozen, with zero wrong signs on them.
- **Cost.** The exact-gradient scores used 1,044,480 oracle tail evaluations plus 184,320 for an independent directional check. The finite score reuses the `v016:C6b` measurements at no additional tail cost; that cost is reported with the checks of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8).

![Figure 7.2: ranker error by second-moment share](../figures/fig7_2_rankers.png)

*Figure 7.2.* Wrong-sign rate of each ranker on the controlled provider panel, overall and by second-moment share of the pair. Every first-order ranker, including the full-state score with the exact continuation gradient, fails most where the second-moment channel dominates. The finite paired-replacement score stays low in every bin. [PDF](../figures/fig7_2_rankers.pdf)

**Why it supports the statement.**

- The full-state first-order score is exact to first order, yet it is wrong on most high-share pairs. This is the failure Remark 7.5 predicts: where the $`v`$-channel matters, the first-order form misjudges it.
- The CRN-measured first-order scores fail in the same bins. The failure therefore comes from linearization, not from estimating the gradient.
- Measuring the finite replacement removes most of these errors. This is the remedy Remark 7.5 names.
- Cosine and parameter-only scores are unreliable across recipes. This is consistent with [Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes.md#s7-3), although the panel does not isolate the amplitude mechanism of Proposition 7.3(b) (the constructed-pair check on the [experiments page of Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#c7-4) does).

**Limits.**

- One small controlled network with a finite panel.
- Rates are descriptive, not estimates over independent tasks, and repeated cells appear in several splits.
- The LESS-style cosine is a proxy for, not a reproduction of, the LESS implementation.
- No real-model claim is made.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C5,C5b
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- The groups take about 22 s (`v016:C5`) and 3 s (`v016:C5b`) on rai.
- `v016:C5` and `v016:C5b` import the frozen controlled-network panel read-only. The pinned files are hashed before and after.
- Seeds: `numpy.SeedSequence([160160, group, subcase, ...])`, PCG64, float64, single BLAS thread.

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)).

## Artifacts

- `theory/v016_checks/results_v016.json` — claim ids `v016:C5`, `v016:C5b`: ranker tables, share bins, reproduction records.
- `theory/v016_checks/results_v016_raw.json.gz` — raw arrays referenced by JSON pointers. Not committed; SHA-256 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`.
- `theory/v016_checks/SPEC.md` (sections on `v016:C5`/`v016:C6` and `v016:C5b`), `c5c6.py`, `c5c6b.py` and their tests.
- `figures/fig7_2_rankers.{png,pdf}` and its spec in `figures/specs/`.
