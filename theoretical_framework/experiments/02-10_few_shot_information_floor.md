# Theorem 2.10 · Few-shot information floor and optimal hard-budget querying

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-10_few_shot_information_floor.md) · [Proof](../proofs/02-10_few_shot_information_floor.md) · [Experiments](02-10_few_shot_information_floor.md) · [中文](02-10_few_shot_information_floor_ZH.md)

**What is tested here.** Three checks test the theorem. [Check 2.4](../experiments/02-10_few_shot_information_floor.md#c2-4) enumerates 100 small cases exhaustively: the Bayes-optimal risk equals (2.7)/(2.9), the best value-adaptive query tree equals greedy, and moving the positive part outside the expectation is wrong in 36 of 100 cases (PASS). [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6) computes 944 exact risk curves and trains actual learners: targeted queries beat extra demonstrations at equal budget, replay gains 0, and no learner goes below the floor (PASS). [Check 2.7](../experiments/02-10_few_shot_information_floor.md#c2-7) reproduces the closed forms on three independent package constructions (PASS).

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID, independent of the bits, and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

**Statement under test ([Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10)).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8):

**(a) Information floor.** Without queries, the minimum achievable expected regret after $`K`$ trajectories is

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

with the convention $`(1-q)^0=1`$. The marginal value of one more trajectory is

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) Optimal queries.** With weights of the unseen bits sorted as $`w_{(1)}\ge\dots\ge w_{(M_K)}`$,

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

and querying the $`n`$ heaviest unseen bits is optimal; adapting the queries to the revealed values cannot help. In the uniform case ($`w_j=1/d`$, each trajectory reveals one uniformly drawn context),

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

In general $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$, and the inequality can be strict.

**Other statements tested on this page.**

- [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9): given the transcript, the unrevealed bits stay independent fair coins (tested through its consequences in [Check 2.4](../experiments/02-10_few_shot_information_floor.md#c2-4)).
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): replay, an environment-independent post-processing, cannot lower the information term ([Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6)).
- The algorithm term $`\mathcal R_{\rm alg}`$ of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2): an actual learner's risk above the floor is its algorithm error ([Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6)).
- [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11): demonstration KL plus query KL must reach $`\mathrm{kl}(1-\delta,\delta)`$ (package construction `pkg:T11` in [Check 2.7](../experiments/02-10_few_shot_information_floor.md#c2-7)).

<a id="c2-4"></a>

## Check 2.4 · Hidden-bit coverage, hard query budget and adaptivity (Lemma 2.9, Theorem 2.10)

**Verdict.** PASS: 348 instances, 26,214 assertions, worst residual 0. In all 100 brute-force cases ($`d\le5`$, $`K\le4`$, $`n\le3`$) the Bayes-optimal risk equals (2.7)/(2.9), and the best label-adaptive query decision tree equals the weight-greedy rule.

**Why.**

- Theorem 2.10(a) claims the Bayes risk is $`\tfrac12\sum_jw_j(1-q_j)^K`$ for *any* within-trajectory dependence of reveal events.
- Theorem 2.10(b) claims that no label-adaptive query policy beats greedy-by-weight, and that the positive part must stay inside the expectation.
- A brute-force Bayes or decision-tree value differing from the formula would falsify these, as would an instance with $`\mathbb E[(M-n)_+]<(\mathbb EM-n)_+`$.

**Setup.**

- 100 cases: $`d=1,\dots,5`$ (20 each), $`K=0,\dots,4`$, $`n=0,\dots,3`$, random rational weights $`w`$ and random reveal-set laws over bitmasks.
- Enumeration covers all $`2^d`$ bit worlds and every ordered reveal history; the largest case has 2,321 transcripts. Bayes actions are taken per posterior. Label-dependent query decision trees are minimized exactly; the largest has 271 query nodes.
- The uniform-occupancy law of $`M_K`$ is computed by two independent methods, a subset recursion and inclusion–exclusion, for $`d=2,\dots,5`$ and $`K\le4`$.
- An adversarial hill-climb (8 restarts × 20 steps) searches for a violation of Jensen's inequality or of greedy optimality.
- Seed `SeedSequence([150150, 5])`; `fractions.Fraction` throughout.

**Result.**

| Quantity | Value | JSON key (entry `v015:V5`) |
|---|---|---|
| Instances / assertions / worst residual | 348 / 26,214 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Cases with Bayes risk equal to (2.7) | 100 / 100 | `data.cases[*].Bayes` vs `.A11` |
| Cases with adaptive optimum equal to greedy | 100 / 100 | `data.cases[*].adaptive` vs `.greedy` |
| Cases with multi-coordinate reveal sets | 80 / 100 | `data.cases[*].reveal` (bitmask keys) |
| Cases where $`\mathbb E[(M-n)_+]>(\mathbb EM-n)_+`$ strictly | 36 / 100 | `data.cases[*].E_positive` vs `.positive_E` |
| Strict witness | $`d=2,K=1,n=1`$: $`\mathbb E[(M-1)_+]=\tfrac12`$ vs $`(\mathbb EM-1)_+=0`$; risk $`\tfrac5{56}`$ | `data.strict_Jensen` |
| Two occupancy methods agree; the uniform curve (2.10) is exact | all $`d\le5`$, $`K\le4`$, $`n\le3`$ | `data.uniform` |
| Largest violation found by adversarial search | $`0`$ | `data.adversarial.best.score` |

**Why it supports the statements.**

- The information floor and the optimal-query value are exactly the closed forms of Theorem 2.10, including correlated reveal events within a trajectory ([Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) allows them).
- Value-adaptive querying never helped, as the independence argument of [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) predicts.
- Moving the positive part outside the expectation would be wrong in more than a third of the cases, so the expectation in (2.9) is essential.

<a id="c2-6"></a>

## Check 2.6 · Demonstration-count curves, replay and an actual learner (Theorem 2.10, Proposition 2.4, Theorem 2.2)

**Verdict.** PASS: 944 exact $`(K,\text{query})`$ cells and 5,664 actual-learner rows, 7,450 assertions, worst residual $`5.6\times10^{-17}`$. Replay gain is 0 in every cell; targeted queries beat extra passive demonstrations at equal budget; no actual learner goes below the information floor.

**Why.** The check separates three things the theory keeps apart:

- independent new evidence (more trajectories, or targeted queries);
- replay of seen evidence, which has zero information value by [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4);
- the gap between information availability and a learner's actual use of it, which is the algorithm term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2).

**Setup.**

- Exact curves of $`R_{\rm info}(K,\cdot)`$ for $`m\in\lbrace 4,8,16\rbrace`$ contexts, uniform and nonuniform $`q`$ and $`w`$, $`K=0,\dots,32`$ and query budgets $`0,1,2,4`$, by integer subset Möbius inversion with rational risks.
- Three access types at equal budget: *active* (queries target unseen contexts), *passive* (the same budget spent on extra random demonstrations) and *replay* (seen labels shown again).
- An actual tabular learner (one logit per context) trained by 12 updates of separable summed cross-entropy with Adam or SGD on the revealed bits; unseen logits stay at 0, and deployment samples Bernoulli actions.
- Seed `SeedSequence([150150, 105])`.

**Result.**

| Quantity | Value | JSON key (entry `v015:E5`) |
|---|---|---|
| Exact cells / assertions / worst residual | 944 / 7,450 / 5.6e-17 | `data.curves`, `n_assertions`, `worst_residual` |
| Replay gain | $`0`$ in all 944 cells | `data.curves[*].replay_gain` |
| $`m=8`$ uniform, $`K=4`$: no query / 4 passive demonstrations / 4 active queries | $`0.2931`$ / $`0.1718`$ / $`0.0431`$ ($`\tfrac{2401}{8192}`$, $`\tfrac{5764801}{33554432}`$, $`\tfrac{353}{8192}`$) | `data.curves[m=8,uniform,K=4]` |
| $`m=8`$ uniform, $`K=8`$: no query / 4 active queries | $`0.1718`$ / $`0.0013`$ | same |
| Actual-learner rows | 5,664 (Adam, SGD × active, passive, replay) | `data.actual_learners` |
| $`\min(\text{learner risk}-\text{information floor})`$ | $`0.0`$ (never below the floor) | `data.actual_learners[*].risk`, `.information_floor` |
| Mean learner error on *revealed* bits after 12 updates | Adam 0.239, SGD 0.372 | `data.actual_learners[*].observed_bit_error` |

![Figure 2.3: information floor in K, actual learners above the floor, and the KL budget](../figures/fig2_3_floor.png)

*Figure 2.3.* (a) Exact hidden-bit information risk against $`K`$ ($`m=8`$, uniform): four targeted queries reduce it far faster than four extra demonstrations, and four replays coincide with "$`K`$ demonstrations only", $`\tfrac12(7/8)^K`$. (b) Every actual Adam or SGD tabular learner lies on or above the information floor; the excess is its algorithm error. (c) For the package construction of [Check 2.7](../experiments/02-10_few_shot_information_floor.md#c2-7), demonstration KL plus query KL always covers $`\mathrm{kl}(1-\delta,\delta)`$, as (2.11) requires: binary-KL lower bound ≤ exact transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$. [PDF](../figures/fig2_3_floor.pdf)

**Why it supports the statements.**

- Every exact curve equals (2.7) and (2.9).
- Replay leaves the floor unchanged in every cell ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)).
- At equal budget, targeted queries remove far more risk than additional passive demonstrations: at $`K=4`$, $`m=8`$, four active queries reach $`0.043`$ against $`0.172`$. This is the coverage effect of Theorem 2.10.
- The actual learner stays above the floor. Its excess comes from imperfect fitting of revealed bits, the algorithm term $`\mathcal R_{\rm alg}`$ of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2), so information availability and information use are visible separately.

<a id="c2-7"></a>

## Check 2.7 · Independent package constructions (Theorem 2.10, Proposition 2.11)

**Verdict.** PASS: three independent constructions from the theory package agree with (2.7), (2.9), (2.10) and (2.11).

**Why.** These are implementations independent of [Check 2.4](../experiments/02-10_few_shot_information_floor.md#c2-4), [Check 2.5](../experiments/02-11_shared_information_budget.md#c2-5) and [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6), on different constructions.

**Setup and result.**

| Check | Construction | Result | JSON key (`docs/direction/v015_revision/results_v015.json`) |
|---|---|---|---|
| `pkg:T09` | $`d=3`$, $`w=(\tfrac12,\tfrac13,\tfrac16)`$, reveal sets $`\varnothing,\lbrace 0,1\rbrace ,\lbrace 1,2\rbrace ,\lbrace 0,2\rbrace`$ with probabilities $`\tfrac1{10},\tfrac2{10},\tfrac3{10},\tfrac4{10}`$ ($`q=(\tfrac35,\tfrac12,\tfrac7{10})`$); full posterior enumeration and decision-tree minimization, $`K\le4`$, $`n\le3`$ | 20/20 cases equal the formula. E.g. $`K=0`$: $`\tfrac12,\tfrac14,\tfrac1{12},0`$; $`K=4`$: $`\tfrac{2099}{120000},\tfrac1{40000},\tfrac1{120000},0`$ | `checks.T09_exact_posterior_and_adaptive_query_optimum.exact_curves` |
| `pkg:T10` | Uniform $`d=4`$, $`K\le6`$, $`n\le4`$ | 35/35 exact. At $`K=3,n=2`$ the correct risk is $`\tfrac1{128}`$; the formula with the positive part outside the expectation gives 0 | `checks.T10_uniform_coverage_and_positive_part` |
| `pkg:T11` | Two worlds, demonstration law $`(\tfrac3{10},\tfrac7{10})`$, two query channels, score-based stopping, $`N_{\max}=4`$, $`K\le4`$ | 10/10 directional rows satisfy binary-KL lower bound ≤ exact transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$, and $`\mathbb EN`$ ≥ the bound (2.12). Lower/upper ratio 0.37–0.56 | `checks.T11_adaptive_query_KL_enumeration.rows` |

**Why it supports the statements.** Three further constructions, with nonuniform weights, multi-bit reveal sets and score-based stopping, agree with Theorem 2.10 and [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5,E5 --output theory/v015_checks/quick/sec2.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: the group of `v015:V5`, `v015:E5`; code `theory/v015_checks/mathematics.py` (`check_V5`; `check_E5` in `experiments.py`).
- `docs/direction/v015_revision/results_v015.json`: `checks.T09_*`, `checks.T10_*`, `checks.T11_*`; code `docs/direction/v015_revision/verify_v015_extensions.py`.
- Figures: `figures/fig2_3_floor.{png,pdf}`; specs `figures/specs/fig2_3.md`.
