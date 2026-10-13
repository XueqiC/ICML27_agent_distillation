# Theorem 2.2 · Regret decomposition and decision ambiguity

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-02_regret_decomposition_decision_ambiguity.md) · [Proof](../proofs/02-02_regret_decomposition_decision_ambiguity.md) · [Experiments](02-02_regret_decomposition_decision_ambiguity.md) · [中文](02-02_regret_decomposition_decision_ambiguity_ZH.md)

**What is tested here.** [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) tests both parts of the theorem on 500 random finite decision problems in exact rational arithmetic: the decomposition residual is exactly 0, every term is nonnegative, and the zero-information criterion matches direct computation in both directions (PASS, 5,810 assertions). [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6) shows the algorithm term in action: actual Adam and SGD learners never go below the information floor, and their excess is their algorithm error (PASS).

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).

**Statement under test ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).** Under [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1):

**(a) Exact decomposition.** The expected regret against the reference optimum is

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) Decision ambiguity.** $`\mathcal R_{\rm info}=0`$ if and only if, for every $`D`$ with $`\Pr(D)>0`$,

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

**Other statements tested on this page.**

- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3): the robust ambiguity value (2.3), a minimum over policy mixtures of the worst-case regret on a class $`C`$, equals its dual (2.4), a maximum over priors on $`C`$.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): an environment-independent post-processing $`D'`$ of the transcript cannot raise the Bayes value, (2.5), and adding independent data leaves it unchanged.

<a id="c2-1"></a>

## Check 2.1 · Regret decomposition, ambiguity criterion, duality and data processing (Theorem 2.2, Propositions 2.3–2.4)

**Verdict.** PASS: 500 random finite decision problems, 5,810 assertions, no violation. Every rational identity holds with residual exactly 0; the worst floating-point residual (from LP solves) is $`1.33\times10^{-15}`$.

**Why.**

- The decomposition (2.1) is an identity, so any error in how $`B(D)`$, the posterior or the learner kernel is defined would show up as a nonzero residual.
- The ambiguity criterion (2.2) is an "if and only if"; a counterexample in either direction would falsify it.
- The duality (2.4) and the data-processing inequality (2.5) are statements that a random or adversarially chosen instance could violate if a hypothesis were missing.

**Setup.**

- 500 random instances: 3–6 environments with a random rational prior; 2–5 policies in $`\Pi`$ plus two extra policies in $`\Pi_{\rm ref}`$.
- Values $`J_E(\pi)\in[0,1]`$ are random rationals with denominator 12. The transcript alphabet has size $`\lvert\mathcal D\rvert=4`$, with a random channel $`P(D\mid E)`$ and a random learner kernel $`\lambda(\cdot\mid D)`$.
- One instance in four is forced to have a common optimum with ties ($`J_E(\pi_1)=J_E(\pi_2)=1`$ for all $`E`$); one in four is forced to have environment-specific optima.
- Exact arithmetic (`fractions.Fraction`). LPs are solved with HiGHS ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)) at tolerance $`10^{-8}`$. Seed `SeedSequence([20261010, 1])`.

**Procedure.** For each instance:

1. Compute $`\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}`$; assert that they sum exactly to the total regret and are each $`\ge0`$.
2. Decide whether a common posterior-optimal policy exists for every $`D`$; assert that this holds exactly when $`\mathcal R_{\rm info}=0`$.
3. Garble $`D`$ through 3-symbol channels, the post-processings of Blackwell's comparison of experiments ([Blackwell, 1953](#ref-blackwell1953equivalent)): one random randomized channel, and on the first 30 instances all $`3^4=81`$ deterministic channels, which are the extreme points of the convex objective. Assert $`B(D')\le B(D)`$.
4. Augment $`D`$ by an independent $`Z`$ and assert $`B(D,Z)=B(D)`$ exactly.
5. On 220 instances, solve the primal min-over-mixtures LP (2.3) and the dual max-over-priors LP (2.4) separately; assert that they are equal and that the value is $`0`$ exactly when a common optimum exists.

**Result.**

| Quantity | Value | JSON key (entry of the check) |
|---|---|---|
| Instances / assertions | 500 / 5,810 | `n_instances`, `.n_assertions` |
| Worst residual (identity, LP equality) | $`1.33\times10^{-15}`$ | `worst_residual` |
| Worst LP feasibility residual | $`1.33\times10^{-15}`$ | `worst_lp_feasibility_residual` |
| Instances with separate primal/dual LPs | 220 | `details.minimax_instances` |
| Largest $`B(D')-B(D)`$ over all garblings | $`0.0`$ | `details.adversarial_garbling_best_lhs_minus_rhs` |
| Status | PASS | `status` |

![Figure 2.1: regret decomposition, garbling and minimax duality](../figures/fig2_1_decomposition.png)

*Figure 2.1.* (a) Exact regret decomposition for four learners on one finite problem with three environments: with the same data, changing the learner changes only the algorithm term. (b) Garbling the transcript through a binary symmetric channel raises the information regret monotonically, from 0.1217 with the original data to 0.1833 with no data. (c) The robust ambiguity value of the two-world instance of [Check 2.2](../experiments/02-03_robust_ambiguity_duality.md#c2-2): the minimum over policy mixtures and the maximum over priors both equal 0.1. [PDF](../figures/fig2_1_decomposition.pdf)

**Why it supports the statements.**

- The decomposition is exact and each term is nonnegative on every instance, as Theorem 2.2(a) states.
- The zero-information criterion matched direct computation in both directions, including tie cases (Theorem 2.2(b)).
- No garbling, even an extreme-point one chosen adversarially, increased the Bayes value ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)), and the dual LP always attained the primal value ([Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3)).
- The residuals are at the level of floating-point LP solves; every rational identity has residual exactly zero.

### [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6) · Demonstration-count curves, replay and an actual learner

- *What it tests for this theorem.* The algorithm term $`\mathcal R_{\rm alg}`$ of (2.1): with the information term known exactly, an actual learner's extra risk is the algorithm error.
- *Setup.* Hidden-bit model with $`m\in\lbrace4,8,16\rbrace`$ contexts; a tabular learner (one logit per context) trained for 12 updates with Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) or SGD on the revealed bits; 5,664 learner rows.
- *Result.* The learner risk minus the exact information floor is never negative (minimum 0.0). The mean error on already revealed bits after 12 updates is 0.239 (Adam) and 0.372 (SGD): that excess is $`\mathcal R_{\rm alg}`$, visible separately from $`\mathcal R_{\rm info}`$.
- *Verdict.* PASS.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E5 --output theory/v015_checks/quick/sec2.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C1`; code `theory/independent_checks/check_v014_independent.py` (`check_C1`).
- `theory/v015_checks/results_v015_independent.json`: the group of `v015:E5`; code `theory/v015_checks/experiments.py` (`check_E5`).
- Figures: `figures/fig2_1_decomposition.{png,pdf}`; specs `figures/specs/fig2_1.md`.
