# Proposition 3.1 · Static value is not learning value

*Experiments*

[← Framework](../README.md) · [Statement](../statements/03-01_static_value_not_learning_value.md) · [Proof](../proofs/03-01_static_value_not_learning_value.md) · [Experiments](03-01_static_value_not_learning_value.md) · [中文](03-01_static_value_not_learning_value_ZH.md)

**What is tested here.** [Check 3.1](../experiments/03-01_static_value_not_learning_value.md#c3-1) verifies both separations as exact identities (60 instances, 75 assertions, residual 0): identical static values $`\tfrac12+p/20`$ with frequency-learner returns exactly $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$ for every $`m=1,\dots,30`$, and equal packet means with a gap of exactly $`\eta^2/8`$ that reverses under $`1-p^2`$ (PASS). [Check 3.2](../experiments/03-01_static_value_not_learning_value.md#c3-2) finds the same separation in a trained controlled network: in all 24 exactly enumerated configurations the two statically equivalent sources give different students, with the predicted sign (controlled finding).

**Setting used on this page.**

- *Aliased student* $`\pi_p`$: one context, plays $`A`$ with probability $`p`$. *Frequency distillation:* $`\widehat p`$ = fraction of $`A`$ among $`m`$ teacher demonstrations. *One SGD step* of size $`\eta`$ on the squared loss.

**Statement under test ([Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1)).** *Construction (a): four-state sources.*

- States $`u,v,w,x`$ share the reward rules below; action $`B`$ always succeeds with probability $`\tfrac12`$.
- The reference source draws the state from $`\rho_0=(\tfrac34,\tfrac14,0,0)`$, the alternative source from $`\rho_1=(0,0,\tfrac14,\tfrac34)`$.
- An optimal teacher observes the state and plays its better action.
- The student is aliased: it has one context and plays $`A`$ with probability $`p`$, giving policy $`\pi_p`$.
- The learner is frequency distillation: from $`m\ge1`$ independent teacher demonstrations, $`\widehat p`$ is the fraction of $`A`$ actions (also the maximizer of empirical cross-entropy).

| State | $`\rho_0`$ | $`\rho_1`$ | $`r(A)`$ | $`r(B)`$ |
|---|---:|---:|---:|---:|
| $`u`$ | $`3/4`$ | $`0`$ | $`19/30`$ | $`1/2`$ |
| $`v`$ | $`1/4`$ | $`0`$ | $`3/10`$ | $`1/2`$ |
| $`w`$ | $`0`$ | $`1/4`$ | $`9/10`$ | $`1/2`$ |
| $`x`$ | $`0`$ | $`3/4`$ | $`13/30`$ | $`1/2`$ |

*Construction (b): two label packets.* A packet is a pair of binary labels $`(Y_1,Y_2)`$.

- Under the *independent* source, $`Y_1,Y_2`$ are IID Bernoulli $`(\tfrac12)`$. Under the *paired* source, $`(Y_1,Y_2)`$ is uniform on $`\lbrace (0,1),(1,0)\rbrace`$.
- The learner takes one gradient step of size $`0<\eta\le1`$ from $`p_0=\tfrac12`$ on the squared loss $`\tfrac14\sum_{i=1}^2(p-Y_i)^2`$:

```math
p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2 .
```

- Deployment is two independent decisions that both must choose action 1, so $`J(p)=p^2`$. The alternative task is $`J'(p)=1-p^2`$.

Then:

- **(a) Identical static values, different students.** In Construction (a), both sources give every aliased student the same value $`J(\pi_p)=\tfrac12+p/20`$ and give the teacher the same value $`\tfrac35`$. The teacher's $`A`$-frequencies are $`\tfrac34`$ and $`\tfrac14`$, and frequency distillation on $`m\ge1`$ demonstrations yields expected returns $`\tfrac{43}{80}`$ (reference) and $`\tfrac{41}{80}`$ (alternative): a gap of $`\tfrac1{40}`$ for every $`m`$.
- **(b) Identical mean updates, different learning values.** In Construction (b), the two sources have the same label marginals, the same mean gradient and the same mean update $`\mathbb Ep^+=\tfrac12`$, yet

```math
\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8, \qquad(3.1)
```

and under $`J'=1-p^2`$ the difference is $`-\eta^2/8`$.

<a id="c3-1"></a>

## Check 3.1 · Identical static values and identical mean updates, different learning values (Proposition 3.1)

**Verdict.** PASS: 60 instances, 75 assertions, worst residual 0. The gap of Proposition 3.1(a) is exactly $`1/40`$ for every $`m=1,\dots,30`$, and the gap of Proposition 3.1(b) is exactly $`\eta^2/8`$ (and $`-\eta^2/8`$ under $`1-p^2`$).

**Why.**

- Proposition 3.1(a) asserts an *exact* equality of static values for all $`p`$ together with an exact learning-value gap for all $`m`$. An arithmetic slip in the table, a non-optimal teacher or an $`m`$-dependent gap would falsify it.
- Proposition 3.1(b) asserts equal means but a curvature-dependent gap whose sign depends on the task.

**Setup.**

- SymPy ([Meurer et al., 2017](#ref-meurer2017sympy)) symbolic computation in $`p`$ and $`\eta`$.
- Exact binomial expectations of the frequency learner in `fractions.Fraction`, for $`m=1,\dots,30`$ and both sources.
- Statewise teacher optimality is checked at all four states.
- Seed `SeedSequence([20261010, 3])`; the check is deterministic.

**Procedure.**

1. Expand $`J_\rho(\pi_p)`$ for both state laws and assert equality with $`\tfrac12+p/20`$.
2. Assert that the teacher's value is $`\tfrac35`$ under both laws and that its action is optimal at every state.
3. For each $`m`$, compute $`\mathbb E\,J(\pi_{\widehat p})`$ exactly as a binomial sum and assert $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$.
4. Enumerate the four equally likely label pairs; assert $`\mathbb Ep^+=\tfrac12`$, the gap $`\eta^2/8`$ for $`p^2`$ and the gap $`-\eta^2/8`$ for $`1-p^2`$.

**Result.**

| Quantity | Value | JSON key (entry `v014:C3`) |
|---|---|---|
| Instances / assertions / worst residual | 60 / 75 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Gap of Proposition 3.1(a) (every $`m=1,\dots,30`$) | $`1/40`$ | `details.prop5_gap` |
| Gap of Proposition 3.1(b) | $`\eta^2/8`$ (and $`-\eta^2/8`$ under $`1-p^2`$) | `details.prop6_gap` |
| Status | PASS | `status` |

**Why it supports the statement.** Both separations are exact identities, not approximations. Two sources can agree on every static quantity a selector might check for the student class (every aliased student's value, the teacher's value, the label marginals and the mean gradient) and still produce different learned students.

<a id="c3-2"></a>

## Check 3.2 · The same separation in a trained network (Proposition 3.1, controlled network)

**Verdict.** Controlled finding: in all 24 exactly enumerated configurations ($`R\le4`$) the two statically equivalent sources of each construction give different trained students, with the sign predicted by Proposition 3.1. Scope: a controlled small network, not a real model and not a benchmark.

**Why.** Proposition 3.1 uses idealized learners (frequency distillation, one SGD step). The question is whether a nonlinear network trained by gradient methods also distinguishes the statically equivalent sources. A zero gap would mean the separation is an artifact of the idealized learner.

**Setup.**

- Network: one-hot context → 4 tanh units → sigmoid action probability, fixed initialization, cross-entropy on label blocks.
- Optimizers: SGD with $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$; Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\beta=(0.9,0.999)`$, $`\epsilon=10^-8`$, bias correction and $`\eta\in\lbrace 0.01,0.1\rbrace`$.
- $`R\in\lbrace 1,4,40\rbrace`$ updates, using the source laws of Constructions (a) and (b).
- For $`R\le4`$ all training histories are enumerated exactly. For $`R=40`$, 2,400 independent training runs per source; SE counts training runs.
- Seed `SeedSequence([20261010, 23])`.

**Result (gap = reference − alternative for (a), independent − paired for (b); exact enumeration unless an SE is given).**

| Construction | Optimizer, $`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$ (SE) |
|---|---|---:|---:|---:|
| (a) | SGD 0.01 | 1.21e-4 | 4.81e-4 | 4.38e-3 (1.7e-5) |
| (a) | SGD 0.1 | 1.21e-3 | 4.48e-3 | 2.10e-2 (9.2e-5) |
| (a) | SGD 1.0 | 1.10e-2 | 1.88e-2 | 2.44e-2 (2.0e-4) |
| (a) | Adam 0.01 | 5.04e-4 | 2.00e-3 | 1.54e-2 (6.9e-5) |
| (a) | Adam 0.1 | 4.91e-3 | 1.54e-2 | 2.40e-2 (1.9e-4) |
| (b) | SGD 0.01 | 3.47e-6 | 1.36e-5 | 3.7e-5 (2.0e-4) |
| (b) | SGD 0.1 | 3.46e-4 | 1.15e-3 | 2.0e-3 (1.1e-3) |
| (b) | SGD 1.0 | 2.86e-2 | 1.86e-2 | 1.61e-2 (2.7e-3) |
| (b) | Adam 0.01 | 9.19e-5 | 6.33e-4 | 1.23e-3 (9.1e-4) |
| (b) | Adam 0.1 | 8.75e-3 | 2.76e-2 | 1.23e-2 (2.4e-3) |

Values are read from `controlled_neural.details.configurations[task∈{Prop5,Prop6}].neural_gap` and `.SE`; the rows with $`\eta=3`$ are in the JSON. At small SGD steps the measured gap matches the network's local tangent prediction: for (a) with SGD $`\eta=0.01`$, $`R=1`$, 1.21170e-4 measured against 1.21171e-4 predicted (`.local_affine_SGD_prediction`).

![Figure 3.1: equal evaluation, different learned students](../figures/fig3_1_static_vs_learning.png)

*Figure 3.1.* (a) Construction (a): every aliased student has the same value $`\tfrac12+p/20`$ under both sources, and the teacher has value $`\tfrac35`$ under both, yet frequency distillation yields $`\tfrac{43}{80}`$ against $`\tfrac{41}{80}`$. (b) Construction (b): identical label marginals, mean gradient and mean update, but a learning-value gap of $`+\eta^2/8`$ under $`J=p^2`$ and $`-\eta^2/8`$ under $`J=1-p^2`$. (c) In the controlled tanh network, the statically equivalent sources still give different trained students (exact enumeration, $`R\le4`$; log scale). [PDF](../figures/fig3_1_static_vs_learning.pdf)

**Why it supports the statement.**

- In a trained network, as in the proposition, statically equivalent feedback sources yield different students. Every exactly enumerated gap is positive, matching the predicted direction: the reference source is better in (a), and independent packets are better under $`p^2`$ in (b).
- The gap grows with the step size, as a learning-consequence effect should; in (a) it also grows with the number of updates for $`\eta\le1`$.
- At $`R=40`$ (Monte Carlo) three gaps of construction (b) are within two standard errors of zero (SGD $`\eta=0.01`$ and $`0.1`$, Adam $`\eta=0.01`$); the exact rows carry the evidence.
- The magnitude depends on the optimizer: at equal $`\eta`$ the Adam gaps exceed the SGD gaps. §5 explains this dependence ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3), cold-start Adam).

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SymPy ([Meurer et al., 2017](#ref-meurer2017sympy)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C3,C23 --output theory/independent_checks/subset.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C3`; code `theory/independent_checks/check_v014_independent.py` (`check_C3`, `check_C23`). Controlled network: `controlled_neural.details.configurations` (filter `task` = `Prop5` or `Prop6`).
- Figures: `figures/fig3_1_static_vs_learning.{png,pdf}`; specs `figures/specs/fig3_1.md`.
