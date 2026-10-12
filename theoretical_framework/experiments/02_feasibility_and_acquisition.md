# §2 Feasibility and acquisition — Experiments

[← Framework](../README.md) · [Proofs](../proofs/02_feasibility_and_acquisition.md) · [中文](02_feasibility_and_acquisition_ZH.md)

**Verdict.**

- [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2), [Proposition 2.3](../proofs/02_feasibility_and_acquisition.md#s2-3), [Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4): PASS. On 500 random finite decision problems in exact rational arithmetic (5,810 assertions) the decomposition residual is 0, the zero-information criterion matches direct computation in both directions, primal and dual LPs agree on 220 instances, and no garbling, even an adversarially chosen one, increases the Bayes value.
- [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6), [Proposition 2.7](../proofs/02_feasibility_and_acquisition.md#s2-7): PASS. On 25 values of $`(\Delta,\delta)`$ (540 assertions, exact binomial sums and exact random-walk recursions) no strategy falls below the lower bound (2.6). An optimally tuned SPRT attains it exactly (ratio 1.000) at two configurations, so its constant cannot be improved in general. Compilation needs a number of labels that depends only on $`\delta_L`$.
- [Lemma 2.9](../proofs/02_feasibility_and_acquisition.md#s2-9), [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10): PASS. In all 100 exhaustively enumerated cases the Bayes-optimal risk equals (2.7) and the best value-adaptive query tree equals weight-greedy querying; 80 cases have multi-bit reveal sets. Moving the positive part outside the expectation is wrong in 36 of the 100 cases.
- [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11): PASS. The budget inequality (2.11) holds on 297 adaptive query-and-stop processes and is attained with equality (ratio 1.0).
- Replay ([Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4)) gains exactly 0 at all 944 exact grid points, and actual Adam and SGD tabular learners never go below the information floor of [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 2.1](#c2-1) | [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2), [Propositions 2.3–2.4](../proofs/02_feasibility_and_acquisition.md#s2-3) | `v014:C1` | PASS |
| [Check 2.2](#c2-2) | [Proposition 2.3](../proofs/02_feasibility_and_acquisition.md#s2-3) | `pkg:robust_ambiguity_minimax` | PASS |
| [Check 2.3](#c2-3) | [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6), [Proposition 2.7](../proofs/02_feasibility_and_acquisition.md#s2-7) | `v014:C2` | PASS |
| [Check 2.4](#c2-4) | [Lemma 2.9](../proofs/02_feasibility_and_acquisition.md#s2-9), [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10) | `v015:V5` | PASS |
| [Check 2.5](#c2-5) | [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11) | `v015:V6` | PASS |
| [Check 2.6](#c2-6) | [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4), the algorithm term of [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2) | `v015:E5` | PASS |
| [Check 2.7](#c2-7) | [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11) | `pkg:T09`, `pkg:T10`, `pkg:T11` | PASS |

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
- Exact arithmetic (`fractions.Fraction`). LPs are solved with HiGHS at tolerance $`10^{-8}`$. Seed `SeedSequence([20261010, 1])`.

**Procedure.** For each instance:

1. Compute $`\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}`$; assert that they sum exactly to the total regret and are each $`\ge0`$.
2. Decide whether a common posterior-optimal policy exists for every $`D`$; assert that this holds exactly when $`\mathcal R_{\rm info}=0`$.
3. Garble $`D`$ through 3-symbol channels: one random randomized channel, and on the first 30 instances all $`3^4=81`$ deterministic channels, which are the extreme points of the convex objective. Assert $`B(D')\le B(D)`$.
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

*Figure 2.1.* (a) Exact regret decomposition for four learners on one finite problem with three environments: with the same data, changing the learner changes only the algorithm term. (b) Garbling the transcript through a binary symmetric channel raises the information regret monotonically, from 0.1217 with the original data to 0.1833 with no data. (c) The robust ambiguity value of the two-world instance of Check 2.2: the minimum over policy mixtures and the maximum over priors both equal 0.1. [PDF](../figures/fig2_1_decomposition.pdf)

**Why it supports the statements.**

- The decomposition is exact and each term is nonnegative on every instance, as Theorem 2.2(a) states.
- The zero-information criterion matched direct computation in both directions, including tie cases (Theorem 2.2(b)).
- No garbling, even an extreme-point one chosen adversarially, increased the Bayes value (Proposition 2.4), and the dual LP always attained the primal value (Proposition 2.3).
- The residuals are at the level of floating-point LP solves; every rational identity has residual exactly zero.

<a id="c2-2"></a>

## Check 2.2 · The canonical two-world ambiguity instance (Proposition 2.3)

**Verdict.** PASS: primal and dual values are both 0.1, with minimax mixture and least-favourable prior both $`(0.5,0.5)`$.

**Why.** A hand-checkable instance of Proposition 2.3 whose optimal mixtures are known in closed form.

**Setup.** Two environments prefer opposite policies, with regret matrix $`M=\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$. Primal and dual LPs are solved with SciPy HiGHS by the package v0.14 verifier, seed 2026101014.

**Result.**

| Quantity | Value | JSON key (entry of the check) |
|---|---|---|
| Primal $`r(C)`$ | 0.1 | `robust_ambiguity_minimax.primal_regret` |
| Dual value | 0.1 | `robust_ambiguity_minimax.dual_regret` |
| Minimax policy mixture | $`(0.5,0.5)`$ | `.policy_mixture` |
| Least-favourable prior | $`(0.5,0.5)`$ | `.environment_mixture` |

**Why it supports the statement.**

- When the two worlds cannot be told apart, every learner's output law is common to both worlds, and the best achievable worst-case expected regret is $`r(C)=0.1>0`$, half of the 0.2 regret of committing to either policy (Proposition 2.3(a)).
- The value is a *mixture-averaged* regret, as the comments after Proposition 2.4 state: a sampled policy has regret 0 or 0.2.

<a id="c2-3"></a>

## Check 2.3 · Query complexity and compilation in the two-world family (Theorem 2.6, Proposition 2.7)

**Verdict.** PASS: 25 configurations $`\Delta\in\lbrace \tfrac18,\tfrac1{10},\tfrac1{16},\tfrac1{32},\tfrac1{64}\rbrace \times\delta\in\lbrace 0.01,0.05,0.1,0.25,0.4\rbrace`$, 540 assertions, worst residual 0. No exactly computed strategy falls below (2.6); the SPRT sits exactly on it at $`(\Delta,\delta)=(\tfrac18,\tfrac1{10})`$ and $`(\tfrac18,\tfrac14)`$.

**Why.**

- Theorem 2.6(b) is a lower bound for *every* adaptive strategy. A strategy with error $`\le\delta`$ in both worlds and fewer expected queries would falsify it.
- Theorem 2.6(c) and Proposition 2.7 are sufficiency claims. An exact error above $`\delta`$ (or $`\delta_L`$) at the stated sample size would falsify them.
- The constant 64 rests on $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$, which is checked analytically and numerically.

**Setup.**

- Exact rational binomial tails by integer-numerator summation, not a normal approximation.
- The SPRT expected sample size and error are computed by exact tridiagonal elimination of the absorbing random walk with boundaries $`\pm h`$, $`h=\lceil\log\frac{1-\delta}{\delta}/\log\frac{1/2+2\Delta}{1/2-2\Delta}\rceil`$.
- The payoff $`J_v(p)`$ and the passive minimax are computed in SymPy.
- The KL inequality is checked on a dense grid of 20,001 points on $`(0,\tfrac14]`$ and by a bounded scalar optimizer that maximizes the ratio adversarially.
- Seed `SeedSequence([20261010, 2])`; the check is deterministic.

**Procedure.**

1. Find the smallest odd $`n`$ for which the threshold test succeeds with probability $`\ge1-\delta`$ in both worlds (ties count as failures); check that $`n-1`$ and $`n-2`$ fail.
2. Compute the SPRT's exact expected stopping time and exact error.
3. Assert that both stopping times are $`\ge`$ the bound (2.6), and that the first form of (2.6) is $`\ge`$ its loosened form $`\mathrm{kl}(1-\delta,\delta)/(64\Delta^2)`$.
4. Assert that the exact error at $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ is $`\le\delta`$.
5. Compilation: compute the exact probability that $`\lvert\widehat p-q\rvert>\tfrac18`$ at $`m=\lceil32\log(2/\delta_L)\rceil`$ for $`q\in\lbrace \tfrac18,\tfrac78\rbrace`$; assert that it is $`\le\delta_L`$ and that every within-tolerance outcome has regret $`\le\Delta/2`$.

**Result (selected rows; all 25 are in the JSON).**

| $`\Delta`$ | $`\delta`$ | Bound (2.6) | Optimal fixed $`n`$ | SPRT $`\mathbb E N`$ | SPRT / bound | $`n_{\rm query}`$ | Exact error at $`n_{\rm query}`$ |
|---|---|---:|---:|---:|---:|---:|---:|
| 1/8 | 0.01 | 8.198 | 19 | 9.918 | 1.210 | 37 | 5.1e-4 |
| 1/8 | 0.1 | 3.200 | 7 | 3.200 | **1.000** | 19 | 8.9e-3 |
| 1/8 | 0.25 | 1.000 | 1 | 1.000 | **1.000** | 12 | 5.4e-2 |
| 1/16 | 0.01 | 35.262 | 85 | 35.282 | 1.0005 | 148 | 1.3e-3 |
| 1/32 | 0.05 | 84.356 | 173 | 87.030 | 1.032 | 384 | 8.0e-3 |
| 1/64 | 0.01 | 575.661 | 1383 | 580.574 | 1.0085 | 2358 | 1.3e-3 |
| 1/64 | 0.1 | 224.703 | 421 | 233.224 | 1.038 | 1179 | 1.6e-2 |

| Summary quantity | Value | JSON key (entry of the check) |
|---|---|---|
| Configurations / assertions | 25 / 540 | `n_instances`, `.n_assertions` |
| Minimum SPRT / bound ratio | 1.0 | `details.minimum_sprt_ratio` |
| Minimum fixed-$`n`$ / bound ratio | 1.0 (at $`\Delta=\tfrac18,\delta=0.25`$) | `details.configurations[*].fixed_ratio` |
| Analytic $`\sup_u\mathrm{kl}(\tfrac12+u,\tfrac12-u)/u^2`$ on $`(0,\tfrac14]`$ | $`8\log3=8.789`$ | `details.analytic_kl_sup` |
| Optimizer's largest ratio | 8.789 (below 16) | `details.optimizer_kl_ratio` |
| Compilation $`m`$ ($`\delta_L=0.01,0.05,0.1,0.25,0.4`$) | 170, 119, 96, 67, 52 | `details.compilation[*].m` |
| Exact compilation error at those $`m`$ | 4.2e-6, 1.2e-4, 2.4e-4, 3.0e-3, 3.9e-3 | `details.compilation[*].error` |
| Worst residual | 0 | `worst_residual` |

![Figure 2.2: query counts against the lower bound, and passive impossibility](../figures/fig2_2_queries.png)

*Figure 2.2.* (a) Exact query counts against $`1/\Delta^2`$ at $`\delta=0.05`$: the SPRT meets the KL lower bound (2.6), the smallest successful fixed-sample test and the sufficient $`n_{\rm query}`$ of Theorem 2.6(c) lie above it, and no strategy falls below. (b) Ratios of each strategy to the bound over all 25 configurations; the SPRT reaches ratio 1 at two of them. (c) For $`\Delta=1/8`$, the two passive regrets $`2\Delta(1-m)`$ and $`2\Delta m`$ trade off in the mean output $`m`$, so the worst case of any passive learner is at least $`\Delta`$. [PDF](../figures/fig2_2_queries.pdf)

**Why it supports the statements.**

- *Passive impossibility (Theorem 2.6(a)).* The symbolic payoff is exactly $`\tfrac34+v\Delta(2p-1)`$, and the two passive regrets intersect at value exactly $`\Delta`$.
- *Lower bound (Theorem 2.6(b)).* No exactly computed strategy beats the bound. The SPRT, which is near-optimal for this problem, sits on it at two configurations; at $`\Delta=\tfrac1{64}`$ and $`\delta\le0.25`$ it is within 5% of it (ratios 1.009–1.047). The bound therefore has the right constant, not only the right order. For $`\delta\le0.25`$, fixed-sample tests need 1.6–2.4 times the bound apart from the trivial case $`n=1`$, as expected for non-sequential designs.
- *Upper bound (Theorem 2.6(c)).* At $`n_{\rm query}`$ the exact error is far below $`\delta`$ in every configuration: Hoeffding is conservative.
- *Compilation (Proposition 2.7).* The number of compiled labels is $`O(\log(1/\delta_L))`$, independent of $`\Delta`$, and the exact tail is two to four orders of magnitude below $`\delta_L`$.
- The two costs, $`\Theta(\Delta^{-2}\log(1/\delta))`$ authentic queries and $`O(\log(1/\delta_L))`$ compiled labels, are thus confirmed separately.

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

- The information floor and the optimal-query value are exactly the closed forms of Theorem 2.10, including correlated reveal events within a trajectory (Lemma 2.9 allows them).
- Value-adaptive querying never helped, as the independence argument of Lemma 2.9 predicts.
- Moving the positive part outside the expectation would be wrong in more than a third of the cases, so the expectation in (2.9) is essential.

<a id="c2-5"></a>

## Check 2.5 · Demonstration information credits the adaptive query budget (Proposition 2.11)

**Verdict.** PASS: 297 instances, 2,969 assertions, worst residual $`2.7\times10^{-15}`$. The inequality (2.11) is never violated and is attained with equality (ratio 1.0) by a single informative query.

**Why.** Proposition 2.11 must hold for *every* common history-dependent query-and-stop rule. A process with $`\mathrm{kl}(1-\delta,\delta)>K\kappa_D+\kappa_X\mathbb E_vN`$ would falsify it; a ratio far below 1 everywhere would mean the bound is loose.

**Setup.**

- Two-world processes with $`K\in\lbrace 1,2\rbrace`$ symmetric informative demonstrations and $`N_{\max}\le4`$, two random binary query channels, and randomized history-dependent selection and stopping rules shared by both worlds.
- Exact rational transcript probabilities. The transcript KL is computed directly and, separately, as the adaptive chain sum; logarithms are in float64 with tolerance $`10^{-8}`$.
- 160 random processes, an adversarial hill-climb (8 restarts × 16 steps), and two constructed cases: a tight case ($`K=0`$, one binary query with channel $`(\tfrac45,\tfrac15)`$, $`\delta=\tfrac15`$) and a zero-information case ($`\kappa_D=\kappa_X=0`$).
- Seed `SeedSequence([150150, 6])`.

**Result.**

| Quantity | Value | JSON key (entry `v015:V6`) |
|---|---|---|
| Instances / assertions / worst residual | 297 / 2,969 / 2.7e-15 | `n_instances`, `n_assertions`, `worst_residual` |
| Random processes satisfying the premise $`0<\delta<\tfrac12`$ | 159 / 160 | `data.random_premise_count` |
| Largest ratio $`\mathrm{kl}(1-\delta,\delta)/(K\kappa_D+\kappa_X\mathbb EN)`$, random processes | 0.9973 | `data.worst_random.score` |
| Largest ratio, adversarial search | 0.9760 | `data.adversarial.best.score` |
| Ratio, constructed tight case | **1.0** (KL $`=\mathrm{kl}(\tfrac45,\tfrac15)=0.8318`$, $`\mathbb EN=1`$) | `best_ratio`, `data.tight` |
| Zero-information case | transcript KL $`=0`$; the two error probabilities sum to 1 | `data.zero_information` |

**Why it supports the statement.**

- The KL chain identity (direct transcript KL equals the accumulated chain) holds exactly under adaptive stopping, which is the core of the proof.
- (2.11) is never violated, and a single informative query attains it with equality, so its constant cannot be improved.
- When demonstrations and queries carry no information, uniform identification is impossible, as the last clause of Proposition 2.11 states.

<a id="c2-6"></a>

## Check 2.6 · Demonstration-count curves, replay and an actual learner (Theorem 2.10, Proposition 2.4, Theorem 2.2)

**Verdict.** PASS: 944 exact $`(K,\text{query})`$ cells and 5,664 actual-learner rows, 7,450 assertions, worst residual $`5.6\times10^{-17}`$. Replay gain is 0 in every cell; targeted queries beat extra passive demonstrations at equal budget; no actual learner goes below the information floor.

**Why.** The check separates three things the theory keeps apart:

- independent new evidence (more trajectories, or targeted queries);
- replay of seen evidence, which has zero information value by Proposition 2.4;
- the gap between information availability and a learner's actual use of it, which is the algorithm term of Theorem 2.2.

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

*Figure 2.3.* (a) Exact hidden-bit information risk against $`K`$ ($`m=8`$, uniform): four targeted queries reduce it far faster than four extra demonstrations, and four replays coincide with "$`K`$ demonstrations only", $`\tfrac12(7/8)^K`$. (b) Every actual Adam or SGD tabular learner lies on or above the information floor; the excess is its algorithm error. (c) For the package construction of Check 2.7, demonstration KL plus query KL always covers $`\mathrm{kl}(1-\delta,\delta)`$, as (2.11) requires: binary-KL lower bound ≤ exact transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$. [PDF](../figures/fig2_3_floor.pdf)

**Why it supports the statements.**

- Every exact curve equals (2.7) and (2.9).
- Replay leaves the floor unchanged in every cell (Proposition 2.4).
- At equal budget, targeted queries remove far more risk than additional passive demonstrations: at $`K=4`$, $`m=8`$, four active queries reach $`0.043`$ against $`0.172`$. This is the coverage effect of Theorem 2.10.
- The actual learner stays above the floor. Its excess comes from imperfect fitting of revealed bits, the algorithm term $`\mathcal R_{\rm alg}`$ of Theorem 2.2, so information availability and information use are visible separately.

<a id="c2-7"></a>

## Check 2.7 · Independent package constructions (Theorem 2.10, Proposition 2.11)

**Verdict.** PASS: three independent constructions from the theory package agree with (2.7), (2.9), (2.10) and (2.11).

**Why.** These are implementations independent of Checks 2.4–2.6, on different constructions.

**Setup and result.**

| Check | Construction | Result | JSON key (`docs/direction/v015_revision/results_v015.json`) |
|---|---|---|---|
| `pkg:T09` | $`d=3`$, $`w=(\tfrac12,\tfrac13,\tfrac16)`$, reveal sets $`\varnothing,\lbrace 0,1\rbrace ,\lbrace 1,2\rbrace ,\lbrace 0,2\rbrace`$ with probabilities $`\tfrac1{10},\tfrac2{10},\tfrac3{10},\tfrac4{10}`$ ($`q=(\tfrac35,\tfrac12,\tfrac7{10})`$); full posterior enumeration and decision-tree minimization, $`K\le4`$, $`n\le3`$ | 20/20 cases equal the formula. E.g. $`K=0`$: $`\tfrac12,\tfrac14,\tfrac1{12},0`$; $`K=4`$: $`\tfrac{2099}{120000},\tfrac1{40000},\tfrac1{120000},0`$ | `checks.T09_exact_posterior_and_adaptive_query_optimum.exact_curves` |
| `pkg:T10` | Uniform $`d=4`$, $`K\le6`$, $`n\le4`$ | 35/35 exact. At $`K=3,n=2`$ the correct risk is $`\tfrac1{128}`$; the formula with the positive part outside the expectation gives 0 | `checks.T10_uniform_coverage_and_positive_part` |
| `pkg:T11` | Two worlds, demonstration law $`(\tfrac3{10},\tfrac7{10})`$, two query channels, score-based stopping, $`N_{\max}=4`$, $`K\le4`$ | 10/10 directional rows satisfy binary-KL lower bound ≤ exact transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$, and $`\mathbb EN`$ ≥ the bound (2.12). Lower/upper ratio 0.37–0.56 | `checks.T11_adaptive_query_KL_enumeration.rows` |

**Why it supports the statements.** Three further constructions, with nonuniform weights, multi-bit reveal sets and score-based stopping, agree with Theorem 2.10 and Proposition 2.11.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1,C2 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5,V6,E5 --output theory/v015_checks/quick/sec2.json
theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py --output theory/v014_repro/results_v014_rerun.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entries of `v014:C1` and `v014:C2` (keys `details.configurations`, `details.compilation`, `details.analytic_kl_sup`, `details.optimizer_kl_ratio`, `details.minimum_sprt_ratio`).
- `theory/independent_checks/SPEC.md` (sections `v014:C1`, `v014:C2`) and `theory/independent_checks/check_v014_independent.py` (`check_C1`, `check_C2`, `binomial_tail`, `walk_recursion`).
- `theory/v014_repro/results_v014_rerun.json`: key `robust_ambiguity_minimax`.
- `theory/v015_checks/results_v015_independent.json`: the groups of `v015:V5`, `v015:V6`, `v015:E5`; code `theory/v015_checks/mathematics.py` (`check_V5`, `check_V6`) and `theory/v015_checks/experiments.py` (`check_E5`).
- `docs/direction/v015_revision/results_v015.json`: `checks.T09_*`, `checks.T10_*`, `checks.T11_*`.
- Figures: `figures/fig2_1_decomposition.{png,pdf}`, `figures/fig2_2_queries.{png,pdf}`, `figures/fig2_3_floor.{png,pdf}`; specs in `figures/specs/fig2_1.md`, `fig2_2.md`, `fig2_3.md`.
