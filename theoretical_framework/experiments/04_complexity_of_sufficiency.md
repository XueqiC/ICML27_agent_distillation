# §4 Complexity of sufficiency — Experiments

[← Framework](../README.md) · [Proofs](../proofs/04_complexity_of_sufficiency.md) · [中文](04_complexity_of_sufficiency_ZH.md)

**Verdict.** Every statement of this section passes its checks. All checks except Check 4.13 are exact rational or exact LP computations; Check 4.13 is a controlled small network.

- **[Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2) (sufficiency duality and transfer): PASS.**
  - The span criterion never disagrees with an independent LP search on 300 random instances.
  - The worst-pair discrepancy equals twice the best uniform approximation error to $`2.2\times10^{-14}`$ (Check 4.1), and to $`1.6\times10^{-15}`$ on 21 structured subspaces (Check 4.10).
- **[Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4) (affine learners, moment order): PASS.**
  - Degree and coefficient bounds hold on 1,080 random and adversarially mutated deployment trees.
  - Moment-matched laws give exactly equal learning values for affine learners with up to 5 updates.
  - Every order is necessary for $`N=1,\dots,14`$; parity keeps gap 1 for $`N=1,\dots,8`$.
  - The grid statistic count equals $`\binom{k+N}N-1`$ in all 18 configurations, and the binary barrier is attained (Checks 4.2–4.5).
- **[Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5) (round-independent bound, averaging rates): PASS.**
  - No adversarial instance exceeds the bound; the closest has gap/bound $`55/57`$ (Check 4.6).
  - The first unmatched cumulant gives both the exponent and the coefficient of the decay in all six cases (Check 4.7).
  - Replay and deterministic descendants leave the gap constant through 256 exposures (Check 4.8).
- **[Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6) (exact order is not approximate complexity): PASS.**
  - The Chebyshev truncation error of $`p^N`$ equals the binomial tail; accuracy $`10^{-3}`$ needs only $`q=4,6,9,13`$ for $`N=4,8,16,32`$.
  - Parity keeps error exactly $`1/2`$ below degree $`N`$ (Check 4.9).
  - At equal degree 3, structured families need 3 statistics and parity needs 7 (Check 4.10).
- **[Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7) (update composition): PASS.**
  - Block degrees equal the composition bound in all 12 random chains, and the squaring witness attains it.
  - The SGD expansion remainder never exceeds $`0.275`$ times its bound (Check 4.11).
  - The geometric accumulation bound is attained with equality, and the linear-in-$`R`$ reading fails without stability (Check 4.12).
  - In a controlled network, the tangent affine-SGD prediction agrees within the preset margin in the small-$`\eta R`$ regime and fails at large step sizes (Check 4.13).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 4.1](#c4-1) | [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(a)–(d) | `v014:C5` | PASS |
| [Check 4.2](#c4-2) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(a) | `v014:C6` | PASS |
| [Check 4.3](#c4-3) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(b) | `v014:C7` | PASS |
| [Check 4.4](#c4-4) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(c) | `v014:C8`, `pkg:constant_gap_parity` | PASS |
| [Check 4.5](#c4-5) | [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(d) | `v014:C9`, `pkg:grid_response_dimensions` | PASS |
| [Check 4.6](#c4-6) | [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(a) | `v014:C10` | PASS |
| [Check 4.7](#c4-7) | [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(b)–(d) | `v014:C11`, `pkg:scalar_exact_decay` | PASS |
| [Check 4.8](#c4-8) | [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(e) | `v014:C18` | PASS |
| [Check 4.9](#c4-9) | [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6)(a),(b) | `v014:C12`, `pkg:scalar_approximation` | PASS |
| [Check 4.10](#c4-10) | [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6); [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(a),(b); [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(d) | `v014:C19` | PASS |
| [Check 4.11](#c4-11) | [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(a)–(d) | `v014:C13` | PASS |
| [Check 4.12](#c4-12) | [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(e) | `v015:V7` | PASS |
| [Check 4.13](#c4-13) | [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(d),(e), regime in a network | `v014:C23` (SGD rows) | regime map |

Code IDs: `v014:` = `theory/independent_checks` (results `results_independent_v014.json`); `v015:` = `theory/v015_checks` (results `results_v015_independent.json`); `pkg:` = package verifier `docs/direction/v014_handoff/verify_environment_information_v014.py`, re-run into `theory/v014_repro/results_v014_rerun.json`.

<a id="c4-1"></a>

## Check 4.1 · Span criterion, duality and transfer on random instances (Theorem 4.2)

**Verdict.** PASS on 300 random finite instances with 6,014 assertions. The span criterion is exact; the duality is an equality to $`2.2\times10^{-14}`$; adversarial search against the transfer bound reaches only $`1.1\times10^{-16}`$; part (d) holds exactly for $`m=2,\dots,8`$.

**Why.** [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2) turns "which distinctions between feedback laws matter" into linear algebra. Each part has a falsifier, and the check searches for each independently:

- (a) a feature-matched pair that separates a response lying in the span, or an out-of-span response the span test misses;
- (b) a feasible pair whose discrepancy exceeds twice the uniform distance $`2\mathrm{dist}_\infty`$, or no pair attaining it — (b) is an equality, not only a bound;
- (c) a pair violating the transfer inequality;
- (d) an event family whose supremum differs from $`\gamma\,\mathrm{TV}`$, or a response dimension different from $`m-1`$.

**Setup.**

- 300 instances. Alphabet size $`m\sim\mathrm U\lbrace 4,\dots,12\rbrace`$, feature count $`d\sim\mathrm U\lbrace 1,\dots,m-2\rbrace`$.
- Feature tables $`\phi\in\mathbb Q^{d\times m}`$ and three responses $`\mathcal G\subset\mathbb Q^m`$ are random rationals.
- Each instance also tests $`g=\phi_1`$, an in-span control with expected discrepancy 0.
- Transfer instances use random $`b`$ and $`r`$, rescaled so that $`g=\tfrac12+b^\top\phi+r\in[0,1]^m`$.
- Part (d) uses random rational laws $`P,Q`$ on $`m=2,\dots,8`$, with $`a=\tfrac15`$ and $`\gamma=\tfrac35`$.
- Seed: `SeedSequence([20261010, 5])`.

**Procedure.**

1. *Exact ranks.* SymPy computes the ranks of $`[\mathbf 1;\phi]`$ and $`[\mathbf 1;\phi;g]`$ to decide $`g\in H_\phi`$, and $`d_0=\mathrm{rank}[\mathbf 1;\mathcal G]-1`$.
2. *Primal LP.* HiGHS solves $`\max(P-Q)g`$ over feature-matched pairs of probability vectors, directly in the variables $`(P,Q)`$.
3. *Approximation LP.* A separate LP computes $`\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$.
4. *Assertions for (a), (b).* Width $`=2\cdot`$ error, and width $`\le10^{-8}`$ exactly when $`g\in H_\phi`$.
5. *Transfer LP.* An LP maximizes $`(P-Q)g-B\lVert\Delta\phi\rVert_\infty-2\tau`$ over all pairs; the optimum must be $`\le0`$.
6. *Part (d).* All $`2^m`$ events are enumerated in exact rationals; $`\max_A\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert`$ is compared with $`\gamma\cdot\tfrac12\lVert P-Q\rVert_1`$, and the rank of $`[\mathbf 1;\lbrace\mathbf 1_A\rbrace]`$ is checked to be $`m`$.

**Result.**

| Quantity | Value | Source (JSON key) |
|---|---:|---|
| Instances / assertions | 300 / 6,014 | entry `v014:C5`, `n_instances`, `.n_assertions` |
| Status | PASS | entry `v014:C5`, `status` |
| $`\max\lvert\text{primal width}-2\cdot\text{approximation error}\rvert`$ | $`2.23\times10^{-14}`$ | entry `v014:C5`, `details.max_duality_residual` |
| Best adversarial transfer score (must be $`\le0`$) | $`1.11\times10^{-16}`$ | entry `v014:C5`, `details.adversarial_transfer_best_lhs_minus_rhs` |
| Worst residual over all assertions | $`5.76\times10^{-14}`$ | entry `v014:C5`, `worst_residual` |
| Part (d): event supremum $`=\gamma\,\mathrm{TV}`$, dimension $`=m-1`$ | exact for $`m=2,\dots,8`$ | assertions `all-event total variation`, `event response dimension` |

All values are from `theory/independent_checks/results_independent_v014.json`.

![Figure 4.1: worst matched-pair discrepancy equals twice the approximation error](../figures/fig4_1_duality.png)

*Figure 4.1.* One exact five-message example shows a response, its best uniform approximation from the feature span, and the feature-matched pair of laws that attains twice the approximation error. The second panel plots the primal LP optimum against twice the approximation error for all structured subspaces of Check 4.10 and the random instances of this check; every point lies on the diagonal. [PDF](../figures/fig4_1_duality.pdf)

**Why it supports Theorem 4.2.**

- The span criterion and the LP separation search never disagree, so the criterion of part (a) is the exact test.
- The two LPs are solved independently, one over pairs of laws and one over approximating functions. Their optima agree to solver tolerance, so part (b) is an equality.
- An adversary optimizing against part (c) cannot push the left side above the right.
- Limit of the evidence: finite alphabets of size at most 12; the theorem itself is stated for finite alphabets.

<a id="c4-2"></a>

## Check 4.2 · Deployment value is a bounded polynomial (Theorem 4.4(a))

**Verdict.** PASS on 1,080 deployment trees (5,400 assertions). The largest ratio of coefficient norm to $`3^N`$ found by adversarial search is $`2/3`$.

**Why.** The affine theory rests on $`J_E`$ being a polynomial of degree at most $`N`$ in the policy coordinates, with $`\sum_\alpha\lvert c_\alpha\rvert\le3^N`$ ([Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(a)). One tree whose exact expansion has a higher degree or a larger norm would falsify it.

**Setup.**

- 90 search rounds with $`k\in\lbrace 1,2,3\rbrace`$ contexts and $`N\in\lbrace 1,\dots,5\rbrace`$ decisions per path.
- Contexts may recur along a path; chance nodes have rational probabilities.
- Each round evaluates 12 candidate trees: every fourth is a fresh random tree, the rest are random leaf, context or chance-probability mutations of the current worst-ratio tree (1,080 trees).
- Seed: `SeedSequence([20261010, 6])`.

**Procedure.**

1. Expand each tree symbolically (SymPy) into an exact rational polynomial.
2. Assert total degree $`\le N`$ and $`\sum\lvert c_\alpha\rvert\le3^N`$.
3. Assert $`J\in[0,1]`$ at 3 random rational points.
4. Hill-climb on $`\sum\lvert c_\alpha\rvert/3^N`$ to search for the worst case.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Trees / assertions | 1,080 / 5,400 | entry `v014:C6`, `n_instances`, `.n_assertions` |
| Largest $`\sum\lvert c_\alpha\rvert/3^N`$ | $`2/3`$ | entry `v014:C6`, `details.exact_best_ratio` |
| Witness | $`N=1`$, $`J=1-p_1`$ | entry `v014:C6`, `details.witness` |
| Status | PASS (worst residual 0) | entry `v014:C6`, `status` |

**Why it supports Theorem 4.4(a).**

- Degree and coefficient bounds hold on every tree, including the adversarially mutated ones.
- The $`3^N`$ bound is valid but conservative on these trees.

<a id="c4-3"></a>

## Check 4.3 · Moment matching is sufficient and necessary (Theorem 4.4(b))

**Verdict.** PASS (46 instances, 385 assertions, worst residual exactly 0). Moment-matched laws give identical learning values in 24 random configurations with up to $`R=5`$ updates. Every single moment of order at most $`N`$ is detected by the corresponding monomial task.

**Why.** [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(b) has two directions, each with a falsifier:

- a moment-matched pair with different learning values for some task or affine learner (against sufficiency);
- a moment whose mismatch no task can detect (against necessity).

**Setup and procedure.**

- *Sufficiency*, 24 configurations ($`k\in\lbrace 1,2\rbrace`$, $`N\in\lbrace 1,2,3\rbrace`$, $`R\in\lbrace 1,\dots,5\rbrace`$):
  1. take a random positive law $`P`$ on the grid $`\lbrace 0,\tfrac1{N+1},\dots,1\rbrace ^k`$;
  2. construct $`Q=P+av`$ with $`v`$ in the exact null space of all moment constraints of degree $`\le N`$, mass included;
  3. draw random convex weights, a random $`p_0`$ and a random task tree;
  4. compute both learning values by exact block-moment convolution, and assert that all matched moments and both values are equal.
- *Necessity*, for $`(k,N)\in\lbrace (1,4),(2,3),(3,2)\rbrace`$ and every nonzero $`\alpha`$ with $`\lvert\alpha\rvert\le N`$:
  1. verify symbolically that the task "visit $`j`$ exactly $`\alpha_j`$ times, pay iff all outcomes are ones" evaluates to $`p^\alpha`$;
  2. construct a law pair that differs only in moment $`\alpha`$;
  3. assert that the one-update learner $`p=Z`$ separates the pair.
- *Two-context example*: equal marginals, cross moments $`\tfrac12`$ versus $`0`$.
- Seed: `SeedSequence([20261010, 7])`.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Instances / assertions | 46 / 385 | entry `v014:C7`, `n_instances`, `.n_assertions` |
| Affine sufficiency configurations | 24, with $`R`$ up to 5 | entry `v014:C7`, `details.affine_configurations` |
| Worst residual | 0 (exact) | entry `v014:C7`, `worst_residual` |
| Status | PASS | entry `v014:C7`, `status` |

**Why it supports Theorem 4.4(b).**

- With all joint moments through order $`N`$ matched, the learning values are exactly equal; the differences are zero, not small.
- Conversely, every single moment through order $`N`$ is detected by some task, so no moment can be dropped.
- Matching marginals only is not enough (two-context example).

<a id="c4-4"></a>

## Check 4.4 · Every order is necessary; a constant-gap witness (Theorem 4.4(c))

**Verdict.** PASS (22 instances, 8,983 assertions, worst residual 0). The scalar witnesses satisfy $`\Delta_N=(-1)^NN!/(2^{N-1}N^N)`$ exactly for $`N=1,\dots,14`$ with all lower moments equal. The parity laws match every joint multi-index of degree below $`N`$ and keep the gap at 1 for $`N=1,\dots,8`$; an independent implementation agrees.

**Why.** [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(c) says the hierarchy is strict:

- for each $`N`$, some laws agree on all lower moments yet differ in the order-$`N`$ learning value;
- in the parity construction the gap does not shrink with $`N`$.

**Setup and procedure.**

- *Scalar witness.* For $`N=1,\dots,14`$, build $`P_N`$ and $`Q_N`$ in exact rationals. Assert unit mass, equality of all moments of order $`\ell<N`$, and $`m_{P_N}(N)-m_{Q_N}(N)=(-1)^NN!/(2^{N-1}N^N)`$.
- *Parity witness.* For $`N=1,\dots,8`$, enumerate $`\lbrace 0,1\rbrace ^N`$:
  1. assert equality of every joint moment over all multi-indices of total degree $`<N`$ (not only coordinate subsets);
  2. assert that $`J_{\rm even}`$ takes the values 1 and 0 under the two laws;
  3. assert at 4 random rational $`p`$ that the closed form equals the enumerated even-parity probability.
- Seed: `SeedSequence([20261010, 8])`.
- *Independent implementation* (`pkg:constant_gap_parity`, written by the theory author): for $`N=1,\dots,8`$, test all $`2^N-1`$ proper coordinate subsets for equal subset moments — binary coordinates are idempotent, so this covers all raw moments of degree $`<N`$ — and compute both values exactly.

**Result.**

| $`N`$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $`\Delta_N`$ (exact) | $`-1`$ | $`1/4`$ | $`-1/18`$ | $`3/256`$ | $`-3/1250`$ | $`5/10368`$ | $`315/16777216`$ | $`567/800000000`$ | $`\approx9.58\times10^{-10}`$ |

Source: entry `v014:C8`, `details.scalar`; all 14 values equal the closed form.

| Quantity | Value | Source |
|---|---:|---|
| Parity orders with all lower multi-indices matched | $`N=1,\dots,8`$ | entry `v014:C8`, `details.parity_all_multiindices` (true) |
| Assertions | 8,983 | entry `v014:C8`, `n_assertions` |
| Status | PASS (worst residual 0) | entry `v014:C8`, `status` |
| Independent parity gap, $`N=1,\dots,8`$ | exactly 1 (reference 1.0, alternative 0.0) | `results_v014_rerun.json → constant_gap_parity` |
| Matching proper-subset tests | $`1,3,7,15,31,63,127,255`$ | same |

The re-run file `theory/v014_repro/results_v014_rerun.json` is identical to the package's own `docs/direction/v014_handoff/results_v014.json` for this entry.

**Why it supports Theorem 4.4(c).**

- Every order $`N`$ is necessary for some scalar task, and the gap formula is exact.
- The scalar gap shrinks exponentially in $`N`$, while the parity gap stays 1 at every order. Exact necessity and fixed-accuracy hardness are therefore different properties; [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6) and Checks 4.9–4.10 separate them.

<a id="c4-5"></a>

## Check 4.5 · Exact statistic count and the binary barrier (Theorem 4.4(d))

**Verdict.** PASS (22 instances, 903 assertions, worst LP residual $`2.3\times10^{-12}`$). The grid rank equals $`\binom{k+N}N`$ in all 18 configurations; deleting any of the 237 nonconstant monomials leaves a separable pair; all 32 random binary interfaces with fewer than $`2^k-1`$ features have worst-pair discrepancy exactly 1.

**Why.** [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(d) instantiates the span criterion of [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(a) in numbers:

- exactly $`\binom{k+N}N-1`$ statistics are necessary and sufficient on the grid;
- on binary messages with arbitrary payoffs, fewer than $`2^k-1`$ statistics leave worst-case error 1.

**Setup and procedure.**

1. *Grid rank.* For all $`(k,N)`$ with $`k\le3`$, $`N\le5`$, and for $`k=4`$, $`N\le3`$ (18 configurations), compute the exact integer rank of the monomial evaluation matrix on the grid and assert that it equals $`\binom{k+N}N`$.
2. *Removed features.* For each nonconstant monomial of each configuration (237 in total), delete it, solve the worst-pair LP, and assert that a feature-matched pair separates the deleted monomial.
3. *Binary barrier.* For $`k=1,\dots,4`$, draw 8 random rational feature tables with $`d<2^k-1`$. Take a null vector of $`[\mathbf 1;\phi]`$, set the event to its positive support, and assert that the LP discrepancy equals 1. For $`k\le3`$, also enumerate all events and assert that the optimum is 1.
4. *Full interface.* With $`2^k-1`$ generic features, assert discrepancy 0 for every indicator response and for a random response.
5. *Independent implementation* (`pkg:grid_response_dimensions`): numerical rank (tolerance $`10^{-10}`$) of the monomial matrix on $`\lbrace 0,\tfrac1N,\dots,1\rbrace ^k`$ in six configurations.

Seed: `SeedSequence([20261010, 9])`.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Grid ranks $`=\binom{k+N}N`$ | 18/18, e.g. $`(k,N)=(3,5)`$: 56; $`(4,3)`$: 35 | entry `v014:C9`, `details.grid_ranks` |
| Dropped nonconstant features, all separable | 237 | entry `v014:C9`, `details.dropped_features` |
| Binary interfaces with $`d<2^k-1`$; worst-pair discrepancy | 32; all $`=1`$ (max deviation $`4.4\times10^{-16}`$) | entry `v014:C9`, `details.binary_barriers` |
| Full $`2^k-1`$ interface discrepancy | 0 | assertion `full generic interface discrepancy` |
| Assertions / worst LP residual | 903 / $`2.3\times10^{-12}`$ | entry `v014:C9`, `n_assertions`, `.worst_residual` |

| $`(k,N)`$ | $`(1,1)`$ | $`(1,4)`$ | $`(2,2)`$ | $`(2,3)`$ | $`(3,2)`$ | $`(3,3)`$ |
|---|---:|---:|---:|---:|---:|---:|
| alphabet size | 2 | 5 | 9 | 16 | 27 | 64 |
| rank $`=\binom{k+N}N`$ | 2 | 5 | 6 | 10 | 10 | 20 |
| minimal nonconstant statistics | 1 | 4 | 5 | 9 | 9 | 19 |

The second table is the independent implementation, `results_v014_rerun.json → grid_response_dimensions`.

![Figure 4.2: every moment order is necessary, and the exact statistic count is a binomial coefficient](../figures/fig4_2_moment_order.png)

*Figure 4.2.* Scalar witnesses separate at every order with exponentially small gaps that match the closed form, while the parity laws keep gap 1 (Check 4.4). The monomial evaluation rank on the grid equals $`\binom{k+N}N`$ in all 18 configurations, and on binary messages with arbitrary payoffs every interface with fewer than $`2^k-1`$ features leaves worst-case error 1 (this check). [PDF](../figures/fig4_2_moment_order.pdf)

**Why it supports Theorem 4.4(d).**

- The binomial count is exact: each nonconstant moment is individually necessary, and together they suffice.
- The binary barrier is attained by an explicit separating event, not only bounded.

<a id="c4-6"></a>

## Check 4.6 · A discrepancy bound that does not grow with the number of updates (Theorem 4.5(a))

**Verdict.** PASS on 288 adversarially searched instances (582 assertions, worst residual 0). The bound $`C_J\delta_N`$ is never exceeded; the tightest instance has gap/bound $`55/57\approx0.965`$. On a fixed problem the gap and the bound are both constant as $`R`$ grows from 1 to 50.

**Why.** [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(a) bounds the learning-value gap of every convex-affine learner by $`C_J\delta_N\le N3^N\delta_N`$, uniformly in the number of updates $`R`$. It is falsified by an instance above the bound, or by a family whose worst ratio grows with $`R`$.

**Setup.**

- 24 search trials, $`k\in\lbrace 1,2\rbrace`$, $`N\in\lbrace 1,2,3\rbrace`$, support $`\lbrace 0,\tfrac12,1\rbrace ^k`$.
- $`R`$ cycles through $`\lbrace 1,2,5,10,25,50\rbrace`$.
- Random laws $`P,Q`$, random convex weights $`w_0,\dots,w_R`$, a random $`p_0`$, and a random task tree whose exact polynomial gives $`C_J`$.
- Seed: `SeedSequence([20261010, 10])`.

**Procedure.**

1. In each trial, run a 12-step rational hill-climb on $`Q`$ (moving mass $`\le\tfrac1{20}`$ between support points) to maximize gap/bound: 288 instances.
2. Compute each learning value exactly by independent-block moment convolution.
3. Assert gap $`\le C_J\delta_N\le N3^N\delta_N`$.
4. *Fixed-problem sweep.* $`P=(\tfrac14,\tfrac34)`$ and $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, $`J=p^2`$, uniform average of $`R`$ blocks, $`R=1,\dots,50`$.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Instances / assertions | 288 / 582 | entry `v014:C10`, `n_instances`, `.n_assertions` |
| Largest gap/bound found | $`55/57\approx0.9649`$ at $`N=1`$, $`R=10`$ | entry `v014:C10`, `details.best_exact_ratio`, `.best_instance` |
| Status | PASS (worst residual 0) | entry `v014:C10`, `status` |

| Fixed problem, $`R`$ | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| exact gap | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| bound $`C_J\delta_N`$ | 1 | 1 | 1 | 1 | 1 | 1 |

Source: entry `v014:C10`, `details.fixed_problem_R_sweep`.

**Why it supports Theorem 4.5(a).**

- No adversarial instance exceeds the bound, and the bound comes within 4% of being attained.
- The number of updates never enters: in the sweep, gap and bound are constant in $`R`$.

<a id="c4-7"></a>

## Check 4.7 · The first unmatched cumulant sets the rate (Theorem 4.5(b)–(d))

**Verdict.** PASS (1,530 instances, 1,797 assertions). The exact scalar law holds in all 210 cells $`N=2,\dots,8`$, $`n=1,\dots,30`$. In all six cumulant cases the exact gap starts at $`n^{1-r}`$ with the predicted coefficient and has no slower term; a vanishing contraction removes the leading term; unequal means converge to $`J(\mu_P)-J(\mu_Q)=13/32`$ at rate $`O(1/n)`$. A cumulant-free implementation reproduces the exact law in 20 cases.

**Why.** [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(b) predicts both the exponent $`1-r`$ and the coefficient $`\langle D^rJ(\mu),\Delta\kappa_r\rangle/r!`$ of the decay in the number $`n`$ of independent blocks. It is falsified by a wrong leading coefficient, by a slower term $`n^{-j}`$ with $`j<r-1`$, by a residual that is not $`O(n^{-r})`$, or by a nonzero leading term when the contraction vanishes. Part (c) is the exact law $`\Delta_N/n^{N-1}`$ for the scalar witnesses, and part (d) gives the limit $`J(\mu_P)-J(\mu_Q)`$ when the means differ.

**Setup.**

- *Exact law, part (c).* The scalar witness laws of Check 4.4 with $`J=p^N`$, $`N=2,\dots,8`$, $`n=1,\dots,30`$.
- *Cumulant cases*, six of them, $`k\in\lbrace 1,2\rbrace`$, $`r\in\lbrace 2,3,4\rbrace`$:
  - $`k=1`$: the scalar witness laws of order $`r`$;
  - $`k=2`$: the bivariate lift $`(x,(x+b)/2)`$ with $`b\in\lbrace 0,1\rbrace`$ equally likely;
  - task $`J=(\tfrac1k\sum_jp_j)^{r+1}`$.
- *Vanishing contraction.* $`J=(p-\tfrac12)^{r+2}`$ for $`r=2,3,4`$; its $`r`$-th derivative at $`\mu=\tfrac12`$ is zero.
- *Unequal means.* $`P=(\tfrac14,\tfrac34)`$, $`Q=(\tfrac34,\tfrac14)`$ on $`\lbrace 0,1\rbrace`$, $`J=p^3`$.

**Procedure.**

1. Compute $`\mathbb EJ(\bar p_n)`$ exactly as a polynomial in $`1/n`$, by sample-index multinomial expansion.
2. Compute the cumulants independently, from the moments.
3. Assert that lower cumulants match, that the coefficient of $`n^{1-r}`$ equals the predicted contraction, and that all slower coefficients are zero.
4. Assert that $`n\cdot\lvert n^{r-1}\text{gap}-\text{coefficient}\rvert`$ stays below the envelope of higher coefficients for $`n=1,\dots,220`$.
5. *Independent implementation* (`pkg:scalar_exact_decay`): convolve the laws $`n`$ times and evaluate $`\mathbb E\bar p_n^N`$ directly, never using cumulants, for $`N=2,\dots,6`$ and $`n\in\lbrace 1,2,4,8\rbrace`$.

**Result.**

| Case $`(k,r)`$ | predicted coefficient | exact gap as a polynomial in $`1/n`$ |
|---|---:|---|
| (1, 2) | $`3/8`$ | $`\tfrac38\,n^{-1}`$ |
| (1, 3) | $`-1/9`$ | $`-\tfrac19\,n^{-2}`$ |
| (1, 4) | $`15/512`$ | $`\tfrac{15}{512}\,n^{-3}`$ |
| (2, 2) | $`27/128`$ | $`\tfrac{27}{128}\,n^{-1}`$ |
| (2, 3) | $`-3/64`$ | $`-\tfrac3{64}\,n^{-2}`$ |
| (2, 4) | $`1215/131072`$ | $`\tfrac{1215}{131072}\,n^{-3}`$ |
| vanishing, $`r=2`$ | 0 | $`\tfrac3{16}n^{-2}-\tfrac18n^{-3}`$ |
| vanishing, $`r=3`$ | 0 | $`-\tfrac5{108}n^{-3}+\tfrac5{162}n^{-4}`$ |
| vanishing, $`r=4`$ | 0 | $`\tfrac{45}{4096}n^{-4}-\tfrac{15}{2048}n^{-5}`$ |
| unequal means | limit $`13/32`$ | $`\tfrac{13}{32}+\tfrac9{32}n^{-1}-\tfrac3{16}n^{-2}`$ |

- Sources: entry `v014:C11`, `details.cumulant_cases[*].predicted_coefficient`, `.exact_gap_polynomial_in_inverse_n`, `.gap` (vanishing cases), entry `v014:C11`, `details.unequal_means_gap`, `.unequal_means_limit`.
- The exact law of part (c) holds in all $`7\times30=210`$ cells $`(N,n)`$; status PASS.
- Independent implementation: all 20 enumerated gaps equal $`\Delta_N/n^{N-1}`$ exactly, e.g. $`N=4`$ gives $`3/256,\ 3/2048,\ 3/16384,\ 3/131072`$ (`results_v014_rerun.json → scalar_exact_decay`; fields `enumerated_gap_exact` and `predicted_gap_exact` identical in every row).

**Why it supports Theorem 4.5(b)–(d).**

- In every case the gap begins exactly at $`n^{1-r}`$ with the predicted coefficient: exponent and coefficient are both set by the first unmatched cumulant.
- A zero contraction makes the decay faster ($`n^{-r}`$), as the expansion predicts.
- With unequal means the limit $`J(\mu_P)-J(\mu_Q)`$ is approached at $`O(1/n)`$.

<a id="c4-8"></a>

## Check 4.8 · Independent evidence versus replay versus synthetic descendants (Theorem 4.5(e))

**Verdict.** PASS (45 instances, 140 assertions). With independent blocks the gap decays exactly as $`\text{gap}_1/n^{N-1}`$ (log-log slopes $`-1,-2,-3`$ for $`N=2,3,4`$). Replaying one block, or training on deterministic descendants of one parent block, leaves the gap constant through 256 exposures.

**Why.** The rates of [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5)(b)–(c) are in the number of independent blocks. Part (e) predicts that two kinds of exposure without new evidence leave the gap at its one-block value: replaying one block $`n`$ times, and training on $`n`$ deterministic descendants of one parent block. This separates information from exposure.

**Setup.**

- Scalar witness laws with $`N=2,3,4`$, and parity laws with $`N=2,3`$.
- Exposure $`n\in\lbrace 1,2,4,\dots,256\rbrace`$.
- Descendant channel $`T(Z)=(Z+\tfrac12)/2`$, a fixed information-preserving map.
- Seed: `SeedSequence([20261010, 18])`.

**Procedure.** For each family, $`N`$ and $`n`$, compute three gaps exactly:

- independent blocks, $`\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)`$;
- replay, $`n`$ copies of one block, so $`\bar p_n=Z_1`$;
- descendants, $`n`$ copies of $`T(Z_1)`$, so $`\bar p_n=T(Z_1)`$.

Then assert that the independent gap equals the one-block gap divided by $`n^{N-1}`$, that the replay gap equals the one-block gap, and that the descendant gap equals the one-block gap divided by $`2^N`$ (the contraction of this channel).

**Result.**

| Family, $`N`$ | gap, $`n=1`$ | independent, $`n=256`$ | replay, $`n=256`$ | descendants, $`n=256`$ | independent log-log slope |
|---|---:|---:|---:|---:|---:|
| scalar, 2 | $`1/4`$ | $`1/1024`$ | $`1/4`$ | $`1/16`$ | $`-1`$ |
| scalar, 3 | $`-1/18`$ | $`-1/1179648`$ | $`-1/18`$ | $`-1/144`$ | $`-2`$ |
| scalar, 4 | $`3/256`$ | $`3/4294967296`$ | $`3/256`$ | $`3/4096`$ | $`-3`$ |
| parity, 2 | $`1`$ | $`1/256`$ | $`1`$ | $`1/4`$ | $`-1`$ |
| parity, 3 | $`1`$ | $`1/65536`$ | $`1`$ | $`1/8`$ | $`-2`$ |

Source: entry `v014:C18`, `details.curves`; slopes from the $`n=1`$ and $`n=256`$ rows.

![Figure 4.3: only independent blocks make the discrepancy decay](../figures/fig4_3_rates.png)

*Figure 4.3.* Independent blocks shrink the gap as $`n^{1-N}`$, while replaying one block or training on its descendants leaves it constant (this check). Rescaled by $`n^{r-1}`$, the exact gap converges to the first-unmatched-cumulant coefficient in all six cases, and a zero contraction removes the leading term (Check 4.7). The round-independent bound holds for $`R`$ up to 50 and is nearly attained (Check 4.6). [PDF](../figures/fig4_3_rates.pdf)

**Why it supports Theorem 4.5(e).**

- Only independent evidence makes the discrepancy decay, at exactly the predicted exponent $`1-N`$.
- Replay and descendants raise training exposure 256-fold and leave the learning-value gap unchanged; descendants rescale it once through the channel and never decay.
- Scope: the channel is the stated deterministic map. It is a constructed control, not a claim about every possible post-processing.

<a id="c4-9"></a>

## Check 4.9 · Chebyshev identity, exact tails and the parity minimax (Proposition 4.6)

**Verdict.** PASS (136 instances, 484 assertions, worst residual $`1.0\times10^{-17}`$). For $`N=1,\dots,16`$ and every $`q<N`$, the truncation error of $`p^N`$ equals the binomial tail and stays below $`2e^{-q^2/N}`$ (largest tail/bound ratio 0.430). An independent implementation to $`N=32`$ finds that accuracy $`10^{-3}`$ needs only $`q=4,6,9,13`$ for $`N=4,8,16,32`$. The best degree-$`(<N)`$ approximation error of parity is exactly $`1/2`$.

**Why.** [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6)(a) makes three falsifiable claims: the Chebyshev expansion is an identity; the truncation error equals the binomial tail $`\Pr\lbrace\lvert B-N\rvert>q\rbrace`$, $`B\sim\mathrm{Bin}(2N,\tfrac12)`$, not merely is bounded by it; and the tail is at most $`2e^{-q^2/N}`$. Part (b) says the best lower-degree error for parity is exactly $`1/2`$; a smaller value would falsify it.

**Setup.**

- Scalar task: $`N=1,\dots,16`$ and all $`q<N`$.
- Parity: $`N=1,\dots,5`$ on all $`2^N`$ cube vertices; 10 random rational points per $`N`$ check $`\lvert J_{\rm even}-\tfrac12\rvert\le\tfrac12`$.
- Seed: `SeedSequence([20261010, 12])`.
- Independent implementation (`pkg:scalar_approximation`): $`N\in\lbrace 4,8,16,32\rbrace`$, $`q=0,\dots,N`$, 8,001-point grid.

**Procedure.**

1. Verify the Chebyshev identity symbolically: $`\mathrm{expand}(\text{RHS}-p^N)=0`$ in SymPy.
2. For each $`(N,q)`$:
   - compute $`\max\lvert p^N-h_q\rvert`$ on a 3,001-point grid that includes $`p=1`$ and assert it equals the coefficient tail;
   - assert in exact rationals that the coefficient tail equals the binomial tail;
   - assert tail $`\le2e^{-q^2/N}`$.
3. Parity: solve the minimax LP $`\min_{h\in\mathcal P_{N-1}}\max_{\rm vertices}\lvert J_{\rm even}-h\rvert`$ over all nonconstant monomials of degree $`<N`$, and assert the optimum is $`0.5`$.
4. Independent implementation: compare $`\lvert x^N-h_q\rvert`$ on the grid with the coefficient tail and with $`\min(1,2e^{-q^2/N})`$.

**Result.**

| Quantity | Value | Source |
|---|---:|---|
| Instances / assertions | 136 / 484 | entry `v014:C12`, `n_instances`, `.n_assertions` |
| Range of $`N`$ for identity and tails | 1–16 | entry `v014:C12`, `details.N_range` |
| Largest tail $`/\,2e^{-q^2/N}`$ | 0.430 | entry `v014:C12`, `details.exhaustive_tail_best_ratio` |
| Parity best degree-$`(<N)`$ error | 0.5 exactly ($`N=1,\dots,5`$) | entry `v014:C12`, `details.parity_minimax_error` |
| Worst residual | $`1.0\times10^{-17}`$ | entry `v014:C12`, `worst_residual` |
| Status | PASS | entry `v014:C12`, `status` |

Independent implementation: all 64 rows satisfy grid error = tail ($`\max\lvert\text{grid}-\text{tail}\rvert=8.9\times10^{-16}`$) and tail $`\le`$ bound. Selected rows:

| $`N`$ | $`q`$ | truncation error | $`2e^{-q^2/N}`$ | parity best degree-$`q`$ error |
|---:|---:|---:|---:|---:|
| 16 | 4 | 0.1102 | 0.7358 | 0.5 |
| 16 | 8 | 0.00210 | 0.0366 | 0.5 |
| 16 | 12 | $`2.56\times10^{-6}`$ | $`2.47\times10^{-4}`$ | 0.5 |
| 32 | 8 | 0.0328 | 0.2707 | 0.5 |
| 32 | 16 | $`2.44\times10^{-5}`$ | $`6.71\times10^{-4}`$ | 0.5 |
| 32 | 20 | $`1.01\times10^{-7}`$ | $`7.45\times10^{-6}`$ | 0.5 |

Source: `results_v014_rerun.json → scalar_approximation` (fields `Chebyshev_truncation_error`, `grid_check_error`, `Hoeffding_upper`, `parity_best_degree_q_error`). The smallest $`q`$ with truncation error $`\le10^{-3}`$ is $`4,6,9,13`$ for $`N=4,8,16,32`$.

**Why it supports Proposition 4.6.**

- The truncation error of $`p^N`$ is exactly a binomial tail and decays like $`e^{-q^2/N}`$; the stated bound is conservative by at least a factor 2.3 in the tested range.
- The order needed for practical accuracy grows like $`\sqrt N`$: doubling $`N`$ adds only a few orders, although exact equality needs order $`N`$.
- Parity cannot be approximated better than the trivial constant by any lower degree, so the $`\sqrt N`$ behaviour is a property of the task, not of the degree.

<a id="c4-10"></a>

## Check 4.10 · Same maximum degree, different response-space complexity (Proposition 4.6; Theorems 4.2 and 4.4(d))

**Verdict.** PASS (21 subspaces, 1,445 assertions). Three response families share maximum degree 3. The scalar and structured families are resolved exactly by 3 statistics; the parity family keeps uniform error at least $`0.0625`$ (discrepancy at least $`0.125`$) until all 7 nonconstant directions are present. Exact span dimensions, including the constant, are 4, 4 and 8. On every selected subspace the discrepancy equals twice the uniform error to $`1.6\times10^{-15}`$.

**Why.** The degree of a response family does not determine its approximate complexity; the operative quantity is the approximation width, which [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(b) identifies with the worst-pair discrepancy. This check compares three families with the same maximum degree, expecting the structured families to be cheap and parity expensive ([Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6)). It also re-tests the duality of Theorem 4.2(b) on structured rather than random responses, and the span dimension that Theorem 4.2(a) and [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(d) predict for a given family.

**Setup.**

- Grid $`\lbrace 0,\tfrac14,\dots,1\rbrace ^3`$ (125 messages); maximum degree 3 in every family.
- *Scalar affine powers* $`(a+bx_1)^3`$: 5 responses.
- *Multivariate parity pullbacks*: $`J_{\rm even}`$ composed with 8 affine maps, plus the raw parity: 9 responses.
- *Structured sum powers* $`(a+b\bar x)^3`$: 5 responses.
- Seed: `SeedSequence([20261010, 19])`.

**Procedure.**

1. Compute the exact rational rank of each family, including the constant.
2. Fix a finite dictionary before fitting: the response vectors, all multilinear monomials, and the scalar or structured Chebyshev bases.
3. For each budget $`d=1,\dots,7`$, select a subspace greedily and compute the uniform error $`\max_g\mathrm{dist}_\infty(g,H)`$ by LP over the full grid, and the worst-pair discrepancy for that subspace by an independent primal LP.
4. Assert discrepancy $`=2\times`$ error. Positive errors are upper bounds on the optimal width, because the greedy subspace need not be optimal; zero errors are exact.

**Result (uniform error; the discrepancy is exactly twice this).**

| feature budget $`d`$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar affine powers | 0.0569 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| structured sum powers | 0.0582 | 0.0068 | **0** | 0 | 0 | 0 | 0 |
| multivariate parity | 0.25 | 0.125 | 0.125 | 0.125 | 0.0625 | 0.0625 | **0** |

- Source: entry `v014:C19`, `details.families[*].curve[*].uniform_error` (and `.discrepancy`, `.optimality`).
- Span dimensions including the constant: 4 (scalar), 8 (parity), 4 (structured) (`span_dimension_including_constant`).
- Duality on all 21 (family, $`d`$) subspaces: $`\max\lvert\text{discrepancy}-2\cdot\text{uniform error}\rvert=1.6\times10^{-15}`$.
- Held-out responses from the same families: the selected scalar and structured subspaces give errors of at most $`0.017`$ at $`d=1`$ and $`0`$ from $`d=3`$ on (`heldout_uniform_error`).

![Figure 4.4: same exact degree, very different approximate information order](../figures/fig4_4_approximation.png)

*Figure 4.4.* Low-order Chebyshev truncations of $`p^{16}`$ are already close to it, and the truncation error equals a binomial tail below $`2e^{-q^2/N}`$, while parity keeps error $`1/2`$ below degree $`N`$ (Check 4.9). On one grid and one maximum degree, the scalar and structured families are resolved exactly by 3 features and parity only by 7 (this check; hollow markers are upper bounds from the best subspace found). [PDF](../figures/fig4_4_approximation.pdf)

**Why it supports the statements.**

- At equal maximum degree the scalar and structured families are captured exactly by 3 statistics, parity needs all $`2^3-1=7`$, and with 3 statistics parity still leaves worst-pair discrepancy $`0.25`$: task structure, not degree or horizon, sets the information order ([Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6)).
- The exact span dimensions are the numbers the span criterion predicts, and they are far below the worst case over all tasks counted by [Theorem 4.4](../proofs/04_complexity_of_sufficiency.md#s4-4)(d).
- The duality of [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(b) holds on structured, non-random responses too, including the parity family, where the discrepancy stays large.

<a id="c4-11"></a>

## Check 4.11 · Degree composition, attainment, the threshold learner and the SGD expansion (Theorem 4.7(a)–(d))

**Verdict.** PASS (72 instances, 371 assertions, worst residual 0). In all 12 random polynomial update chains the actual block degrees equal the bound $`D_i=Nb_i\prod_{t>i}a_t`$, and blockwise moment matching through $`D_i`$ preserves the learning value exactly. The squaring witness attains the bound at degrees 1, 2, 4, 8. The threshold learner separates two equal-mean laws, $`2/3`$ against $`0`$. The SGD expansion remainder never exceeds 0.275 of its $`\eta^3`$ bound.

**Why.** [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(a) claims that the response degree in block $`i`$ is at most $`Nb_i\prod_{t>i}a_t`$ and that this many moments suffice; a wrong degree count would make the moment sufficiency false. Part (b) claims the bound is attained, so a loose count would be detected. Part (c) claims that no moment order suffices for a discontinuous update. Part (d), the local SGD expansion, is the first-order description of SGD used in [§5](05_optimizer_state.md), so its remainder bound must hold.

**Setup.**

- 12 random chains with $`R\in\lbrace 1,2,3\rbrace`$, $`N\in\lbrace 1,2\rbrace`$, degrees $`a_i,b_i\in\lbrace 1,2\rbrace`$, and $`p_i=\lambda_ip_{i-1}^{a_i}+(1-\lambda_i)Z_i^{b_i}`$ with random $`\lambda_i\in\lbrace \tfrac14,\tfrac12,\tfrac34\rbrace`$.
- Exact symbolic pullback (SymPy) and exact rational block moments; block $`i`$ uses the scalar witness laws of order $`D_i+1`$.
- The squaring witness for $`R=1,\dots,4`$, and the threshold laws of part (c).
- Part (d): 15 random cubic $`V`$ on $`\mathbb R^3`$ with random discrete gradient laws on 6 atoms, and the same with a logistic $`V(\theta)=\sigma(w^\top\theta)`$; step sizes $`\eta\in\lbrace 0.01,0.1,0.5,1\rbrace`$.

**Procedure.**

- Expand $`J(p_R)`$, read off each block degree, and compare it with $`D_i`$; evaluate both learning values with moments matched through $`D_i`$.
- For part (d), compute the exact difference $`(K^P-K^Q)V(\theta)`$ and the first two expansion terms, and compare the remainder with $`\tfrac{M_3\eta^3}6(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)`$:
  - for cubics, $`M_3`$ is the Frobenius norm of the constant third-derivative tensor, which upper-bounds the operator norm;
  - for the logistic, $`M_3=\lVert w\rVert^3/8`$, because $`\lvert\sigma'''\rvert\le1/8`$ globally.

**Result.** Source: `theory/independent_checks/results_independent_v014.json`, `v014:C13`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 72, 371, 0.0 |
| Declared vs actual block degrees (`details.degrees`, 12 chains) | equal in every chain, e.g. $`a=(2,2,2)`$, $`b=(2,2,1)`$, $`N=2`$: $`D=(16,8,2)`$, actual $`(16,8,2)`$ |
| Value equality under blockwise matching | exact in all 12 chains |
| Squaring witness (`details.amplification`) | $`R=1,2,3,4`$; degree $`1,2,4,8`$; gap $`-1`$, $`1/4`$, $`3/256`$, $`315/16777216`$ |
| Threshold learner (assertions in `check_C13`) | equal mean $`1/2`$; values $`2/3`$ vs $`0`$ |
| Logistic remainder/bound, max over 15 instances (`details.logistic`) | $`\eta=0.01`$: 0.274; $`0.1`$: 0.274; $`0.5`$: 0.270; $`1`$: 0.265 |
| Largest remainder/bound overall (`best_ratio`) | 0.2743 |

**Why it supports Theorem 4.7(a)–(d).**

- The degree bound is exact on random chains and attained by the squaring witness, so the order a source must match grows with the composition of updates.
- The threshold learner shows that no fixed moment order suffices once the update is discontinuous.
- The expansion of part (d) holds with room to spare at every tested step size.

<a id="c4-12"></a>

## Check 4.12 · Accumulation of per-step errors (Theorem 4.7(e))

**Verdict.** PASS (717 instances, 1,435 assertions, worst residual 0). The geometric bound holds in every instance and is attained with equality. The witness $`\eta=\tfrac12`$, $`L=C=1`$, $`R=4`$ has $`e_4=65/32`$, above the linear value $`RC\eta^2=1`$.

**Why.** A one-step $`O(\eta^2)`$ error does not by itself give an $`O(R\eta^2)`$ training error. [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(e) states a geometric bound and claims it is sharp, so that linear accumulation needs a stability condition. Both claims are checked exactly.

**Setup.** Exact rational recursions $`e_{t+1}=s\cdot((1+\eta L)e_t+C\eta^2)`$ with slack $`s\in(0,1]`$, random $`\eta\in(0,1]`$, $`L\in[0,1]`$, $`C\in(0,1]`$ and $`R\le40`$; fixed-seed random restarts and hill-climbing over $`(\eta,L,C,R,s)`$ supplement the random cases.

**Procedure.** Assert $`e_R\le C\eta^2\sum_{s<R}(1+\eta L)^s`$, with equality at $`s=1`$. Search for the largest ratio of the bound to the linear simplification $`RC\eta^2`$. Evaluate the witness $`\eta=\tfrac12`$, $`L=C=1`$, $`R=4`$.

**Result.** Source: `theory/v015_checks/results_v015_independent.json`, `v015:V7`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 717, 1,435, 0.0 |
| Geometric bound attained (`data.geometric_adversarial.best.score`) | 1.0 (e.g. $`\eta=4/5`$, $`L=1/5`$, $`C=7/10`$, $`R=13`$) |
| Witness (`data.linear_failure`) | $`e_4=65/32`$, bound $`65/32`$, linear $`RC\eta^2=1`$ |
| Largest bound/linear ratio found by the search (`best_ratio`) | $`3.78\times10^{30}`$ ($`\eta=2`$, $`L=2`$, $`R=47`$) |

**Why it supports Theorem 4.7(e).**

- The geometric bound is exact: it holds everywhere and is attained.
- The linear-in-$`R`$ reading fails as soon as $`\eta LR`$ is not small, so accumulating local expansions along training requires the stated stability or short-horizon condition.

<a id="c4-13"></a>

## Check 4.13 · Where the local affine SGD description applies in a controlled network (Theorem 4.7(d),(e))

**Verdict.** Regime map, consistent with the theorem. The tangent affine-SGD prediction agrees with the actual network in all 35 SGD settings with $`\eta\le0.1`$ and at most 4 updates, or $`\eta=0.01`$ over 40 updates; the largest deviation there is $`2.5\times10^{-4}`$. At $`\eta\ge1`$ it departs in 35 of 42 settings, by up to 0.53.

**Why.** [Theorem 4.7](../proofs/04_rates_and_composition.md#s4-7)(d),(e) predicts that for SGD the response is governed by the mean consumed gradient up to $`O(\eta^2)`$ per step, with errors that accumulate geometrically. In a nonlinear network, a fixed initial-tangent affine-SGD prediction should therefore be accurate at small $`\eta R`$ and inaccurate at large $`\eta`$ or long horizons. This is a test of where the description applies, not of a bound.

**Setup.** Controlled small network, not a large model and not an external benchmark.

- Network: one-hot context, 4 tanh units, sigmoid output; binary cross-entropy; float64.
- Training: plain SGD per sampled block, $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$, $`R\in\lbrace 1,4,40\rbrace`$ updates.
- Tasks: the two source pairs of [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1), moment laws $`N=2,3,4`$, parity laws $`N=2,3`$ (7 tasks).
- Estimation: complete finite-history enumeration where enumerable, otherwise 2,400 independent training runs per source.
- Comparator: the initial-tangent affine-SGD response of the actual policy.
- Agreement margin, fixed in code before execution: $`10^{-4}+0.1\lvert\text{prediction}\rvert+4\,\mathrm{SE}`$.

**Procedure.** For each setting, compute the actual final expected-return gap between the two sources and the tangent-affine prediction, and record agreement under the margin.

**Result.** Source: `theory/independent_checks/results_independent_v014.json → controlled_neural.details.configurations` (SGD rows); cohort totals in `details.observed_affine_comparison_cohorts`.

| $`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$ |
|---|---|---|---|
| 0.01 | 7/7 agree (max dev. $`1.1\times10^{-6}`$) | 7/7 ($`4.1\times10^{-6}`$) | 7/7 ($`2.4\times10^{-5}`$) |
| 0.1 | 7/7 ($`1.1\times10^{-4}`$) | 7/7 ($`2.5\times10^{-4}`$) | 5/7 ($`1.9\times10^{-3}`$) |
| 1 | 3/7 (0.013) | 0/7 (0.053) | 3/7 (0.084) |
| 3 | 0/7 (0.52) | 0/7 (0.53) | 1/7 (0.51) |

- Cohort "SGD, step $`\le0.1`$, at most four updates": 28/28 agreements, largest absolute deviation $`2.52\times10^{-4}`$.
- The additive one-step null (identical examples and weights, regrouped) has parameter residual exactly $`0.0`$ (`details.additive_null_parameter_residual`).
- The Adam rows of the same experiment are analysed in [§5](05_optimizer_state.md).

![Figure 4.5: regime map of the local SGD expansion and exact composition checks](../figures/fig4_5_regime.png)

*Figure 4.5.* The tangent affine-SGD prediction matches the actual controlled network for $`\eta\le0.1`$ and short horizons and fails at $`\eta\ge1`$ (this check). The third-order remainder of the SGD expansion stays below its bound (Check 4.11). Per-step errors accumulate geometrically, and the linear-in-$`R`$ reading fails without a stability condition (Check 4.12). [PDF](../figures/fig4_5_regime.pdf)

**Why it supports Theorem 4.7(d),(e).**

- The affine first-order description of SGD is accurate exactly where the expansion and the stability condition say it should be: small step size and few updates, or a very small step over many updates.
- The departure grows with $`\eta`$ and with the horizon, consistent with the curvature and accumulation terms.
- Limit of the evidence: one small architecture and seven tasks; this locates the regime boundary, it does not certify a real model.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C5,C6,C7,C8,C9,C10,C11,C12,C13,C18,C19,C23 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V7 \
  --output theory/v015_checks/quick/v7.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py \
  --output theory/v014_repro/results_v014_rerun.json
```

`--only` preserves each claim's random stream (`SeedSequence([20261010, claim number])`), so a subset run reproduces the full-run numbers.

## Artifacts

- Results:
  - `theory/independent_checks/results_independent_v014.json` (`v014:C5`–`v014:C13`, `v014:C18`, `v014:C19`; `controlled_neural`)
  - `theory/v015_checks/results_v015_independent.json` (entry `v015:V7`)
  - `theory/v014_repro/results_v014_rerun.json` (`constant_gap_parity`, `grid_response_dimensions`, `scalar_exact_decay`, `scalar_approximation`)
- Code: `theory/independent_checks/check_v014_independent.py` (`check_C5` … `check_C13`, `check_C18`, `check_C19`, `check_C23`), `theory/v015_checks/mathematics.py` (`check_V7`), `docs/direction/v014_handoff/verify_environment_information_v014.py`
- Specification: `theory/independent_checks/SPEC.md`
- Existing plots: `theory/independent_checks/figs/C18_evidence_exposure.png`, `theory/independent_checks/figs/C19_response_complexity.png`, `theory/independent_checks/figs/C23_controlled_neural.png`
- Figure specs: `theoretical_framework/figures/specs/fig4_1.md` … `fig4_5.md`
