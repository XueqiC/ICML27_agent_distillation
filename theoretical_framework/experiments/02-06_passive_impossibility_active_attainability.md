# Theorem 2.6 · Passive impossibility, active attainability

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-06_passive_impossibility_active_attainability.md) · [Proof](../proofs/02-06_passive_impossibility_active_attainability.md) · [Experiments](02-06_passive_impossibility_active_attainability.md) · [中文](02-06_passive_impossibility_active_attainability_ZH.md)

**What is tested here.** [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability.md#c2-3) computes every quantity of the theorem exactly on 25 values of $`(\Delta,\delta)`$ (540 assertions, worst residual 0). No exactly computed strategy falls below the lower bound (2.6); an optimally tuned SPRT attains it exactly (ratio 1.000) at two configurations, so its constant cannot be improved in general; the sufficient number of queries of (c) always achieves error far below $`\delta`$; and the passive regrets intersect exactly at $`\Delta`$ (PASS).

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).

**Statement under test ([Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5):

- **(a) Passive impossibility.** Under the passive interface every learner satisfies $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$.
- **(b) Active lower bound.** Under the query interface, let $`0<\delta<1/2`$, and suppose that in each world, with probability at least $`1-\delta`$, the learner outputs a student with regret at most $`\Delta/2`$. Then for each $`v`$,

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) Active upper bound.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ independent queries suffice to output the optimal endpoint $`p\in\lbrace 0,1\rbrace`$ with probability at least $`1-\delta`$ in each world.

**Other statements tested on this page.**

- [Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7): after the sign is estimated, $`m\ge\lceil32\log(2/\delta_L)\rceil`$ compiled labels toward $`q\in\lbrace\tfrac18,\tfrac78\rbrace`$ put the frequency learner within $`\tfrac18`$ of $`q`$ with probability at least $`1-\delta_L`$.

<a id="c2-3"></a>

## Check 2.3 · Query complexity and compilation in the two-world family (Theorem 2.6, Proposition 2.7)

**Verdict.** PASS: 25 configurations $`\Delta\in\lbrace \tfrac18,\tfrac1{10},\tfrac1{16},\tfrac1{32},\tfrac1{64}\rbrace \times\delta\in\lbrace 0.01,0.05,0.1,0.25,0.4\rbrace`$, 540 assertions, worst residual 0. No exactly computed strategy falls below (2.6); the SPRT sits exactly on it at $`(\Delta,\delta)=(\tfrac18,\tfrac1{10})`$ and $`(\tfrac18,\tfrac14)`$.

**Why.**

- Theorem 2.6(b) is a lower bound for *every* adaptive strategy. A strategy with error $`\le\delta`$ in both worlds and fewer expected queries would falsify it.
- Theorem 2.6(c) and [Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7) are sufficiency claims. An exact error above $`\delta`$ (or $`\delta_L`$) at the stated sample size would falsify them.
- The constant 64 rests on $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$, which is checked analytically and numerically.

**Setup.**

- Exact rational binomial tails by integer-numerator summation, not a normal approximation.
- The expected sample size and error of the SPRT ([Wald, 1945](#ref-wald1945sequential)) are computed by exact tridiagonal elimination of the absorbing random walk with boundaries $`\pm h`$, $`h=\lceil\log\frac{1-\delta}{\delta}/\log\frac{1/2+2\Delta}{1/2-2\Delta}\rceil`$.
- The payoff $`J_v(p)`$ and the passive minimax are computed in SymPy ([Meurer et al., 2017](#ref-meurer2017sympy)).
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
- *Upper bound (Theorem 2.6(c)).* At $`n_{\rm query}`$ the exact error is far below $`\delta`$ in every configuration: Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)) is conservative.
- *Compilation ([Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7)).* The number of compiled labels is $`O(\log(1/\delta_L))`$, independent of $`\Delta`$, and the exact tail is two to four orders of magnitude below $`\delta_L`$.
- The two costs, $`\Theta(\Delta^{-2}\log(1/\delta))`$ authentic queries and $`O(\log(1/\delta_L))`$ compiled labels, are thus confirmed separately.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)), SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) and SymPy ([Meurer et al., 2017](#ref-meurer2017sympy)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C2 --output theory/independent_checks/subset.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C2`; code `theory/independent_checks/check_v014_independent.py` (`check_C2`, `binomial_tail`, `walk_recursion`).
- `theory/independent_checks/SPEC.md` (section `v014:C2`).
- Figures: `figures/fig2_2_queries.{png,pdf}`; specs `figures/specs/fig2_2.md`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-meurer2017sympy"></a>

**Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, Anthony Scopatz (2017).** SymPy: Symbolic computing in Python. *PeerJ Computer Science* 3, e103. DOI [10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

<a id="ref-wald1945sequential"></a>

**Abraham Wald (1945).** Sequential tests of statistical hypotheses. *The Annals of Mathematical Statistics* 16, no. 2, 117–186. DOI [10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118).
