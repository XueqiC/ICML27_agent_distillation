# Proposition 2.7 · Compilation cost

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-07_compilation_cost.md) · [Proof](../proofs/02-07_compilation_cost.md) · [Experiments](02-07_compilation_cost.md) · [中文](02-07_compilation_cost_ZH.md)

**What is tested here.** The compilation part of [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability.md#c2-3) computes the exact failure probability of the frequency learner at $`m=\lceil32\log(2/\delta_L)\rceil`$ for five values of $`\delta_L`$: it is two to four orders of magnitude below $`\delta_L`$, and the number of labels depends only on $`\delta_L`$ (PASS).

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).
- $`\widehat v`$ is the estimated sign; the frequency learner sets $`\widehat p`$ to the empirical frequency of $`m`$ Bernoulli $`(q)`$ labels.

**Statement under test ([Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7)).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5), after the sign is estimated as $`\widehat v`$:

- choose the target $`q=7/8`$ if $`\widehat v=+1`$ and $`q=1/8`$ otherwise;
- generate $`m`$ Bernoulli $`(q)`$ target labels, and let the student be their empirical frequency $`\widehat p`$.

If $`m\ge\lceil32\log(2/\delta_L)\rceil`$, then $`\lvert\widehat p-q\rvert\le1/8`$ with probability at least $`1-\delta_L`$. If $`\widehat v`$ is correct with probability at least $`1-\delta`$ ([Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)), the compiled student has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

### [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability.md#c2-3) · Query complexity and compilation in the two-world family

- *What it tests for this proposition.* That $`m=\lceil32\log(2/\delta_L)\rceil`$ compiled labels give $`\lvert\widehat p-q\rvert\le\tfrac18`$ with probability at least $`1-\delta_L`$ for both targets $`q\in\lbrace\tfrac18,\tfrac78\rbrace`$, and that every outcome within tolerance has regret at most $`\Delta/2`$.
- *Setup.* Exact rational binomial tails (integer-numerator summation, no normal approximation) for $`\delta_L\in\lbrace0.01,0.05,0.1,0.25,0.4\rbrace`$.
- *Result.* $`m=170,119,96,67,52`$ labels; exact failure probabilities $`4.2\times10^{-6}`$, $`1.2\times10^{-4}`$, $`2.4\times10^{-4}`$, $`3.0\times10^{-3}`$, $`3.9\times10^{-3}`$, all far below $`\delta_L`$, the slack of the Hoeffding bound ([Hoeffding, 1963](#ref-hoeffding1963probability)) on which $`m=\lceil32\log(2/\delta_L)\rceil`$ rests. The label count does not involve $`\Delta`$, whereas the identification cost of the same check grows like $`\Delta^{-2}`$.
- *Verdict.* PASS.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)), SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)) and SymPy ([Meurer et al., 2017](#ref-meurer2017sympy)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C2 --output theory/independent_checks/subset.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C2`; code `theory/independent_checks/check_v014_independent.py` (`check_C2`, `binomial_tail`). Keys `details.compilation[*].m`, `details.compilation[*].error`.

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
