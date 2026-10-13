# Proposition 9.1 · Coupling width

*Experiments*

[← Framework](../README.md) · [Statement](../statements/09-01_coupling_width.md) · [Proof](../proofs/09-01_coupling_width.md) · [Experiments](09-01_coupling_width.md) · [中文](09-01_coupling_width_ZH.md)

**What is tested here.** All four parts of the proposition, by exact finite computation and in a trained controlled network.

- **Exact checks: PASS.** On 320 random packet-response tables the width is exactly zero for additive tables and positive for non-additive ones, the bound (b) holds with median slack 0.231, and the restricted-pairing and strict-null statements hold exactly (worst residual $`9.9\times10^{-16}`$). An independent implementation agrees ([Check 9.1](#c9-1), [Check 9.2](#c9-2)).
- **Trained network.** One common-state SGD step is exactly regrouping-invariant (parameter difference 0). Sequential updates on the same labels, grouped differently, change the final return by $`9.6\times10^{-6}`$ to $`0.75`$, growing with step size and horizon under SGD and with the opposite sign under Adam ([Check 9.3](#c9-3)).

**Setting used on this page.**

- *Packets and marginals.* Two slots of a training packet are filled from finite unit sets $`\mathcal Y_1,\mathcal Y_2`$ with strictly positive marginals $`\mu,\nu`$; the coupling polytope $`\mathcal C(\mu,\nu)`$ is the set of nonnegative tables with row sums $`\mu`$ and column sums $`\nu`$, i.e. all pairings that keep the per-unit content fixed.
- *Packet response.* $`H(i,j)`$ is the expected final return after training on packet $`(i,j)`$ (here a random table, or the exact return of a small network). A pairing $`\Gamma`$ has expected value $`\langle\Gamma,H\rangle`$.
- *Coupling width.* $`w(H)=\max_\Gamma\langle\Gamma,H\rangle-\min_\Gamma\langle\Gamma,H\rangle`$ over $`\Gamma\in\mathcal C(\mu,\nu)`$; *additive residual* $`\inf_{a,b}\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty`$.
- *Regrouping.* The same examples, with the same weights, split into different packets.

**Statement under test ([Proposition 9.1](../statements/09-01_coupling_width.md#s9-1)).** (a) $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some $`a,b`$. (b) For all couplings $`\Gamma,\Gamma'`$, $`\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty`$, possibly strictly. (c) With forbidden pairings, zero feasible width does not imply additivity. (d) One SGD step at a common parameter on the union of all packets, with identical per-example weights, is invariant to regrouping; sequential steps are not, and for quadratic losses the difference is of order $`\eta^2`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 9.1](#c9-1) | (a)–(d) | `v014:C17` | PASS |
| [Check 9.2](#c9-2) | (a), (b), (d) | `pkg:coupling_and_additive_null` | PASS |
| [Check 9.3](#c9-3) | (d) in a trained network | `v014:C23` (regrouping) | exact null; sequential effects present |

![Figure 9.1: coupling width — exact null and sequential regrouping effects](../figures/fig9_1_coupling.png)

*Figure 9.1.* (A) In a controlled network, grouping the same four labels as $`\lbrace(0,0),(1,1)\rbrace`$ versus $`\lbrace(0,1),(0,1)\rbrace`$ changes the final return. The gap is exactly zero for one common-state step and grows with step size and horizon under SGD; under Adam it has the opposite sign and different dynamics. (B) The range of the expected packet response over the coupling polytope: a single point for an additive table, positive width for a non-additive one. [PDF](../figures/fig9_1_coupling.pdf)

<a id="c9-1"></a>

## Check 9.1 · Coupling polytope, additive characterization and strict null (Proposition 9.1)

**Verdict.** PASS: 320 instances, 2,567 assertions, worst residual $`9.85\times10^{-16}`$ (LP feasibility). Each of the four parts holds on every instance.

**Why.** Proposition 9.1 makes four claims: the "if and only if" characterization of zero width, the bound (b) by the additive residual, the failure of that characterization under forbidden pairings, and the single-step null. Each would fail visibly on some random instance if it were wrong.

**Setup.**

- $`\lvert\mathcal Y_1\rvert,\lvert\mathcal Y_2\rvert\in[2,5]`$, with strictly positive marginals.
- Random additive tables $`H=a+b`$ and random non-additive tables.
- Seed `SeedSequence([20261010, 17])`.
- Procedure:
  1. Compute the width with two linear programs: the maximum and the minimum of $`\langle\Gamma,H\rangle`$ over the coupling polytope, the feasible set of a transportation problem ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)).
  2. Compute $`\inf_{a,b}\lVert H-a-b\rVert_\infty`$ with a third linear program and check the bound (b).
  3. Construct the forbidden-pairing example of part (c).
  4. Check the null exactly in rational arithmetic: one SGD step of a linear model at a common parameter, with the same examples regrouped. Then run two sequential updates with regrouping.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C17`):

| Quantity | Value |
|---|---:|
| status | PASS |
| instances / assertions | 320 / 2,567 |
| worst residual (LP feasibility) | $`9.85\times10^{-16}`$ |
| slack of (b): median / max | 0.231 / 0.712 |
| forbidden-pairing table with zero feasible width | $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$ |
| one-step regrouping difference (linear model, exact) | 0 |
| two-step sequential regrouping difference (linear model, exact) | $`1/32`$ |

**Why it supports the statement.**

- Zero width holds exactly for the additive tables, and positive width for the non-additive ones, as part (a) states.
- The bound (b) is valid but generally strict; the typical slack is about a quarter of the return range, consistent with the strictness example in the proof.
- The single-step null is exact in rational arithmetic, so any measured single-step effect would be an implementation artefact.
- Sequential updates create a genuine regrouping effect. It is the $`\eta^2H_{P_2}g_{P_1}`$ term in the proof of part (d).

**Limits.** Small random tables; the linear-model null is checked in exact arithmetic, not in a nonlinear network ([Check 9.3](#c9-3) does that).

<a id="c9-2"></a>

## Check 9.2 · Independent implementation (Proposition 9.1)

**Verdict.** PASS: the additive table has width 0, a non-additive quadratic pair has width 0.25, and the frozen-gradient null holds to $`2.8\times10^{-17}`$.

**Why.** An implementation written separately from [Check 9.1](#c9-1) guards against a shared coding error.

**Setup.** Finite packet linear programs and frozen-gradient arithmetic; no neural optimizer.

**Result** (`theory/v014_repro/results_v014_rerun.json`, `coupling_and_additive_null`):

| Quantity | Value |
|---|---:|
| width of an additive table | 0.0 |
| quadratic pair: min / max / width | 0.25 / 0.50 / 0.25 |
| strict frozen-gradient additive null, max residual | $`2.8\times10^{-17}`$ |

**Why it supports the statement.** The additive table has zero width and the non-additive pair positive width, as in part (a). The frozen-gradient null of part (d) is exact up to machine precision.

**Limits.** A handful of instances; its value is independence from [Check 9.1](#c9-1), not coverage.

<a id="c9-3"></a>

## Check 9.3 · Regrouping in a controlled network (Proposition 9.1(d))

**Verdict.** The single common-state step is exactly regrouping-invariant (parameter difference 0.0). After sequential updates, regrouping changes the final return by up to 0.749 under SGD and by up to 0.073 in absolute value under Adam.

**Why.** The width is defined through actual learning consequences. A trained network fed the same labels, grouped differently, must show an exactly zero difference at one common-state step; after sequential updates it may show a real difference.

**Setup.**

- Network: one-hot context, 4 tanh units, sigmoid output, float64. This is a controlled small network, not Gemma and not an external benchmark.
- Source A packets are $`\lbrace (0,0),(1,1)\rbrace`$ and source B packets are $`\lbrace (0,1),(0,1)\rbrace`$. Both contain the same four labels, so the marginals are identical.
- Optimizers: SGD with $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$; Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$ and $`\eta\in\lbrace 0.01,0.1\rbrace`$.
- 1, 4 or 40 cycles, giving 2, 8 or 80 updates.
- Exact enumeration of all deterministic histories.
- Procedure: compute the final expected-return gap between the two groupings. Separately, compute the parameter difference after one common-state step with identical weights, and check the manual gradients against central differences.

**Result** (`theory/independent_checks/results_independent_v014.json`, `controlled_neural.details`):

- `additive_null_parameter_residual` = **0.0** for the single common-state step;
- `gradient_max_residual` = $`8.7\times10^{-11}`$ for the manual gradients against central differences.

Return gap A − B (`regrouping`):

| optimizer | $`\eta`$ | 2 updates | 8 updates | 80 updates |
|---|---:|---:|---:|---:|
| SGD | 0.01 | $`9.6\times10^{-6}`$ | $`3.8\times10^{-5}`$ | $`3.2\times10^{-4}`$ |
| SGD | 0.1 | $`9.2\times10^{-4}`$ | $`3.2\times10^{-3}`$ | $`7.9\times10^{-3}`$ |
| SGD | 1.0 | 0.040 | 0.062 | 0.076 |
| SGD | 3.0 | 0.658 | 0.747 | 0.749 |
| Adam | 0.01 | −0.0094 | −0.0136 | −0.0032 |
| Adam | 0.1 | −0.0729 | −0.0503 | −0.0029 |

**Why it supports the statement.**

- The effect is exactly zero where part (d) says it must be: one step at a common parameter.
- In an actual nonlinear learner, the joint organization of identical-marginal feedback changes the learned student once updates are sequential. Under SGD the effect grows with step size and horizon; under Adam it has the opposite sign and other dynamics.
- So the coupling width is a real, optimizer-dependent quantity of the learning consequence. Packet organization belongs to the response that [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) measures (step 3), not to a fixed property of the data.

**Limits.** One small network and one pair of groupings; no real-model or benchmark result.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C17,C23 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # package check
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)); linear programs are solved with SciPy's interface to HiGHS ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)).

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entry `v014:C17`, `controlled_neural.details.regrouping`, `controlled_neural.details.additive_null_parameter_residual`.
- `theory/v014_repro/results_v014_rerun.json`: `coupling_and_additive_null`.
- Figure: `../figures/fig9_1_coupling.{pdf,png}` (spec `../figures/specs/fig9_1.md`).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
