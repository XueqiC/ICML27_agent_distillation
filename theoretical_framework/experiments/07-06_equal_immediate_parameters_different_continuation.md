# Proposition 7.6 · Equal immediate parameters do not fix the continuation

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-06_equal_immediate_parameters_different_continuation.md) · [Proof](../proofs/07-06_equal_immediate_parameters_different_continuation.md) · [Experiments](07-06_equal_immediate_parameters_different_continuation.md) · [中文](07-06_equal_immediate_parameters_different_continuation_ZH.md)

**What is tested here.**

- **(c), the exact instance ([Check 7.8](#c7-8)).** PASS. The ratio identity holds exactly in rational arithmetic ($`\tfrac{1960}{1249}`$ for both gradients); in floating point the two second-step parameters agree to below $`10^{-12}`$, the moments differ, and the third-step parameters differ by $`-0.0712027`$.
- **(d), positive $`\epsilon`$ ([Check 7.8](#c7-8)).** PASS. At $`\epsilon=10^{-8}`$ a root $`g_B(\epsilon)\approx2.662025475`$ restores equal second-step parameters (agreement below $`10^{-12}`$), and the third-step difference stays $`-0.0712027`$.
- **(e), warm-state identities ([Check 7.9](#c7-9)).** PASS. The relative change of $`v`$ matches $`(1-\beta_2)(g^2/v-1)`$ to $`2.0\times10^{-16}`$ over 1,000 random states and gradients, with relative changes up to $`0.119`$ although $`1-\beta_2=0.001`$; the momentum shares are $`0.1`$ at once and $`0.651`$ within ten updates.
- **Real-model counterpart.** The state-transplant test is pre-registered; its result is not yet available, and no real-model result is reported on this page.

**Setting used on this page.**

- One coordinate of Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) with $`\beta_1=0.9`$, $`\beta_2=0.999`$, bias correction $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$, $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$, no weight decay, learning rate $`\eta=1`$; $`\epsilon=0`$ unless stated.
- A consumed gradient $`g`$ at step counter $`s`$ gives $`m'=\beta_1m+(1-\beta_1)g`$, $`v'=\beta_2v+(1-\beta_2)g^2`$ and $`\theta'=\theta-\eta q_{1,s}m'/(\sqrt{q_{2,s}v'}+\epsilon)`$.
- The checkpoint $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$ is reached from $`(0,0,0)`$ by one step with gradient $`1`$. The two second-step messages have consumed gradients $`g_A=\tfrac12`$ and $`g_B=\tfrac{2103}{790}`$; the third step uses a common gradient $`1`$.

**Statement under test ([Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation.md#s7-6)).**

- (c) At $`z_1`$, $`g_A`$ and $`g_B`$ satisfy $`(0.9+g)^2/(0.999+g^2)=\tfrac{1960}{1249}`$; they give the same second-step parameter $`\theta_2\approx-1.932179639`$ and different moments, and after a common third step $`\theta_3^A-\theta_3^B\approx-0.0712027`$, with sigmoid-score difference $`\approx-0.00364994`$.
- (d) For small $`\epsilon>0`$ a gradient $`g_B(\epsilon)`$ near $`\tfrac{2103}{790}`$ again equalizes the second-step parameters, and the third-step difference stays nonzero.
- (e) At any state with $`v>0`$,

```math
\frac{v'-v}{v}=(1-\beta_2)\Big(\frac{g^2}{v}-1\Big),
```

and, with $`v`$ fixed and tail gradients independent of $`\theta`$, the first $`K`$ updates carry the fraction $`1-\beta_1^{\,K}`$ of a gradient perturbation's total first-order displacement.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 7.8](#c7-8) | (c) and (d) | `v017:C14.memory` | PASS |
| [Check 7.9](#c7-9) | (e) | `v017:C16.warm` | PASS |

<a id="c7-8"></a>

## Check 7.8 · Equal immediate parameters, different next step (Proposition 7.6(c),(d))

**Verdict.** PASS. The two gradients give the same squared second step $`\tfrac{1960}{1249}`$ exactly, equal second-step parameters in floating point, different moments, and a third-step parameter difference of $`-0.0712027`$. The construction survives $`\epsilon=10^{-8}`$.

**Why.**

- Part (c) claims an exact coincidence, so the check verifies it in exact rational arithmetic, not only in floating point.
- Part (d) claims that the coincidence is not an artefact of $`\epsilon=0`$. The check finds the perturbed gradient by a root search and confirms that the next-step difference is unchanged to the digits shown.

**Setup.**

- Exact rational check: $`f(g)=(\tfrac9{10}+g)^2/(\tfrac{999}{1000}+g^2)`$ evaluated with Python fractions at $`g=\tfrac12`$ and $`g=\tfrac{2103}{790}`$; the assertion is exact equality.
- Floating-point run of the three Adam steps of the instance, in float64.
- Positive $`\epsilon`$: all three steps rerun with $`\epsilon=10^{-8}`$; $`g_B(\epsilon)`$ is the root in $`[2,4]`$ of $`\theta_2(g,\epsilon)-\theta_2(\tfrac12,\epsilon)`$, found by Brent's method with tolerance $`10^{-15}`$.
- Assertions: exact ratio equality; second-step parameter difference below $`10^{-12}`$ at $`\epsilon=0`$ and at $`\epsilon=10^{-8}`$; sigmoid-score difference larger than $`3\times10^{-3}`$ in absolute value; third-step difference larger than $`0.07`$ in absolute value at $`\epsilon=10^{-8}`$.

**Result.**

| Quantity | Value |
|---|---|
| $`(0.9+g)^2/(0.999+g^2)`$ at $`g_A`$ and at $`g_B`$ | $`\tfrac{1960}{1249}`$ and $`\tfrac{1960}{1249}`$ (exact) |
| checkpoint $`z_1=(\theta_1,m_1,v_1)`$ | $`(-1,\;0.1,\;0.001)`$ |
| second-step parameter $`\theta_2`$, both sides | $`-1.932179639`$ |
| moments $`(m_2,v_2)`$ after $`g_A`$ | $`(0.14,\;0.001249)`$ |
| moments $`(m_2,v_2)`$ after $`g_B`$ | $`(0.356203,\;0.00808538)`$ |
| third-step parameter difference $`\theta_3^A-\theta_3^B`$ | $`-0.0712027`$ |
| sigmoid-score difference | $`-0.00364994`$ |
| $`g_B(\epsilon)`$ at $`\epsilon=10^{-8}`$ | $`2.662025475`$ |
| third-step difference at $`\epsilon=10^{-8}`$ | $`-0.0712027`$ |
| bias correction of $`v`$ at step counter 2, $`1/(1-\beta_2^{\,2})`$ | $`500.25`$ |

**Why it supports the statement.**

- The exact equality rules out a rounding coincidence: the two messages leave identical parameters by algebra.
- The moments differ by a factor of about $`2.5`$ in $`m`$ and $`6.5`$ in $`v`$, and the next update, run with one common loss, already moves the two parameters apart by $`0.071`$. Any interface that records only the parameters after the message assigns the two messages the same consequence.
- The last row shows why this is a construction near a cold start: at step counter 2 one gradient dominates $`v`$ because of a bias-correction factor of about $`500`$.

**Limits.** One coordinate and three steps. The check verifies the construction, not the size of the memory effect in a real network; that is the purpose of the pre-registered transplant test below.

<a id="c7-9"></a>

## Check 7.9 · What the memory carries in a warm state (Proposition 7.6(e))

**Verdict.** PASS. The relative change of $`v`$ follows $`(1-\beta_2)(g^2/v-1)`$ to rounding error, and it reaches $`0.119`$ in the sample although the mixing coefficient is $`0.001`$. The momentum delivers $`0.1`$ of a perturbation's first-order displacement at once and $`0.651`$ within ten updates.

**Why.**

- The identity for $`v`$ corrects a common reading of $`1-\beta_2`$ as the size of one update's effect on $`v`$: it is the mixing coefficient, and the effect scales with $`g^2/v`$.
- The momentum shares say how much of a message's first-order effect an immediate-update interface sees.

**Setup.**

- 1,000 random pairs $`(v,g)`$ with $`v`$ uniform on $`[0.1,2]`$ and $`g`$ normal with standard deviation $`2`$; the exact update $`v'=\beta_2v+(1-\beta_2)g^2`$ against the identity.
- Momentum shares: the partial sums of $`(1-\beta_1)\beta_1^{\,k}`$ for $`\beta_1=0.9`$, and the full sum over 20,000 terms against $`1`$.

**Result.**

| Quantity | Value |
|---|---|
| largest error of the relative-change identity | $`2.0\times10^{-16}`$ |
| largest relative change of $`v`$ in the sample | $`0.119`$ |
| share of the first-order displacement in the message's own update | $`0.100`$ |
| share within the first ten updates | $`0.651`$ |

**Why it supports the statement.** The identity is exact, and the sample shows that a single update can move $`v`$ by more than a hundred times the mixing coefficient. The shares show that nine tenths of a gradient's first-order displacement under momentum arrive after the message's own step, so an interface that reads only that step sees a tenth of it; under the premises of (e) the rest is a constant multiple, which is why rankings can change only through curvature or through $`v`$.

**Limits.** The momentum shares assume $`v`$ held fixed, no bias correction and tail gradients that do not depend on $`\theta`$, as part (e) states.

## Real-model counterpart (pre-registered, result pending)

The state-transplant test ([pre-registration](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md), in Chinese) is the real-model test of this proposition's mechanism.

- At a warm checkpoint of the ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)) distillation recipe, a reference branch and an alternative-feedback branch run the same number of updates. Their parameters, momentum and second moment are then recombined in a 2×2 design: the reference parameters continued with the reference or the alternative momentum, and with the reference or the alternative second moment, plus a duplicate reference arm and the natural alternative state.
- All arms continue with the same reference updates; the primary readout is a held-out surrogate score, and the decision threshold was fixed before any training.
- Its result is not yet available. This page will report it, positive or negative, when the run is complete.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entries `v017:C14.memory` and `v017:C16.warm`.
- `theory/v017_checks/check_v017.py`: functions `c_memory` and `c_warm`, with the shared one-coordinate Adam step `adam_step`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
