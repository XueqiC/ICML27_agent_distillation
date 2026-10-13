# Proposition 8.4 · Measuring the mean increment at the actual states

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-04_measuring_mean_increment_actual_states.md) · [Proof](../proofs/08-04_measuring_mean_increment_actual_states.md) · [Experiments](../experiments/08-04_measuring_mean_increment_actual_states.md) · [中文](08-04_measuring_mean_increment_actual_states_ZH.md)

**In one paragraph.** A first-order audit compares two feedback laws through their mean optimizer increments. This proposition shows that the mean increment can be estimated at an actual state from gradient evaluations alone, with an error of order one over the square root of the sample size in the Hilbert norm, and turns the estimate into a bound on the one-round effect and on the whole-run gap. It also makes explicit what the samples do not measure: the gradient norm and the curvature of the continuation value.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

Use the objects of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): consumed gradient $`G_t(z,Y,\xi)`$, increment $`\Delta_t(z,g)`$, mean increment $`\mu^P_t(z)`$, second increment moment $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$ and continuation $`V^P_{t+1}`$. The norm is the Hilbert norm of the state space (Euclidean in finite dimension).

- **Bounded increments (B).** There is $`b<\infty`$ with $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$ almost surely, for the compared laws at the audited states. Examples: clipped SGD on the parameter block, $`b=\eta c`$; sign-type updates on $`p`$ parameters, $`b=\eta\sqrt p`$; SGD with gradients bounded by $`\bar G`$, $`b=\eta\bar G`$.
- **Curvature (S).** At the audited state, $`V:=V^P_{t+1}`$ is twice differentiable along the segments $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ reached by the increments, with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$; write $`\Lambda=\lVert\nabla V(z)\rVert`$.
- **Audit samples.** At an audited state $`z`$, draw $`n_P`$ messages $`Y\sim P_t(\cdot\mid z)`$ and $`n_Q`$ messages $`Y\sim Q_t(\cdot\mid z)`$, independently and with independent update randomness, and compute the increments $`X=\Delta_t(z,G_t(z,Y,\xi))`$. Their empirical means are $`\bar X_P,\bar X_Q`$ and their empirical second moments $`\hat m^P_2,\hat m^Q_2`$. This takes $`n_P+n_Q`$ consumed-gradient evaluations at $`z`$ and no training tail.

<a id="s8-4"></a>

**Proposition 8.4 (Measuring the mean increment at the actual states).**

**(a) Hilbert-space concentration.** Let $`X_1,\dots,X_n`$ be i.i.d. in a real Hilbert space with $`\lVert X\rVert\le b`$ almost surely and mean $`\mu`$. With probability at least $`1-\delta`$,

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

For two independent samples of sizes $`n_P,n_Q`$, with probability at least $`1-\delta`$,

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) First-order audit at an actual state.** Fix an audited state $`z`$, conditionally on the history that produced it, and assume (B) and (S) there. With probability at least $`1-\delta`$ over the audit samples,

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

where

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) Run-level audit along actual runs.** Draw $`J`$ independent runs of protocol $`Q`$. At every round $`t`$ of every run, apply (b) at the visited state with confidence $`\delta_{\rm a}/(JR)`$, and let $`\hat\varepsilon_{j,t}`$ be the right side of (b), capped at $`1`$. Suppose (B) and (S) hold at every visited state with the stated constants, and audit samples are drawn without changing the run. Then with probability at least $`1-\delta_{\rm a}-\delta_{\rm mc}`$,

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**What is measured and what is not.**

- The data-dependent parts are $`\bar X_P-\bar X_Q`$ and $`\hat m_2`$. Each needs only consumed-gradient evaluations at the visited state.
- $`\Lambda`$ and $`M_2`$ are properties of the continuation value, not of the messages, and the samples do not estimate them. They must be bounded analytically, or calibrated against full-tail audits ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)) with the calibration error added to the bound.

**What it says.** The mean optimizer increment of a feedback law can be estimated from gradient evaluations alone, with an error that falls like one over the square root of the sample size in the Hilbert norm. Combined with the continuation's gradient norm and curvature, this bounds the one-round effect and, along sampled runs, the whole-run gap.

**Intuition.** The data side of a first-order certificate is cheap; the continuation side (its gradient and curvature) is not measured by these samples at all.

**Role in the framework.** It is the cheap half of first-order auditing and explains why first-order audits still need calibration constants.

**Derivation outline.**

- In a Hilbert space the cross terms of $`\lVert\bar X-\mu\rVert^2`$ vanish, so the mean deviation is at most $`b/\sqrt n`$; McDiarmid's inequality adds the deviation term.
- A second-order Taylor bound turns the one-round effect into the first-order term plus a remainder, and Cauchy–Schwarz bounds the first-order term by $`\Lambda`$ times the mean-increment difference.
- The run-level version combines the telescoping identity, a union bound over audits and Hoeffding over runs.

**Uses.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) — first-order expansion of the one-round effect
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — which increment each learner produces
- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — telescoping identity for whole runs
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) — full-tail audits to calibrate the constants

**Used by.**

- [Remark 8.5](../statements/08-05_norms_constants.md#s8-5) — norms and constants of the bound
- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) — first-order certificate and its cap

## Related work

- [Theorems 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)–[8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) and [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) use Hoeffding's inequality ([Hoeffding, 1963](#ref-hoeffding1963probability)), union bounds and McDiarmid's inequality ([McDiarmid, 1989](#ref-mcdiarmid1989method)). Selecting by a lower confidence bound to guarantee no degradation is the logic of high-confidence policy improvement ([Thomas et al., 2015](#ref-thomas2015high)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-mcdiarmid1989method"></a>

**Colin McDiarmid (1989).** On the method of bounded differences. In *Surveys in Combinatorics, 1989*, London Mathematical Society Lecture Note Series 141, 148–188, Cambridge University Press. DOI [10.1017/CBO9781107359949.008](https://doi.org/10.1017/CBO9781107359949.008).

<a id="ref-thomas2015high"></a>

**Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh (2015).** High confidence policy improvement. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2380–2388. [proceedings.mlr.press/v37/thomas15](https://proceedings.mlr.press/v37/thomas15.html).
