# Proposition 8.4 · Measuring the mean increment at the actual states

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-04_measuring_mean_increment_actual_states.md) · [Proof](../proofs/08-04_measuring_mean_increment_actual_states.md) · [Experiments](../experiments/08-04_measuring_mean_increment_actual_states.md) · [中文](08-04_measuring_mean_increment_actual_states_ZH.md)

**What is tested here.**

- **[Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) (measuring the mean increment) holds in all 144 cells, up to $`p=10^5`$.** The realized deviation quantile is at most 0.615 of the bound and does not grow with $`p`$ at a fixed increment bound $`b`$, as [Remark 8.5](../statements/08-05_norms_constants.md#s8-5) states.

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

**Statement under test ([Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4)).**

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

<a id="c8-13"></a>

## Check 8.13 · Concentration of the mean increment (Proposition 8.4(a), Remark 8.5)

**Verdict.** In all 144 cells (20,000 trials each, 2,880,000 instances) the failure rate is far below $`\delta`$, and the realized deviation quantile never exceeds 0.615 of the bound. The ratio does not grow from $`p=1`$ to $`p=100{,}000`$ at a fixed increment bound $`b`$.

**Why.** [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4)(a) states that the empirical mean increment concentrates in the Hilbert norm at rate $`b(1+\sqrt{2\log(1/\delta)})/\sqrt n`$, with no explicit dimension. LoRA adapters have tens of millions of parameters. An implicit dimension dependence would show as a failure rate above $`\delta`$, or as a realized quantile that rises toward the bound as $`p`$ grows.

**Setup.**

- Dimensions $`p\in\lbrace1,10,1000,100000\rbrace`$, sample sizes $`n\in\lbrace8,32,128\rbrace`$, confidence $`\delta\in\lbrace0.05,0.2\rbrace`$, and one- and two-sample statistics.
- Three bounded laws with $`b=1`$ (`data.laws`):
  - *sign:* a biased scalar sign times a fixed unit sign vector, i.e. a normalized sign-type increment with $`\eta=1/\sqrt p`$, with different means under $`P`$ and $`Q`$;
  - *clipped Gaussian:* $`N(0,I/p)`$ clipped at norm $`1`$;
  - *sparse spikes:* a uniformly chosen coordinate with a fair sign, norm $`1`$.
- High-dimensional resultants are sampled exactly, by isotropy and a Beta-angle recursion, without forming $`p`$-vectors (`data.implicit_sampling`).
- 20,000 independent trials per cell; seed `SeedSequence([150150, 9])`.
- In each cell: the fraction of trials whose deviation exceeds the bound (asserted ≤ $`\delta`$ plus four binomial standard errors) and the empirical $`(1-\delta)`$ quantile of deviation / bound.

**Result** (`theory/v015_checks/results_v015_independent.json`, group `v015:V9`, `data.rows`):

| $`p`$ | cells | worst failure rate, $`\delta=0.05`$ | worst failure rate, $`\delta=0.2`$ | lowest coverage | largest quantile/bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.00095 | 0.00765 | 0.99235 | 0.615 |
| 10 | 36 | 0.00030 | 0.00315 | 0.99685 | 0.574 |
| 1,000 | 36 | 0.00050 | 0.00355 | 0.99645 | 0.574 |
| 100,000 | 36 | 0.00040 | 0.00335 | 0.99665 | 0.574 |

By law, the largest quantile/bound is 0.574 (sign), 0.408 (clipped Gaussian) and 0.615 (sparse spikes); the overall maximum is `data.max_quantile_over_bound` = 0.615. For the one-sample clipped Gaussian at $`\delta=0.05`$ the ratio falls with $`p`$, from 0.406 to 0.291, at every $`n`$.

**Why it supports the statement.**

- The bound is valid and conservative by a factor of at least 1.6 in every cell, and nothing in it deteriorates with dimension at fixed $`b`$.
- As [Remark 8.5](../statements/08-05_norms_constants.md#s8-5) states, $`b`$ itself can grow with dimension: for unnormalized sign-type increments $`b=\eta\sqrt p`$. The first-order differences then have the same scale, so the relative accuracy is still set by $`1/\sqrt n`$ (`data.caveat`).
- The check covers the estimation of the mean increment only. Ranking also needs the continuation gradient and the remainder ([Checks 8.15](../experiments/08-09_first_order_version_margin_criterion.md#c8-15)–[8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16)).

The concentration result is panel A of [Figure 8.4](../experiments/08-09_first_order_version_margin_criterion.md#c8-15).

![Figure 8.4: First-order measurement and the first-order margin](../figures/fig8_4_first_order_audit.png)

*Figure 8.4.* Left: the empirical deviation of the mean increment stays below 0.62 of the bound at every dimension from 1 to 100,000 ([Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states.md#c8-13)). Middle: on the frozen held-out panel, the first-order score computed with the **oracle** continuation gradient has no wrong signs once its margin exceeds a quarter of the remainder bound, while static proxies remain wrong 42% of the time at large margins ([Check 8.15](../experiments/08-09_first_order_version_margin_criterion.md#c8-15)). Right: inside the margin first order decides; outside it only measured tails are reliable. Controlled small network. [PDF](../figures/fig8_4_first_order_audit.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V9 --output theory/v015_checks/quick/s8.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: group `v015:V9`; specification `theory/v015_checks/SPEC.md`.
- Figure: `../figures/fig8_4_first_order_audit` (panel A).
