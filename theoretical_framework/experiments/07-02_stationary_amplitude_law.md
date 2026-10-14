# Corollary 7.2 · Stationary amplitude law (restricted mean-field model)

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-02_stationary_amplitude_law.md) · [Proof](../proofs/07-02_stationary_amplitude_law.md) · [Experiments](07-02_stationary_amplitude_law.md) · [中文](07-02_stationary_amplitude_law_ZH.md)

**What is tested here.** All three parts of the corollary, on exact one-dimensional Adam with stochastic gradients, against the finite response measured with common random numbers; and, on the real model, the prediction obtained from (a).

- **(a), exact mean-field response.** The formula matches the measured finite response within 2.7% of the response scale at all 213 grid points (**PASS**). The predicted zero crossing agrees with the measured one within 3.9% in 25 of the 26 settings where a crossing exists, and neither crosses in the other 7. One setting placed next to the crossing criterion ($`\beta_2=0.99`$, $`\rho=0.245`$, $`T=50`$) misses the 10% tolerance: predicted 68.6, measured $`77.0\pm4.0`$, relative error 10.9% (**FAIL**, reported as is).
- **(b), quadratic law.** Inside its range the response error is at most 1.5% and $`2\lambda^\star`$ is within 0.6% of the measured crossing (**PASS**). Outside its range $`2\lambda^\star`$ underestimates the crossing by up to a factor 5.7, as the corollary warns.
- **(c), large-amplitude criterion.** The sign of the exact limit classifies crossing versus no crossing correctly in 33/33 settings (**PASS**).
- **Real model ([Check 7.6](#c7-6)).** The prediction of (a), computed before the measurement from the saved optimizer state of a warm checkpoint, is **not supported**: the measured responses to one batch at five magnitudes are positive and inside their diagnostic radii, the predictions negative and 28 to 339 times larger in absolute value.

**Setting used on this page.**

- One coordinate of Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) without bias correction, $`\epsilon=0`$, moments $`m,v`$, $`\beta_1,\beta_2`$, learning rate $`\eta`$; terminal score $`F`$ affine in the parameter.
- Tail gradients are IID with mean $`\bar g`$ and mean square $`\bar v`$; the signal-to-noise ratio is $`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$, and $`\alpha=-(\nabla F)\,\mathrm{sign}(\bar g)`$ is positive when moving against $`\bar g`$ improves the score.
- A message $`g=\lambda\sqrt{v}`$ (amplitude $`\lambda`$ in units of the running RMS) is injected at a warm state and followed by $`T`$ tail updates. The **measured finite response** is the change of $`F`$ against a zero-gradient message, averaged over replicates; the **formula** is the mean-field response $`\Delta F(\lambda)`$ of (a).
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$, $`c_j=(1-\beta_2)\beta_2^{\,j}`$, $`\omega_T=1-\beta_2^{\,T+1}`$, $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$, $`\kappa_{\rm iso}(T)=\sum_jb_j/\sqrt{c_j}`$, $`\Lambda(T)=\sum_j(1-b_j)/\sqrt{1-c_j}`$.
- $`\mathrm{Val}`$ is the first-order value of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1), computed with the exact stochastic costates.

**Statement under test ([Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2)).** In the warm, stationary, linear-response regime of one coordinate (tail gradients independent of $`\theta`$, $`\epsilon=0`$, no bias correction, no decay):

(a) the mean-field response is, for every $`\lambda`$,

```math
\Delta F(\lambda)=\alpha\eta\sum_{j=0}^{T}\Big[\frac{\rho+b_j(\lambda-\rho)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho(1-b_j)}{\sqrt{1-c_j}}\Big];
```

(b) for small increments it follows the concave quadratic $`Q(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho\,\omega_T\lambda^2`$ up to a relative error $`11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$, with optimum $`\lambda^\star`$ and a sign change near $`2\lambda^\star`$, the threshold $`2\lambda^\star`$ being usable only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$;

(c) as $`\lambda\to\infty`$, $`\Delta F\to\alpha\eta[\kappa_{\rm iso}(T)-\rho\Lambda(T)]`$, so for $`\alpha>0`$ large same-direction messages are worse than a zero message when $`\rho>\kappa_{\rm iso}(T)/\Lambda(T)`$ (and then $`\Delta F`$ has a positive zero), better when $`\rho<\kappa_{\rm iso}(T)/\Lambda(T)`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 7.2](#c7-2) | (a) and (c) | `v016:C4.amplitude` | FAIL (1 of 26 crossings) |
| [Check 7.3](#c7-3) | (b) | `v016:C4.amplitude` | PASS |
| [Check 7.6](#c7-6) | the real-model prediction of (a) | `v017:R3.r3` | prediction not supported (record check PASS) |

[Check 7.2](#c7-2) and [Check 7.3](#c7-3) share one code ID: its single FAIL assertion is the crossing comparison of [Check 7.2](#c7-2); every assertion of [Check 7.3](#c7-3) passes.

<a id="c7-2"></a>

## Check 7.2 · Exact mean-field response and the large-amplitude criterion (Corollary 7.2(a),(c))

**Verdict.**

- The formula of (a) matches the measured finite response on the whole amplitude grid: worst error 2.74% of the response scale (tolerance 5%).
- The predicted zero crossing matches the measured crossing within 3.9% in 25 of the 26 settings with a crossing. In one setting next to the criterion boundary the error is 10.9%: **FAIL**.
- The sign of the exact large-amplitude limit predicted crossing versus no crossing correctly in 33/33 settings.

**Why.**

- (a) is exact under the mean-field substitution $`m_k/\sqrt{v_k}\to\mathbb Em_k/\sqrt{\mathbb Ev_k}`$. The check measures what that substitution costs on actual stochastic Adam trajectories, at every amplitude, including the large amplitudes where the quadratic law fails.
- (c) predicts from one number, $`\kappa_{\rm iso}(T)-\rho\Lambda(T)`$, whether a very large same-direction message ends up worse than a zero message.

**Setup.**

- Exact one-dimensional Adam without bias correction: $`\beta_1=0.9`$, $`\epsilon=0`$, $`\eta=\alpha=1`$, terminal score $`F=-\theta`$, so the response is exactly linear in $`\theta`$.
- Tail gradients IID $`\mathcal N(\rho,1-\rho^2)`$, so $`\bar g=\rho`$ and $`\bar v=1`$.
- Burn-in of $`\lceil12/(1-\beta_2)\rceil`$ steps from zero. The message $`g=\lambda\sqrt{v}`$ is injected at each replicate's actual warmed state; the noisy state is not replaced by its mean.
- 4,096 replicates. Every message and the zero-message comparator share each replicate's future gradients (common random numbers).
- Settings:
  - **Core**, 24 settings: $`\beta_2\in\lbrace0.99,0.999\rbrace`$, $`\rho\in\lbrace0.3,0.7,0.95\rbrace`$, $`T\in\lbrace50,200,1000,5000\rbrace`$, amplitudes $`\lambda\in\lbrace-2,0.5,1,2,4,8,16\rbrace`$.
  - **Criterion straddle**, 6 settings fixed in advance: $`\beta_2=0.99`$, $`T\in\lbrace20,50,100\rbrace`$, $`\rho=0.8`$ or $`1.2`$ times $`\kappa_{\rm iso}(T)/(T+1)`$. These lie on both sides of the exact critical ratio $`\kappa_{\rm iso}(T)/\Lambda(T)`$.
  - **Boundary**, 3 settings: $`\beta_2=0.99999`$, $`\rho=0.05`$, $`\lambda=20`$, $`T\in\lbrace0,1,5\rbrace`$, 128 replicates (1.2 million burn-in steps).
- Zero crossings are found by doubling from $`2\lambda^\star`$ and Brent's method, separately for the measured rollout and for the formula, on the same search bounds.
  - Core roots use the first 512 replicates; criterion and boundary roots use all paths.
  - Root standard errors come from implicit differentiation.
- Assertions and tolerances were fixed before the run:
  - grid error at most 5% of the setting's largest response;
  - crossing error at most 10% when the measured crossing is bracketed;
  - when it is not bracketed, the formula must not cross up to the same bound;
  - the exact limit's sign must agree with both crossing statuses;
  - the finite formula at $`\lambda=10^8/\sqrt{c_T}`$ must lie within $`10^{-5}`$ of the limit.

**Result.**

| Quantity | Settings | Worst | Tolerance | Status |
|---|---:|---:|---:|---|
| grid error of (a), share of response scale | 33 (213 points) | 2.74% | 5% | PASS |
| crossing error of (a), measured crossing bracketed | 26 | 10.93% | 10% | FAIL (1 setting) |
| no crossing up to the same bound, both formula and measurement | 7 | — | — | PASS |
| finite formula vs limit of (c) | 33 | $`1.0\times10^{-7}`$ | $`10^{-5}`$ | PASS |
| limit sign predicts the measured crossing | 33 | 0 errors | 0 | PASS |
| limit sign predicts the formula crossing | 33 | 0 errors | 0 | PASS |
| limit sign agrees with the stochastic-branch limit | 33 | 0 errors | 0 | PASS |

- **Grid error by group.** Core settings at most 1.69%; criterion settings at most 2.74%; boundary settings at most $`2.5\times10^{-5}`$.
- **Crossings.** In the 24 core settings, the largest crossing error is 3.85% ($`\beta_2=0.999`$, $`\rho=0.7`$, $`T=50`$: formula 312.6, measured $`325.1\pm19.0`$). All other core errors are below 3%, most below 0.5%.
- **The FAIL setting.** It is the criterion-straddle case $`\beta_2=0.99`$, $`T=50`$, $`\rho=0.245`$, which is 1.18 times the exact critical ratio 0.2074. The formula's crossing is 68.62 and the measured crossing is $`77.03\pm3.98`$ (standard error). The relative error of 10.9% is 2.1 standard errors.
  - The other two criterion settings that cross have errors of 3.2% ($`T=20`$) and 3.1% ($`T=100`$).
  - The tolerance was not changed.
  - A likely reason, not tested separately: near the criterion boundary the crossing moves fast with any model error, and the mean-field substitution ignores the fluctuation of $`v`$ around $`\bar v`$. That fluctuation is largest for the short averaging window of $`\beta_2=0.99`$.
- **Exact versus approximate criterion.** At $`\beta_2=0.99`$, $`T=50`$ the exact critical ratio $`\kappa_{\rm iso}/\Lambda`$ is 0.2074, against the approximation $`\kappa_{\rm iso}/(T+1)=0.2042`$. The two differ by 4.0% at $`T=20`$, by 1.6% at $`T=50`$, and by less than 0.1% for $`T\ge1000`$.
  - The 7 settings without a crossing are exactly those with $`\rho`$ below the exact critical ratio: $`\beta_2=0.999`$, $`\rho=0.3`$, $`T=50`$; the three low-$`\rho`$ criterion cases; and the three boundary settings, whose critical ratio exceeds 1.

![Figure 7.1: amplitude law, measured vs formula](../figures/fig7_1_amplitude_law.png)

*Figure 7.1.* Measured finite response $`\Delta F(\lambda)`$ (points with 95% CRN intervals) against the exact mean-field formula of (a) and the quadratic law of (b), for representative $`(\beta_2,\rho,T)`$. The quadratic law tracks the response only at small amplitudes; the exact formula tracks it at all amplitudes. The second panel compares predicted and measured zero crossings for every setting with a crossing, on $`y=x`$ with a ±10% band. The single setting outside the band ($`\beta_2=0.99`$, $`\rho=0.245`$, $`T=50`$) is marked. [PDF](../figures/fig7_1_amplitude_law.pdf)

**Why it supports the statement.**

- The exact formula is the only approximation-free statement about large messages, and it reproduces the measured response at amplitudes where the response has already turned over.
- The FAIL shows the limit of the mean-field substitution: next to the criterion boundary, the crossing location is sensitive to the fluctuations the formula ignores. The crossing *classification* of (c) was still correct there.

**Limits.** One coordinate, exogenous Gaussian gradients, $`\epsilon=0`$, no bias correction. These are exactly the corollary's assumptions, so the check tests the formula, not its transfer to networks; that transfer is examined on the experiments page of [Remark 7.5](07-05_where_first_order_rankers_fail.md#c7-5).

<a id="c7-3"></a>

## Check 7.3 · The quadratic law and its sign-change threshold (Corollary 7.2(b))

**Verdict.** PASS. Inside the range of (b), the first-order value and the finite response follow the quadratic law within 0.31% and 1.53%. Where $`(1-\beta_2)(2\lambda^\star)^2\le0.1`$, $`2\lambda^\star`$ is within 0.6% of both measured crossings. Outside that range, $`2\lambda^\star`$ is wrong by up to a factor 5.7, as (b) warns.

**Why.**

- (b) asserts a quadratic law with an explicit error bound that shrinks with $`(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$. It says the threshold $`2\lambda^\star`$ may be used only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$.
- The check measures both sides of that boundary, for the first-order value $`\mathrm{Val}`$ (exact stochastic costates and exact message increments) and for the finite rollout.

**Setup.** Same runs as [Check 7.2](#c7-2).

- Quadratic-law points are asserted (5% relative tolerance) only where $`(1-\beta_2)\lambda^2\le0.02`$ and $`\lvert\lambda\rvert\le4`$: 96 points.
- Crossings are compared with $`2\lambda^\star`$ (10% tolerance) only where $`(1-\beta_2)(2\lambda^\star)^2\le0.1`$: 10 settings.
- All other points and settings are reported without assertion.

**Result.**

- Eligible points: worst relative error 0.31% for $`\mathrm{Val}`$ and 1.53% for the rollout.
- Eligible settings: the measured $`\mathrm{Val}`$ crossing is within 0.24% of $`2\lambda^\star`$ and the measured rollout crossing within 0.54%.
- Outside the range. On the 9 settings with a crossing and $`(1-\beta_2)(2\lambda^\star)^2`$ between 0.5 and 18, the measured rollout crossing divided by $`2\lambda^\star`$ is:

| $`\beta_2`$ | $`\rho`$ | $`T`$ | $`(1-\beta_2)(2\lambda^\star)^2`$ | $`2\lambda^\star`$ | measured crossing | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.7 | 50 | 0.50 | 7.09 | 7.86 | 1.11 |
| 0.99 | 0.3 | 200 | 0.59 | 7.69 | 7.70 | 1.00 |
| 0.999 | 0.3 | 200 | 1.34 | 36.6 | 49.9 | 1.36 |
| 0.999 | 0.95 | 50 | 1.77 | 42.1 | 76.6 | 1.82 |
| 0.99 | 0.3 | 50 | 2.74 | 16.5 | 29.9 | 1.81 |
| 0.99 | 0.526 | 20 | 3.17 | 17.8 | 75.4 | 4.23 |
| 0.999 | 0.7 | 50 | 3.27 | 57.2 | 325.1 | 5.69 |
| 0.99 | 0.245 | 50 | 4.11 | 20.3 | 77.0 | 3.80 |
| 0.99 | 0.124 | 100 | 6.35 | 25.2 | 78.1 | 3.10 |

- **The first-order value follows the quadratic law past its range.** At $`\beta_2=0.999`$, $`\rho=0.7`$, $`T=50`$, the measured $`\mathrm{Val}`$ crossing is 55.0, close to $`2\lambda^\star=57.2`$, while the finite response crosses only at 325.1. The first-order value turns negative five times too early: the linearization overstates the harm done through the second-moment channel at large amplitudes.
- **Boundary settings** ($`\beta_2=0.99999`$, $`\rho=0.05`$; $`(1-\beta_2)(2\lambda^\star)^2\sim10^6`$). The quadratic crossing errors are 28% ($`T=1`$) and 33% ($`T=5`$), reported only; neither the formula nor the rollout crosses.

**Why it supports the statement.**

- The law and its threshold hold where the corollary says they hold, with errors far below the conservative constant of the proof.
- They break down where the corollary says they may break down. The size of the breakdown, up to a factor 5.7, is the reason the corollary routes large amplitudes to (a) and (c).
- The early sign change of the first-order value is the mechanism behind [Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5), observed directly.

**Limits.** As for [Check 7.2](#c7-2): the corollary's own one-coordinate, exogenous-gradient setting.

<a id="c7-6"></a>

## Check 7.6 · The real-model prediction of (a) at a warm checkpoint (Corollary 7.2(a))

**Verdict.** The prediction is **not supported**. With clipping off, the measured response to one batch is positive at all five magnitudes and lies inside its diagnostic radius. The predicted response is negative at all five magnitudes, lies outside every measured interval, and is 28 to 339 times larger in absolute value. Under the rule written before the measurement, which required all five measured responses to be negative and beyond the diagnostic radius, the outcome is "not supported". The record check `v017:R3.r3`, which recomputes these numbers from the stored run outputs, passes.

**Why.** Corollary 7.2(a) is exact inside its model. The real-model test asks whether that model predicts one real training step. The prediction was computed and written down before any measurement, by inserting the saved optimizer state into (a) coordinate by coordinate and summing over coordinates.

**Setup.**

- *State.* The warm checkpoint $`z_{10}`$ of the K20 Seed-SFT run on ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)), the starting state of the real-model tests; configurations are in the [pre-registration](../../docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md) (in Chinese).
- *Messages.* One fixed reference batch, its loss multiplied by $`c\in\lbrace0.5,1,2,4,8\rbrace`$, applied as one update with global-norm clipping turned off for that update. Only the magnitude changes, so every cosine score is the same across $`c`$. The comparator is a zero-gradient message. Two further arms keep clipping on ($`c=1`$ and $`c=4`$) and consume identical clipped gradients; comparing them is a mechanism check of [Remark 7.4](../statements/07-04_relation_prior_work_role_clipping.md#s7-4) and is not part of this check.
- *Measurement.* After the message, $`T=10`$ reference tail updates with common random numbers across arms, in 16 independent replicas. The score $`\tilde F`$ is the per-token mean teacher-forced log-likelihood on 40 held-out teacher trajectories that never enter training (40 instead of the pre-registered 80, a deviation recorded before any measurement). $`\delta\tilde F(c)`$ is the mean paired difference against the zero message. The radius is the radius of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) with the observed range of the paired scores in place of the a-priori range, which makes it a diagnostic, not a certificate.
- *Prediction.* For every coordinate $`i`$, the bias-corrected $`m_i`$ and $`v_i`$ of the saved state are taken as the stationary tail mean and mean square; $`\rho_i=\lvert m_i\rvert/\sqrt{v_i}`$, $`\lambda_i(c)=c\,g_i\,\mathrm{sign}(m_i)/\sqrt{v_i}`$ and $`\alpha_i=-\partial_i\tilde F\,\mathrm{sign}(m_i)`$ are inserted into (a) with $`T=10`$, and the coordinate responses are summed. This uses every premise of the corollary: no evolving bias correction, $`\epsilon=0`$, the mean-field substitution, tail gradients that do not depend on $`\theta`$, and $`\tilde F`$ linearized at the restored state, with no cross-coordinate curvature.

**Result.** Per-token score, clipping off.

| $`c`$ | measured $`\delta\tilde F`$ | diagnostic radius | predicted $`\delta\tilde F`$ | measured inside radius | $`\lvert\text{predicted}/\text{measured}\rvert`$ |
|---:|---:|---:|---:|---|---:|
| 0.5 | +0.000498 | 0.00637 | −0.0191 | yes | 38 |
| 1 | +0.001231 | 0.00741 | −0.0346 | yes | 28 |
| 2 | +0.001092 | 0.00440 | −0.0544 | yes | 50 |
| 4 | +0.000726 | 0.00541 | −0.0711 | yes | 98 |
| 8 | +0.000236 | 0.00586 | −0.0800 | yes | 339 |

- Every predicted value lies outside its measured interval $`\delta\tilde F\pm\text{radius}`$. The smallest predicted magnitude, $`0.0191`$ at $`c=0.5`$, is more than twice the largest radius, $`0.0074`$.
- The sign of the true response is not determined, since every measured value is inside its radius. What the measurement does determine is that the response is far smaller than predicted and not clearly negative.

**Why it supports the statement.** The check does not contradict the corollary's mathematics, whose premises are not those of the real step. It does show that the corollary does not predict a real training step at this state, which is why the statement page classifies it as a restricted model and keeps it out of Theorem B. The most likely failed premise is linear response: in the real network the tail gradients move with $`\theta`$, which the model holds fixed. This is a hypothesis; it was not tested separately.

**Limits.** One state, one batch, five magnitudes, 16 replicas, and a surrogate score instead of task success. At this state the batch's direct effect on the surrogate is negative, so the prediction has no interior peak; a state with a positive direct effect would test the optimal amplitude, which this check does not.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- The amplitude group (`v016:C4.amplitude`) takes about 29 s on rai.
- Seeds: `numpy.SeedSequence([160160, group, subcase, ...])`, PCG64, float64, single BLAS thread.

The real-model record check of [Check 7.6](#c7-6) recomputes the reported numbers from the stored run outputs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v016_checks/results_v016.json` — claim id `v016:C4.amplitude`: assertions, the single failure record, per-setting maps.
- `theory/v016_checks/results_v016_raw.json.gz` — raw arrays referenced by JSON pointers. Not committed; SHA-256 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`.
- `theory/v016_checks/SPEC.md` (section on `v016:C4`), `check_v016.py` and its tests.
- `figures/fig7_1_amplitude_law.{png,pdf}` and its spec in `figures/specs/`.
- `theory/v017_checks/results_v017.json` — entry `v017:R3.r3` (function `r_r3` in `theory/v017_checks/check_v017.py`), computed from `theory/v017_checks/records/r3/r3_eval_summary.json` and `theory/v017_checks/records/r3/prediction.json`.

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
