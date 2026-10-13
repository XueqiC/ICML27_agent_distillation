# Corollary 8.9 · The first-order version, its margin criterion, and its cap

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-09_first_order_version_margin_criterion.md) · [Proof](../proofs/08-09_first_order_version_margin_criterion.md) · [Experiments](../experiments/08-09_first_order_version_margin_criterion.md) · [中文](08-09_first_order_version_margin_criterion_ZH.md)

**What is tested here.**

- **[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) (first-order version and its cap).**
  - With the oracle continuation gradient the first-order score has zero wrong signs on the 56 of 396 rows where its margin exceeds the remainder bound, as (a) states.
  - The measured first-order certificate (b) certifies 0 of 1,584 smooth pairs. The second-order remainder alone exceeds the measured first-order score on 69% of pairs and the true gap on 73%; by the cap (c) those pairs can never be certified at first order.

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

*The paired-replacement score* ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)).

- Fix an audited state $`z`$ at round $`t`$ (a complete learner state, including the optimizer state), conditionally on the history that produced it. The candidates are $`j=0,1,\dots,M`$ with $`M\ge1`$; $`j=0`$ is the reference. Candidate $`j`$ has message law $`P^j=P^j_t(\cdot\mid z)`$, and a message $`Y`$ produces the increment $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$, update randomness included.
- A *surrogate tail* $`T_\omega`$ maps a state to a final state by running a declared (possibly truncated) reference tail with randomness $`\omega`$. A *surrogate objective* $`\tilde F`$ scores the final state (for example, the teacher-forced log-likelihood of held-out successful trajectories, mapped to $`[0,1]`$). Write

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

where $`\omega`$ is independent of $`\Delta^j`$. $`\tilde L(P^j)`$ is the surrogate outcome of one round with candidate $`j`$ followed by the surrogate tail.

- *Replicas* $`\ell=1,\dots,r`$, $`r\ge2`$: draw a fresh $`\omega_\ell`$ and, for every candidate, one message and its increment $`\Delta^j_\ell`$. All candidates in replica $`\ell`$ share $`\omega_\ell`$ (*common random numbers*); their messages may also be coupled through common random numbers, as long as each $`\Delta^j_\ell`$ has its correct marginal law. Replicas are independent of each other.
- If a candidate's message law has finite support, the draw may be replaced by the exact weighted average over the support; the randomness then comes only from $`\omega`$.
- The score of the ordered pair $`(j,k)`$ is

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- On the task side, $`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ is the outcome of one round with candidate $`j`$ followed by the full reference tail and the task objective. In the notation of the setting, $`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$, so $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$.

*Its assumptions* ([Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7)).

- **(I) Independent replicas.** Conditionally on the history that produced $`z`$, the vectors $`(Z^{jk}_\ell)_{j,k}`$, $`\ell=1,\dots,r`$, are i.i.d.; $`r`$ is fixed before the scores are observed.
- **(B) Bounded scores.** For every pair, $`Z^{jk}_\ell`$ lies almost surely in a known interval of length $`B`$. For example $`\tilde F\in[0,1]`$ gives $`B\le2`$.
- **(A) Surrogate alignment.** There are $`\alpha>0`$ and $`\varepsilon_{\rm sur}\ge0`$ such that, for all pairs,

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ absorbs both the difference between the surrogate and the task objective and the truncation of the tail.

**Statement under test ([Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)).** Use the setting of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6) and let $`\mu^j=\mathbb E\Delta^j`$ and $`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$.

- **(a) Margin criterion.** Let $`V`$ be a continuation that is twice differentiable along the reached segments with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$. If $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$, then $`(K^j_t-K^k_t)V(z)`$ has the sign of $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$. At run level, let $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ and $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ be the first-order term and the remainder bound of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) with $`P_j`$ in the role of $`P`$; here $`d^k_t`$ is the state law at round $`t`$ of the process driven by $`P_k`$. If $`\lvert T_{jk}\rvert>B_{jk}`$, then $`L(P_j)-L(P_k)`$ has the sign of $`T_{jk}`$.
- **(b) Measured first-order certificate.** Let $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ be twice differentiable along $`\lbrace su:s\in[0,1]\rbrace`$ for the reached increments $`u`$, with $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$, and let $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$. For a step $`h\in(0,1]`$ define $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$ with mean $`\hat S^{jk}(h)`$, and assume (I), (A), and (B) with range length $`B_h`$. Let $`e^{jk}_{\rm stat}(h)`$ be the radius of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) with $`B_h`$ in place of $`B`$, and

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

Then with probability at least $`1-\delta`$, simultaneously for all ordered pairs,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) Cap.** On the event of (b), a pair is certified by (b) only if $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. A pair whose first-order term does not exceed the second-order remainder bound is never certified at first order, whatever its true gap. In particular this holds for pairs whose mean increments agree along $`\nabla\tilde V(0)`$, which differ only through second and higher moments of the increment; [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) can certify such pairs. In population terms, (b) certifies the lower bound $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$, the quantity that Theorem 8.8 estimates directly.
- **(d) Oracle scores are not certificates.** A score $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ computed with the exact continuation gradient of the task, as in controlled experiments that differentiate through the whole reference tail, satisfies (a) wherever the margin holds. It is not a measurable certificate: $`\nabla V(z)`$ and $`M_2`$ are not available from feedback samples, and checking the margin needs both. A cheap score ranks correctly by (a) only if (i) its difference equals the first-order term with the optimizer-transformed mean increment of the actual learner ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)) at the actual state, and (ii) the margin holds. Scores built from the raw mean gradient, or with coefficients shared across states and tasks, violate (i); any score used where the margin fails violates (ii).

<a id="c8-15"></a>

## Check 8.15 · The first-order margin with the oracle continuation gradient (Corollary 8.9)

**Verdict.** On the 56 of 396 smooth rows where the first-order margin exceeds the remainder bound, the first-order score computed with the **oracle** continuation gradient has zero wrong signs. On the same rows every static proxy is wrong 35.7% of the time. Outside the margin the oracle score is wrong on 34.1% of nonzero-gold rows.

**Why.** [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9)(a) states that a first-order score ranks two feedback laws correctly whenever its margin exceeds the second-order remainder. Corollary 8.9(d) adds that a cheap score can rank correctly only if (i) it pairs the optimizer-transformed mean increment with the continuation gradient at the actual state and (ii) it is used where the margin holds. The check tests both conditions on the exact rows of the frozen held-out panel, without refitting anything. It uses the exact continuation gradient of each row's reference tail, which a real audit does not have; [Checks 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) and [8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16) are the measured counterparts.

**Setup.**

- The held-out panels and frozen proxy models of [§9 Experiments](../statements/09-03_derivation_design_rules.md), read unchanged; their SHA-256 hashes match before and after (`data.frozen_sha256_before`, `_after`).
- Each row is one candidate update at a learner state followed by the declared seven reference updates, with the exact full-tail gold contrast.
- Oracle score $`\langle\nabla V^0(z),\mu^{\rm cand}-\mu^{\rm ref}\rangle`$, with $`\nabla V^0`$ the exact continuation gradient of that row's reference tail.
- The remainder bound uses sampled, unpadded segment curvature, so "inside the margin" is an empirical classification, not a certified one.
- The 36 rows of the threshold task have a discontinuous return; they violate the smoothness assumption and are ineligible.
- Seed `SeedSequence([150150, 112])`.
- Rows are split by the ratio |score| / remainder bound; wrong-sign rates are reported for the oracle, every frozen proxy, measured short tails and random selection. A zero prediction is not counted as a wrong sign, and only rows with nonzero gold enter the rates.

**Result — overall** (`theory/v015_checks/results_v015_independent.json`, group `v015:E12`, `data.summary[split=overall]`):

| Quantity | Value |
|---|---:|
| rows / eligible (smooth) rows | 432 / 396 |
| rows with margin $`>`$ remainder bound | 56 (14.1% of eligible) |
| oracle wrong-sign rate inside the margin | **0** |
| oracle wrong-sign rate outside the margin | 0.341 (101 of 296 nonzero-gold rows) |
| oracle wrong-sign rate, all eligible nonzero-gold rows | 0.287 |
| static proxies inside the margin (input moment, mean gradient, sign, $`\Delta\theta`$, $`\Delta\theta`$ with $`m,v`$) | 0.357 each |
| measured short tail of 1 / 4 updates, inside the margin | 0 / 0 |
| measured short tail of 1 / 4 updates, outside the margin | 0.203 / 0.030 |
| random selection, inside the margin | 0.536 |
| full-tail gold | 0 |

**Result — by ratio |oracle score| / remainder bound** (`data.bins`):

| ratio | rows | oracle | mean gradient | $`\Delta\theta`$ with $`m,v`$ | short tail 1 | short tail 4 | random |
|---|---:|---:|---:|---:|---:|---:|---:|
| $`[0,0.25)`$ | 316 | 0.371 | 0.371 | 0.485 | 0.221 | 0.033 | 0.533 |
| $`[0.25,0.5)`$ | 12 | 0 | 0.250 | 0.083 | 0 | 0 | 0.333 |
| $`[0.5,1)`$ | 12 | 0 | 0 | 0 | 0 | 0 | 0.667 |
| $`[1,2)`$ | 8 | 0 | 0 | 0 | 0 | 0 | 0.625 |
| $`[2,\infty)`$ | 48 | 0 | 0.417 | 0.417 | 0 | 0 | 0.521 |

**Result — by split** (`data.summary`):

| split | eligible rows | margin rows | oracle wrong signs inside / overall |
|---|---:|---:|---|
| held-out states | 72 | 16 | 0 / 0.234 |
| held-out $`\epsilon`$ | 72 | 16 | 0 / 0.281 |
| held-out learning rate | 72 | 0 | — / 0.313 |
| held-out $`\epsilon`$ and learning rate | 72 | 0 | — / 0.313 |
| fresh → warm | 72 | 16 | 0 / 0.234 |
| warm → fresh | 36 | 8 | 0 / 0.406 |
| held-out threshold task | 0 (36 ineligible) | — | — |

**Why it supports the statement.**

- **The margin criterion holds.** Wherever the first-order margin exceeds the remainder bound, the oracle score never ranks wrongly, in each of the four splits that contain margin rows. It is never wrong once the ratio passes 0.25, so the empirical remainder bound is conservative.
- **Inside the margin, static proxies fail for lack of the continuation gradient.** On the 48 rows with ratio ≥ 2, where first order decides, the static proxies are wrong 41.7% of the time and the oracle 0%. They use the same update statistics without the state- and task-specific continuation gradient, so they violate condition (i) of Corollary 8.9(d).
- **Outside the margin the tail decides.** On the 316 rows with ratio below 0.25 even the oracle is wrong 37.1% of the time, while measured four-update tails are wrong 3.3% of the time; condition (ii) fails there.
- **The margin is rare.** Only 14% of eligible rows meet it, and none in the held-out learning-rate splits, whose step sizes are ten times those of the pilot recipes (Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) $`\eta=0.1`$, SGD $`\eta=1`$; `theory/v015_checks/proxy_split_freeze.json`, `recipe_shift_axes`). This is the remainder scaling of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1).
- **Limit of the evidence.** The score uses the oracle gradient, so the check shows what a first-order score can achieve at best; as Corollary 8.9(d) states, it is not a measurable certificate.

![Figure 8.4: First-order measurement and the first-order margin](../figures/fig8_4_first_order_audit.png)

*Figure 8.4.* Left: the empirical deviation of the mean increment stays below 0.62 of the bound at every dimension from 1 to 100,000 ([Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states.md#c8-13)). Middle: on the frozen held-out panel, the first-order score computed with the **oracle** continuation gradient has no wrong signs once its margin exceeds a quarter of the remainder bound, while static proxies remain wrong 42% of the time at large margins ([Check 8.15](../experiments/08-09_first_order_version_margin_criterion.md#c8-15)). Right: inside the margin first order decides; outside it only measured tails are reliable. Controlled small network. [PDF](../figures/fig8_4_first_order_audit.pdf)

<a id="c8-16"></a>

## Check 8.16 · The measured first-order certificate and its cap (Corollary 8.9)

**Verdict.** Measured without any oracle gradient, the first-order certificate of Corollary 8.9(b) certifies 0 of 1,584 smooth pairs, at every step size $`h`$, for both surrogates; it therefore makes no certification error, vacuously. The causes are the two predicted by Corollary 8.9(c) and Proposition 8.10(b):

- the second-order remainder $`e_2`$ alone exceeds the measured first-order score on 69% of pairs (and the true gap on 73%);
- with eight groups the statistical radius is about six times the typical gap.

The measurement also costs 8.3 times the full-tail paired audit, 95% of it in curvature search.

**Why.** Corollary 8.9(b) says that a first-order certificate must add a finite-difference term $`e_{\rm fd}`$ and the second-order remainder $`e_2`$ to the statistical radius. By the cap (c), a pair is certified only if its first-order term exceeds $`e_2+\varepsilon_{\rm sur}`$, whatever its true gap. Measuring every term on the same panel as [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) shows how much of the panel this cap leaves.

**Setup.**

- Same panel, gold and calibration cells as [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14).
- Each provider has one or two support atoms, so the mean increment is integrated exactly; $`G=8`$ groups, each with one fresh seven-update reference tail shared by all candidates (CRN).
- Group score $`Z_g=[\tilde F(T_\omega(z+h\mu_j))-\tilde F(T_\omega(z+h\mu_k))]/h`$ for $`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$, with the $`h`$ of smallest task-unit radius chosen per pair without consulting gold.
- Radius $`e_{\rm stat}+e_{\rm fd}+\varepsilon_{\rm sur}+e_2/\alpha`$:
  - $`e_{\rm stat}`$ by empirical Bernstein ([Maurer & Pontil, 2009](#ref-maurer2009empirical)) with $`\log(4M^2/\delta)=8.5409`$ and the observed range of the eight group scores (this observed range is not a proven support bound);
  - $`e_{\rm fd}`$ from measured ray curvature of the CRN continuation (five-point stencil with Richardson extrapolation);
  - $`e_2=\tfrac12M_2(m_2^j+m_2^k)`$ from measured curvature search along the actual increment rays.
- Surrogates as in [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14). Short-surrogate calibration: $`\alpha=0.902`$ and $`\varepsilon_{\rm sur}=0.0078`$ at every $`h`$.
- The 144 threshold pairs are ineligible (nonsmooth).

**Result — coverage** (`theory/v016_checks/results_v016.json`, check `v016:C6`, `data[1..2]`):

| surrogate | eligible pairs | certified (every $`h`$) | wrong certified | large-gold pairs missed | reference rows certified (of 396) |
|---|---:|---:|---:|---:|---:|
| aligned | 1,584 | 0 | 0 | 281 / 281 | 0 |
| short | 1,584 | 0 | 0 | 281 / 281 | 0 |

The chosen $`h`$ is $`10^{-3}`$ for 1,501 aligned pairs and $`10^{-2}`$ for 83.

**Result — radius components** (aligned surrogate, chosen $`h`$, medians over the 1,584 eligible pairs, from the raw companion):

| Quantity | Median | Remark |
|---|---:|---|
| $`\lvert{\rm gold}\rvert`$ | 0.0035 | true pair gap |
| $`\lvert\text{first-order score}\rvert`$ | 0.0035 | |
| $`e_{\rm stat}`$ | 0.0195 | exceeds the score on 94.8% of pairs |
| of which the range term $`7B\log(4M^2/\delta)/(3(G-1))`$ | 0.0164 | $`=2.85\times`$ the observed range $`B`$ when $`G=8`$ |
| $`e_{\rm fd}`$ | $`1.5\times10^{-5}`$ | but above 1 on 16.7% of pairs (curvature blow-up) |
| $`e_2`$ | 0.0045 | exceeds the measured score on 68.9% of pairs and the true gap on 73.0% |

**Result — cost** (check `v016:C6`, `data[0].costs`):

| Item | Trajectories |
|---|---:|
| held-out score and finite-difference endpoints (per surrogate) | 43,200 |
| remainder curvature search | 2,995,200 |
| calibration (short surrogate) | 18,880 |
| all executed tail trajectories | 3,151,680 |
| curvature search share | 95.0% (96.4% of tail updates) |
| measurement / full-tail paired audit (365,760) | 8.31 (aligned), 8.36 (short) |

**Why it supports the statement.**

- **The cap is binding.** By Corollary 8.9(c) a pair can be certified at first order only if its first-order term exceeds $`e_2`$. The measured score exceeds $`e_2`$ on only 31.1% of pairs, and $`e_2+e_{\rm fd}`$ on 28.0%; the rest are ruled out before any statistics, even with unlimited replicas.
- **The remaining pairs fail on statistics.** With $`G=8`$ the empirical Bernstein range term is 2.85 times the observed range, and $`e_{\rm stat}`$ exceeds the score on 95% of pairs. This is the small-sample regime of Proposition 8.10(b): bringing the range term below $`0.2B`$ at $`M=8`$, $`\delta=0.05`$ needs $`r\ge101`$ replicas, not 8 groups.
- **The curvature terms are expensive and unstable.** Curvature search takes 95% of the tail cost and gives very large $`e_{\rm fd}`$ on a sixth of the pairs.
- **The two failure causes are independent.** Theorem 8.8 removes the second-order cause entirely; [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14) shows that the statistical cause then reduces to the choice of range $`B`$.
- **No certificate is wrong**, but with zero coverage this is vacuous; the PASS status certifies the implementation, not usefulness.

![Figure 8.5: Measurable certificates on the controlled panel](../figures/fig8_5_certificate.png)

*Figure 8.5.* First-order certificate ([Check 8.16](../experiments/08-09_first_order_version_margin_criterion.md#c8-16)): the median radius components $`e_{\rm stat}`$, $`e_{\rm fd}`$ and $`e_2`$ against the median true gap; the remainder alone is comparable to the gap, so nothing is certified. Finite CRN paired replacement ([Check 8.14](../experiments/08-08_crn_paired_replacement_certificate.md#c8-14)): the certified fraction under the a-priori range $`B=2`$ (strict) and under the observed range (diagnostic only), with wrong certificates, for both surrogates and $`r\in\lbrace64,256\rbrace`$. The remaining panels show the distribution of the CRN variance ratio and the trajectory cost relative to the full-tail paired audit. Controlled small network. [PDF](../figures/fig8_5_certificate.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E12 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C6 --output theory/v016_checks/subset.json
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: group `v015:E12`; frozen panel inputs `theory/v015_checks/proxy_split_freeze.json` and `proxy_models_frozen.json`.
- `theory/v016_checks/results_v016.json`: check `v016:C6`; raw companion `results_v016_raw.json.gz`; code `c5c6.py`.
- Figures: `../figures/fig8_4_first_order_audit`, `fig8_5_certificate`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
