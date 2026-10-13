# Theorem 8.3 · Actual-state adaptive audit; fidelity versus improvement

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-03_actual_state_adaptive_audit_fidelity.md) · [Proof](../proofs/08-03_actual_state_adaptive_audit_fidelity.md) · [Experiments](../experiments/08-03_actual_state_adaptive_audit_fidelity.md) · [中文](08-03_actual_state_adaptive_audit_fidelity_ZH.md)

**What is tested here.**

- **[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) (actual-state adaptive audit) holds at its stated strength.**
  - In 5 configurations × 100,000 runs the gain stays above both guarantees.
  - The signed identity is exact on 50 rational instances.
  - The lower confidence bound and the unconditional form are both necessary: there are witnesses with exact expected gain $`-78/15625`$ and a conditioning bias of $`1/4\to119/494`$.
  - The sufficient Hoeffding cost is 1,016 paired replicas per candidate.

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

**Adaptive audit.** A *provider policy* $`S`$ chooses at each round a kernel index $`J_t`$ as a function of the history.

- Let $`\mathcal H_t`$ be the history just before the audit of round $`t`$. Conditional on $`\mathcal H_t`$, freeze the actual state $`Z_t`$, a finite set of $`M_t`$ non-reference candidate kernels, a replica count $`n_t`$ and a confidence allocation $`\delta_t`$, with $`\sum_t\delta_t\le\delta`$ almost surely.
- For each candidate, run $`n_t`$ independent paired experiments from $`Z_t`$. Each applies the candidate update or the reference update, then the **full** reference tail and real deployment. Each arm has its correct marginal law, and the paired contrast lies in $`[-1,1]`$.
- Let $`\widehat A_t(j)`$ be the mean contrast and

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- Select $`J_t\in\arg\max_j\ell_t(j)`$ and set $`\Gamma_t=\ell_t(J_t)\in[0,1]`$. Then apply a **fresh** update from $`K^{J_t}_t`$, discarding the audit replicas. Unaudited rounds use the reference and set $`\Gamma_t=0`$.
- Audits leave the actual learner and environment state unchanged, and the reference ignores audit records. $`\mathcal F_t\supseteq\mathcal H_t`$ is the history after the audit and the selection, before the fresh update.

**Statement under test ([Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)).**

- **(a) Fidelity.** If $`L_E(0)-J_E(\pi_0)\ge\gamma`$ and $`L_E(0)-L_E(S)\le\varepsilon`$, then $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$. Here $`J_E(\pi_0)`$ is the value of the initial student; $`\varepsilon`$ can be the certificate of [Theorem 8.2(c)](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2), and $`\gamma`$ the measured gain of reference training from the $`K`$ demonstrations.
- **(b) Signed identity.** For every history-dependent provider policy $`S`$ with the common initial law,

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) Adaptive audit.** (i) With probability at least $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$. (ii) Unconditionally over audits and training,

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) Cost.** Radius $`e_t\le e`$ needs $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ paired replicas per candidate. For $`M=4`$, $`\delta=0.05`$, $`e=0.1`$ this is $`\lceil2\log160/0.01\rceil=1016`$. The count is sufficient, not necessary.

<a id="c8-9"></a>

## Check 8.9 · Adaptive audit on a finite learning process (Theorem 8.3(c))

**Verdict.** In 5 configurations, each with 100,000 independent training runs, the measured gain of the adaptive audit stays above both guarantees, $`-\delta`$ and $`\mathbb E\sum_t\Gamma_t-2\delta`$. The selected drift exceeds $`\sum_t\Gamma_t`$ with frequency 0.990–0.999, above every nominal $`1-\delta`$.

**Why.** [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3)(c) makes a high-probability drift statement and two unconditional expected-value bounds. A violation would appear as a measured gain below $`-\delta`$ or below $`\mathbb E\sum\Gamma-2\delta`$, or as a drift-success frequency below $`1-\delta`$.

**Setup.**

- A small finite learning process with 4 states, $`R\le6`$, $`M\in\lbrace 2,3\rbrace`$ candidate kernels per state, and known true advantages.
- Audits sample bounded paired contrasts from the true model: full reference tail plus Bernoulli deployment.
- Selection by the lower confidence bound with $`e_t=\sqrt{2\log(2M/\delta_t)/n_t}`$, followed by a fresh update.
- Stress settings: small $`n_t`$, large $`\delta_t\in[0.1,0.4]`$, and candidates with slightly negative mean and high variance.
- Seed `SeedSequence([20261010, 16])`.
- For each configuration, 100,000 complete trainings (audits plus updates); the gain is estimated with its standard error and the two bounds are checked with a 4-SE margin.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C16`, `details.configurations`):

| $`R`$ | $`M`$ | $`n`$ | $`\delta`$ | radius | gain (SE) | $`\mathbb E\sum\Gamma`$ | drift frequency | ≥ $`1-\delta`$ |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 2 | 8 | 0.4 | 0.759 | 0.00084 (0.00091) | 0.00090 | 0.99412 | ✓ |
| 2 | 2 | 12 | 0.4 | 0.707 | −0.00067 (0.00102) | 0.00088 | 0.99002 | ✓ |
| 3 | 3 | 20 | 0.2 | 0.671 | 0.00003 (0.00106) | 0.00054 | 0.99407 | ✓ |
| 6 | 3 | 64 | 0.1 | 0.429 | 0.00115 (0.00108) | 0.00026 | 0.99628 | ✓ |
| 2 | 3 | 1000 | 0.2 | 0.090 | **0.13638** (0.00097) | 0.05153 | 0.99897 | ✓ |

Status PASS, with 500,450 instances and 468 assertions. An adversarial search over 327 unique instances found a best margin (bound − gain) of $`-0.199`$, so the bound is always below the gain (`adversarial_best_bound_minus_gain`).

Necessity witnesses (entry `v014:C16`, `details`):

| Witness | Value | Meaning |
|---|---|---|
| `point_estimate_exact_gain` | $`-78/15625`$ | selecting by $`\widehat A`$ without $`-e_t`$ can lose in expectation, so the lower confidence bound is needed |
| `zero_drift_conditioning_witness` | conditional value $`119/494`$ against unconditional $`1/4`$ (audit failure at state 1: $`9/128`$) | conditioning on audit success biases the terminal value, so the guarantee must be unconditional |
| `conditioning_probe_found_45_violation` | false (5 configurations searched) | the unconditional bounds are not violated even though conditional values are biased |
| `replicas_required_per_candidate` | 1016 | Hoeffding allocation for $`M=4,\delta=0.05,e=0.1`$ |

**Why it supports the statement.** The guarantees hold at their stated strength and no stronger:

- with few replicas and large $`\delta`$, the audit almost never selects, and the gain stays within the $`-\delta`$ floor;
- with enough replicas ($`n=1000`$), it selects the beneficial candidate and gains $`0.136`$, well above $`\mathbb E\sum\Gamma-2\delta`$.

<a id="c8-10"></a>

## Check 8.10 · Signed identity and fidelity versus improvement (Theorem 8.3(a), (b))

**Verdict.** The signed telescoping identity is exact on 50 rational processes (residual 0), and the fidelity bound holds with equality in three examples, including one where a faithful copy of an ineffective reference loses $`0.11`$.

**Why.** Theorem 8.3(b) is an exact identity for history-dependent provider choices, and (a) separates fidelity from improvement. Both are checked exactly.

**Setup.**

- 50 rational finite processes with state-dependent kernel selection, plus an adversarial search on $`\mathrm{LHS}-\mathrm{RHS}`$.
- Three fidelity examples: a good reference ($`\gamma>0`$), an ineffective one ($`\gamma<0`$) and a neutral one ($`\gamma=0`$), each paired with a candidate of fidelity $`\varepsilon=1/100`$.

**Result** (entry `v014:C21`): PASS, 834 assertions, worst residual 0, `adversarial_best_lhs_minus_rhs` = 0.

| initial | reference | candidate | $`\gamma`$ | $`\varepsilon`$ | candidate gain | gain ≥ $`\gamma-\varepsilon`$ |
|---:|---:|---:|---:|---:|---:|---|
| 1/2 | 4/5 | 79/100 | 3/10 | 1/100 | 29/100 | ✓ (equality) |
| 1/2 | 2/5 | 39/100 | −1/10 | 1/100 | −11/100 | ✓ (equality) |
| 1/2 | 1/2 | 49/100 | 0 | 1/100 | −1/100 | ✓ (equality) |

**Why it supports the statement.**

- The signed identity is exact.
- A high-fidelity copy of an ineffective reference does not improve the initial student: its gain is $`-0.11`$. Fidelity inherits only the reference's own gain.

<a id="c8-11"></a>

## Check 8.11 · Exact adaptive-gate expectations (Theorem 8.3(c))

**Verdict.** Exact expectations, with no Monte Carlo, satisfy both bounds; a slightly inferior candidate passes an optimistic gate with probability 0.00458, so the expected change can be strictly negative.

**Why.** An independent exact computation of the adaptive gate, and of why the guarantee carries $`-\delta`$ rather than $`0`$.

**Result** (`theory/v014_repro/results_v014_rerun.json`, `adaptive_gate_expectations`; $`n=400`$ per candidate, $`\delta=0.05`$, radius $`0.136`$):

| Quantity | Value |
|---|---:|
| unconditional expected gain | 0.200 |
| $`\mathbb E\sum`$ selected lower confidence bound | 0.132 |
| lower bound $`-\delta`$ | −0.050 |
| lower bound $`\mathbb E\sum\Gamma-2\delta`$ | 0.032 |
| failure probability | $`1.5\times10^{-8}`$ |
| no exact non-degradation: false-selection probability / expected gain ($`n=20`$, $`\delta=0.9`$, candidate 0.49 vs reference 0.50) | 0.00458 / $`-4.6\times10^{-5}`$ |

**Why it supports the statement.** Both bounds hold with the exact expectations, and the counterexample shows that "no worse than the reference" can only hold up to $`\delta`$.

<a id="c8-12"></a>

## Check 8.12 · Audit cost (Theorem 8.3(d))

**Verdict.** The Hoeffding allocation ([Hoeffding, 1963](#ref-hoeffding1963probability)) of 1,016 paired replicas per candidate ($`M=4`$, $`\delta=0.05`$, $`e=0.1`$) gives radius 0.09995, as stated; it is sufficient, not necessary. In the controlled network, deployment noise is 77–89% of the paired variance, and an empirical-Bernstein radius ([Maurer & Pontil, 2009](#ref-maurer2009empirical)) is *forecast* to need 559 replicas.

**Why.** The theorem fixes a valid but expensive allocation, and its cost must be known before any real-model use. The paired-contrast variance is measured in the controlled network to see how much a variance-sensitive radius could save.

**Setup (`v015:E7`).**

- The controlled network, one predeclared audit round, and four candidate providers against a shared reference.
- One reference replica is reused across candidates, with the cross-candidate covariance retained.
- 2,400 independent paired pilot replicas with inverse-CDF coupling.
- The variance is decomposed exactly into block sampling, suffix training and deployment. Seed `SeedSequence([150150, 107])`.

**Result** (`docs/direction/v015_revision/results_v015.json` `checks.T13_*`; `theory/v015_checks/results_v015_independent.json` group `v015:E7`, `data`):

| Quantity | Value |
|---|---|
| `pkg:T13`: sufficient Hoeffding pairs ($`M=4,\delta=.05,e=.1`$) | 1016; radius at 1016 = 0.09995; `is_necessary_lower_bound` = false |
| `v015:E7` Hoeffding allocation | 1016 per candidate; 5,080 training branches with a shared reference (8,128 without reuse) |
| `v015:E7` empirical-Bernstein **forecast** | 559 per candidate; radius 0.0999 at $`n=559`$, from the maximum pilot variance 0.129 |

Variance decomposition of the paired contrast (`v015:E7`):

| candidate | mean | block | suffix | deployment | total | deployment share |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.084 | 0.0133 | 0.0041 | 0.0908 | 0.1082 | 84% |
| 1 | −0.042 | 0.0084 | 0.0020 | 0.0454 | 0.0559 | 81% |
| 2 | −0.143 | 0.0096 | 0.0060 | 0.1202 | 0.1358 | 89% |
| 3 | 0.057 | 0.0096 | 0.0027 | 0.0410 | 0.0533 | 77% |

**Why it supports the statement.**

- The Hoeffding allocation is sufficient, as the theorem states.
- Most of the paired variance comes from deployment noise, not from the training update. A variance-sensitive fixed-$`n`$ radius is therefore forecast to need about 45% fewer replicas.
- The 559 is a forecast. An actual audit must compute its own radius at a fixed $`n`$, without optional stopping. `v015:E7` also records projections for a 12B model from measured cluster timings (`data.forecast`); those are planning inputs, not measurements.

![Figure 8.3: Actual-state adaptive audit](../figures/fig8_3_adaptive_audit.png)

*Figure 8.3.* The measured adaptive-audit gain (±2 SE, 100,000 trainings per configuration, [Check 8.9](../experiments/08-03_actual_state_adaptive_audit_fidelity.md#c8-9)) never falls below $`-\delta`$ or $`\mathbb E\sum\Gamma-2\delta`$, and it is large only when enough replicas allow selection. The second panel shows replicas per candidate needed for radius $`e`$: Hoeffding's 1,016 at $`e=0.1`$ is sufficient, and the empirical-Bernstein curve is a forecast from pilot variance. The third panel decomposes the paired-contrast variance in the controlled network ([Check 8.12](../experiments/08-03_actual_state_adaptive_audit_fidelity.md#c8-12)); deployment noise dominates. [PDF](../figures/fig8_3_adaptive_audit.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C16,C21 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E7 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # adaptive_gate_expectations
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T13
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entries `v014:C16`, `v014:C21`.
- `theory/v015_checks/results_v015_independent.json`: group `v015:E7`.
- `theory/v014_repro/results_v014_rerun.json`: `adaptive_gate_expectations`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T13_*`.
- Figure: `../figures/fig8_3_adaptive_audit`.
