# §8 Verification and Theorem C — Statements

[← Framework](../README.md) · [Proofs](../proofs/08_verification.md) · [Experiments](../experiments/08_verification.md) · [中文](08_verification_ZH.md)

### §8 · Verification and Theorem C — exact statements

[Proofs](../proofs/08_verification.md) · [Experiments](../experiments/08_verification.md)

**What this section does.**

- It asks how finitely many measurements can certify that a feedback law is good enough, or rank candidate laws, and it compares four kinds of measurement: feature means on frozen cells, full-tail paired contrasts at the actual states, mean optimizer increments, and CRN paired-replacement scores on a surrogate tail.
- It shows that a frozen audit needs an explicit transport term, and that auditing at the actual states with full tails is reliable but expensive.
- Its main result, **Theorem C** ([Theorem 8.8](../proofs/08_certificate.md#s8-8)), measures the finite effect of a one-round replacement directly, so the certificate needs no continuation gradient and no second-order remainder.
- It states exactly where first-order scores can be trusted ([Corollary 8.9](../proofs/08_certificate.md#s8-9)) and what remains open ([Remark 8.12](../proofs/08_certificate.md#s8-12)).

**Setting.** Rounds $`t<R`$, complete learner states $`z`$, candidate kernels $`K^j_t`$ with reference $`K^0_t`$, reference continuation $`V^0_{t+1}`$ and response $`g_{t,z}`$ as in [Definitions 1.3–1.4](../README.md#s1-3). The *local advantage* of candidate $`j`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1],
```

the effect on the final score of replacing one reference round by one round of candidate $`j`$. By [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), whole-run differences are sums of such terms over reached states.

#### [Theorem 8.1](../proofs/08_verification.md#s8-1) · Stability–approximation tradeoff

For a response $`g`$ on a finite message alphabet, features $`\phi_1,\dots,\phi_d`$ (matrix $`\Phi`$) and tolerances $`\epsilon_j\ge0`$, let $`W_{\mathbf{\epsilon}}(g)`$ be the largest $`\lvert(P-Q)g\rvert`$ over pairs of laws with $`\lvert(P-Q)\phi_j\rvert\le\epsilon_j`$ for all $`j`$. Then

```math
W_{\mathbf{\epsilon}}(g)=\min_{b,c}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert\Big]=\min_b\Big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].
```

- With $`\mathbf{\epsilon}=0`$ this is twice the best uniform approximation error by $`\mathrm{span}\lbrace\mathbf 1,\phi\rbrace`$.
- If $`\phi_1=\alpha g`$ for a non-constant binary $`g`$, then $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$.
- On a simultaneous confidence event for measured feature means, the theorem applies with $`\epsilon_j`$ equal to the measured difference plus both confidence radii.

*Derivation outline.*

- Differences of probability laws are exactly the vectors with zero sum and $`\ell_1`$ norm at most $`2`$, and the support function of that set is the oscillation.
- The feature constraints make $`W_{\mathbf{\epsilon}}`$ a linear program; weak duality gives the upper bound for every coefficient vector $`b`$, and Sion's minimax theorem applied to the Lagrangian gives equality.
- The minimum is attained because the dual objective is convex polyhedral and bounded below.

*Illustration.* A feature that equals the response scaled by $`\alpha=0.01`$ represents it exactly, yet with tolerance $`\epsilon=0.005`$ it certifies only $`W=0.5`$: the coefficient $`1/\alpha`$ multiplies the measurement error.

*Uses* [Theorem 4.2(b)](../proofs/04_complexity_of_sufficiency.md#s4-2), [Proposition 4.6](../proofs/04_rates_and_composition.md#s4-6). *Used by* [Theorem 8.2](../proofs/08_verification.md#s8-2), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/08_verification.md#s8-1) · [Experiments](../experiments/08_verification.md) · proved

#### [Theorem 8.2](../proofs/08_verification.md#s8-2) · Frozen finite-interface selection with transport

Freeze $`M`$ candidates, a finite set of audit cells and $`d`$ features per cell in $`[0,1]`$. Let $`\widehat D_j`$ be candidate $`j`$'s largest measured feature-mean discrepancy from the reference, with radii $`e_0,e_S`$ from Hoeffding and a union bound, and select $`\widehat j`$ minimizing $`\widehat D_j`$. Suppose each certified state $`z\in\mathcal C_t`$ carries a cell, a transport bound $`a_{t,j}(z)`$ on how far the feature discrepancy at $`z`$ differs from that in its cell, and a response decomposition with coefficient bound $`B_t`$ and residual $`\tau_t`$. Let $`\sigma`$ be the first exit from the certified states and $`\rho`$ a bound on its probability. Then, with probability at least $`1-\delta`$,

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- The measured discrepancies concentrate uniformly, so selecting on them loses at most $`2(e_0+e_S)`$.
- With exact transport, or for IID blocks under convex-affine updates with moment features, the bound simplifies; in the latter case no transport is needed.
- The transport term is necessary: there are instances with zero audited discrepancy, an exact response decomposition and learning gap $`1`$.

*Derivation outline.*

- Hoeffding plus a union bound over all cells, features and candidates gives simultaneous concentration.
- At a certified state, the one-round effect is the law difference applied to the response; the decomposition splits it into a constant (which cancels), a linear part (bounded by the coefficient norm times the transported discrepancy) and a residual.
- A switched process that falls back to the reference after the first exit differs from the trained process by at most $`\rho`$, and its telescoped gap ([Theorem 3.2](../proofs/03_learning_consequence.md#s3-2)) involves only pre-exit, certified states.

*Illustration.* Audit a state where both laws emit the same message, then train from a state where they emit different messages. The audit reports zero discrepancy and the gap is $`1`$; only a transport term of $`1`$, or counting the uncovered state as an exit, makes the bound correct.

*Uses* [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Theorem 4.5](../proofs/04_rates_and_composition.md#s4-5), [Theorem 8.1](../proofs/08_verification.md#s8-1). *Used by* [Theorem 8.3](../proofs/08_verification.md#s8-3), [Remark 8.11](../proofs/08_certificate.md#s8-11), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/08_verification.md#s8-2) · [Experiments](../experiments/08_verification.md) · proved

#### [Theorem 8.3](../proofs/08_verification.md#s8-3) · Actual-state adaptive audit; fidelity versus improvement

At each audited round, conditional on the history, freeze the actual state $`Z_t`$, $`M_t`$ candidates, $`n_t`$ paired full-tail replicas and a confidence share $`\delta_t`$ with $`\sum_t\delta_t\le\delta`$. Select by the lower confidence bound $`\ell_t(j)=\widehat A_t(j)-e_t`$, $`e_t=\sqrt{2\log(2M_t/\delta_t)/n_t}`$, with $`\ell_t(0)=0`$, and let $`\Gamma_t`$ be the selected bound. Then:

- **Fidelity.** If reference training gains $`\gamma`$ over the initial student and the candidate protocol loses at most $`\varepsilon`$ against the reference, the candidate gains at least $`\gamma-\varepsilon`$.
- **Signed identity.** $`L_E(S)-L_E(0)=\sum_t\mathbb E_{d^S_t}A_t(Z_t,J_t)`$ for every history-dependent provider policy.
- **No degradation.** With probability $`1-\delta`$, $`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$, and unconditionally

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **Cost.** Radius $`e`$ needs $`\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ paired replicas per candidate; for $`M=4`$, $`\delta=0.05`$, $`e=0.1`$ this is $`1016`$.

*Derivation outline.*

- Conditional on the pre-audit history, the paired contrasts are i.i.d. in $`[-1,1]`$ with mean $`A_t(Z_t,j)`$; Hoeffding and a union bound over candidates give a per-round event, and the failure probabilities add up to at most $`\delta`$ along the run.
- On each round's event the selected advantage is at least $`\Gamma_t\ge0`$.
- The fresh update makes $`A_t`$ the conditional drift of $`V^0`$, so the terminal score telescopes into drifts plus martingale differences; taking expectations gives the unconditional bounds.

*Illustration.* Selecting by the point estimate instead of the lower bound can lose in expectation even when every audit uses the exact full tail: a slightly worse candidate wins the noisy comparison often enough.

*Uses* [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Theorem 8.2](../proofs/08_verification.md#s8-2). *Used by* [Proposition 8.4](../proofs/08_certificate.md#s8-4), [Proposition 8.10](../proofs/08_certificate.md#s8-10), [Remark 8.11](../proofs/08_certificate.md#s8-11), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/08_verification.md#s8-3) · [Experiments](../experiments/08_verification.md) · proved

#### [Proposition 8.4](../proofs/08_certificate.md#s8-4) · Measuring the mean increment at the actual states

Let increments be bounded, $`\lVert\Delta\rVert\le b`$, in the Hilbert norm of the state space.

- For $`n`$ i.i.d. increments, with probability $`1-\delta`$,

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big),
```

and the analogous bound holds for the difference of two independent samples.

- If the continuation $`V`$ has gradient norm $`\Lambda`$ at $`z`$ and curvature at most $`M_2`$ along the increments, then with probability $`1-\delta`$ the one-round effect $`\lvert(K^P_t-K^Q_t)V(z)\rvert`$ is at most $`\Lambda(\lVert\bar X_P-\bar X_Q\rVert+e_1)+\tfrac{M_2}2(\hat m^P_2+\hat m^Q_2+e_2)`$, with explicit radii $`e_1,e_2`$.
- Applying this at every visited state of $`J`$ sampled runs bounds the whole-run gap, up to a Monte Carlo term $`R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$.
- The samples measure only the increments; $`\Lambda`$ and $`M_2`$ belong to the continuation and must be bounded or calibrated separately.

*Derivation outline.*

- In a Hilbert space the cross terms of $`\lVert\bar X-\mu\rVert^2`$ vanish, so the mean deviation is at most $`b/\sqrt n`$; McDiarmid's inequality adds the deviation term.
- A second-order Taylor bound turns the one-round effect into the first-order term plus a remainder, and Cauchy–Schwarz bounds the first-order term by $`\Lambda`$ times the mean-increment difference.
- The run-level version combines the telescoping identity, a union bound over audits and Hoeffding over runs.

*Uses* [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10), [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Theorem 8.3](../proofs/08_verification.md#s8-3). *Used by* [Corollary 8.9](../proofs/08_certificate.md#s8-9).

[Proof](../proofs/08_certificate.md#s8-4) · [Experiments](../experiments/08_verification.md) · proved

#### [Remark 8.5](../proofs/08_certificate.md#s8-5) · Norms and constants

- The concentration bound holds in the Hilbert norm; it does not carry over to arbitrary norms with the same constant (the sup norm on $`\mathbb R^p`$ adds a $`\sqrt{\log p}`$ factor).
- No dimension appears explicitly, but $`b`$ can grow with dimension: a cold-start Adam step moves every coordinate by about $`\eta`$, so $`\lVert\Delta\theta\rVert`$ can reach $`\eta\sqrt p`$.
- A ranking needs only the scalar projection $`\langle\nabla V,\Delta\rangle`$, and Theorem 8.8 needs no bound on $`\lVert\Delta\rVert`$ at all.

[Proof](../proofs/08_certificate.md#s8-5) · remark

#### [Theorem 8.8](../proofs/08_certificate.md#s8-8) · CRN paired-replacement certificate; Theorem C

*Setting* ([Definition 8.6](../proofs/08_certificate.md#s8-6), [Assumption 8.7](../proofs/08_certificate.md#s8-7)). At an audited state $`z`$, candidates $`j=0,\dots,M`$ ($`0`$ is the reference) emit messages with increments $`\Delta^j`$. A surrogate tail $`T_\omega`$ and objective $`\tilde F`$ give $`\varphi_\omega(u)=\tilde F(T_\omega(z+u))`$ and $`\tilde L(P^j)=\mathbb E\varphi_\omega(\Delta^j)`$. In each of $`r`$ independent replicas all candidates share one $`\omega_\ell`$ (common random numbers), and

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_\ell Z^{jk}_\ell .
```

The scores lie in a known interval of length $`B`$, and the surrogate is aligned with the task up to $`(\alpha,\varepsilon_{\rm sur})`$: task differences $`L_F(P^j)-L_F(P^k)`$ equal $`\alpha`$ times surrogate differences up to $`\alpha\varepsilon_{\rm sur}`$. Here $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$.

*Statement.* $`\mathbb EZ^{jk}_\ell=\tilde L(P^j)-\tilde L(P^k)`$ exactly. With $`\hat V^{jk}`$ the sample variance and

```math
e^{jk}_{\rm stat}=\sqrt{\frac{2\hat V^{jk}\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)},
```

with probability at least $`1-\delta`$, simultaneously for all ordered pairs,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}-e^{jk}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

- Declaring $`j`$ better than $`k`$ whenever $`\hat S^{jk}>e^{jk}_{\rm stat}+\varepsilon_{\rm sur}`$ makes no wrong declaration with probability $`1-\delta`$; with $`k=0`$ it certifies $`A_t(z,j)>0`$.
- No continuation gradient, no curvature constant and no second-order remainder enter; only $`(\alpha,\varepsilon_{\rm sur})`$ must be calibrated.

*Derivation outline.*

- Exactness is linearity of expectation: each term depends only on the marginal law of one candidate's increment and the independent tail randomness, so coupling the candidates changes nothing in the mean.
- Maurer and Pontil's empirical Bernstein bound for $`[0,1]`$ variables, rescaled to an interval of length $`B`$, gives each one-sided bound; there are at most $`2M^2`$ ordered pairs, which yields $`\log(4M^2/\delta)`$ after a union bound.
- The alignment assumption transfers the surrogate bound to the task.

*Illustration.* Two candidates whose mean increments coincide but whose second moments differ have zero first-order difference; Theorem 8.8 still sees their finite difference, because nothing is linearized.

*Uses* [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Theorem 8.3](../proofs/08_verification.md#s8-3) (calibration). *Used by* [Corollary 8.9](../proofs/08_certificate.md#s8-9), [Proposition 8.10](../proofs/08_certificate.md#s8-10), [Remark 8.11](../proofs/08_certificate.md#s8-11), [Remark 7.5](../proofs/07_second_moment_channel.md#s7-5), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/08_certificate.md#s8-8) · [Experiments](../experiments/08_verification.md) · proved (the rigor of a concrete use rests on a known range $`B`$ and calibrated $`(\alpha,\varepsilon_{\rm sur})`$; see Remark 8.12)

#### [Corollary 8.9](../proofs/08_certificate.md#s8-9) · The first-order version, its margin criterion, and its cap

Let $`\mu^j=\mathbb E\Delta^j`$ and $`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$.

- **Margin criterion.** If a continuation $`V`$ with curvature at most $`M_2`$ satisfies $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}2(m^j_2+m^k_2)`$, the one-round effect has the sign of the first-order term; the analogous statement holds for whole runs.
- **Measured first-order certificate.** Scaling the increments by a step $`h`$ and dividing the paired scores by $`h`$ estimates the first-order surrogate difference $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$. With surrogate curvature $`\tilde M`$, the certified bound is

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\Big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-\tfrac12h\tilde M(m^j_2+m^k_2)-\tfrac12\tilde M(m^j_2+m^k_2)-\varepsilon_{\rm sur}\Big).
```

- **Cap.** A pair can be certified this way only if $`\tilde s^{jk}>\tfrac12\tilde M(m^j_2+m^k_2)+\varepsilon_{\rm sur}`$. A pair whose first-order term is below the remainder bound, for example two candidates whose mean increments agree along $`\nabla\tilde V(0)`$, is never certified at first order, whatever its true gap.
- **Oracle scores.** A first-order score computed with the exact task continuation gradient obeys the margin criterion but is not a measurable certificate, because neither $`\nabla V`$ nor $`M_2`$ is available from feedback samples.

*Derivation outline.*

- The margin criterion is the second-order Taylor bound: the remainder is smaller than the first-order term, so it cannot flip the sign.
- For the measured version, Taylor's theorem along $`s\mapsto\tilde V(shu)`$ gives a finite-difference bias of order $`h`$ and, at $`h=1`$, the remainder; empirical Bernstein bounds the sampling error as in Theorem 8.8.
- The cap follows by chaining these inequalities on the confidence event.

*Illustration.* Under Adam the moment increments are of order one while the continuation depends on them only through steps of size $`\eta`$; the first-order term and the remainder are then of the same order, so a small learning rate does not make the margin hold.

*Uses* [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10), [Proposition 8.4](../proofs/08_certificate.md#s8-4), [Theorem 8.8](../proofs/08_certificate.md#s8-8). *Used by* [Remark 7.5](../proofs/07_second_moment_channel.md#s7-5), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/08_certificate.md#s8-9) · [Experiments](../experiments/08_verification.md) · proved

#### [Proposition 8.10](../proofs/08_certificate.md#s8-10) · Cost, and what common random numbers buy

- One audited state costs $`(M+1)r`$ surrogate tails plus the calibration tails.
- The range term of $`e_{\rm stat}`$ is at most $`\zeta B`$ if and only if $`r\ge1+7\log(4M^2/\delta)/(3\zeta)`$; for $`M=8`$, $`\delta=0.05`$, $`\zeta=0.2`$ this means $`r\ge101`$.
- If $`\varphi_\omega`$ is pathwise Lipschitz with constant $`\lambda_\omega`$, then $`\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\lVert\Delta^j_\ell-\Delta^k_\ell\rVert`$ and $`\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E[\lambda_\omega^2\lVert\Delta^j-\Delta^k\rVert^2]`$, whereas independent tails keep the tail noise in the variance. Uniform bounds $`\lambda_\omega\le\bar\lambda`$ and $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$ give the valid range $`B=2\bar\lambda b_{jk}`$.

*Derivation outline.* Counting; rearranging the range term; the Lipschitz condition at the paired increments and the law of total variance for independent arms.

*Uses* [Theorem 8.3](../proofs/08_verification.md#s8-3), [Theorem 8.8](../proofs/08_certificate.md#s8-8). *Used by* [Remark 8.12](../proofs/08_certificate.md#s8-12), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/08_certificate.md#s8-10) · [Experiments](../experiments/08_verification.md) · proved

#### [Remark 8.11](../proofs/08_certificate.md#s8-11) · Calibrating the surrogate; transfer along training

- $`(\alpha,\varepsilon_{\rm sur})`$ are fitted by regressing paired full-tail task contrasts on the surrogate scores of a set of calibration pairs. Using them on new pairs assumes the pairs come from the same family; this is an assumption, not a theorem.
- Inside the adaptive audit of Theorem 8.3, the certified lower bound of Theorem 8.8 can replace the full-tail lower bound, and Theorem 8.3(c) then holds unchanged.
- The certificate holds at the audited state; using the selection later needs the transport term of Theorem 8.2 or a re-audit at the next checkpoint.

[Proof](../proofs/08_certificate.md#s8-11) · remark (calibration transfer is an assumption)

#### [Remark 8.12](../proofs/08_certificate.md#s8-12) · Open problems

- **A provable range.** With the a-priori range ($`B\le2`$), the range term of order $`B\log(M^2/\delta)/r`$ can exceed the gaps between nearby candidates for any affordable $`r`$, and the observed range of the scores is not a valid substitute. A provable, much smaller range, for instance from a certified pathwise Lipschitz constant, is open.
- **Calibration transfer.** A testable condition under which calibrated $`(\alpha,\varepsilon_{\rm sur})`$ remain valid on new pairs and later states is open.
- Until both are settled, Theorem 8.8 is in practice an unbiased, low-variance ranking score, and a rigorous certificate only where $`B`$ and $`(\alpha,\varepsilon_{\rm sur})`$ are justified independently.

[Proof](../proofs/08_certificate.md#s8-12) · open
