# Theorem 8.2 · Frozen finite-interface selection with transport

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-02_frozen_finite_interface_selection_transport.md) · [Proof](../proofs/08-02_frozen_finite_interface_selection_transport.md) · [Experiments](../experiments/08-02_frozen_finite_interface_selection_transport.md) · [中文](08-02_frozen_finite_interface_selection_transport_ZH.md)

**In one paragraph.** A frozen audit fixes in advance a finite set of audit cells and features, measures how far each candidate feedback law is from the reference on those cells, and selects the closest candidate. This theorem says what such an audit certifies about the learning outcome of training with the selected candidate: the measured discrepancy, plus statistical error, plus an explicit transport term for the states training actually visits, plus the probability of leaving the certified states. It also proves that the transport term cannot be dropped.

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

**Frozen audit.** Using only independent pilot data, freeze before the audit:

- $`M`$ executable candidate feedback laws;
- a finite set $`\mathcal U`$ of audit cells $`u`$, each specifying a complete learner state, a task, a query, the update and the reference tail (the letter $`u`$ denotes a cell only in this subsection);
- $`d`$ features $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$ per cell.

$`P^0_u`$ and $`P^j_u`$ are the message laws of the reference and of candidate $`j`$ in cell $`u`$. The audit draws $`n_0`$ independent reference messages per cell and $`n_S`$ independent messages per candidate and cell. Let $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$, with empirical means $`\widehat\mu_{j,u,\ell}`$, and

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

The selected candidate $`\widehat j`$ minimizes $`\widehat D_j`$ up to a known optimization error $`\eta_{\rm opt}`$.

**Certified states.** For each round $`t`$ there is a certified set $`\mathcal C_t`$ of complete states. Each $`z\in\mathcal C_t`$ carries the following.

- An audited cell $`u=u(t,z)`$ whose features are expressed in the message coordinates at $`z`$.
- A *transport bound*: for every candidate that can be selected on the audit event,

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z).\qquad\text{(T)}
```

- A *response decomposition*

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

Let $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$ ($`\sigma=\infty`$ if no such $`t`$), a stopping time of the training process. **Fresh training:** conditional on the audit result, the selected candidate is trained with randomness independent of the audit samples, and $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$.

<a id="s8-2"></a>

**Theorem 8.2 (Frozen finite-interface selection with transport).** All bounds below may be capped at $`1`$.

**(a) Audit concentration.** With probability at least $`1-\delta`$, simultaneously for all $`j`$,

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) Local transport bound.** For $`z\in\mathcal C_t`$ and $`u=u(t,z)`$,

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) Learning-value bound.** On the event of (a),

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) Specializations.** With exact transport ($`a\equiv0`$), $`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$. Under the state-independent IID-block and convex-affine assumptions of [Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5), with the audited features equal to the moments $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$.
- **(e) Feature drift.** If state-dependent features $`\phi_z`$ satisfy $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$, then $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$, and (b)–(c) hold with $`\phi_z`$ in (D) and $`a`$ replaced by $`a+2\alpha_{zu}`$.
- **(f) Necessity of transport.** For every audit size there is an instance with $`D_j=0`$, an exact response decomposition ($`B=1`$, $`\tau=0`$) and learning gap $`1`$. In it the transport-free expression $`B(\widehat D+e_0+e_S)`$ is strictly below the gap, while the bound of (c) remains valid and is tight.

**What it says.** Fix audit cells and features in advance, measure each candidate's feature discrepancy from the reference, and select the smallest. The selected candidate's learning gap is bounded by the measured discrepancy plus statistical error plus an explicit transport term for the states training actually visits, plus the probability of leaving the certified states. The transport term cannot be dropped.

**Intuition.** An audit sees only the states it covers. If training goes elsewhere, the audit can report zero discrepancy while the learning gap is maximal.

**Role in the framework.** It gives the correct form of any finite, frozen audit and identifies its blind spot, which the actual-state audits of Theorems 8.3 and 8.8 avoid.

**Comments on the statement.**

- **Complete states.** Cells, transport and certification refer to complete learner states. Two states with the same parameters but different optimizer moments produce different next states and can have different responses ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)). Parameter proximity alone does not establish (T).
- **Exact transport is weaker than equality of laws.** $`a=0`$ requires only that the two feature-discrepancy vectors agree, not the four laws.
- **What is certified.** (c) bounds the population expected value of the selected frozen procedure under fresh training, not one training seed's score. Building a new candidate from the audit samples, or training on them, changes the experiment and needs a new independent audit.
- **Further confidence failures.** If transport bounds, coefficients or residuals are themselves estimated, their failure probabilities add to $`\delta`$.

**Derivation outline.**

- Hoeffding plus a union bound over all cells, features and candidates gives simultaneous concentration.
- At a certified state, the one-round effect is the law difference applied to the response; the decomposition splits it into a constant (which cancels), a linear part (bounded by the coefficient norm times the transported discrepancy) and a residual.
- A switched process that falls back to the reference after the first exit differs from the trained process by at most $`\rho`$, and its telescoped gap ([Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)) involves only pre-exit, certified states.

**Illustration.** Audit a state where both laws emit the same message, then train from a state where they emit different messages. The audit reports zero discrepancy and the gap is $`1`$; only a transport term of $`1`$, or counting the uncovered state as an exit, makes the bound correct.

**Uses.**

- [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2) — run-level gap as a sum of one-round effects
- [Theorem 4.5](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5) — moment features for affine learners
- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) — coefficient-weighted measurement error

**Used by.**

- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) — audit at the actual states removes transport
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer.md#s8-11) — transfer of a certificate along training
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2) — coverage step of the method

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
