# Proposition 8.4 · Measuring the mean increment at the actual states

*Proof*

[← Framework](../README.md) · [Statement](../statements/08-04_measuring_mean_increment_actual_states.md) · [Proof](../proofs/08-04_measuring_mean_increment_actual_states.md) · [Experiments](../experiments/08-04_measuring_mean_increment_actual_states.md) · [中文](08-04_measuring_mean_increment_actual_states_ZH.md)

**What is proved here.** Hilbert-space concentration of the empirical mean increment (Jensen plus McDiarmid), a first-order audit bound at an actual state (Taylor expansion and Cauchy–Schwarz on a joint confidence event), and its run-level version along sampled runs.

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

**Statement ([Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4)).**

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

*Proof.*

**(a)** Let $`f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert`$.

- *Mean.* By Jensen's inequality and independence, $`\mathbb Ef\le(\mathbb Ef^2)^{1/2}=(\mathbb E\lVert X-\mu\rVert^2/n)^{1/2}`$. The cross terms $`\mathbb E\langle X_i-\mu,X_l-\mu\rangle`$, $`i\ne l`$, vanish by independence; this uses the inner product. In a Hilbert space $`\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2`$, so $`\mathbb Ef\le b/\sqrt n`$.
- *Deviation.* Replacing $`X_i`$ by $`X_i'`$ changes $`f`$ by at most $`\lVert X_i-X_i'\rVert/n\le2b/n`$. McDiarmid's inequality ([McDiarmid, 1989](../references.md#mcdiarmid1989method)) with $`c_i=2b/n`$ gives $`\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp(-2s^2/\sum_ic_i^2)=\exp(-ns^2/(2b^2))`$. Setting this to $`\delta`$ gives $`s=b\sqrt{2\log(1/\delta)/n}`$.
- *Two samples.* Apply the one-sample bound to each sample with $`\delta/2`$, use $`\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert`$, and take a union bound.

**(b)** Allocate $`\delta/2`$ to the two-sample bound of (a), which gives $`e_1`$ since $`\log(2/(\delta/2))=\log(4/\delta)`$. Allocate $`\delta/4`$ to each second moment: $`\lVert X\rVert^2\in[0,b^2]`$, so one-sided Hoeffding ([Hoeffding, 1963](#ref-hoeffding1963probability)) gives $`m^P_2\le\hat m^P_2+b^2\sqrt{\log(4/\delta)/(2n_P)}`$ with probability at least $`1-\delta/4`$, and likewise for $`Q`$. On the intersection of the three events, $`\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1`$ and $`m^P_2+m^Q_2\le\hat m^P_2+\hat m^Q_2+e_2`$.

- By (S) and Taylor's theorem with integral remainder, $`V(z+\Delta)=V(z)+\langle\nabla V(z),\Delta\rangle+\rho(\Delta)`$ with $`\lvert\rho(\Delta)\rvert\le\tfrac{M_2}2\lVert\Delta\rVert^2`$. Since $`(K^P_tV)(z)=\mathbb E_PV(z+\Delta)`$, this is the one-boundary case of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1):

```math
\big\lvert(K^P_t-K^Q_t)V(z)\big\rvert\le\big\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\big\rvert+\tfrac{M_2}{2}\big(m^P_2+m^Q_2\big).
```

- Cauchy–Schwarz gives $`\lvert\langle\nabla V(z),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert`$; substitute the two inequalities. The samples are drawn after the state is fixed, so the argument holds conditionally on the history.

**(c)** Let $`\varepsilon_t(z)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\rvert\rbrace`$.

- *Audit event.* Each audit $`(j,t)`$ succeeds with conditional probability at least $`1-\delta_{\rm a}/(JR)`$; the conditional failure probabilities sum to at most $`\delta_{\rm a}`$. On the event that all succeed, $`\varepsilon_t(Z_{j,t})\le\hat\varepsilon_{j,t}`$ for all $`j,t`$, using $`\varepsilon_t\le1`$.
- *Telescoping.* [Theorem 3.2(a)](../statements/03-02_continuation_response_characterization.md#s3-2) with $`P`$ as the reference and the triangle inequality inside the expectation give $`\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d^Q_t}\varepsilon_t(Z_t)`$. The cap at $`1`$ is valid because each term is a difference of two $`[0,1]`$-valued expectations.
- *Monte Carlo.* The run totals $`S_j=\sum_t\varepsilon_t(Z_{j,t})`$ are i.i.d. in $`[0,R]`$ with mean $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t`$. One-sided Hoeffding gives $`\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\tfrac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}`$ with probability at least $`1-\delta_{\rm mc}`$. On both events $`\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}`$; a union bound completes the proof. $`\square`$

## References

- [McDiarmid, 1989](../references.md#mcdiarmid1989method). McDiarmid, C. (1989). On the method of bounded differences. Surveys in Combinatorics, LMS Lecture Note Series 141, 148–188.
