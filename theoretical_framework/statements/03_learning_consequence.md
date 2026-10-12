# §3 Learning consequence — Statements

[← Framework](../README.md) · [Proofs](../proofs/03_learning_consequence.md) · [Experiments](../experiments/03_learning_consequence.md) · [中文](03_learning_consequence_ZH.md)

### §3 · Learning consequence — exact statements

[Proofs](../proofs/03_learning_consequence.md) · [Experiments](../experiments/03_learning_consequence.md)

**What this section does.**

- It shows that static measures of feedback (teacher quality, policy value, label marginals, the mean gradient) do not determine what the student learns ([Proposition 3.1](../proofs/03_learning_consequence.md#s3-1)).
- It identifies the right object: the continuation response, the expected final task return after one message, the actual update and the declared rest of training ([Theorem 3.2](../proofs/03_learning_consequence.md#s3-2)).
- It shows that two message laws are interchangeable exactly when they agree on every response, and that coverage failures are paid once ([Theorem 3.2(b),(c)](../proofs/03_learning_consequence.md#s3-2)).
- It fixes the response class $`\mathcal G`$ as the object that §4–§8 measure ([Remark 3.3](../proofs/03_learning_consequence.md#s3-3)).

#### Proposition 3.1 · Static value is not learning value

- **(a)** There are two sources over four states, with an optimal teacher and an aliased student $`\pi_p`$, that give every student the same value $`J(\pi_p)=\tfrac12+p/20`$ and give the teacher the same value $`\tfrac35`$. Frequency distillation from $`m\ge1`$ demonstrations yields expected returns $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$, a gap of $`\tfrac1{40}`$ for every $`m`$.
- **(b)** There are two label packets with the same marginals, the same mean gradient and the same mean update whose learning values under one SGD step differ by $`\eta^2/8`$ when $`J(p)=p^2`$; under $`J'(p)=1-p^2`$ the sign reverses.

*Derivation outline.*

- (a) Both state laws give the action $`A`$ the same average reward $`\tfrac{11}{20}`$, so every aliased student has the same value. The teacher, however, plays $`A`$ with frequency $`\tfrac34`$ under one law and $`\tfrac14`$ under the other, and frequency distillation copies that frequency in expectation.
- (b) The two packets have equal first moments but different variances of the mean label ($`\tfrac18`$ against $`0`$). The deployment value $`p^2`$ reads the variance of the updated student, so $`\mathbb E(p^+)^2`$ differs by $`\eta^2/8`$; for $`1-p^2`$ the sign flips.

*Illustration.* Static evaluation of the data (how good the teacher is, how good each student looks on the data's states, the mean gradient) can be identical while learning reads a different statistic: the action frequency in (a), the second moment in (b). Which statistic matters is decided by the composition of the update with the deployment objective.

*Uses* [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2). *Used by* [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2), [Proposition 7.3(a)](../proofs/07_second_moment_channel.md#s7-3), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/03_learning_consequence.md#s3-1) · [Experiments](../experiments/03_learning_consequence.md#c3-1) · proved

#### Theorem 3.2 · Continuation-response characterization

**Setting ([Definitions 1.3–1.4](../README.md#s1-3)).** The complete learner state $`z`$ evolves by $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$ with messages $`Y_t\sim P_t(\cdot\mid Z_t)`$; integrating gives a learning kernel $`K_t`$, with $`K_t^0`$ the reference and $`K_t^S`$ an alternative protocol from the same initial law. The reference continuation is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$, the one-boundary response is $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$, $`L_E(S)=\mathbb EF_E(Z_R^S)`$, and $`d_t^S`$ is the law of $`Z_t^S`$.

**(a)** The learning gap telescopes exactly:

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big],
```

so $`\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t)`$ whenever the per-round discrepancy is at most $`\varepsilon_t`$.

**(b)** Two message laws give equal value for every one-boundary replacement in the declared family if and only if $`\mathbb E_Pg=\mathbb E_Qg`$ for every $`g\in\mathcal G`$.

**(c)** If the discrepancy bound holds only on certified sets and the alternative process leaves them before round $`R`$ with probability $`\rho`$, then $`\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau_{\rm exit}}\varepsilon_t+\rho`$: the penalty is $`\rho`$, not $`R\rho`$.

*Derivation outline.*

- (a) Write the gap as $`\sum_t(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S))`$. By the Markov property each term is the reference kernel minus the alternative kernel applied to $`V_{t+1}^0`$, at the states the alternative actually visits.
- (b) A one-boundary replacement draws one message, applies the actual update and then follows the reference tail, so its value is $`\mathbb E_Pg_{t,z}`$; values differ exactly by $`(P-Q)g`$, and the replacement that separates two laws is itself a member of the family.
- (c) Stop the telescoping at the first exit time. Before exit each term is controlled; after exit, the rest of training contributes one bounded difference of returns, at most 1, on an event of probability $`\rho`$.

*Illustration.* Per-round discrepancies can cancel: with two states, the reference kernel the identity, the alternative the swap, two rounds and $`F=(0,1)`$, the final gap is 0 while the total absolute drift is 2. This is why (b) quantifies over all one-boundary replacements: matching one final mean does not require pointwise matching, but interchangeability in every replacement does.

*Uses* [Definitions 1.3–1.4](../README.md#s1-3), [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1). *Used by* [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2), [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Theorem 8.2](../proofs/08_verification.md#s8-2), [Theorem 8.3](../proofs/08_verification.md#s8-3), [Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/03_learning_consequence.md#s3-2) · [Experiments](../experiments/03_learning_consequence.md#c3-3) · proved

#### Remark 3.3 · The response class is derived, not chosen

$`\mathcal G`$ is generated from the declared objects: start from $`F_E`$, apply the reference kernels backward, and pull back through the actual update. Enlarging the declared task, learner, suffix or snapshot family can only enlarge $`\mathcal G`$ and therefore only refine the equivalence of Theorem 3.2(b). The response class is fixed by the learner and the task, not chosen for convenience.

[Proof](../proofs/03_learning_consequence.md#s3-3)
