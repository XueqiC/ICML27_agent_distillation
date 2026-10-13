# Theorem 3.2 · Continuation-response characterization

*Statement*

[← Framework](../README.md) · [Statement](03-02_continuation_response_characterization.md) · [Proof](../proofs/03-02_continuation_response_characterization.md) · [Experiments](../experiments/03-02_continuation_response_characterization.md) · [中文](03-02_continuation_response_characterization_ZH.md)

**In one paragraph.** If static quantities do not decide what a student learns, what does? This theorem identifies the invariant: the continuation response, the expected final return after one message, the actual update and the declared rest of training. The gap between two training processes is exactly a sum over rounds of one-round replacement effects, two message laws are interchangeable exactly when they agree on every response, and leaving a certified region is paid once.

**Setting used on this page.**

- The complete learner state $`z\in\mathcal Z`$: parameters, optimizer state, buffers, sampler state, randomness and history ([Definition 1.3](../README.md#s1-3)).
- At round $`t<R`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$ and the actual update gives $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$, with update randomness $`\xi_t`$. Integrating $`Y_t`$ and $`\xi_t`$ gives the learning kernel $`K_t`$, acting on bounded functions by $`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$.
- $`K_t^0`$ is the reference protocol and $`K_t^S`$ an alternative protocol $`S`$; both start from the same initial law $`\nu`$. $`Z_t^S`$ is the chain driven by $`K^S`$, and its law $`d_t^S`$ is the occupancy at round $`t`$.
- $`F_E(z)\in[0,1]`$ is the real-task return of the student deployed from state $`z`$; $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ is the learning value of protocol $`S`$, and $`L_E(0)`$ that of the reference.
- The reference continuation value is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$: the expected final score when training continues from a state under the reference protocol ([Definition 1.4](../README.md#s1-4)).
- The one-boundary response at a snapshot $`(t,z)`$ is $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$: the expected final return after message $`y`$, the actual update and the declared reference tail. The response class $`\mathcal G`$ collects $`g_{t,z}`$ over a declared family $`\mathcal I`$ of snapshots.
- A *one-boundary replacement* $`I(t,z,P)`$ starts at state $`z`$ at round $`t`$, draws one message $`Y\sim P`$, applies the actual update, follows the declared reference tail to round $`R`$ and scores $`F_E`$.

<a id="s3-2"></a>

**Theorem 3.2 (Continuation-response characterization).** In this setting:

**(a) Exact composition.**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(3.2)
```

Consequently, if $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for $`d_t^S`$-almost every $`z`$,

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(3.3)
```

**(b) Operational characterization.** Two message laws $`P,Q`$ yield equal value for every one-boundary replacement in the declared family if and only if

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{for every }g\in\mathcal G. \qquad(3.4)
```

**(c) Stopped composition.** Let $`C_t\subseteq\mathcal Z`$ be certified sets, $`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$ (with $`\tau=R`$ if there is no such $`t`$), and $`\rho=\Pr(\tau<R)`$. If $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for all $`z\in C_t`$, then

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(3.5)
```

**What it says.**

- The gap between the reference and an alternative training process is exactly the sum over rounds of the effect of replacing that round's reference step by the alternative step, measured by the reference continuation value at the states the alternative actually visits.
- Two message laws are interchangeable for every one-boundary replacement if and only if they have the same expectation under every response.
- When the per-round guarantee holds only on certified sets, the probability of leaving them is paid once, not once per round.

**Intuition.**

- The response $`g(y)`$ answers "if this round feeds the student $`y`$ and training then continues as declared, what is the final score"; it builds in the update rule, the rest of training and the deployment task.

**Role in the framework.**

- It fixes the object of sufficiency, the response class $`\mathcal G`$. [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2) measures its size, [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) expands it to first order, and [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) and [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) check it with finite data.

**Derivation outline.**

- (a) Write the gap as $`\sum_t(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S))`$. By the Markov property each term is the reference kernel minus the alternative kernel applied to $`V_{t+1}^0`$, at the states the alternative actually visits.
- (b) A one-boundary replacement draws one message, applies the actual update and then follows the reference tail, so its value is $`\mathbb E_Pg_{t,z}`$; values differ exactly by $`(P-Q)g`$, and the replacement that separates two laws is itself a member of the family.
- (c) Stop the telescoping at the first exit time. Before exit each term is controlled; after exit, the rest of training contributes one bounded difference of returns, at most 1, on an event of probability $`\rho`$.

**Illustration.** Per-round discrepancies can cancel: with two states, the reference kernel the identity, the alternative the swap, two rounds and $`F=(0,1)`$, the final gap is 0 while the total absolute drift is 2. This is why (b) quantifies over all one-boundary replacements: matching one final mean does not require pointwise matching, but interchangeability in every replacement does.

**Comments.**

- **The object.** $`g_{t,z}(y)`$ is the expected *final real-task return* after receiving $`y`$, performing the *actual* update and completing the *declared reference learning tail*. It is not a training loss, a teacher confidence or a frozen-policy action value. §4 studies the size and approximation of $`\mathcal G`$, §5–§7 its dependence on the optimizer, and §8 its measurement.
- **Orientation and signs.** The occupancy in (3.2) is that of the *alternative* process, $`d_t^S`$. Each drift term can have either sign. Any difference in update rules between the two protocols must be included in the compared kernels.
- **Exit penalty.** The uncontrolled tail after the first exit contributes at most one bounded-return discrepancy, so the penalty is $`\rho`$, not $`R\rho`$. If $`Z_0\notin C_0`$, then $`\tau=0`$ and this event is part of $`\rho`$. A finite bank of stored checkpoints does not by itself make $`\rho`$ small for a continuous trajectory; [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) makes coverage quantitative.
- **Responsive tails.** The reference suffix must react to the branch's own current student. Coupling random seeds across branches is legitimate. Replaying identical labels or trajectories where the protocol would query a changed student measures a different continuation.

**Uses.**

- [Definition 1.3](../README.md#s1-3): complete learner state, feedback law, kernel.
- [Definition 1.4](../README.md#s1-4): continuation value and response.
- [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1): static criteria are insufficient.

**Used by.**

- [Remark 3.3](../statements/03-03_response_class_derived_not_chosen.md#s3-3): the response class is derived.
- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2): size of the response class.
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): first-order expansion.
- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2): frozen audit with transport.
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3): adaptive audit at actual states.
- [Proposition 9.1](../statements/09-01_coupling_width.md#s9-1): coupling width.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): select by measured consequence.

## Related work

- The telescoping identity (3.2) has the form of the performance-difference lemma of reinforcement learning ([Kakade & Langford, 2002](#ref-kakade2002approximately)), with the training process in place of the MDP: the occupancy is that of the alternative process, and the role of the advantage is played by the one-round replacement effect under the reference continuation.
- Coupling the random seeds of the branches, which the comments allow, is the method of common random numbers ([Glasserman, 2004](#ref-glasserman2004monte)).
