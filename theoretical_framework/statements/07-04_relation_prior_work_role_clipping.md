# Remark 7.4 · Relation to prior work; the role of clipping

*Statement*

[← Framework](../README.md) · [Statement](07-04_relation_prior_work_role_clipping.md) · [中文](07-04_relation_prior_work_role_clipping_ZH.md)

**In one paragraph.** This remark places the Adam results of §7 among existing work on how a minibatch perturbation travels through AdamW's state and on data attribution under Adam, and it names one mechanism suggested by the amplitude law: global-norm gradient clipping caps the amplitude of every message, and with it the damage a message can do through the second moment. The mechanism is stated as a testable hypothesis, not as a result.

**Setting used on this page.**

- *Amplitude.* For one coordinate of warm Adam, the amplitude $`\lambda`$ of a message is its gradient in units of the running RMS $`\sqrt{v}`$, in the direction of the mean gradient ([Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)).
- *Second-moment channel.* The part of a message's first-order value that acts through the second moment $`v`$, i.e. through all later step sizes ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)).
- *Parameter-only and cosine scores.* The two classes of influence scores shown to be inconsistent rankers in [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3)(b): a score that pairs the continuation gradient with the message's own parameter step, and a score that depends only on the direction of the message gradient.
- *Global-norm clipping* ([Pascanu et al., 2013](#ref-pascanu2013difficulty)) at threshold $`c`$ rescales a gradient $`g`$ to $`c\,g/\lVert g\rVert`$ whenever $`\lVert g\rVert>c`$.

<a id="s7-4"></a>

**Remark 7.4 (Relation to prior work; the role of clipping).**

- [Liu and Li (2026)](../references.md#liu2026finite) write AdamW as an input–state–output system and analyze how a minibatch perturbation propagates through the momentum and second-moment states over short horizons (8–32 steps). They give no amplitude law, optimal amplitude, threshold, or implication for data selection; [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2) supplies these for the long-horizon stationary case.
- In-Run Data Shapley for Adam ([Ding et al., 2026](../references.md#ding2026inrun)) uses a fixed-state approximation: only the current preconditioned update is scored. It therefore belongs to the parameter-only class of [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes.md#s7-3). So do TracIn-style scores ([Pruthi et al., 2020](../references.md#pruthi2020estimating)) evaluated with Adam steps. LESS ([Xia et al., 2024](../references.md#xia2024less)) ranks by the cosine between Adam-preconditioned training gradients and validation gradients, the cosine class of [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes.md#s7-3).
- Global-norm clipping bounds the amplitude $`\lambda`$ of any single message. It therefore bounds the loss that the second-moment channel can cause ([Corollary 7.2(a),(c)](07-02_stationary_amplitude_law.md#s7-2)). This is a testable mechanism for why clipping helps; it is not claimed as a result here.

**What it says.**

- The amplitude law, the optimal amplitude and the sign-change threshold of Corollary 7.2 are not in the closest prior analyses of AdamW's state dynamics.
- Three common attribution scores fall into the two classes that Proposition 7.3 proves inconsistent.
- Clipping is a natural candidate explanation for why large messages do less harm in practice.

**Role in the framework.**

- It fixes what is and is not claimed as new in §7.
- It motivates recording the clipping activation rate and controlling message magnitude in the method ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), step 4), and the real-model test R3, which compares messages with clipping on and off ([Remark 9.4](09-04_scope_of_evidence.md#s9-4)).

**Uses.**

- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2): the amplitude law and its large-amplitude limit.
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3): the two inconsistent score classes.

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): clipping caps $`\lambda`$ (step 4).
- [Remark 9.4](09-04_scope_of_evidence.md#s9-4): the real-model test R3 of the clipping cap.

## References

- [Liu & Li, 2026](../references.md#liu2026finite): finite-horizon input–output dynamics of minibatch perturbations in AdamW.
- [Ding et al., 2026](../references.md#ding2026inrun): In-Run Data Shapley for Adam.
- [Pruthi et al., 2020](../references.md#pruthi2020estimating): TracIn.
- [Xia et al., 2024](../references.md#xia2024less): LESS.
