# Remark 7.5 · Where first-order rankers fail

*Statement*

[← Framework](../README.md) · [Statement](07-05_where_first_order_rankers_fail.md) · [Experiments](../experiments/07-05_where_first_order_rankers_fail.md) · [中文](07-05_where_first_order_rankers_fail_ZH.md)

**In one paragraph.** The state-channel decomposition of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) is exact, but using it to rank candidate feedback laws means keeping only its first-order term. This remark says where that is least reliable: where the second-moment channel carries a large share of the difference between two candidates. There the messages are large relative to $`v`$, which is the saturating regime of the amplitude law, and every first-order score, even one computed with the exact continuation gradient, misjudges the difference. In that regime candidates must be ranked by a finite measurement.

**Setting used on this page.**

- *Candidates.* Two feedback laws $`j,k`$ applied at the same complete Adam state $`z=(\theta,m,v,s)`$ ([Definition 1.3](../README.md#s1-3)), with mean increments $`\mu^j,\mu^k`$ of the complete state; their *increment contrast* is $`\mu^j-\mu^k`$.
- *Channel contractions.* Pairing the contrast with the continuation gradient $`\nabla V=(\chi^\theta,\chi^m,\chi^v)`$ of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) gives three numbers: the $`\theta`$-, $`m`$- and $`v`$-channel contractions $`\langle\chi^\theta,\mu^j_\theta-\mu^k_\theta\rangle`$, $`\langle\chi^m,\mu^j_m-\mu^k_m\rangle`$, $`\langle\chi^v,\mu^j_v-\mu^k_v\rangle`$. Their sum is the first-order difference.
- *First-order score.* Any ranker that orders candidates by such a first-order difference, or by part of it: the parameter-only score, the full-state score with the exact continuation gradient, or a measured directional difference along the mean increment.
- *Finite-replacement measurement.* The CRN paired-replacement score of [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8): replace one round of training by each candidate, run a common reference tail, and compare the outcomes.
- *Saturating regime.* The large-amplitude regime of [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)(a),(c), where a message is large relative to the running RMS $`\sqrt v`$.

<a id="s7-5"></a>

**Remark 7.5 (Where first-order rankers fail).**

- Call the second-moment share of a candidate pair the absolute $`v`$-channel contraction of the pair's increment contrast, divided by the sum of the absolute $`\theta`$-, $`m`$- and $`v`$-channel contractions.
- A high share typically means that the increment is large relative to $`v`$. That is the nonlinear, saturating regime of [Corollary 7.2(a),(c)](07-02_stationary_amplitude_law.md#s7-2), where the first-order value keeps following the quadratic law while the actual response does not. There the linearization overstates the harm done through the $`v`$-channel.
- The decomposition of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) is exact, but its first-order use is least reliable exactly where the $`v`$-channel matters.
- Where the second-moment share is high, a ranking should use a finite-replacement measurement ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)) or a nonlinear formula such as [Corollary 7.2(a)](07-02_stationary_amplitude_law.md#s7-2), not any first-order score. Neither a parameter-only score nor a full-state score with the exact continuation gradient is sufficient.
- [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes.md#s7-3) is a statement inside the small-increment regime of [Corollary 7.2(b)](07-02_stationary_amplitude_law.md#s7-2).

**What it says.**

- The second-moment share of a pair is a computable diagnostic of whether a first-order ranking of that pair can be trusted.
- When the share is high, rank by measuring the finite effect of each candidate, or by the nonlinear amplitude formula.

**Intuition.**

- A large message both takes a large step and inflates $`v`$. The first-order view extrapolates both effects linearly; the real effect on $`v`$ saturates, so the linear view charges the message far more harm than it actually does.
- In the one-coordinate amplitude law the first-order value already turns negative several times earlier than the actual response.

**Role in the framework.**

- It limits the use of first-order scores in the method to pairs where the first-order term dominates and the share is low ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), steps 4–5), and routes the rest to the finite measurement of [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8).
- It explains why first-order data-attribution methods that are accurate under SGD can fail under Adam.

**Uses.**

- [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1): the three channel contractions.
- [Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2): the saturating regime and the early sign change of the first-order value.
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3): the small-increment inconsistency.
- [Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8): the finite paired-replacement measurement.

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2) (steps 4–5) and [Remark 9.3](09-03_derivation_design_rules.md#s9-3) (Rules 3 and 5).
