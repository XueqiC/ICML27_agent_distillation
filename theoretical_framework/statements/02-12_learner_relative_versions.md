# Remark 2.12 · Learner-relative versions

*Statement*

[← Framework](../README.md) · [Statement](02-12_learner_relative_versions.md) · [中文](02-12_learner_relative_versions_ZH.md)

**In one paragraph.** §2 measures the decision regret of the student's policy. Distillation cares about something more specific: the first-order consequence of a feedback law for one declared learner. This remark points to the §6 results that restate the aliasing bound and the demonstration–query budget in that learner-relative form.

**Setting used on this page.**

- $`\Psi`$ is the learner-relative functional of [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): the first-order effect of the oracle feedback on the declared task, as seen by the actual learner.
- $`e_D(h)`$ and $`e_Q(h)`$ are the Fisher information of one demonstration and of one query in direction $`h`$, and $`c`$ is the gradient of $`\Psi`$, which the learner determines.

<a id="s2-12"></a>

**Remark 2.12 (Learner-relative versions).** §6 restates the results of this section for the quantity that matters for distillation: the first-order consequence of a feedback law for one declared learner.

- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7) generalizes [Theorem 2.6(a)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) and [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4). If the demonstrations have the same law in two environments that differ on the learner-relative functional $`\Psi`$, then every distiller, including replay, rewriting and synthesis, has worst-case error at least $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$, for every $`K`$.
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8) generalizes [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11). Asymptotically, $`K\,e_D(h)+n\,e_Q(h)\ge(c^\top h)^2/(16\varepsilon^2)`$ for every direction $`h`$, where $`c`$ is determined by the learner. The rate at which demonstrations substitute for queries therefore depends on the learner, not only on the data.
- The difference in object: this section measures the decision regret of the student's policy; §6 measures the fidelity of a feedback law relative to a learner's update. Both show that missing information can only be supplied by new experiments.

**What it says.**

- The passive impossibility of [Theorem 2.6(a)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) and the data-processing result of [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4) become [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): if demonstrations cannot distinguish two environments that differ on $`\Psi`$, no distiller, including replay, rewriting and synthesis, beats half the difference, for any $`K`$.
- The budget of [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11) becomes the per-direction budget of [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8), where the directions are weighted by the learner.

**Role in the framework.**

- It links the decision-theoretic results of §2 to the learner-relative information bound (Theorem A) of §6.

**Uses.**

- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): post-processing.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): passive impossibility.
- [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11): shared budget.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): learner-relative aliasing.
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8): learner-relative exchange rate.

**Used by.**

- none
