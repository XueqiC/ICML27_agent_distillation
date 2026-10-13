# Definition 2.8 · Hidden-bit model

*Statement*

[← Framework](../README.md) · [Statement](02-08_hidden_bit_model.md) · [中文](02-08_hidden_bit_model_ZH.md)

**In one paragraph.** This definition makes the number $`K`$ of demonstrations explicit. Each demonstration reveals some of the facts the student needs, by a mechanism that does not depend on the facts; a hard budget of queries can reveal more. The model is simple enough that the information left after $`K`$ demonstrations, and the best use of the queries, can be computed exactly.

**Setting used on this page.**

- [Definition 1.2](../README.md#s1-2): the $`K`$ trusted demonstrations and further experiments (queries).
- The utility of a policy is the deployment-weighted probability of choosing the correct action in each context.
- The information term $`R_{\rm info}`$ is the minimum expected regret any learner can reach from the data ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).

<a id="s2-8"></a>

**Definition 2.8 (Hidden-bit model).** Every probability is conditional on a fixed, declared pretrained information state $`I_{\rm pre}`$. Extra teacher feedback is never counted as part of the $`K`$ demonstrations.

- **Bits and deployment.** The unknown bits $`\Theta=(\Theta_1,\dots,\Theta_d)`$ are independent and fair. Context $`j`$ occurs at deployment with weight $`w_j\ge0`$, $`\sum_jw_j=1`$, and the correct decision at context $`j`$ is $`\Theta_j`$. The student class contains every context-to-action table, so $`\mathcal R_{\rm rep}=0`$. The utility is $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$, so the informed optimum is 1.
- **Non-leaking demonstrations.** Trajectory $`k`$ reveals a random subset $`S_k\subseteq[d]`$ together with the labels $`\Theta_{S_k}`$. The subsets $`S_1,\dots,S_K`$ are IID with law $`L`$ and independent of $`\Theta`$; dependence among the coordinates revealed within one trajectory is allowed. Any further transcript content has a conditional law that depends only on the revealed labels and independent randomness. Write $`q_j=\Pr(j\in S)`$.
- **Hard query budget.** After the demonstrations, the learner may make at most $`n`$ noiseless unit-cost queries, each revealing one chosen bit. The choices may depend on everything observed so far. $`n`$ is a per-transcript hard budget, not an expected budget.
- $`U_K`$ is the set of bits revealed by none of the $`K`$ trajectories, and $`M_K=\lvert U_K\rvert`$.

**What it says.**

- Facts are independent fair bits, one per context; demonstrations reveal random subsets of them; queries reveal chosen bits one at a time, up to a hard budget $`n`$.

**Intuition.**

- A demonstration is a random sample of facts; whatever it misses stays completely unknown.

**Role in the framework.**

- It is the setting of [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) and [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10), and the model case for the learner-relative coverage model of [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10).

**Uses.**

- none

**Used by.**

- [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9): unrevealed bits stay uniform.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10): information floor and optimal queries.

## Related work

- The reveal sets are independent of the bits, so the unrevealed labels are *missing completely at random* in the sense of [Rubin, 1976](#ref-rubin1976inference). [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) turns this assumption into the absence of leakage; reveal sets chosen on the basis of the bits would make the missingness informative.
