# Lemma 2.9 · Demonstrations leak nothing about unrevealed bits

*Statement*

[← Framework](../README.md) · [Statement](02-09_no_leakage_about_unrevealed_bits.md) · [Proof](../proofs/02-09_no_leakage_about_unrevealed_bits.md) · [Experiments](../experiments/02-09_no_leakage_about_unrevealed_bits.md) · [中文](02-09_no_leakage_about_unrevealed_bits_ZH.md)

**In one paragraph.** In the hidden-bit model, do the demonstrations say anything about the facts they did not reveal? This lemma shows they do not: given everything observed, the unrevealed bits are still independent fair coins. It is the key step behind the exact information floor and the optimality of greedy querying.

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID, independent of the bits, and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

<a id="s2-9"></a>

**Lemma 2.9 (Demonstrations leak nothing about unrevealed bits).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8), conditional on the full transcript $`T`$, the unseen bits $`(\Theta_j)_{j\in U_K}`$ are IID fair and independent of $`T`$ and of the learner's private randomness.

**What it says.**

- Given the complete transcript, the bits no demonstration revealed are independent, fair and independent of the transcript and of the learner's randomness.

**Intuition.**

- Demonstrations that reveal some facts by a mechanism independent of the facts carry no evidence about the rest.

**Role in the framework.**

- It makes every unrevealed bit a coin flip for any learner, which gives the information floor of [Theorem 2.10(a)](../statements/02-10_few_shot_information_floor.md#s2-10), and it makes value-adaptive querying useless in [Theorem 2.10(b)](../statements/02-10_few_shot_information_floor.md#s2-10).

**Derivation outline.**

- The joint law of reveal sets, bits and the rest of the transcript factorizes, and the only factor that depends on the unrevealed bits is the constant $`2^{-d}`$.

**Uses.**

- [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8): the hidden-bit model.

**Used by.**

- [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10): exact floor and greedy optimality.
