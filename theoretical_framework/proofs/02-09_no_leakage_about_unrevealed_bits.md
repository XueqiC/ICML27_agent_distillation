# Lemma 2.9 · Demonstrations leak nothing about unrevealed bits

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-09_no_leakage_about_unrevealed_bits.md) · [Proof](02-09_no_leakage_about_unrevealed_bits.md) · [Experiments](../experiments/02-09_no_leakage_about_unrevealed_bits.md) · [中文](02-09_no_leakage_about_unrevealed_bits_ZH.md)

**What is proved here.** The joint law of the reveal sets, the bits and the rest of the transcript factorizes so that the unrevealed bits enter only through the constant $`2^{-d}`$; their conditional law given the transcript is therefore uniform.

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID, independent of the bits, and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

**Statement ([Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9)).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8), conditional on the full transcript $`T`$, the unseen bits $`(\Theta_j)_{j\in U_K}`$ are IID fair and independent of $`T`$ and of the learner's private randomness.

**Proof.**

The joint law factorizes as

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

where $`X`$ is the additional transcript content. Fix the observed $`(S_{1:K},\Theta_{\cup S_k},X)`$. The only factor that depends on $`\Theta_{U_K}`$ is $`2^{-d}`$, which is constant. Hence $`\Theta_{U_K}`$ is uniform on $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ given $`T`$. The learner's randomness is independent of everything else. $`\square`$
