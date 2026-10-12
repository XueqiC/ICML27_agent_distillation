# Theorem 2.10 · Few-shot information floor and optimal hard-budget querying

*Statement*

[← Framework](../README.md) · [Statement](02-10_few_shot_information_floor.md) · [Proof](../proofs/02-10_few_shot_information_floor.md) · [Experiments](../experiments/02-10_few_shot_information_floor.md) · [中文](02-10_few_shot_information_floor_ZH.md)

**In one paragraph.** How much is still unknown after $`K`$ demonstrations, and how should a hard query budget be spent? In the hidden-bit model this theorem gives the exact information floor after $`K`$ demonstrations and shows that the best queries take the heaviest unseen facts first, with no benefit from adapting to the values found.

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID, independent of the bits, and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

<a id="s2-10"></a>

**Theorem 2.10 (Few-shot information floor and optimal hard-budget querying).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8):

**(a) Information floor.** Without queries, the minimum achievable expected regret after $`K`$ trajectories is

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

with the convention $`(1-q)^0=1`$. The marginal value of one more trajectory is

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) Optimal queries.** With weights of the unseen bits sorted as $`w_{(1)}\ge\dots\ge w_{(M_K)}`$,

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

and querying the $`n`$ heaviest unseen bits is optimal; adapting the queries to the revealed values cannot help. In the uniform case ($`w_j=1/d`$, each trajectory reveals one uniformly drawn context),

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

In general $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$, and the inequality can be strict.

**What it says.**

- After $`K`$ demonstrations the information floor is $`\tfrac12\sum_jw_j(1-q_j)^K`$: half the expected weight of the facts no demonstration revealed.
- With a hard budget of $`n`$ queries, querying the heaviest unseen facts is optimal, and adapting the queries to the revealed values does not help. The expectation must stay outside the positive part.

**Intuition.**

- Each demonstration reveals a fact with some probability; what it leaves unrevealed must be queried, and the most valuable gaps first.

**Role in the framework.**

- It puts $`K`$ into the theory and is the model case for the query allocation of [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10) and the acquisition step of [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2).

**Derivation outline.**

- *(a)* Any action guesses an unrevealed bit correctly with probability one half ([Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9)), and revealed bits can be copied, so the conditional regret is half the weight of the unseen bits. Bit $`j`$ stays unrevealed in all $`K`$ independent demonstrations with probability $`(1-q_j)^K`$.
- *(b)* Induct on the remaining budget. After a query, Lemma 2.9 makes the remaining problem independent of the value found, so branching on values gains nothing, and the greedy choice of the heaviest unseen bit is optimal at every step. Since $`x\mapsto(x-n)_+`$ is convex, Jensen's inequality orders the two expressions, strictly in small examples.

**Illustration.** With $`d=8`$ equally weighted contexts, each demonstration revealing one of them uniformly, the floor is $`\tfrac12(7/8)^K`$: a fact keeps escaping the demonstrations with probability $`(7/8)^K`$, and a targeted query removes one unseen fact for sure, whereas one more demonstration removes each unseen fact only with probability $`1/8`$.

**Comments.**

- **Replay adds zero.** Re-presenting already revealed labels leaves $`U_K`$ unchanged, so it does not change $`R_{\rm info}`$. This is [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4) specialized to this model. Replay can still reduce the learner's own error $`\mathcal R_{\rm alg}`$.
- **No universal rate in $`K`$.** A coordinate with $`q_j=0`$ leaves an irreducible passive error $`\tfrac12w_j`$, and a small $`q_j>0`$ makes convergence arbitrarily slow. Any rate in $`K`$ needs a coverage assumption on $`(q_j)`$.
- **What each assumption does.**
  - Independence of the bits makes queries non-interacting. With correlated bits one query informs another, and the greedy result changes.
  - The hard budget is essential: an expected-budget constraint allows allocation across transcripts.
  - The non-leaking reveal mechanism is essential: reveal sets chosen based on the bits would themselves leak information.
- **Floor, not achieved risk.** $`R_{\rm info}(K)`$ is a Bayes information floor. An actual learner's risk can be higher; the excess is its algorithm term $`\mathcal R_{\rm alg}`$ of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2).

**Uses.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2): the information term.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): replay adds nothing.
- [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8): the hidden-bit model.
- [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9): unrevealed bits stay uniform.

**Used by.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): learner-relative coverage model.
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2): allocate queries before compiling.
