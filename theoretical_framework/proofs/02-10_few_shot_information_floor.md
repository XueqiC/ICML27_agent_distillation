# Theorem 2.10 · Few-shot information floor and optimal hard-budget querying

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-10_few_shot_information_floor.md) · [Proof](02-10_few_shot_information_floor.md) · [Experiments](../experiments/02-10_few_shot_information_floor.md) · [中文](02-10_few_shot_information_floor_ZH.md)

**What is proved here.** Part (a) combines the lemma that unrevealed bits stay fair coins with the independence of the reveal sets. Part (b) is an induction on the remaining budget in which value-adaptive branching gains nothing and the greedy choice is optimal; Jensen's inequality and a small example settle the positive part.

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID, independent of the bits, and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

**Statement ([Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10)).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8):

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

**Proof.**

**(a)**

- *Lower bound.* Fix any learner. For $`j\in U_K`$, [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) gives $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$ for every possibly randomized action $`A_j`$. For $`j\notin U_K`$ the success probability is at most 1. Hence the conditional regret is at least $`\tfrac12\sum_{j\in U_K}w_j`$.
- *Achievability.* The table that copies every revealed label and guesses the rest attains this bound, and it lies in the student class.
- *Taking expectations.* The $`S_k`$ are IID, so $`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$ and $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$, which is (2.7).
- *Marginal value.* $`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$ gives (2.8).

**(b)** Condition on the demonstration transcript and let $`V(U,b)`$ be the optimal conditional regret with unseen set $`U`$ and remaining budget $`b`$. We show by induction on $`b`$ that

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

where $`\mathrm{top}_b(U)`$ is the sum of the $`\min(b,\lvert U\rvert)`$ largest weights in $`U`$.

- *Base case.* For $`b=0`$ the claim is (a), conditionally.
- *Inductive step.* With $`b\ge1`$ the learner may stop, query a known bit, or query some $`j\in U`$.
  - Stopping gives $`V(U,0)`$ and querying a known bit gives $`V(U,b-1)`$. Both are at least the right-hand side by the induction hypothesis, since $`\mathrm{top}_b`$ is monotone in $`b`$.
  - Querying $`j\in U`$ reveals $`\Theta_j`$. By [Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) the remaining unseen bits are still IID fair and independent of the revealed value, so the continuation problem is $`V(U\setminus\lbrace j\rbrace ,b-1)`$ whatever value is observed; value-adaptive branching gains nothing.

Hence

```math
V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
```

The maximum equals $`\mathrm{top}_b(U)`$, attained by taking $`j`$ of largest weight. This completes the induction and shows that the greedy rule is optimal. Averaging $`V(U_K,n)`$ over the demonstrations gives (2.9).

- *Uniform case.* With weights $`1/d`$, $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$, which gives (2.10); for $`n=0`$, (a) with $`q_j=1/d`$ gives $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$.
- *Jensen.* Since $`x\mapsto(x-n)_+`$ is convex, $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$.
- *Strictness.* Take $`d=3`$, $`K=3`$, $`n=1`$. Then $`\Pr(M_3=0)=6/27`$, $`\Pr(M_3=1)=18/27`$, $`\Pr(M_3=2)=3/27`$ and $`\mathbb EM_3=8/9<1`$. The right-hand side is $`0`$, while $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$. $`\square`$
