# N03 · Few-shot information floor — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N03_few_shot_information_floor.md) · [中文](N03_few_shot_information_floor_ZH.md)

## Setting and assumptions

Every probability below is conditional on a fixed, declared pretrained information state $`I_{\rm pre}`$. Extra teacher feedback is never counted as part of the $`K`$ demonstrations.

**Assumption 3.1 (hidden-bit model).**

- The unknown bits $`\Theta=(\Theta_1,\dots,\Theta_d)`$ are independent and fair.
- Context $`j`$ occurs at deployment with weight $`w_j\ge0`$, where $`\sum_jw_j=1`$. The correct decision at context $`j`$ is $`\Theta_j`$.
- The student class contains every context-to-action table, so $`\mathcal R_{\rm rep}=0`$. The utility is $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$, so the informed optimum is 1.

**Assumption 3.2 (non-leaking demonstrations).**

- Trajectory $`k`$ reveals a random subset $`S_k\subseteq[d]`$ together with the labels $`\Theta_{S_k}`$.
- The subsets $`S_1,\dots,S_K`$ are IID with law $`L`$ and independent of $`\Theta`$. Dependence among the coordinates revealed within one trajectory is allowed.
- Any further transcript content has a conditional law that depends only on the revealed labels and independent randomness.
- Write $`q_j=\Pr(j\in S)`$.

**Assumption 3.3 (hard query budget).** After the demonstrations, the learner may make at most $`n`$ noiseless unit-cost queries, each revealing one chosen bit. The choices may depend on everything observed so far. $`n`$ is a per-transcript hard budget, not an expected budget.

**Assumption 3.4 (two-world adaptive acquisition).** The world is $`v\in\lbrace 0,1\rbrace`$, and the two worlds require different final decisions.

- Each of $`K`$ IID demonstrations satisfies $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$.
- A common history-dependent rule selects queries and stops at $`N\le N_{\max}<\infty`$. The rule's selection, stopping and private randomization kernels do not depend on $`v`$.
- For every allowed query $`x`$ and every preceding history $`h`$ on the relevant support, $`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$.
- The final decision errs with probability at most $`\delta`$ in each world, where $`0<\delta<1/2`$.

## Statement

**(a)** **Proposition (information floor).** Under Assumptions 3.1–3.2, the minimum achievable expected regret after $`K`$ trajectories is

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(A11)
```

with the convention $`(1-q)^0=1`$. The marginal value of one more trajectory is

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(A12)
```

**(b)** **Proposition (optimal queries).** Let $`U_K`$ be the set of bits revealed by none of the $`K`$ trajectories, with weights sorted as $`w_{(1)}\ge\dots\ge w_{(\lvert U_K\rvert)}`$. Under Assumptions 3.1–3.3,

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{\lvert U_K\rvert}w_{(r)}\Big], \qquad(A13)
```

and querying the $`n`$ heaviest unseen bits is optimal. Adapting the queries to the revealed values cannot help.

In the uniform case ($`w_j=1/d`$, and each trajectory reveals one uniformly drawn context), write $`M_K=\lvert U_K\rvert`$. Then

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(A14)
```

In general $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$, and the inequality can be strict.

**(c)** **Proposition (demonstrations credit the query budget).** Under Assumption 3.4,

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(A15)
```

If $`0<\kappa_X^{(v)}<\infty`$,

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(A16)
```

If $`\kappa_X^{(v)}=0`$ and $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$, no query budget achieves the stated errors.

## Proof

**Lemma 3.0 (unseen bits stay fair).** Under Assumptions 3.1–3.2, conditional on the full transcript $`T`$, the unseen bits $`(\Theta_j)_{j\in U_K}`$ are IID fair and independent of $`T`$ and of the learner's private randomness.

*Proof.* The joint law factorizes as

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

where $`X`$ is the additional transcript content. Fix the observed $`(S_{1:K},\Theta_{\cup S_k},X)`$. The only factor that depends on $`\Theta_{U_K}`$ is $`2^{-d}`$, which is constant. Hence $`\Theta_{U_K}`$ is uniform on $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ given $`T`$. The learner's randomness is independent of everything else. $`\square`$

**(a)**

- *Lower bound.* Fix any learner. For $`j\in U_K`$, by Lemma 3.0, $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$ for every possibly randomized action $`A_j`$. For $`j\notin U_K`$, success probability is at most 1. Hence the conditional regret is at least $`\tfrac12\sum_{j\in U_K}w_j`$.
- *Achievability.* The table that copies every revealed label and guesses the rest attains this, and it lies in the student class.
- *Taking expectations.* Since the $`S_k`$ are IID, $`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$. Hence $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$, which gives (A11).
- *(A12).* Subtract consecutive terms: $`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$. $`\square`$

**(b)**

*Conditional value.* Condition on the demonstration transcript, and let $`V(U,b)`$ be the optimal conditional regret with unseen set $`U`$ and remaining budget $`b`$. We show by induction on $`b`$ that

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

where $`\mathrm{top}_b(U)`$ is the sum of the $`\min(b,\lvert U\rvert)`$ largest weights in $`U`$.

*Base case.* For $`b=0`$ the claim is (a) conditionally.

*Inductive step.* With $`b\ge1`$ the learner may stop, query a known bit, or query some $`j\in U`$.

- Stopping gives $`V(U,0)`$, and querying a known bit gives $`V(U,b-1)`$. Both are at least the right-hand side by the induction hypothesis, since $`\mathrm{top}`$ is monotone in $`b`$.
- Querying $`j\in U`$ reveals $`\Theta_j`$. By Lemma 3.0 the remaining unseen bits are still IID fair and independent of the revealed value. The continuation problem is therefore $`V(U\setminus\lbrace j\rbrace ,b-1)`$ whatever value is observed, so value-adaptive branching gains nothing.

*Minimization.* Hence

```math
V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
```

The maximum equals $`\mathrm{top}_b(U)`$, attained by taking $`j`$ of largest weight. This completes the induction and shows that the greedy rule is optimal.

Averaging $`V(U_K,n)`$ over the demonstrations gives (A13).

*Uniform case.* With weights $`1/d`$, $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$, which gives (A14). For $`n=0`$, (a) with $`q_j=1/d`$ gives $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$. Since $`x\mapsto(x-n)_+`$ is convex, Jensen's inequality gives $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$.

*Strictness.* Take $`d=3`$, $`K=3`$, $`n=1`$. Then $`\Pr(M_3=0)=6/27`$, $`\Pr(M_3=1)=18/27`$, $`\Pr(M_3=2)=3/27`$, and $`\mathbb EM_3=8/9<1`$. The right-hand side is therefore $`0`$, while $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$. $`\square`$

**(c)**

*Padding.* Pad the transcript after stopping with a fixed null symbol to length $`N_{\max}`$. Write it as $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$, together with the stop indicators and the private randomness.

*Likelihood ratio.* The selection, stopping and randomization kernels are the same in both worlds, so they cancel in the likelihood ratio:

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

The finite KL upper bounds imply absolute continuity on the relevant support, so every term is finite almost surely.

*Taking $`\mathbb E_v`$.*

- The first sum has expectation $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$.
- In the second sum, $`\lbrace t\le N\rbrace`$ and $`X_t`$ are $`(H_{t-1},\text{randomness})`$-measurable. Conditioning term by term, each summand has expectation $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$.
- The sum is finite because $`N_{\max}<\infty`$. Summing over $`t`$ gives $`\kappa_X^{(v)}\mathbb E_vN`$.

Hence $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$.

*Data processing.* Let $`\mathsf{dec}`$ be the final decision, and apply data processing to the indicator $`1\lbrace \mathsf{dec}=\text{correct decision for }v\rbrace`$. Its probabilities are $`x\ge1-\delta`$ under $`v`$ and $`y\le\delta`$ under $`1-v`$. Hence $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$, by the monotonicity of binary kl on the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$. Combining the two bounds gives (A15). Rearranging gives (A16).

*The case $`\kappa_X^{(v)}=0`$.* The left side of (A15) equals $`K\kappa_D^{(v)}`$. If this is below $`\mathrm{kl}(1-\delta,\delta)`$, (A15) is violated for every $`N`$, so no such algorithm exists. $`\square`$

## Remarks

1. **Replay adds zero.** Re-presenting already revealed labels leaves $`U_K`$ unchanged, so it does not change $`R_{\rm info}`$. This is [N01(d)](N01_decomposition_and_ambiguity.md) of N01 specialized to this model. It can still reduce the learner's own error $`\mathcal R_{\rm alg}`$.
2. **No universal rate in $`K`$.** A coordinate with $`q_j=0`$ leaves an irreducible passive error $`\tfrac12w_j`$. A small $`q_j>0`$ makes convergence arbitrarily slow. Any rate in $`K`$ needs a coverage assumption on $`(q_j)`$.
3. **What each assumption does.** Independence of the bits is what makes queries non-interacting. Correlated bits would let one query inform another, and the greedy result changes. The hard budget is essential: an expected-budget constraint allows allocation across transcripts. The non-leaking reveal mechanism is essential: reveal sets chosen based on the bits would themselves leak information.
4. **Scope of (A15).** It is a binary-identification bound. A policy-learning version needs disjoint near-optimal output sets. With a zero right-hand side, (A16) does not establish achievability.
5. **Floor, not achieved risk.** $`R_{\rm info}(K)`$ is a Bayes information floor. An actual learner's risk can be higher (experiments page, E5).

## Dependencies and source

- **Uses:** [N01](N01_decomposition_and_ambiguity.md) (the $`\mathcal R_{\rm info}`$ term and [N01(d)](N01_decomposition_and_ambiguity.md)), [N02](N02_passive_vs_active_acquisition.md) (the change-of-measure technique).
- **Used by:** [N17](N17_ced_method.md) (design rule 1: acquire before compiling).

**Source:** [v0.15 §4](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#4-make-the-number-of-demonstrations-explicit) — Props. A5–A7; eqs. (A11)–(A16). [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md), §4, Propositions A5–A7, eqs. (A11)–(A16). Review: [Proof_Review_v015_EN.md](../../docs/direction/v015_revision/Proof_Review_v015_EN.md), V15.3–V15.4.
