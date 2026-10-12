# §2 Feasibility and acquisition — Statements

[← Framework](../README.md) · [Proofs](../proofs/02_feasibility_and_acquisition.md) · [Experiments](../experiments/02_feasibility_and_acquisition.md) · [中文](02_feasibility_and_acquisition_ZH.md)

### §2 · Feasibility and acquisition — exact statements

[Proofs](../proofs/02_feasibility_and_acquisition.md) · [Experiments](../experiments/02_feasibility_and_acquisition.md)

**What this section does.**

- It asks whether the available data can determine what the student should do, separately from whether the student learns well ([Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2)).
- It shows that no processing of existing data can supply a missing fact ([Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4)), and that even optimal teacher demonstrations can leave one ([Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6)).
- It prices the missing information: active queries at order $`\Delta^{-2}\log(1/\delta)`$, compilation separately ([Proposition 2.7](../proofs/02_feasibility_and_acquisition.md#s2-7)).
- It makes the number $`K`$ of demonstrations explicit and shows that demonstrations and queries share one information budget ([Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11)).

**Setting ([Definition 2.1](../proofs/02_feasibility_and_acquisition.md#s2-1)).** A finite set $`\mathcal E`$ of environments carries a prior conditional on the common initial information $`I_0`$. The student class $`\Pi`$ sits inside a reference class $`\Pi_{\rm ref}`$, with values $`J_E(\pi)\in[0,1]`$. The transcript $`D`$ contains every experiment the learner may run, and the learner outputs $`\widehat\pi\sim\lambda(\cdot\mid D)`$. Write $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$, $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$, and let $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ be the Bayes value.

#### Theorem 2.2 · Regret decomposition and decision ambiguity

**(a)** For any learner,

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\underbrace{\mathbb E[J_E^{\rm ref}-J_E^\Pi]}_{\mathcal R_{\rm rep}}+\underbrace{\mathbb E J_E^\Pi-\mathbb E_D B(D)}_{\mathcal R_{\rm info}}+\underbrace{\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big]}_{\mathcal R_{\rm alg}},
```

and all three terms are nonnegative.

**(b)** $`\mathcal R_{\rm info}=0`$ if and only if, for every transcript $`D`$ of positive probability, the environments of positive posterior probability share an optimal policy: $`\bigcap_{E:\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing`$.

*Derivation outline.*

- The identity follows by adding and subtracting $`\mathbb EJ_E^\Pi`$ and $`\mathbb E_DB(D)`$.
- The representation term is nonnegative because $`\Pi\subseteq\Pi_{\rm ref}`$. The information term is nonnegative because $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$ for every policy. The algorithm term is nonnegative because, given $`D`$, the learner's output is a convex combination of policies and cannot beat the best of them.
- For (b), $`\mathcal R_{\rm info}`$ is a sum over transcripts of nonnegative brackets $`\sum_E\mu(E\mid D)(J_E^\Pi-J_E(\pi_D))`$; a bracket vanishes exactly when a posterior-optimal policy is optimal in every environment of positive posterior probability.

*Uses* [Definition 2.1](../proofs/02_feasibility_and_acquisition.md#s2-1). *Used by* [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6), [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-2) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-1) · proved

#### Proposition 2.3 · Robust ambiguity duality

For a finite class $`C`$ of environments,

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big].
```

If no allowed experiment distinguishes the members of $`C`$, every learner has worst-case expected regret at least $`r(C)`$ on $`C`$, and the minimax mixture attains it; $`r(C)=0`$ if and only if some policy is optimal in every $`E\in C`$.

*Derivation outline.*

- The regret matrix $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ defines a finite zero-sum game: the learner picks a mixture of policies, the adversary a prior over $`C`$.
- Von Neumann's minimax theorem exchanges min and max; the inner minimum is attained at a pure policy, which gives the dual form.
- When the experiments cannot tell the members of $`C`$ apart, the learner's output law is the same in all of them, so its worst-case regret is at least the game value.

*Illustration.* Two environments prefer opposite policies, with regret matrix $`\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$. The minimax mixture is $`(\tfrac12,\tfrac12)`$ and the least favourable prior is $`(\tfrac12,\tfrac12)`$; both sides equal $`0.1`$. This is a mixture-averaged regret: a sampled policy has regret 0 or 0.2.

*Uses* [Definition 2.1](../proofs/02_feasibility_and_acquisition.md#s2-1). *Used by* [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6) (an instance with $`r(C)=\Delta`$).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-3) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-2) · proved

#### Proposition 2.4 · Post-processing cannot reduce information regret

If $`D'\sim\kappa(\cdot\mid D)`$ with a kernel $`\kappa`$ that does not depend on $`E`$, then $`\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D]`$, that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. Adding to $`D`$ a variable $`Z`$ with $`Z\perp E\mid(D,I_0)`$ leaves $`B`$ and $`\mathcal R_{\rm info}`$ unchanged.

*Derivation outline.*

- Because $`E\to D\to D'`$ is a Markov chain, $`\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D]\,\big|\,D'\big]`$.
- A maximum of conditional expectations is at most the conditional expectation of the maximum, so the Bayes value under $`D'`$ is at most that under $`D`$.

*Illustration.* Replaying, paraphrasing or rewriting the $`K`$ demonstrations by an environment-independent procedure is such a $`\kappa`$. It can make the student fit the existing information better (lower $`\mathcal R_{\rm alg}`$) but cannot lower $`\mathcal R_{\rm info}`$.

*Uses* [Definition 2.1](../proofs/02_feasibility_and_acquisition.md#s2-1). *Used by* [Theorem 2.6(a)](../proofs/02_feasibility_and_acquisition.md#s2-6), [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10) (replay adds zero), [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-4) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-1) · proved

#### Theorem 2.6 · Passive impossibility, active attainability

**Setting ([Definition 2.5](../proofs/02_feasibility_and_acquisition.md#s2-5)).** An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds with known gap $`0<\Delta\le1/8`$. A uniform hidden state $`s\in\lbrace0,1\rbrace`$ is visible to the teacher but not to the aliased student, who plays action $`A`$ with probability $`p`$. The teacher plays the action that succeeds with probability 1 in each state, so it is optimal in both worlds; the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$, so its optimal action depends on $`v`$. Under the query interface the learner may reset to $`s=1`$, execute $`A`$ and observe the outcome, adaptively, with a stopping time $`N`$.

- **(a)** Every learner that sees only demonstrations (and post-processing of them) has $`\max_v\mathbb E[J_v^\Pi-J_v(\widehat p)]\ge\Delta`$.
- **(b)** If, in each world, the learner outputs a student with regret at most $`\Delta/2`$ with probability at least $`1-\delta`$, then

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}.
```

- **(c)** $`\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ queries suffice.

*Derivation outline.*

- (a) In both worlds a demonstration is, with probability one half each, "choose $`A`$ at $`s=0`$ and succeed" or "choose $`B`$ at $`s=1`$ and succeed", so the law of the demonstrations and of the learner's output does not depend on $`v`$. The value is affine in $`p`$; the two regrets are $`2\Delta(1-m)`$ and $`2\Delta m`$ with $`m`$ the mean output, and their maximum is at least their average $`\Delta`$.
- (b) Reduce "$`\Delta/2`$-optimal in both worlds" to a binary test of the event $`\lbrace\widehat p>1/2\rbrace`$. With a stopping time, Wald's identity makes the KL divergence of the whole transcript equal to the expected number of queries times the KL of one query, and data processing makes it at least $`\mathrm{kl}(1-\delta,\delta)`$. The constant 64 comes from $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$.
- (c) Average independent query outcomes and apply the one-sided Hoeffding inequality.

*Uses* [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2), [Propositions 2.3–2.4](../proofs/02_feasibility_and_acquisition.md#s2-3). *Used by* [Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10), [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11) (the change-of-measure technique), [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-6) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-3) · proved

#### Proposition 2.7 · Compilation cost

After the sign is estimated, generating $`m\ge\lceil32\log(2/\delta_L)\rceil`$ labels from the target $`q\in\lbrace\tfrac18,\tfrac78\rbrace`$ and fitting their frequency gives $`\lvert\widehat p-q\rvert\le\tfrac18`$ with probability at least $`1-\delta_L`$. The compiled student then has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

*Derivation outline.* The empirical frequency of the compiled labels is controlled by the two-sided Hoeffding inequality, and a union bound combines the identification and compilation failures. The number of labels does not depend on $`\Delta`$.

*Illustration.* Identification costs authentic environment queries, order $`\Delta^{-2}\log(1/\delta)`$; compilation costs synthetic labels, order $`\log(1/\delta_L)`$. At small $`\Delta`$ almost the whole budget is acquisition, and the two must be accounted for separately.

*Uses* [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6). *Used by* [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-7) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-3) · proved

#### Lemma 2.9 and Theorem 2.10 · Hidden bits, the information floor and optimal querying

**Setting ([Definition 2.8](../proofs/02_feasibility_and_acquisition.md#s2-8)).** Bits $`\Theta_1,\dots,\Theta_d`$ are independent and fair; context $`j`$ has deployment weight $`w_j`$ and correct decision $`\Theta_j`$. Each of the $`K`$ demonstrations reveals a random subset of bits, drawn independently of the bits, and reveals bit $`j`$ with probability $`q_j`$. A hard budget allows $`n`$ queries, each revealing one chosen bit. $`U_K`$ is the set of bits no demonstration revealed and $`M_K=\lvert U_K\rvert`$.

**[Lemma 2.9](../proofs/02_feasibility_and_acquisition.md#s2-9).** Given the complete transcript, the unrevealed bits remain independent, fair and independent of the transcript.

**[Theorem 2.10](../proofs/02_feasibility_and_acquisition.md#s2-10).**

- **(a)** After $`K`$ demonstrations without queries the minimum achievable regret is

```math
R_{\rm info}(K)=\tfrac12\textstyle\sum_jw_j(1-q_j)^K .
```

- **(b)** With $`n`$ queries, $`R_{\rm info}(K,n)=\tfrac12\,\mathbb E\big[\sum_{r>n}w_{(r)}\big]`$ over the sorted unseen weights; querying the heaviest unseen bits is optimal and value-adaptive querying gains nothing. In general $`\mathbb E[(M_K-n)_+]\ne(\mathbb EM_K-n)_+`$.

*Derivation outline.*

- *Lemma 2.9.* The only factor of the joint law that depends on the unrevealed bits is the constant $`2^{-d}`$.
- *(a)* Any action guesses an unrevealed bit correctly with probability one half, and revealed bits can be copied, so the conditional regret is half the weight of the unseen bits. Bit $`j`$ stays unrevealed in all $`K`$ independent demonstrations with probability $`(1-q_j)^K`$.
- *(b)* Induct on the remaining budget. After a query, Lemma 2.9 makes the remaining problem independent of the value found, so branching on values gains nothing, and the greedy choice of the heaviest unseen bit is optimal at every step. Since $`x\mapsto(x-n)_+`$ is convex, Jensen's inequality orders the two expressions, strictly in small examples.

*Illustration.* With $`d=8`$ equally weighted contexts, each demonstration revealing one of them uniformly, the floor is $`\tfrac12(7/8)^K`$: a fact keeps escaping the demonstrations with probability $`(7/8)^K`$, and a targeted query removes one unseen fact for sure, whereas one more demonstration removes each unseen fact only with probability $`1/8`$.

*Uses* [Theorem 2.2](../proofs/02_feasibility_and_acquisition.md#s2-2), [Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4). *Used by* [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10) (the learner-relative coverage model), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-10) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-4) · proved

#### Proposition 2.11 · Demonstrations and queries share one information budget

In two-world adaptive acquisition with $`K`$ IID demonstrations of KL information at most $`\kappa_D`$ each, queries of KL information at most $`\kappa_X`$ each given any history, a world-independent query-and-stop rule with $`N\le N_{\max}`$, and final error at most $`\delta<\tfrac12`$ in each world,

```math
K\kappa_D+\kappa_X\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta).
```

If $`\kappa_X=0`$ and $`K\kappa_D<\mathrm{kl}(1-\delta,\delta)`$, no query budget suffices.

*Derivation outline.* The query-selection, stopping and randomization kernels are the same in both worlds and cancel in the likelihood ratio, leaving the demonstration terms and the query-outcome terms. In expectation the former are at most $`K\kappa_D`$ and the latter at most $`\kappa_X\,\mathbb EN`$; the data-processing inequality requires their sum to be at least $`\mathrm{kl}(1-\delta,\delta)`$.

*Uses* [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6) (the technique). *Used by* [Corollary 6.8](../proofs/06_learner_relative_information.md#s6-8).

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-11) · [Experiments](../experiments/02_feasibility_and_acquisition.md#c2-5) · proved

#### Remark 2.12 · Learner-relative versions

[Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7) generalizes Theorem 2.6(a) and Proposition 2.4: if demonstrations cannot distinguish two environments that differ on the learner-relative functional $`\Psi`$, no distiller, including replay, rewriting and synthesis, can do better than half the difference, for any $`K`$. [Corollary 6.8](../proofs/06_learner_relative_information.md#s6-8) generalizes Proposition 2.11 to a per-direction budget in which the directions are weighted by the learner. The object changes from the decision regret of a policy to the fidelity of a feedback law relative to the learner's update.

[Proof](../proofs/02_feasibility_and_acquisition.md#s2-12)
