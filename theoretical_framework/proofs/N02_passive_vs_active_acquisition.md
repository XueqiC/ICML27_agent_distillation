# N02 · Passive impossibility, active attainability — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N02_passive_vs_active_acquisition.md) · [中文](N02_passive_vs_active_acquisition_ZH.md)

## Setting and assumptions

**The two-world aliased family.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects the world. A gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform.
- The student has one aliased context, so it cannot see $`s`$, and makes one decision: action $`A`$ with probability $`p`$, otherwise $`B`$.
- Success probabilities:

| State | Success of $`A`$ | Success of $`B`$ |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

- The teacher observes $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$.

**Assumption 2.1 (passive interface).** The learner sees only teacher demonstrations, possibly with $`s`$ revealed, together with randomized post-processing of them.

**Assumption 2.2 (query interface).** In addition, the learner may reset to $`s=1`$, execute $`A`$, and observe the Bernoulli outcome. Queries may be chosen adaptively. The number of queries $`N`$ is a stopping time with $`\mathbb E_vN<\infty`$. The learner's algorithm and private randomness are the same in both worlds.

Write $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$.

## Statement

**Lemma 2.0 (values).** $`J_v(p)=\tfrac34+v\Delta(2p-1)`$, and $`J_v^\Pi=\tfrac34+\Delta`$, attained at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$. The teacher succeeds with probability 1 and is strictly optimal in both worlds.

**(a)** **Theorem (passive impossibility).** Under Assumption 2.1, every learner satisfies

```math
\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta.
```

**(b)** **Theorem (active lower bound).** Under Assumption 2.2, let $`0<\delta<1/2`$. Suppose that in each world, with probability at least $`1-\delta`$, the learner outputs a student with regret at most $`\Delta/2`$. Then for each $`v`$,

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(8)
```

**(c)** **Theorem (active upper bound).** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ independent queries suffice to output the optimal endpoint $`p\in\lbrace 0,1\rbrace`$ with probability at least $`1-\delta`$ in each world.

**(d)** **Proposition (compilation).** After the sign is estimated, choose the target $`q=7/8`$ (if $`\widehat v=+1`$) or $`q=1/8`$ (otherwise). Generate $`m`$ Bernoulli $`(q)`$ target labels, and let the student be their empirical frequency $`\widehat p`$. If $`m\ge\lceil32\log(2/\delta_L)\rceil`$, then $`\lvert\widehat p-q\rvert\le1/8`$ with probability at least $`1-\delta_L`$. The compiled student then has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

## Proof

**Proof of Lemma 2.0.** Average over $`s`$:

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big].
```

The first bracket equals $`\tfrac{1+p}{2}-2v\Delta(1-p)`$, and the second equals $`1-\tfrac p2+2v\Delta p`$. Their sum is $`\tfrac32+2v\Delta(2p-1)`$, so $`J_v(p)=\tfrac34+v\Delta(2p-1)`$. This is linear in $`p`$, so its maximum over $`[0,1]`$ is $`\tfrac34+\Delta`$ at the stated endpoint.

The teacher always plays the action with success probability 1. Every other state–action success probability equals $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$, because $`2\Delta\le\tfrac14`$. The teacher is therefore strictly optimal in both worlds. $`\square`$

**Proof of (a).** A demonstration records $`(s,\text{action},\text{outcome})`$, equal to $`(0,A,1)`$ or $`(1,B,1)`$ with probability $`\tfrac12`$ each, in both worlds. Hence the demonstration law does not depend on $`v`$. Any randomized post-processing of it, including the learner's output $`\widehat p`$, also has a $`v`$-independent law. Let $`m=\mathbb E\widehat p`$, the same in both worlds. Because $`J_v`$ is affine in $`p`$, $`\mathbb EJ_v(\widehat p)=J_v(m)`$, and the two expected regrets are

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m.
```

Their maximum is at least their average, which is $`\Delta`$. Equality holds at $`m=\tfrac12`$, for example with $`\widehat p\equiv\tfrac12`$. $`\square`$

**(b)**

*Step 1 (reduction to a binary test).* In world $`+1`$, regret $`2\Delta(1-\widehat p)\le\Delta/2`$ holds exactly when $`\widehat p\ge3/4`$. In world $`-1`$, $`2\Delta\widehat p\le\Delta/2`$ holds exactly when $`\widehat p\le1/4`$. Let $`A=\lbrace \widehat p>1/2\rbrace`$. Then $`P_{+1}(A)\ge1-\delta`$ and $`P_{-1}(A)\le\delta`$.

*Step 2 (KL of the stopped transcript).* Let $`T`$ be the full transcript: demonstrations, query outcomes $`Y_1,\dots,Y_N`$ and the learner's private randomness. Write $`P_v^T`$ for its law. The demonstrations and the private randomness have the same law in both worlds. Each queried outcome is Bernoulli $`(\tfrac12+2v\Delta)`$ given the past. Write $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$.

Let $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ be the per-query log-likelihood ratio. It is bounded: $`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$. The demonstration and randomization factors cancel, so the log-likelihood ratio of the transcript is $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$. Let $`\mathcal F_{t-1}`$ be the history before query $`t`$. The event $`\lbrace t\le N\rbrace`$ is $`\mathcal F_{t-1}`$-measurable, and $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$ on that event. Moreover $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$, so Fubini's theorem applies. This is Wald's identity:

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta.
```

*Step 3 (data processing).* The event $`A`$ is a function of $`T`$, so

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

The second inequality holds because, for $`y\le x`$, $`\mathrm{kl}(x,y)`$ is nondecreasing in $`x`$ and nonincreasing in $`y`$. On the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$, the minimum is therefore at $`(1-\delta,\delta)`$. Combining Steps 2 and 3 gives the first inequality in (8) for $`v=+1`$. For $`v=-1`$ the same argument applies, since $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$.

*Step 4 (the constant).* For $`0<u\le\tfrac14`$,

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

The series has positive coefficients and is increasing in $`u`$. At $`u=\tfrac14`$ it equals $`2\mathrm{artanh}(\tfrac12)=\log3`$. Hence $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$. With $`u=2\Delta\le\tfrac14`$, this gives $`d_\Delta\le64\Delta^2`$, which is the second inequality in (8). $`\square`$

**Proof of (c).** Draw $`n`$ independent query outcomes and set $`\widehat v=+1`$ if $`\bar Y>\tfrac12`$ and $`\widehat v=-1`$ otherwise; a tie counts as an error in the analysis. Output $`\widehat p=1\lbrace \widehat v=+1\rbrace`$. In world $`+1`$, $`\mathbb E\bar Y=\tfrac12+2\Delta`$. The one-sided Hoeffding inequality gives

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

whenever $`n\ge\log(1/\delta)/(8\Delta^2)`$. World $`-1`$ is symmetric. On the success event the output is the optimal endpoint, with regret $`0`$. $`\square`$

**Proof of (d).** The two-sided Hoeffding inequality gives

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

when $`m\ge32\log(2/\delta_L)`$. Suppose $`\widehat v=v=+1`$ and $`\lvert\widehat p-q\rvert\le\tfrac18`$. Then $`\widehat p\ge\tfrac34`$, and the regret is $`2\Delta(1-\widehat p)\le\Delta/2`$. The case $`v=-1`$ is symmetric, with $`\widehat p\le\tfrac14`$. A union bound over the identification failure ($`\le\delta`$) and the compilation failure ($`\le\delta_L`$) gives the stated probability. $`\square`$

## Remarks

1. **Perfect demonstrations can be information-insufficient.** The teacher succeeds with probability 1 in both worlds, yet $`\mathcal R_{\rm info}\ge\Delta`$ for every passive learner (N01). The missing fact, the sign of $`v`$, is not in the demonstrations even when they reveal $`s`$.
2. **Acquisition and compilation are separate costs.** Identification costs $`\Theta(\Delta^{-2}\log(1/\delta))`$ authentic environment queries, by (8) and (c), which match up to constants. Compilation then costs $`O(\log(1/\delta_L))`$ *synthetic* target labels. These labels are compiled, not new teacher executions, and must be accounted for separately.
3. **Tightness.** Exact computation (experiments page) shows that an SPRT attains the bound (8) exactly at $`(\Delta,\delta)=(\tfrac18,\tfrac1{10})`$ and $`(\tfrac18,\tfrac14)`$, and comes within 1% of it at $`(\tfrac1{16},0.01)`$ and $`(\tfrac1{64},0.01)`$.
4. **Assumptions used.** Knowledge of $`\Delta`$ enters the stopping rule of (c). (b) needs the declared query interface, a world-independent algorithm, and $`\mathbb E_vN<\infty`$. The argument is a standard change of measure applied to a student–teacher information mismatch; it is not a new testing inequality.

## Dependencies and source

- **Uses:** [N01](N01_decomposition_and_ambiguity.md) (the decomposition and [N01(d)](N01_decomposition_and_ambiguity.md): post-processing of passive data cannot remove $`\mathcal R_{\rm info}`$).
- **Used by:** [N03](N03_few_shot_information_floor.md) (makes $`K`$ and query budgets explicit).

**Source:** [v0.14.1 §3](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#3-a-complete-acquisition-example-perfect-demonstrations-can-be-insufficient) — Thm. 4; eqs. (6)–(9); compilation paragraph. [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md), §3, Theorem 4, eqs. (6)–(9), compilation paragraph. Audit: [Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md), A5–A8.
