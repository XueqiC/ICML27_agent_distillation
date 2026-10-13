# Theorem 2.6 · Passive impossibility, active attainability

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-06_passive_impossibility_active_attainability.md) · [Proof](02-06_passive_impossibility_active_attainability.md) · [Experiments](../experiments/02-06_passive_impossibility_active_attainability.md) · [中文](02-06_passive_impossibility_active_attainability_ZH.md)

**What is proved here.** The passive part holds because the demonstrations have the same law in both worlds, so the output does, and an affine value forces regret at least $`\Delta`$ in one world. The query lower bound is a change of measure on the stopped transcript (Wald's identity plus data processing), and the upper bound is Hoeffding's inequality.

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).

**Statement ([Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5):

- **(a) Passive impossibility.** Under the passive interface every learner satisfies $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$.
- **(b) Active lower bound.** Under the query interface, let $`0<\delta<1/2`$, and suppose that in each world, with probability at least $`1-\delta`$, the learner outputs a student with regret at most $`\Delta/2`$. Then for each $`v`$,

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) Active upper bound.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ independent queries suffice to output the optimal endpoint $`p\in\lbrace 0,1\rbrace`$ with probability at least $`1-\delta`$ in each world.

**Proof.**

**(a)** A demonstration records $`(s,\text{action},\text{outcome})`$, equal to $`(0,A,1)`$ or $`(1,B,1)`$ with probability $`\tfrac12`$ each, in both worlds. The demonstration law does not depend on $`v`$, and neither does the law of any randomized post-processing of it, including the learner's output $`\widehat p`$. Let $`m=\mathbb E\widehat p`$, the same in both worlds. Because $`J_v`$ is affine in $`p`$, $`\mathbb EJ_v(\widehat p)=J_v(m)`$, and the two expected regrets are

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m .
```

Their maximum is at least their average, $`\Delta`$. Equality holds at $`m=\tfrac12`$, for example with $`\widehat p\equiv\tfrac12`$.

**(b)** *Step 1 (reduction to a binary test).* In world $`+1`$, regret $`2\Delta(1-\widehat p)\le\Delta/2`$ holds exactly when $`\widehat p\ge3/4`$. In world $`-1`$, $`2\Delta\widehat p\le\Delta/2`$ holds exactly when $`\widehat p\le1/4`$. Let $`A=\lbrace \widehat p>1/2\rbrace`$. Then $`P_{+1}(A)\ge1-\delta`$ and $`P_{-1}(A)\le\delta`$.

*Step 2 (KL of the stopped transcript).* Let $`T`$ be the full transcript: demonstrations, query outcomes $`Y_1,\dots,Y_N`$ and the learner's private randomness, with law $`P_v^T`$.

- The demonstrations and the private randomness have the same law in both worlds. Each queried outcome is Bernoulli $`(\tfrac12+2v\Delta)`$ given the past. Write $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$.
- Let $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ be the per-query log-likelihood ratio. It is bounded: $`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$.
- The demonstration and randomization factors cancel, so the log-likelihood ratio of the transcript is $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$.
- Let $`\mathcal F_{t-1}`$ be the history before query $`t`$. The event $`\lbrace t\le N\rbrace`$ is $`\mathcal F_{t-1}`$-measurable, and $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$ on that event. Moreover $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$, so Fubini's theorem applies (Wald's identity):

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta .
```

*Step 3 (data processing).* The event $`A`$ is a function of $`T`$, so

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

The second inequality holds because, for $`y\le x`$, $`\mathrm{kl}(x,y)`$ is nondecreasing in $`x`$ and nonincreasing in $`y`$; on the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$ the minimum is at $`(1-\delta,\delta)`$. Steps 2 and 3 give the first inequality in (2.6) for $`v=+1`$. For $`v=-1`$ the same argument applies, since $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$.

*Step 4 (the constant).* For $`0<u\le\tfrac14`$,

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

The series has positive coefficients and is increasing in $`u`$; at $`u=\tfrac14`$ it equals $`2\,\mathrm{artanh}(\tfrac12)=\log3`$. Hence $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$. With $`u=2\Delta\le\tfrac14`$ this gives $`d_\Delta\le64\Delta^2`$, the second inequality in (2.6).

**(c)** Draw $`n`$ independent query outcomes and set $`\widehat v=+1`$ if $`\bar Y>\tfrac12`$ and $`\widehat v=-1`$ otherwise (a tie counts as an error in the analysis). Output $`\widehat p=1\lbrace \widehat v=+1\rbrace`$. In world $`+1`$, $`\mathbb E\bar Y=\tfrac12+2\Delta`$, and the one-sided Hoeffding inequality gives

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

whenever $`n\ge\log(1/\delta)/(8\Delta^2)`$. World $`-1`$ is symmetric. On the success event the output is the optimal endpoint, with regret 0. $`\square`$

## Related work

- The KL divergence of the stopped transcript is computed with Wald's identity for sums with a stopping time ([Wald, 1944](#ref-wald1944cumulative)).
- The sufficiency bounds use Hoeffding's inequality for bounded independent variables ([Hoeffding, 1963](#ref-hoeffding1963probability)).
- The change-of-measure lower bound for adaptive sampling with a stopping time is the standard argument of best-arm identification ([Kaufmann, Cappé & Garivier, 2016](#ref-kaufmann2016complexity)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-kaufmann2016complexity"></a>

**Emilie Kaufmann, Olivier Cappé, Aurélien Garivier (2016).** On the complexity of best-arm identification in multi-armed bandit models. *Journal of Machine Learning Research* 17, no. 1, 1–42. [jmlr.org/papers/v17/kaufman16a](https://jmlr.org/papers/v17/kaufman16a.html).

<a id="ref-wald1944cumulative"></a>

**Abraham Wald (1944).** On cumulative sums of random variables. *The Annals of Mathematical Statistics* 15, no. 3, 283–296. DOI [10.1214/aoms/1177731235](https://doi.org/10.1214/aoms/1177731235).
