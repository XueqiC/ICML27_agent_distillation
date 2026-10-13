# Proposition 2.11 · Demonstrations and queries share one information budget

*Statement*

[← Framework](../README.md) · [Statement](02-11_shared_information_budget.md) · [Proof](../proofs/02-11_shared_information_budget.md) · [Experiments](../experiments/02-11_shared_information_budget.md) · [中文](02-11_shared_information_budget_ZH.md)

**In one paragraph.** Demonstrations and queries both carry information about which world the student is in. This proposition shows that they draw on a single budget: to decide between two worlds with error at most $`\delta`$, the KL information of the demonstrations plus that of the queries must reach $`\mathrm{kl}(1-\delta,\delta)`$, for every adaptive query-and-stop rule.

**Setting used on this page.**

- Two worlds $`v\in\lbrace0,1\rbrace`$ that require different final decisions.
- $`K`$ IID demonstrations; $`\kappa_D^{(v)}`$ bounds the KL divergence $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})`$ of one demonstration.
- A query rule chooses queries $`X_t`$ from the history $`H_{t-1}`$, observes outcomes $`Y_t`$, and stops at $`N\le N_{\max}`$; its kernels are the same in both worlds. $`\kappa_X^{(v)}`$ bounds the KL divergence of one query outcome given any history.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence; $`\delta<\tfrac12`$ is the allowed error in each world.

<a id="s2-11"></a>

**Proposition 2.11 (Demonstrations and queries share one information budget).** Consider two-world adaptive acquisition.

- The world is $`v\in\lbrace 0,1\rbrace`$, and the two worlds require different final decisions.
- Each of $`K`$ IID demonstrations satisfies $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$.
- A common history-dependent rule selects queries and stops at $`N\le N_{\max}<\infty`$. The rule's selection, stopping and private randomization kernels do not depend on $`v`$.
- For every allowed query $`x`$ and every preceding history $`h`$ on the relevant support, $`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$.
- The final decision errs with probability at most $`\delta`$ in each world, where $`0<\delta<1/2`$.

Then

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

If $`0<\kappa_X^{(v)}<\infty`$,

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

If $`\kappa_X^{(v)}=0`$ and $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$, no query budget achieves the stated errors.

**What it says.**

- To tell two worlds apart with error at most $`\delta`$, the KL information of the $`K`$ demonstrations plus the KL information per query times the expected number of queries must reach $`\mathrm{kl}(1-\delta,\delta)`$.
- If queries carry no information and the demonstrations do not suffice, no query budget helps.

**Intuition.**

- Demonstrations and queries pay into the same account; informative demonstrations reduce the number of queries needed, and replays pay nothing.

**Role in the framework.**

- It is the exchange rate between $`K`$ and the query budget. [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8) makes the rate depend on the learner.

**Derivation outline.**

- The query-selection, stopping and randomization kernels are the same in both worlds and cancel in the likelihood ratio, leaving the demonstration terms and the query-outcome terms.
- In expectation the former are at most $`K\kappa_D`$ and the latter at most $`\kappa_X\,\mathbb EN`$; the data-processing inequality requires their sum to be at least $`\mathrm{kl}(1-\delta,\delta)`$.

**Comments.**

- **Scope of (2.11).** It is a binary-identification bound. A policy-learning version needs disjoint near-optimal output sets. A zero right-hand side in (2.12) does not establish achievability.

**Uses.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): the change-of-measure argument.

**Used by.**

- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8): learner-relative exchange rate.

## Related work

- (2.11) extends the change-of-measure inequality of best-arm identification, which bounds the expected KL information collected up to a stopping time from below by the binary KL of the probabilities of an event ([Kaufmann et al., 2016](#ref-kaufmann2016complexity)), by the demonstration term $`K\kappa_D^{(v)}`$.
- The lower bound $`\mathrm{kl}(1-\delta,\delta)`$ comes from the data-processing inequality for KL divergence, applied to the final decision ([Cover & Thomas, 2006](#ref-cover2006elements)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-cover2006elements"></a>

**Thomas M. Cover, Joy A. Thomas (2006).** Elements of Information Theory. 2nd edition, Wiley-Interscience, Hoboken, NJ. DOI [10.1002/047174882X](https://doi.org/10.1002/047174882X).

<a id="ref-kaufmann2016complexity"></a>

**Emilie Kaufmann, Olivier Cappé, Aurélien Garivier (2016).** On the complexity of best-arm identification in multi-armed bandit models. *Journal of Machine Learning Research* 17, no. 1, 1–42. [jmlr.org/papers/v17/kaufman16a](https://jmlr.org/papers/v17/kaufman16a.html).
