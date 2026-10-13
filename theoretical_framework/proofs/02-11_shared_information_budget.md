# Proposition 2.11 · Demonstrations and queries share one information budget

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-11_shared_information_budget.md) · [Proof](02-11_shared_information_budget.md) · [Experiments](../experiments/02-11_shared_information_budget.md) · [中文](02-11_shared_information_budget_ZH.md)

**What is proved here.** Pad the stopped transcript to a fixed length, write its log-likelihood ratio between the two worlds, and note that the query-selection and stopping kernels cancel. Bounding the expectation of the remaining terms and applying data processing to the correctness event gives the budget inequality.

**Setting used on this page.**

- Two worlds $`v\in\lbrace0,1\rbrace`$; $`K`$ IID demonstrations with per-demonstration KL at most $`\kappa_D^{(v)}`$; queries with per-outcome KL at most $`\kappa_X^{(v)}`$ given any history $`h`$; a world-independent query-and-stop rule with $`N\le N_{\max}`$.
- $`\mathrm{kl}(a,b)`$ is the binary KL divergence; $`H_{t-1}`$ is the history before query $`t`$.

**Statement ([Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11)).** Consider two-world adaptive acquisition.

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

**Proof.**

*Padding.* Pad the transcript after stopping with a fixed null symbol to length $`N_{\max}`$, and write it as $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$ together with the stop indicators and the private randomness.

*Likelihood ratio.* The selection, stopping and randomization kernels are the same in both worlds, so they cancel:

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

The finite KL bounds imply absolute continuity on the relevant support, so every term is finite almost surely.

*Taking $`\mathbb E_v`$.*

- The first sum has expectation $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$.
- In the second sum, $`\lbrace t\le N\rbrace`$ and $`X_t`$ are measurable with respect to $`(H_{t-1},\text{randomness})`$. Conditioning term by term, each summand has expectation $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$.
- The sum is finite because $`N_{\max}<\infty`$. Summing over $`t`$ gives $`\kappa_X^{(v)}\mathbb E_vN`$.

Hence $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$.

*Data processing.* Apply data processing to the indicator that the final decision is correct for $`v`$. Its probabilities are $`x\ge1-\delta`$ under $`v`$ and $`y\le\delta`$ under $`1-v`$, so $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$ by the monotonicity of the binary kl on the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$. Combining the two bounds gives (2.11), and rearranging gives (2.12).

*The case $`\kappa_X^{(v)}=0`$.* The left side of (2.11) equals $`K\kappa_D^{(v)}`$. If this is below $`\mathrm{kl}(1-\delta,\delta)`$, (2.11) fails for every $`N`$, so no such algorithm exists. $`\square`$

## Related work

- The change-of-measure lower bound for adaptive sampling with a stopping time is the standard argument of best-arm identification ([Kaufmann, Cappé & Garivier, 2016](#ref-kaufmann2016complexity)).
- The data-processing inequality for KL divergence is standard ([Cover & Thomas, 2006](#ref-cover2006elements)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-cover2006elements"></a>

**Thomas M. Cover, Joy A. Thomas (2006).** Elements of Information Theory. 2nd edition, Wiley-Interscience, Hoboken, NJ. DOI [10.1002/047174882X](https://doi.org/10.1002/047174882X).

<a id="ref-kaufmann2016complexity"></a>

**Emilie Kaufmann, Olivier Cappé, Aurélien Garivier (2016).** On the complexity of best-arm identification in multi-armed bandit models. *Journal of Machine Learning Research* 17, no. 1, 1–42. [jmlr.org/papers/v17/kaufman16a](https://jmlr.org/papers/v17/kaufman16a.html).
