# Remark 8.5 · Norms and constants

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-05_norms_constants.md) · [Experiments](../experiments/08-05_norms_constants.md) · [中文](08-05_norms_constants_ZH.md)

**In one paragraph.** The concentration bound of Proposition 8.4 has no explicit dimension, which is easy to over-read. This remark states exactly in which norm it holds, why the increment bound can still grow with dimension, and why a ranking decision needs less than the full bound.

**Setting used on this page.**

Use the objects of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): consumed gradient $`G_t(z,Y,\xi)`$, increment $`\Delta_t(z,g)`$, mean increment $`\mu^P_t(z)`$, second increment moment $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$ and continuation $`V^P_{t+1}`$. The norm is the Hilbert norm of the state space (Euclidean in finite dimension).

- **Bounded increments (B).** There is $`b<\infty`$ with $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$ almost surely, for the compared laws at the audited states. Examples: clipped SGD on the parameter block, $`b=\eta c`$; sign-type updates on $`p`$ parameters, $`b=\eta\sqrt p`$; SGD with gradients bounded by $`\bar G`$, $`b=\eta\bar G`$.
- **Curvature (S).** At the audited state, $`V:=V^P_{t+1}`$ is twice differentiable along the segments $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ reached by the increments, with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$; write $`\Lambda=\lVert\nabla V(z)\rVert`$.
- **Audit samples.** At an audited state $`z`$, draw $`n_P`$ messages $`Y\sim P_t(\cdot\mid z)`$ and $`n_Q`$ messages $`Y\sim Q_t(\cdot\mid z)`$, independently and with independent update randomness, and compute the increments $`X=\Delta_t(z,G_t(z,Y,\xi))`$. Their empirical means are $`\bar X_P,\bar X_Q`$ and their empirical second moments $`\hat m^P_2,\hat m^Q_2`$. This takes $`n_P+n_Q`$ consumed-gradient evaluations at $`z`$ and no training tail.

<a id="s8-5"></a>

**Remark 8.5 (Norms and constants).**

- The concentration bound of [Proposition 8.4(a)](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) holds in the Hilbert norm of the state space; its proof uses the inner product (vanishing cross terms). It does not extend to arbitrary norms with the same constant; for example, in the sup norm on $`\mathbb R^p`$ the analogous bound carries a $`\sqrt{\log p}`$ factor.
- The bound contains no explicit dimension, but $`b`$ itself can grow with the dimension. Under cold-start Adam every coordinate moves by about $`\eta`$ ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)), so $`\lVert\Delta\theta\rVert`$ can reach $`\eta\sqrt p`$.
- A ranking decision needs only the scalar projection $`\langle\nabla V(z),\Delta\rangle`$, whose fluctuation can be far smaller than $`\lVert\nabla V(z)\rVert\,b`$. [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) goes further: it measures the finite replacement effect directly and needs no bound $`b`$.
- **Cost comparison.** A full-tail audit at one state needs the replica count of [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) per candidate; (b) needs $`n_P+n_Q`$ gradient evaluations per candidate. What (b) cannot do alone is supply $`\Lambda`$ and $`M_2`$.

**What it says.** The bound of Proposition 8.4 holds in the Hilbert norm of the state space and not in arbitrary norms with the same constant; the increment bound can grow with dimension (cold-start Adam moves every coordinate); a ranking needs only a scalar projection; and Theorem 8.8 needs no increment bound at all.

**Intuition.** Dimension-free means no explicit dimension in the constant, not that the dimension has no effect.

**Role in the framework.** It keeps the first-order measurement honest about its constants and points to the finite measurement of Theorem 8.8.

**Uses.**

- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) — the concentration bound being qualified
- [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3) — size of a cold-start Adam step

**Used by.**

- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) — first-order certificate
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) — measurement that needs no increment bound

## Related work

- The Hilbert-norm bound being qualified combines Jensen's inequality with McDiarmid's bounded-difference inequality ([McDiarmid, 1989](#ref-mcdiarmid1989method)); the inner product enters only through the vanishing cross terms.
- The growth of $`b`$ with dimension under cold-start Adam reflects the sign-like first step of Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)); the sign view of Adam is studied by [Balles & Hennig, 2018](#ref-balles2018dissecting).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-mcdiarmid1989method"></a>

**Colin McDiarmid (1989).** On the method of bounded differences. In *Surveys in Combinatorics, 1989*, London Mathematical Society Lecture Note Series 141, 148–188, Cambridge University Press. DOI [10.1017/CBO9781107359949.008](https://doi.org/10.1017/CBO9781107359949.008).
