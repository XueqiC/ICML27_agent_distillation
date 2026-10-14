# Proposition 4.11 · Static sufficiency is not dynamic sufficiency; finite-horizon closure error

*Statement*

[← Framework](../README.md) · [Statement](04-11_static_not_dynamic_closure_error.md) · [Proof](../proofs/04-11_static_not_dynamic_closure_error.md) · [Experiments](../experiments/04-11_static_not_dynamic_closure_error.md) · [中文](04-11_static_not_dynamic_closure_error_ZH.md)

**In one paragraph.** A set of features that predicts the current deployment score exactly can still fail to predict the score after one more update. This proposition gives the smallest example: two states with the same current score, which one update sends to scores 0 and 1. So a compressed interface that is sufficient now must also be checked for closure under the next update. When closure holds only approximately, the error after a finite horizon is an explicit sum of the propagated closure residuals; no contraction is assumed, and large propagation gains are reported, not hidden.

**Setting used on this page.**

- Complete learning states, controls with Markov kernels, and test spaces as in [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10). A *feature map* is a bounded measurable $`\psi:\mathcal Z\to\mathbb R^d`$; for a state law $`\mu`$, $`q(\mu)=\mathbb E_\mu\psi`$.
- $`\psi`$ is *statically sufficient* for a score $`F`$ if $`F=c^\top\psi`$ for some $`c`$, and *closed* under a kernel $`K`$ if $`K\psi=A\psi`$ for some matrix $`A`$.
- For part (b): a fixed legal control sequence with kernels $`K_0,\dots,K_{H-1}`$, matrices $`A_t`$, residual functions $`r_t=K_t\psi-A_t\psi`$, a score decomposition $`F=c^\top\psi+e_F`$, the true predictive states $`q_t=\mathbb E\psi(Z_t)`$ along the chain, and model predictions $`\hat q_{t+1}=A_t\hat q_t`$ from some $`\hat q_0`$. Products are $`A_{s:u}=A_sA_{s-1}\cdots A_u`$, with the empty product equal to the identity.

<a id="s4-11"></a>

**Proposition 4.11 (Static sufficiency is not dynamic sufficiency; finite-horizon closure error).**

**(a) A statically sufficient interface that is not closed.** Let $`z=(s,b)\in\lbrace0,1\rbrace^2`$, $`F(z)=s`$ and the deterministic update $`U(s,b)=(b,b)`$, with kernel $`Kf(s,b)=f(b,b)`$. The features $`\psi=(\mathbf 1,s)`$ are statically sufficient for $`F`$, but $`KF=b\notin\mathrm{span}\lbrace\mathbf 1,s\rbrace`$. The states $`(0,0)`$ and $`(0,1)`$ have the same $`q`$ and the same current score, and scores $`0`$ and $`1`$ after one update. Adding $`b`$ gives the closed space $`\mathrm{span}\lbrace\mathbf 1,s,b\rbrace`$.

**(b) Finite-horizon error under approximate closure.** In the setting above, exactly

```math
q_H-\hat q_H=A_{H-1:0}\,(q_0-\hat q_0)+\sum_{t=0}^{H-1}A_{H-1:t+1}\,\mathbb E\,r_t(Z_t), \qquad(4.21)
```

and for any norm $`\lVert\cdot\rVert`$ on $`\mathbb R^d`$ with dual norm $`\lVert\cdot\rVert_*`$,

```math
\big\lvert\mathbb EF(Z_H)-c^\top\hat q_H\big\rvert\le\mathbb E\lvert e_F(Z_H)\rvert+\lVert A_{H-1:0}^\top c\rVert_*\,\lVert q_0-\hat q_0\rVert+\sum_{t<H}\lVert A_{H-1:t+1}^\top c\rVert_*\,\mathbb E\lVert r_t(Z_t)\rVert. \qquad(4.22)
```

**What it says.**

- Predicting the present exactly does not make a feature set an adequate state: the next update can depend on a direction the features ignore, and the resulting future error can be the maximal 1.
- With an approximately closed linear model, the predictive error after $`H`$ steps is the initial error and every step's closure residual, each propagated through the remaining model maps.
- The bound uses the actual gains $`\lVert A_{H-1:t+1}^\top c\rVert_*`$. No gain is assumed below 1; if the gains are large, the bound is large, and that must be reported.

**Intuition.** Compression for continued training must compress the dynamics, not only the snapshot. A coordinate that is invisible to today's score but drives tomorrow's update is exactly what a static fit throws away.

**Role in the framework.**

- It is the reason the framework separates current-deployment sufficiency from sufficiency under continued training, and the reason [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12) asks for tests closed under one more legal step.
- The residuals $`r_t`$ are the feature-closure errors that enter the Bellman residual of [Proposition 3.4(c)](03-04_joint_decomposition_approximate_value.md#s3-4); fitting errors of $`A_t`$ must be included in $`r_t`$.

**Derivation outline.**

- (a) $`KF(s,b)=b`$, and $`b`$ is not an affine function of $`s`$ on the four states, because $`s`$ and $`b`$ vary independently.
- (b) The Markov property gives $`q_{t+1}=\mathbb E(K_t\psi)(Z_t)=A_tq_t+\mathbb Er_t(Z_t)`$, so the error obeys $`q_{t+1}-\hat q_{t+1}=A_t(q_t-\hat q_t)+\mathbb Er_t(Z_t)`$; unrolling gives (4.21). For (4.22), write $`\mathbb EF(Z_H)-c^\top\hat q_H=\mathbb Ee_F(Z_H)+c^\top(q_H-\hat q_H)`$ and apply Hölder's inequality term by term, with $`\lVert\mathbb Er_t\rVert\le\mathbb E\lVert r_t\rVert`$.

**Scope.**

- (b) is stated for one fixed open-loop control sequence. Observation-adaptive protocols need the branch-wise conditioned version of [Theorem 4.10(c)](04-10_hankel_rank_predictive_realization.md#s4-10), or the general Bellman-residual form of [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4); the open-loop formula must not be applied to them without justification.
- The counterexample of (a) uses no approximation, no optimizer anomaly and no distribution tail. A static low-rank fit cannot replace a closure check.

**Uses.**

- [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10): predictive states and closure.

**Used by.**

- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): closure is needed to keep using the predictive state.
- [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4): closure residuals inside Bellman residuals.

## Related work

- That a predictive state must predict its own next value, and not only the current output, is the closure requirement of predictive state representations and value-directed compression ([Singh, James & Rudary, 2004](#ref-singh2004predictive); [Poupart & Boutilier, 2002](#ref-poupart2002value)); (a) is a minimal witness of it in the learning setting.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
