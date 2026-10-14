# Theorem 4.10 · Hankel rank and linear predictive realization

*Statement*

[← Framework](../README.md) · [Statement](04-10_hankel_rank_predictive_realization.md) · [Proof](../proofs/04-10_hankel_rank_predictive_realization.md) · [Experiments](../experiments/04-10_hankel_rank_predictive_realization.md) · [中文](04-10_hankel_rank_predictive_realization_ZH.md)

**In one paragraph.** A compressed description of the learning process is useful for continued training only if it can predict its own next value: matching what the tests see now is not enough if the next update moves the process somewhere the description cannot follow. Treat training as a controlled system whose controls are the executable feedback providers and whose outputs are the declared deployment scores. Then the right compressed state is a vector of predictions of future tests, and the classical theory of predictive state representations applies: when the test space is closed under one more control step, a finite basis gives an exact linear update; the rank of the task Hankel matrix is the smallest dimension of such a linear predictor; and recorded observations update the predictive state by a normalized linear map. The theorem is a classical instantiation; what is specific here is that the state is the complete learning state and the outputs are scores after training.

**Setting used on this page.**

- *Controls.* A declared finite set $`\mathcal B`$ of executable controls: feedback providers, query modes or training interventions. Whether a change of learning rate, optimizer or task is a control, part of a test, or a different system must be declared, not decided afterwards.
- *State and kernels.* The state is the complete learning state of [Definition 1.3](../README.md#s1-3), with the clock included; once the training budget is used up the state is absorbing, so every control then acts as the identity. Control $`b`$ has the actual Markov kernel $`K_b`$, acting on bounded functions. Finite horizons and nonstationary schedules are covered; no infinite-horizon stability is assumed.
- *Words.* For a control word $`w=b_1\cdots b_\ell`$, $`K_w=K_{b_1}\cdots K_{b_\ell}`$ ($`b_1`$ is applied first); the empty word gives the identity. For a state law $`\mu`$, $`\mu K_w`$ is the law after executing $`w`$.
- *Outputs and tests.* Declared bounded deployment scores $`F_1,\dots,F_k`$. A test is a pair $`(w,j)`$: execute $`w`$, then score $`F_j`$; its function is $`K_wF_j`$. The test space for a declared set $`\mathcal W`$ of test words is $`\mathcal V=\mathrm{span}\lbrace\mathbf 1,K_wF_j:w\in\mathcal W,\ j\le k\rbrace`$.
- *Observations.* When control $`b`$ also produces an observation $`o`$ in a finite set, $`T_{b,o}f(z)=\mathbb E[\mathbf 1\lbrace O=o\rbrace f(Z')\mid Z=z,b]`$; then $`\sum_oT_{b,o}=K_b`$.

<a id="s4-10"></a>

**Theorem 4.10 (Hankel rank and linear predictive realization).**

**(a) Closed tests give exact linear prediction.** Suppose $`\mathcal V`$ has a finite basis $`\psi=(\psi_1,\dots,\psi_d)`$ with $`\psi_1=\mathbf 1`$, contains every $`F_j`$, and is invariant under every $`K_b`$. Then there are matrices $`A_b`$ and vectors $`c_j`$ with $`K_b\psi=A_b\psi`$ and $`F_j=c_j^\top\psi`$, and for every state law $`\mu`$ the predictive state $`q(\mu)=\mathbb E_\mu\psi`$ satisfies

```math
q(\mu K_b)=A_b\,q(\mu),\qquad \mathbb E_\mu F_j=c_j^\top q(\mu). \qquad(4.18)
```

Invariance holds whenever $`\mathcal W`$ contains the empty word and is closed under one-step prefixing ($`bw\in\mathcal W`$ for $`w\in\mathcal W`$, $`b\in\mathcal B`$).

**(b) Hankel rank is the minimal linear dimension.** Fix an initial law $`\mu_0`$. Let $`\mathcal P`$ be a set of control prefixes that contains the empty word and is closed under one-step extension ($`pb\in\mathcal P`$), and let $`\mathcal T`$ be a set of test columns that contains the constant test and every $`(\varnothing,j)`$ and is closed under one-step prefixing. The task Hankel matrix is $`H(p,(w,j))=\mu_0K_pK_wF_j`$, with the constant column equal to 1. If $`\mathrm{rank}\,H=d<\infty`$:

- every linear realization of all columns of $`H`$, the constant column included, i.e. vectors $`q(p)\in\mathbb R^{d'}`$ and coefficients $`c`$ for every column with $`H(p,\cdot)=c^\top q(p)`$, has $`d'\ge d`$ (without the constant column the bound can fail: if every $`F_j\equiv0`$, the rank is 1 and $`d'=0`$ reproduces every $`(w,j)`$ entry);
- there is a $`d`$-dimensional realization with

```math
q(pb)=A_b\,q(p),\qquad H(p,(w,j))=c_j^\top A_w\,q(p),\qquad A_w=A_{b_\ell}\cdots A_{b_1}. \qquad(4.19)
```

**(c) Conditioning on observations.** Suppose $`\mathrm{span}\,\psi`$, with $`\psi_1=\mathbf 1`$, is invariant under every $`T_{b,o}`$: $`T_{b,o}\psi=M_{b,o}\psi`$. Let $`\mu`$ be a reachable state law with predictive state $`q=\mathbb E_\mu\psi`$, and let $`e_1=(1,0,\dots,0)^\top`$. Then the probability of observing $`o`$ after $`b`$ is $`e_1^\top M_{b,o}q`$, and when it is positive the predictive state of the posterior law is

```math
q'=\frac{M_{b,o}\,q}{e_1^\top M_{b,o}\,q}. \qquad(4.20)
```

**What it says.**

- If the declared tests are closed under one more legal control step and span a finite-dimensional space, the vector of their current predictions is an exact state: it updates linearly under every control, and it determines every declared expectation.
- The rank of the task Hankel matrix is the smallest dimension any linear predictor of these input–output expectations can have, and it is attained.
- With recorded observations, the predictive state is updated by a linear map followed by normalization by the observation's probability.

**Intuition.** The state of a learning process that matters for the declared tasks is not its parameters but what it would score under every legal continuation. A finite set of such predictions is enough exactly when they keep predicting each other one step ahead.

**Role in the framework.**

- It is the dynamic counterpart of the response operator of [Definition 4.8](04-08_response_operator.md#s4-8). The fixed-checkpoint response $`g_a(y)`$ is one local input–suffix block of such a Hankel object: first this round's message, then a future test. The local response spectrum of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) and the dynamic Hankel spectrum are related, but they are in general not the same singular values.
- [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11) shows that closure cannot be dropped and bounds the error when it holds only approximately; [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12) connects (a) with the response operator.

**Derivation outline.**

- (a) Invariance gives the coefficients $`A_b`$; linearity of the integral moves them through $`\mathbb E_\mu`$.
- (b) Any $`d'`$-dimensional realization factors $`H`$ through $`\mathbb R^{d'}`$, so the rank is at most $`d'`$. Conversely, take the row space of $`H`$ as the state space: prefix extension acts on rows as the column shift $`(w,j)\mapsto(bw,j)`$, which is linear and well defined because the columns are closed under prefixing; a basis of the row space gives (4.19).
- (c) The numerator is the joint unnormalized expectation of the observation event and the next features; the denominator is the event's probability; the first coordinate of $`\psi`$ is the constant, so it reads off the probability.

**Four boundaries.**

1. The minimality in (b) is over linear representations of the input–output expectations reachable from $`\mu_0`$ with the declared experiments. It is not minimality over all hidden states or over nonlinear representations, and a space invariant at every single state, as in (a), can be larger.
2. A low-rank submatrix of a few observed rows and columns does not prove that the whole $`H`$ has low rank; its rank is only a lower bound. Suffix-extension checks and held-out predictions are needed, and a direction that was never excited looks like a missing mode.
3. When the clock is known, report the predictive dimension of each time layer, or state the clock's cost. Rank added by concatenating time layers is not environment complexity.
4. Open-loop control words predict interventional expectations. Updating the state online from new observations needs (c), and a test class without the observation branches does not support arbitrary observation-adaptive protocols.

**Scope.**

- The probability interpretation of (4.20) applies to reachable predictive states; an arbitrary $`q\in\mathbb R^d`$ need not correspond to any probability law.
- No global lower bound on observation probabilities is assumed. Rare observations amplify any error in $`M_{b,o}`$; that cost must be paid explicitly, or the analysis must stay with unnormalized quantities.
- Exact realizations come from a real probabilistic process and are therefore legal predictors. A truncated singular value decomposition of an estimated matrix is not automatically positive, normalized, or realizable by legal feedback; realizing feedback is a separate problem ([Proposition 6.14](06-14_convex_realizability_legal_pool.md#s6-14)).

**Uses.**

- [Definition 1.3](../README.md#s1-3): the complete learning state and kernels.
- [Definition 4.8](04-08_response_operator.md#s4-8): the local response operator that this result extends in time.

**Used by.**

- [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11): closure is necessary; the approximate-closure error.
- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): responses factor through the next predictive state.
- [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4): a predictive-state model is one source of the approximate value.

## Related work

- Predictive state representations describe a controlled process by predictions of future tests and update them linearly with normalization ([Littman, Sutton & Singh, 2001](#ref-littman2001predictive); [Singh, James & Rudary, 2004](#ref-singh2004predictive)); (b) and (c) are their realization and conditioning results in this setting.
- Compressing a partially observable process while keeping only what predicts value, through invariant subspaces of this Krylov type, is value-directed compression ([Poupart & Boutilier, 2002](#ref-poupart2002value)). Learning such states from data is the subject of predictive state temporal-difference learning ([Boots & Gordon, 2010](#ref-boots2010predictive)). What changes here is the system (the complete learning process) and the outputs (scores after training); the construction is not new.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
