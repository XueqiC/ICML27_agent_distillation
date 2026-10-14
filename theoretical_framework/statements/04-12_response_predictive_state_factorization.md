# Proposition 4.12 · Responses factor through the next predictive state

*Statement*

[← Framework](../README.md) · [Statement](04-12_response_predictive_state_factorization.md) · [Proof](../proofs/04-12_response_predictive_state_factorization.md) · [Experiments](../experiments/04-12_response_predictive_state_factorization.md) · [中文](04-12_response_predictive_state_factorization_ZH.md)

**In one paragraph.** The response operator of [Definition 4.8](04-08_response_operator.md#s4-8) and the predictive states of [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10) describe the same thing in two languages. If the declared tests' continuation values lie in a finite-dimensional space spanned by predictive features, the response operator is a fixed observation matrix applied to the expected change of the next predictive state. When the tests span that space, two feedback laws have equal responses on every declared test exactly when they produce the same expected next predictive state, and if the space is also closed under the declared continuation, they then agree on every later declared expectation. This lemma is close to the definition of a predictive state; its use is to make the compression of §4, the closure requirement and the measurements of §8 talk about one object.

**Setting used on this page.**

- A checkpoint $`(t,z)`$, the reference message measure $`\rho`$, $`H=L_0^2(\rho)`$, finitely many declared tests $`a=1,\dots,p`$ with weights $`\omega_a>0`$, and the response operator $`\mathcal R_z`$ ([Definition 4.8](04-08_response_operator.md#s4-8)).
- $`V_a`$ is the continuation value of test $`a`$ from the state after this round's update, so $`g_a(y)=\mathbb E[V_a(U_t(z,y,\xi_t))\mid z,y]`$.
- *Finite predictive basis.* Bounded features $`\psi:\mathcal Z\to\mathbb R^d`$ with $`V_a=k_a+c_a^\top\psi`$ for constants $`k_a`$ and vectors $`c_a`$. This is a finite-dimensional assumption for this proposition only, not a default low-rank assumption of the framework. $`C\in\mathbb R^{p\times d}`$ has rows $`c_a^\top`$.
- The *expected next predictive state* after message $`y`$ is $`\Gamma_z(y)=\mathbb E[\psi(U_t(z,y,\xi_t))\mid z,y]\in\mathbb R^d`$, and $`\mathcal T_z:H\to\mathbb R^d`$ is $`\mathcal T_zh=\mathbb E_\rho[\Gamma_z(Y)h(Y)]`$. For laws with densities $`p,q`$, $`\mathcal T_z(p-q)=\mathbb E_P\Gamma_z-\mathbb E_Q\Gamma_z`$.

<a id="s4-12"></a>

**Proposition 4.12 (Responses factor through the next predictive state).** In this setting:

**(a) Factorization.**

```math
\mathcal R_z=C\,\mathcal T_z,\qquad\text{i.e. }(\mathcal R_zh)(a)=c_a^\top\mathcal T_zh\ \text{ for every test }a\text{ and every }h\in H. \qquad(4.23)
```

**(b) Equal responses are equal next predictive states.** If $`C`$ has full column rank $`d`$, i.e. the declared tests span the predictive basis, then $`\ker\mathcal R_z=\ker\mathcal T_z`$. For two laws with $`L^2(\rho)`$ densities, equal expectations on every declared test is equivalent to $`\mathbb E_P\Gamma_z=\mathbb E_Q\Gamma_z`$.

**(c) Continuing.** If in addition $`\mathrm{span}\lbrace\mathbf 1,\psi\rbrace`$ is invariant under the kernels of the declared legal continuation, as in [Theorem 4.10(a)](04-10_hankel_rank_predictive_realization.md#s4-10), then equal expected next predictive states give equal expectations of every declared test after every declared open-loop control word. For observation-adaptive continuations the same holds when the span is invariant under every observation operator, as in [Theorem 4.10(c)](04-10_hankel_rank_predictive_realization.md#s4-10).

**(d) Residual version.** If $`V_a=k_a+c_a^\top\psi+e_a`$ with bounded residuals $`e_a`$, let $`\bar e_a(y)=\mathbb E[e_a(U_t(z,y,\xi_t))\mid z,y]`$. Then $`\mathcal R_z=C\mathcal T_z+\mathcal E_z`$ with $`(\mathcal E_zh)(a)=\langle\bar e_a-\mathbb E_\rho\bar e_a,h\rangle_\rho`$, and

```math
\lVert\mathcal E_zh\rVert_\omega\le\epsilon_{\rm pred}\,\lVert h\rVert_\rho,\qquad \epsilon_{\rm pred}^2=\sum_a\omega_a\,\mathrm{Var}_\rho\,\bar e_a\le\sum_a\omega_a\,\mathbb E_\rho\bar e_a^2 . \qquad(4.24)
```

**What it says.**

- At a checkpoint, the effect of a feedback difference on the declared tests passes through one intermediate object: the change it causes in the expected next predictive state.
- $`\mathcal T_z`$ says which predictive memory the feedback can change; $`C`$ says which of that memory the declared future tests can see; the response operator is their composition.
- When the tests see the whole predictive basis, matching responses and matching the next predictive state are the same condition, and closure turns that one-step match into a match for the declared continuation.

**Intuition.** A feedback law matters for the future only through what it leaves behind in the learner's state; the tests measure that residue through a fixed lens. The spectrum of the response operator mixes the two; this factorization separates them.

**Role in the framework.**

- It is the lemma that lets §4, §3 and §8 speak about one object. The spectral compression of [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9) selects directions of the composition $`C\mathcal T_z`$; the closure of [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11) decides whether those directions remain usable in the next round; the acquisition of [Proposition 6.16](06-16_information_compilation_radius.md#s6-16) collects evidence on the directions $`C`$ sees but the data have not resolved.
- It is the one-step instance of the factorization of a Hankel operator into observability and controllability ([Proposition 4.13](04-13_finite_horizon_gramians.md#s4-13)).

**Derivation outline.** Conditioning on the message gives $`g_a(y)=k_a+c_a^\top\Gamma_z(y)`$. Since $`h`$ has $`\rho`$-mean zero, the constant drops out and $`\langle\tilde g_a,h\rangle_\rho=c_a^\top\mathcal T_zh`$. Full column rank makes $`C`$ injective, which gives the equal kernels. Invariance under the continuation kernels propagates equality of $`\mathbb E\psi`$ through every declared control word. The residual version applies the Cauchy–Schwarz inequality test by test.

**What it does not give.**

- *Near-definitional.* When the declared tests span the predictive basis, "same responses on every test" and "same predictive state" are almost the same statement; the content is the separation into $`\mathcal T_z`$ and $`C`$ and the conditions under which it holds, not a new equivalence.
- *No quantitative inverse.* Full column rank gives equal kernels. Turning a small response error into a small predictive-state error needs the smallest singular value of $`C`$, which can be poor; predictive coordinates are defined only up to invertible linear maps, so guarantees should be stated in the task norm.
- *One checkpoint is not a run.* A match at one checkpoint does not replace matches at all states the training actually visits; over a run, [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) needs a match at every visited state, with the coverage cost of its part (c).
- *Cost.* A continuation protocol run once outputs several metrics. Spanning a $`d`$-dimensional basis therefore needs at least $`\lceil d/p_{\rm branch}\rceil`$ distinct continuation protocols when each branch yields $`p_{\rm branch}`$ metrics, not $`d`$ of them; identifying closure needs extension tests on top. How much independent information one full continuation branch provides is open ([Remark 8.17](08-17_open_problems.md#s8-17)).

**Uses.**

- [Definition 4.8](04-08_response_operator.md#s4-8): the response operator.
- [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10): predictive states, closure and conditioning.

**Used by.**

- [Proposition 4.13](04-13_finite_horizon_gramians.md#s4-13): the finite-horizon version of the factorization.

## Related work

- Representing state by predictions of future tests is the premise of predictive state representations ([Littman, Sutton & Singh, 2001](#ref-littman2001predictive)); (b) restates that premise for the response operator of a learning process.
- Requiring a model to be correct only on the value functions that matter is the value-equivalence principle ([Grimm et al., 2020](#ref-grimm2020value)); here the value functions are the declared continuation values and the model is the learner's next predictive state.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).
