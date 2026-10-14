# Proposition 3.4 · Joint decomposition with an approximate value

*Statement*

[← Framework](../README.md) · [Statement](03-04_joint_decomposition_approximate_value.md) · [Proof](../proofs/03-04_joint_decomposition_approximate_value.md) · [Experiments](../experiments/03-04_joint_decomposition_approximate_value.md) · [中文](03-04_joint_decomposition_approximate_value_ZH.md)

**In one paragraph.** [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) writes the gap between two training processes through the exact reference continuation value $`V^0`$, which nobody has. In practice one has an approximate value $`\widehat V`$, produced by a predictive-state model or by any other means. This proposition keeps the bookkeeping exact with $`\widehat V`$ in place of $`V^0`$: the gap splits into a local response term, the Bellman residuals of $`\widehat V`$ under the reference kernel on both processes, and the terminal errors on both processes. A stopped version charges each process's exit probability once. It is an identity of the performance-difference family, not a source of small errors: every term must still be estimated, and the absolute-value bound throws away a cancellation that an estimator should keep.

**Setting used on this page.**

- The complete learner state $`z\in\mathcal Z`$, feedback laws, the actual update $`U_t(z,y,\xi_t)`$ and the learning kernels are those of [Definition 1.3](../README.md#s1-3). $`K_t^0`$ is the reference protocol, $`K_t^S`$ an alternative protocol $`S`$; both start from the same initial law. $`\mathbb E_0`$ and $`\mathbb E_S`$ are expectations along the reference and alternative chains $`Z_t^0`$, $`Z_t^S`$.
- $`F_E(z)\in[0,1]`$ is the final real-task score and $`L_E(S)=\mathbb E_SF_E(Z_R)`$ ([Definition 1.4](../README.md#s1-4)).
- An *approximate value* is any family of bounded measurable functions $`\widehat V_0,\dots,\widehat V_R`$ on $`\mathcal Z`$, fixed before the comparison. It may come from a predictive-state model ([Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10)) or from anything else.
- The *reference Bellman residual*, the *terminal error* and the *local response difference* of $`\widehat V`$ are

```math
r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t,\qquad e_R=F_E-\widehat V_R,\qquad d_t=(K_t^0-K_t^S)\widehat V_{t+1}\quad(t<R). \qquad(3.6)
```

- When the two protocols share the update map and differ only in their feedback laws $`P_t^0,P_t^S`$, $`d_t(z)=\int\bar g_{t,z}\,d(P_t^0-P_t^S)(\cdot\mid z)`$ with the *approximate response* $`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$. The update inside $`\bar g`$ is the actual update; only the continuation is approximated.
- $`d_t^S`$ is the law of $`Z_t^S`$, the alternative's occupancy at round $`t`$, as in [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2). It is a measure on states and should not be confused with the local response difference $`d_t`$.

<a id="s3-4"></a>

**Proposition 3.4 (Joint decomposition with an approximate value).** In this setting:

**(a) Exact identity.**

```math
L_E(0)-L_E(S)=\mathbb E_S\sum_{t<R}d_t(Z_t)+\mathbb E_0\sum_{t<R}r_t^0(Z_t)-\mathbb E_S\sum_{t<R}r_t^0(Z_t)+\mathbb E_0e_R(Z_R)-\mathbb E_Se_R(Z_R). \qquad(3.7)
```

**(b) Absolute bound.** If $`\lvert d_t(z)\rvert\le\beta_t(z)`$ for $`d_t^S`$-almost every $`z`$, then

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_{t<R}\mathbb E_S\beta_t(Z_t)+\sum_{t<R}\big(\mathbb E_0\lvert r_t^0(Z_t)\rvert+\mathbb E_S\lvert r_t^0(Z_t)\rvert\big)+\mathbb E_0\lvert e_R(Z_R)\rvert+\mathbb E_S\lvert e_R(Z_R)\rvert. \qquad(3.8)
```

**(c) Closure-residual form.** If $`\widehat V_t=c_t^\top\psi_t`$ for bounded features $`\psi_t:\mathcal Z\to\mathbb R^{d_t}`$ and the reference kernel satisfies $`K_t^0\psi_{t+1}=A_t\psi_t+\epsilon_t`$ componentwise, with matrices $`A_t`$ and residual functions $`\epsilon_t`$, then

```math
r_t^0=\big(A_t^\top c_{t+1}-c_t\big)^\top\psi_t+c_{t+1}^\top\epsilon_t .
```

**(d) Stopped version.** Let $`\widehat V_t\in[0,1]`$ for every $`t`$. For each process $`j\in\lbrace0,S\rbrace`$ choose certified sets $`C_t^j\subseteq\mathcal Z`$ and let $`\tau^j`$ be the first $`t<R`$ with $`Z_t^j\notin C_t^j`$ on that process's own chain ($`\tau^j=R`$ if there is none). If $`\lvert d_t(z)\rvert\le\beta_t(z)`$ for $`z\in C_t^S`$, then

```math
\begin{aligned}
\lvert L_E(0)-L_E(S)\rvert\le{}&\mathbb E_S\sum_{t<\tau^S}\big(\beta_t+\lvert r_t^0\rvert\big)(Z_t)+\mathbb E_0\sum_{t<\tau^0}\lvert r_t^0(Z_t)\rvert\\
&+\mathbb E_0\big[\lvert e_R(Z_R)\rvert\mathbf 1\lbrace\tau^0=R\rbrace\big]+\mathbb E_S\big[\lvert e_R(Z_R)\rvert\mathbf 1\lbrace\tau^S=R\rbrace\big]+\Pr_0(\tau^0<R)+\Pr_S(\tau^S<R).
\end{aligned} \qquad(3.9)
```

Taking $`C_t^0=\mathcal Z`$ removes the reference exit term.

**What it says.**

- With any approximate continuation value, the learning gap is exactly a local response term on the alternative's visited states, plus the difference of the reference Bellman residuals accumulated along the two processes, plus the difference of terminal errors.
- The response term uses the actual update; only the continuation is replaced by $`\widehat V`$.
- When $`\widehat V`$ is linear in predictive features, its residual splits into a value-recursion error and a feature-closure error.
- When guarantees hold only on certified sets, each process pays its own exit probability once.

**Intuition.** Theorem 3.2 measures each round's replacement with the exact continuation. Using an approximate continuation is legitimate as long as one also pays for how badly it fails to satisfy the reference recursion, on both processes, because the two processes visit different states.

**Role in the framework.**

- It is the bookkeeping tool behind every composed guarantee: [Corollary 8.16](08-16_composed_guarantee.md#s8-16) uses part (d), inserting the interface and estimation errors of §8 into the response term $`d_t`$. When $`\widehat V`$ is built from predictive features, the closure residual of [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11) enters $`r_t^0`$ through (c).
- It does not make any term small. Estimating all of its terms can cost more than running the two protocols and comparing them directly, and (b) is loose: the two residual sums enter with opposite signs in (3.7), and taking absolute values separately discards that cancellation. An estimator should estimate the signed terms of (3.7) with paired runs.

**Derivation outline.**

- For each process $`j`$, telescope $`\widehat V_t(Z_t^j)`$ from $`t=0`$ to $`R`$. By the Markov property each increment has conditional mean $`(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t^j)`$, which is $`r_t^0`$ for $`j=0`$ and $`r_t^0-d_t`$ for $`j=S`$. Subtract the two expansions; the common initial term cancels. This gives (a), and the triangle inequality gives (b).
- (c) is linearity of $`K_t^0`$ applied to $`c_{t+1}^\top\psi_{t+1}`$.
- (d) stops each telescoping sum at its own exit time. Before the exit the increments are controlled as in (a); after it, the remaining difference $`F_E(Z_R)-\widehat V_{\tau}(Z_{\tau})`$ lies in $`[-1,1]`$ because both functions take values in $`[0,1]`$.

**Scope.**

- With $`\widehat V=V^0`$ every residual and terminal error vanishes; (a) becomes (3.2) and (b) becomes (3.3). For stopped bounds with the exact value and the reference certified sets equal to the whole state space, (3.9) coincides with (3.5): both pay only the alternative's exit.
- If $`\widehat V`$ is clipped to $`[0,1]`$ for (d), the residuals must be recomputed for the clipped function; clipping does not preserve the linear form used in (c).
- Residuals measured only on an audit distribution need a separate transport or coverage term to reach the visited states ([Theorem 8.2](08-02_frozen_finite_interface_selection_transport.md#s8-2)).

**Uses.**

- [Definitions 1.3–1.4](../README.md#s1-3): learner state, kernels, continuation value.
- [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2): the exact case and the stopping argument.

**Used by.**

- [Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11): closure residuals as Bellman residuals.
- [Corollary 8.16](08-16_composed_guarantee.md#s8-16): the composed guarantee from reuse, estimation and compilation.

## Related work

- The identity belongs to the performance-difference and simulation-lemma family of reinforcement learning ([Kakade & Langford, 2002](#ref-kakade2002approximately)), here with an arbitrary approximate value and with training processes in place of policies. It is a bookkeeping identity, not a new lemma.
- Combining a value model with residual corrections along the evaluated process is the structure of doubly robust off-policy evaluation ([Jiang & Li, 2016](#ref-jiang2016doubly)), which also gives matching lower bounds in some settings. A cost theory for the value-model route of this framework must be compared with that line.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
