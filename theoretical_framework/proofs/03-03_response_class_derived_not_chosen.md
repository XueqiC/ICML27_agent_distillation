# Remark 3.3 · The response class is derived, not chosen

*Proof*

[← Framework](../README.md) · [Statement](../statements/03-03_response_class_derived_not_chosen.md) · [Proof](03-03_response_class_derived_not_chosen.md) · [中文](03-03_response_class_derived_not_chosen_ZH.md)

**What is proved here.** Every response has the generated form by the definitions of the continuation value and the response, and every function of that form is a response; the inclusion of response classes under enlargement of the family gives the refinement.

**Setting used on this page.**

- The complete learner state $`z\in\mathcal Z`$: parameters, optimizer state, buffers, sampler state, randomness and history ([Definition 1.3](../README.md#s1-3)).
- At round $`t<R`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$ and the actual update gives $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$, with update randomness $`\xi_t`$. Integrating $`Y_t`$ and $`\xi_t`$ gives the learning kernel $`K_t`$, acting on bounded functions by $`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$.
- $`K_t^0`$ is the reference protocol and $`K_t^S`$ an alternative protocol $`S`$; both start from the same initial law $`\nu`$. $`Z_t^S`$ is the chain driven by $`K^S`$, and its law $`d_t^S`$ is the occupancy at round $`t`$.
- $`F_E(z)\in[0,1]`$ is the real-task return of the student deployed from state $`z`$; $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ is the learning value of protocol $`S`$, and $`L_E(0)`$ that of the reference.
- The reference continuation value is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$: the expected final score when training continues from a state under the reference protocol ([Definition 1.4](../README.md#s1-4)).
- The one-boundary response at a snapshot $`(t,z)`$ is $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$: the expected final return after message $`y`$, the actual update and the declared reference tail. The response class $`\mathcal G`$ collects $`g_{t,z}`$ over a declared family $`\mathcal I`$ of snapshots.

**Statement ([Remark 3.3](../statements/03-03_response_class_derived_not_chosen.md#s3-3)).** The class $`\mathcal G`$ is generated recursively from the declared objects:

- start from $`\mathcal V_R=\lbrace F_E\rbrace`$;
- apply the reference kernels, $`\mathcal V_t=K_t^0\mathcal V_{t+1}`$;
- pull back through the actual update, $`g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)`$ with $`v\in\mathcal V_{t+1}`$.

Enlarging the declared task, learner, suffix or snapshot family can therefore only refine the equivalence (3.4) of [Theorem 3.2(b)](../statements/03-02_continuation_response_characterization.md#s3-2); the response class is fixed by the learner and the task.

**Proof.**

Every $`g\in\mathcal G`$ has this form by the definitions of $`V^0`$ and $`g_{t,z}`$, and every function of this form is the response of the corresponding snapshot. If $`\mathcal I\subseteq\mathcal I'`$ then $`\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')`$, so every law pair that satisfies (3.4) for $`\mathcal G(\mathcal I')`$ also satisfies it for $`\mathcal G(\mathcal I)`$. Enlarging the declared task, learner, suffix or snapshot family can therefore only refine the equivalence (3.4). The response class is fixed by the learner and the task; it is not a modelling convenience.

## Related work

- The inclusion argument is the one behind the monotonicity of value-equivalence classes in model-based reinforcement learning ([Grimm et al., 2020](#ref-grimm2020value), Property 3).
