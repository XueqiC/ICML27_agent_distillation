# Theorem 3.2 · Continuation-response characterization

*Proof*

[← Framework](../README.md) · [Statement](../statements/03-02_continuation_response_characterization.md) · [Proof](03-02_continuation_response_characterization.md) · [Experiments](../experiments/03-02_continuation_response_characterization.md) · [中文](03-02_continuation_response_characterization_ZH.md)

**What is proved here.** Part (a) is a telescoping sum over rounds combined with the Markov property of the alternative chain. Part (b) computes the value of a one-boundary replacement as the expectation of the response. Part (c) stops the telescoping at the first exit from the certified sets and charges the rest of training at most one bounded return difference.

**Setting used on this page.**

- The complete learner state $`z\in\mathcal Z`$: parameters, optimizer state, buffers, sampler state, randomness and history ([Definition 1.3](../README.md#s1-3)).
- At round $`t<R`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$ and the actual update gives $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$, with update randomness $`\xi_t`$. Integrating $`Y_t`$ and $`\xi_t`$ gives the learning kernel $`K_t`$, acting on bounded functions by $`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$.
- $`K_t^0`$ is the reference protocol and $`K_t^S`$ an alternative protocol $`S`$; both start from the same initial law $`\nu`$. $`Z_t^S`$ is the chain driven by $`K^S`$, and its law $`d_t^S`$ is the occupancy at round $`t`$.
- $`F_E(z)\in[0,1]`$ is the real-task return of the student deployed from state $`z`$; $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ is the learning value of protocol $`S`$, and $`L_E(0)`$ that of the reference.
- The reference continuation value is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$: the expected final score when training continues from a state under the reference protocol ([Definition 1.4](../README.md#s1-4)).
- The one-boundary response at a snapshot $`(t,z)`$ is $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$: the expected final return after message $`y`$, the actual update and the declared reference tail. The response class $`\mathcal G`$ collects $`g_{t,z}`$ over a declared family $`\mathcal I`$ of snapshots.
- A one-boundary replacement $`I(t,z,P)`$: from state $`z`$ at round $`t`$, draw $`Y\sim P`$, apply the actual update, follow the declared reference tail, score $`F_E`$.
- Certified sets $`C_t\subseteq\mathcal Z`$, the first exit time $`\tau`$ and the exit probability $`\rho=\Pr(\tau<R)`$ are defined in part (c).

**Statement ([Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)).** In this setting:

**(a) Exact composition.**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(3.2)
```

Consequently, if $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for $`d_t^S`$-almost every $`z`$,

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(3.3)
```

**(b) Operational characterization.** Two message laws $`P,Q`$ yield equal value for every one-boundary replacement in the declared family if and only if

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{for every }g\in\mathcal G. \qquad(3.4)
```

**(c) Stopped composition.** Let $`C_t\subseteq\mathcal Z`$ be certified sets, $`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$ (with $`\tau=R`$ if there is no such $`t`$), and $`\rho=\Pr(\tau<R)`$. If $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for all $`z\in C_t`$, then

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(3.5)
```

**Proof.**

**(a)**

- *Endpoints.* Iterating the recursion $`V_t^0=K_t^0V_{t+1}^0`$ gives $`V_0^0=K_0^0K_1^0\cdots K_{R-1}^0F_E`$, hence $`L_E(0)=\nu V_0^0=\mathbb E\,V_0^0(Z_0^S)`$, using the common initial law. Also $`L_E(S)=\mathbb EF_E(Z_R^S)=\mathbb EV_R^0(Z_R^S)`$.
- *Telescoping.*

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\Big(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S)\Big).
```

- *Each term.* By the Markov property of the $`S`$ chain, $`\mathbb E[V_{t+1}^0(Z_{t+1}^S)\mid Z_t^S]=(K_t^SV_{t+1}^0)(Z_t^S)`$, and $`V_t^0=K_t^0V_{t+1}^0`$ by definition. Each term therefore equals $`\mathbb E[(K_t^0-K_t^S)V_{t+1}^0(Z_t^S)]`$, which is (3.2).
- *Bound.* The triangle inequality inside the sum gives (3.3).

**(b)** Fix $`(t,z)\in\mathcal I`$ and a law $`P`$. After the boundary the state is $`Z_{t+1}=U_t(z,Y,\xi_t)`$, and the remaining process follows the reference kernels. By the Markov property and the definition of $`V^0`$, the expected terminal score given $`Z_{t+1}`$ is $`V_{t+1}^0(Z_{t+1})`$. Integrating over $`\xi_t`$ and $`Y\sim P`$,

```math
\mathrm{val}\,I(t,z,P)=\sum_yP(y)\,\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))=\mathbb E_P\,g_{t,z}.
```

Hence $`\mathrm{val}\,I(t,z,P)-\mathrm{val}\,I(t,z,Q)=(P-Q)g_{t,z}`$.

- *Sufficiency.* If (3.4) holds, every difference vanishes.
- *Necessity.* If $`(P-Q)g\ne0`$ for some $`g=g_{t,z}\in\mathcal G`$, the replacement $`I(t,z,\cdot)`$ is itself a member of the family with unequal values.

The same argument applies with state-dependent laws $`P_{t,z},Q_{t,z}`$.

**(c)** $`\tau`$ is a stopping time: $`\lbrace \tau>t\rbrace =\lbrace Z_0\in C_0,\dots,Z_t\in C_t\rbrace`$ is $`\sigma(Z_0,\dots,Z_t)`$-measurable. Decompose

```math
L_E(0)-L_E(S)=\mathbb E\big[V_0^0(Z_0)-V_\tau^0(Z_\tau)\big]+\mathbb E\big[V_\tau^0(Z_\tau)-F_E(Z_R)\big].
```

- *First term.* Write it as $`\mathbb E\sum_{t=0}^{R-1}1\lbrace t<\tau\rbrace \big(V_t^0(Z_t)-V_{t+1}^0(Z_{t+1})\big)`$. Since $`1\lbrace t<\tau\rbrace`$ is $`\sigma(Z_{0:t})`$-measurable, conditioning on $`Z_{0:t}`$ gives

```math
\mathbb E\big[1\lbrace t<\tau\rbrace (V_t^0(Z_t)-V_{t+1}^0(Z_{t+1}))\big]=\mathbb E\big[1\lbrace t<\tau\rbrace (K_t^0-K_t^S)V_{t+1}^0(Z_t)\big].
```

On $`\lbrace t<\tau\rbrace`$ we have $`Z_t\in C_t`$, so this is at most $`\mathbb E[1\lbrace t<\tau\rbrace \varepsilon_t(Z_t)]`$ in absolute value.

- *Second term.* On $`\lbrace \tau=R\rbrace`$, $`V_R^0(Z_R)-F_E(Z_R)=0`$. On $`\lbrace \tau<R\rbrace`$, both $`V_\tau^0(Z_\tau)`$ and $`F_E(Z_R)`$ lie in $`[0,1]`$, so their difference lies in $`[-1,1]`$. Hence the second term is at most $`\Pr(\tau<R)=\rho`$ in absolute value.

Adding the two bounds gives (3.5). $`\square`$

## Related work

- The telescoping identity is of the performance-difference type used in reinforcement learning ([Kakade & Langford, 2002](../references.md#kakade2002approximately)).

## References

- [Kakade & Langford, 2002](../references.md#kakade2002approximately)
