# §3 Learning consequence — Proofs

[← Framework](../README.md) · [Experiments](../experiments/03_learning_consequence.md) · [中文](03_learning_consequence_ZH.md)

**Summary.**

- [Proposition 3.1](#s3-1) shows that static criteria do not determine what a student learns.
  - Two data sources can have equally good teachers and give every student in the class the same policy value, yet distill different students.
  - Two label packets can have the same marginals and the same mean gradient, yet different learning values, and the sign of the difference reverses on another task.
- [Theorem 3.2](#s3-2) identifies the invariant. The difference between two training processes is exactly a sum over rounds of one-boundary replacement effects, each determined by the continuation response. Two message laws are interchangeable for every one-boundary replacement if and only if they agree on the expectation of every response. When guarantees hold only on part of the state space, leaving it is paid once.
- [Remark 3.3](#s3-3) records that the response class is derived from the learner, the reference tail and the task, not chosen.
- **Uses.** The setting of [Definitions 1.3–1.4](../README.md#s1-3) (complete learner state, feedback law, learning kernel, continuation value, response); the decomposition of [Theorem 2.2](02_feasibility_and_acquisition.md#s2-2) (the gaps of Proposition 3.1 live in what the learner extracts, not in representation).
- **Used by.** [Theorem 4.2](04_complexity_of_sufficiency.md#s4-2) (the size of the response class), [Proposition 5.1](05_optimizer_state.md#s5-1) (its first-order expansion), [Theorem 8.2](08_verification.md#s8-2) and [Theorem 8.3](08_verification.md#s8-3) (finite and actual-state audits), [Proposition 9.1](09_mechanism_and_method.md#s9-1) (coupling responses), [Proposition 7.3(a)](07_second_moment_channel.md#s7-3) (task-blind statistics) and the method ([Remark 9.3](09_mechanism_and_method.md#s9-3)).

## Static evaluation does not determine learning

<a id="s3-1"></a>

**Proposition 3.1 (Static value is not learning value).**

*Construction (a): four-state sources.*

- States $`u,v,w,x`$ share the reward rules below; action $`B`$ always succeeds with probability $`\tfrac12`$.
- The reference source draws the state from $`\rho_0=(\tfrac34,\tfrac14,0,0)`$, the alternative source from $`\rho_1=(0,0,\tfrac14,\tfrac34)`$.
- An optimal teacher observes the state and plays its better action.
- The student is aliased: it has one context and plays $`A`$ with probability $`p`$, giving policy $`\pi_p`$.
- The learner is frequency distillation: from $`m\ge1`$ independent teacher demonstrations, $`\widehat p`$ is the fraction of $`A`$ actions (also the maximizer of empirical cross-entropy).

| State | $`\rho_0`$ | $`\rho_1`$ | $`r(A)`$ | $`r(B)`$ |
|---|---:|---:|---:|---:|
| $`u`$ | $`3/4`$ | $`0`$ | $`19/30`$ | $`1/2`$ |
| $`v`$ | $`1/4`$ | $`0`$ | $`3/10`$ | $`1/2`$ |
| $`w`$ | $`0`$ | $`1/4`$ | $`9/10`$ | $`1/2`$ |
| $`x`$ | $`0`$ | $`3/4`$ | $`13/30`$ | $`1/2`$ |

*Construction (b): two label packets.* A packet is a pair of binary labels $`(Y_1,Y_2)`$.

- Under the *independent* source, $`Y_1,Y_2`$ are IID Bernoulli $`(\tfrac12)`$. Under the *paired* source, $`(Y_1,Y_2)`$ is uniform on $`\lbrace (0,1),(1,0)\rbrace`$.
- The learner takes one gradient step of size $`0<\eta\le1`$ from $`p_0=\tfrac12`$ on the squared loss $`\tfrac14\sum_{i=1}^2(p-Y_i)^2`$:

```math
p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2 .
```

- Deployment is two independent decisions that both must choose action 1, so $`J(p)=p^2`$. The alternative task is $`J'(p)=1-p^2`$.

Then:

- **(a) Identical static values, different students.** In Construction (a), both sources give every aliased student the same value $`J(\pi_p)=\tfrac12+p/20`$ and give the teacher the same value $`\tfrac35`$. The teacher's $`A`$-frequencies are $`\tfrac34`$ and $`\tfrac14`$, and frequency distillation on $`m\ge1`$ demonstrations yields expected returns $`\tfrac{43}{80}`$ (reference) and $`\tfrac{41}{80}`$ (alternative): a gap of $`\tfrac1{40}`$ for every $`m`$.
- **(b) Identical mean updates, different learning values.** In Construction (b), the two sources have the same label marginals, the same mean gradient and the same mean update $`\mathbb Ep^+=\tfrac12`$, yet

```math
\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8, \qquad(3.1)
```

and under $`J'=1-p^2`$ the difference is $`-\eta^2/8`$.

*Proof.* **(a)**

- *Static values.* For $`\pi_p`$ under a state law $`\rho`$,

```math
J_\rho(\pi_p)=\sum_s\rho(s)\big[p\,r_s(A)+(1-p)\tfrac12\big]=\tfrac12+p\Big(\sum_s\rho(s)r_s(A)-\tfrac12\Big).
```

- For $`\rho_0`$, $`\sum_s\rho_0(s)r_s(A)=\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac3{10}=\tfrac{57}{120}+\tfrac9{120}=\tfrac{11}{20}`$. For $`\rho_1`$, $`\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac{13}{30}=\tfrac9{40}+\tfrac{13}{40}=\tfrac{11}{20}`$. Hence $`J_{\rho_0}(\pi_p)=J_{\rho_1}(\pi_p)=\tfrac12+\tfrac p{20}`$ for all $`p\in[0,1]`$.
- *Teacher.* The teacher plays $`A`$ where $`r(A)>\tfrac12`$, at $`u`$ ($`\tfrac{19}{30}`$) and $`w`$ ($`\tfrac9{10}`$), and $`B`$ at $`v`$ ($`\tfrac3{10}`$) and $`x`$ ($`\tfrac{13}{30}`$). Its value is $`\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac12=\tfrac35`$ under $`\rho_0`$ and $`\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac12=\tfrac35`$ under $`\rho_1`$. Its $`A`$-frequency is $`\rho_0(u)=\tfrac34`$ under $`\rho_0`$ and $`\rho_1(w)=\tfrac14`$ under $`\rho_1`$.
- *Learned students.* $`m\widehat p`$ is Binomial $`(m,f)`$ with $`f\in\lbrace \tfrac34,\tfrac14\rbrace`$, so $`\mathbb E\widehat p=f`$ for every $`m\ge1`$. The value is affine in $`p`$ and is the same function under both sources, so

```math
\mathbb EJ(\pi_{\widehat p})=\tfrac12+\tfrac{\mathbb E\widehat p}{20}=\tfrac12+\tfrac3{80}=\tfrac{43}{80}\quad\text{or}\quad\tfrac12+\tfrac1{80}=\tfrac{41}{80}.
```

**(b)**

- *Equal first moments.* Each $`Y_i`$ is Bernoulli $`(\tfrac12)`$ under both sources, so the marginals agree. The gradient at $`p_0`$ is $`p_0-(Y_1+Y_2)/2`$, which has mean $`0`$ under both sources. Hence $`\mathbb Ep^+=\tfrac12`$ for both.
- *Variances.* $`\mathrm{Var}\,p^+=\eta^2\mathrm{Var}\frac{Y_1+Y_2}2`$. Under independence this is $`\eta^2\cdot\tfrac14(\tfrac14+\tfrac14)=\eta^2/8`$. Under pairing $`Y_1+Y_2\equiv1`$, so it is $`0`$.
- *Values.* With $`J(p)=p^2`$, $`\mathbb EJ(p^+)=(\mathbb Ep^+)^2+\mathrm{Var}\,p^+`$, so the difference is $`\eta^2/8-0=\eta^2/8`$. For $`J'=1-p^2`$, $`\mathbb EJ'(p^+)=1-(\mathbb Ep^+)^2-\mathrm{Var}\,p^+`$, and the difference is $`-\eta^2/8`$. $`\square`$

**Comments on Proposition 3.1.**

- **What is equal and what is not.**
  - In (a) the static evaluation agrees on the whole restricted policy class $`\Pi=\lbrace \pi_p\rbrace`$ and on the teacher. Learning, however, consumes the teacher's *action frequencies*, which the values do not determine.
  - In (b) the label marginals and the mean gradient agree. The deployment's curvature, however, reads the packet's *second moment*.
  - In both cases the quantity that decides the outcome is the composition of the actual update with the deployment objective. Theorem 3.2 makes this composition the governing object.
- **Fidelity is not improvement.** The sign of (3.1) depends on the task: the same variance helps under $`p^2`$ and hurts under $`1-p^2`$. Lower gradient variance, higher fidelity and better task performance are distinct properties.
- **Scope.** (a) concerns the stated aliased class; it does not contradict full Bellman value equivalence over unrestricted policy and value tests. The gap is specific to the learning algorithm: a learner that ignores both datasets and outputs $`p=1`$ has no gap. That is the point: sufficiency must be defined relative to the declared learner.

## The continuation response

*Setting* (recalled from [Definitions 1.3–1.4](../README.md#s1-3)).

- The complete learner state $`z\in\mathcal Z`$ includes the parameters, the optimizer state, buffers, sampler state, the relevant randomness and the interaction history; full history may be used when no smaller Markov state is available. $`\mathcal Z`$ is finite or, more generally, a measurable space, and all functions below are bounded and measurable.
- At round $`t\in\lbrace 0,\dots,R-1\rbrace`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$, and the actual update is $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$, where $`\xi_t`$ is the declared update randomness.
- Integrating $`Y_t`$ and $`\xi_t`$ gives a Markov kernel $`K_t`$ on $`\mathcal Z`$, acting on bounded functions by $`(K_tf)(z)=\mathbb E[f(U_t(z,Y,\xi))]`$ with $`Y\sim P_t(\cdot\mid z)`$ and $`\xi`$ from its declared law. $`K_t^0`$ is the reference protocol and $`K_t^S`$ an alternative protocol $`S`$. Both processes start from the same initial law $`\nu`$.
- Every final student is scored by the same $`F_E:\mathcal Z\to[0,1]`$, the real task return of the deployed policy $`\pi_z`$.
- The reference continuation value is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$. The one-boundary response at a snapshot $`(t,z)`$ is

```math
g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0\big(U_t(z,y,\xi_t)\big).
```

- The learning value of a protocol is $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$, where $`Z^S`$ is the chain driven by $`K^S`$ from $`Z_0\sim\nu`$; $`L_E(0)`$ is the reference. The occupancy $`d_t^S`$ is the law of $`Z_t^S`$.
- A declared family is a finite or measurable set $`\mathcal I`$ of snapshots $`(t,z)`$, each with its declared reference suffix. The response class is $`\mathcal G=\lbrace g_{t,z}:(t,z)\in\mathcal I\rbrace`$.
- A *one-boundary replacement* $`I(t,z,P)`$ starts at state $`z`$ at round $`t`$, draws $`Y\sim P`$, applies the actual update $`U_t(z,Y,\xi_t)`$, follows the declared reference suffix to round $`R`$ and scores $`F_E`$.

<a id="s3-2"></a>

**Theorem 3.2 (Continuation-response characterization).** In this setting:

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

*Proof.* **(a)**

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

**Comments on Theorem 3.2.**

- **The object.** $`g_{t,z}(y)`$ is the expected *final real-task return* after receiving $`y`$, performing the *actual* update and completing the *declared reference learning tail*. It is not a training loss, a teacher confidence or a frozen-policy action value. §4 studies the size and approximation of $`\mathcal G`$, §5–§7 its dependence on the optimizer, and §8 its measurement.
- **Orientation and signs.** The occupancy in (3.2) is that of the *alternative* process, $`d_t^S`$. Each drift term can have either sign. Any difference in update rules between the two protocols must be included in the compared kernels.
- **Pointwise matching is sufficient, not necessary, for one fixed final mean.** Errors at different rounds can cancel. Take two states, $`K^0`$ the identity, $`K^S`$ the swap, two rounds, $`F=(0,1)`$ and initial state $`0`$: the final gap is $`0`$ while $`\sum_t\mathbb E\lvert\text{drift}\rvert=2`$. Part (b) quantifies over *all* one-boundary replacements, which is why matching every $`g`$ becomes necessary there.
- **Exit penalty.** The uncontrolled tail after the first exit contributes at most one bounded-return discrepancy, so the penalty is $`\rho`$, not $`R\rho`$. If $`Z_0\notin C_0`$, then $`\tau=0`$ and this event is part of $`\rho`$. A finite bank of stored checkpoints does not by itself make $`\rho`$ small for a continuous trajectory; [Theorem 8.2](08_verification.md#s8-2) makes coverage quantitative.
- **Responsive tails.** The reference suffix must react to the branch's own current student. Coupling random seeds across branches is legitimate. Replaying identical labels or trajectories where the protocol would query a changed student measures a different continuation.

<a id="s3-3"></a>

**Remark 3.3 (The response class is derived, not chosen).** The class $`\mathcal G`$ is generated recursively from the declared objects:

- start from $`\mathcal V_R=\lbrace F_E\rbrace`$;
- apply the reference kernels, $`\mathcal V_t=K_t^0\mathcal V_{t+1}`$;
- pull back through the actual update, $`g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)`$ with $`v\in\mathcal V_{t+1}`$.

Every $`g\in\mathcal G`$ has this form by the definitions of $`V^0`$ and $`g_{t,z}`$, and every function of this form is the response of the corresponding snapshot. If $`\mathcal I\subseteq\mathcal I'`$ then $`\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')`$, so every law pair that satisfies (3.4) for $`\mathcal G(\mathcal I')`$ also satisfies it for $`\mathcal G(\mathcal I)`$. Enlarging the declared task, learner, suffix or snapshot family can therefore only refine the equivalence (3.4). The response class is fixed by the learner and the task; it is not a modelling convenience.

## Related work

- The telescoping identity (3.2) is the performance-difference style decomposition used in reinforcement learning and in the analysis of Markov chains: a difference of terminal expectations under two kernels written as a sum of one-step kernel differences against the continuation value.
- The first-order expansion of the response in [Proposition 5.1](05_optimizer_state.md#s5-1) relates this object to influence functions and training-data attribution; the response itself is defined without any expansion.
