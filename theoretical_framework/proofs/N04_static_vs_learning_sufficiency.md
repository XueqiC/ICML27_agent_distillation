# N04 · Static value is not learning value — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N04_static_vs_learning_sufficiency.md) · [中文](N04_static_vs_learning_sufficiency_ZH.md)

## Setting and assumptions

**Construction 4.1 (Proposition 5: four-state sources).**

- States $u,v,w,x$ share the reward rules below. Action $B$ always has success probability $\tfrac12$.
- The reference source draws the state from $\rho_0=(\tfrac34,\tfrac14,0,0)$, and the alternative source from $\rho_1=(0,0,\tfrac14,\tfrac34)$.
- An optimal teacher observes the state and plays its better action.
- The student is aliased: it has one context and plays $A$ with probability $p$, giving policy $\pi_p$.
- The learner is frequency distillation: from $m\ge1$ independent teacher demonstrations, set $\widehat p$ to the fraction of $A$ actions. This is also the maximizer of empirical cross-entropy.

| State | $\rho_0$ | $\rho_1$ | $r(A)$ | $r(B)$ |
|---|---:|---:|---:|---:|
| $u$ | $3/4$ | $0$ | $19/30$ | $1/2$ |
| $v$ | $1/4$ | $0$ | $3/10$ | $1/2$ |
| $w$ | $0$ | $1/4$ | $9/10$ | $1/2$ |
| $x$ | $0$ | $3/4$ | $13/30$ | $1/2$ |

**Construction 4.2 (Proposition 6: two label packets).**

- A packet is a pair of binary labels $(Y_1,Y_2)$.
- Under the *independent* source, $Y_1,Y_2$ are IID Bernoulli$(\tfrac12)$. Under the *paired* source, $(Y_1,Y_2)$ is uniform on $\lbrace (0,1),(1,0)\rbrace $.
- The learner takes one gradient step of size $0<\eta\le1$ from $p_0=\tfrac12$ on the squared loss $\tfrac14\sum_{i=1}^2(p-Y_i)^2$:
$$p^+=p_0-\eta\big(p_0-\tfrac{Y_1+Y_2}2\big)=\frac{1-\eta}2+\eta\,\frac{Y_1+Y_2}2.$$
- Deployment is two independent decisions that both must choose action 1, so $J(p)=p^2$. The alternative task is $J'(p)=1-p^2$.

## Statement

**Proposition 4.1 (identical static values, different students).** In Construction 4.1:

- both sources give every aliased student the same value $J(\pi_p)=\tfrac12+p/20$;
- both give the teacher value $\tfrac35$;
- the teacher's $A$-frequencies are $\tfrac34$ and $\tfrac14$;
- frequency distillation on $m\ge1$ demonstrations yields expected returns
$$\mathbb E\,J(\pi_{\widehat p})=\tfrac{43}{80}\quad\text{(reference)},\qquad\tfrac{41}{80}\quad\text{(alternative)},\tag{10}$$
a gap of $\tfrac1{40}$ for every $m$.

**Proposition 4.2 (identical mean updates, different learning values).** In Construction 4.2:

- the two sources have the same label marginals, the same mean gradient, and the same mean update $\mathbb Ep^+=\tfrac12$;
- yet
$$\mathbb E_{\rm indep}J(p^+)-\mathbb E_{\rm paired}J(p^+)=\frac{\eta^2}8,\tag{11}$$
- and for $J'=1-p^2$ the difference is $-\eta^2/8$.

## Proof

**Proof of Proposition 4.1.**

*Static values.* For $\pi_p$ under a state law $\rho$,
$$J_\rho(\pi_p)=\sum_s\rho(s)\big[p\,r_s(A)+(1-p)\tfrac12\big]=\tfrac12+p\Big(\sum_s\rho(s)r_s(A)-\tfrac12\Big).$$
For $\rho_0$, $\sum_s\rho_0(s)r_s(A)=\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac3{10}=\tfrac{57}{120}+\tfrac9{120}=\tfrac{11}{20}$. For $\rho_1$, $\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac{13}{30}=\tfrac9{40}+\tfrac{13}{40}=\tfrac{11}{20}$. Hence $J_{\rho_0}(\pi_p)=J_{\rho_1}(\pi_p)=\tfrac12+\tfrac p{20}$ for all $p\in[0,1]$.

*Teacher.* The teacher plays $A$ where $r(A)>\tfrac12$, which is at $u$ ($\tfrac{19}{30}$) and $w$ ($\tfrac9{10}$). It plays $B$ at $v$ ($\tfrac3{10}$) and $x$ ($\tfrac{13}{30}$). Its value is
$$\tfrac34\cdot\tfrac{19}{30}+\tfrac14\cdot\tfrac12=\tfrac35\quad\text{under }\rho_0,\qquad\tfrac14\cdot\tfrac9{10}+\tfrac34\cdot\tfrac12=\tfrac35\quad\text{under }\rho_1.$$
Its $A$-frequency is $\rho_0(u)=\tfrac34$ under $\rho_0$ and $\rho_1(w)=\tfrac14$ under $\rho_1$.

*Learned students.* $m\widehat p$ is Binomial$(m,f)$ with $f\in\lbrace \tfrac34,\tfrac14\rbrace $, so $\mathbb E\widehat p=f$ for every $m\ge1$. The value is affine in $p$, and it is the same function under both sources, so
$$\mathbb EJ(\pi_{\widehat p})=\tfrac12+\tfrac{\mathbb E\widehat p}{20}=\tfrac12+\tfrac3{80}=\tfrac{43}{80}\quad\text{or}\quad\tfrac12+\tfrac1{80}=\tfrac{41}{80}.\qquad\square$$

**Proof of Proposition 4.2.**

*Equal first moments.* Each $Y_i$ is Bernoulli$(\tfrac12)$ under both sources, so the marginals agree. The gradient at $p_0$ is $p_0-(Y_1+Y_2)/2$, which has mean $0$ under both sources. Hence $\mathbb Ep^+=\tfrac12$ for both.

*Variances.* $\operatorname{Var}p^+=\eta^2\operatorname{Var}\frac{Y_1+Y_2}2$.

- Under independence, $\operatorname{Var}\frac{Y_1+Y_2}2=\tfrac14(\tfrac14+\tfrac14)=\tfrac18$.
- Under pairing, $Y_1+Y_2\equiv1$, so the variance is $0$.

*Values.* With $J(p)=p^2$, $\mathbb EJ(p^+)=(\mathbb Ep^+)^2+\operatorname{Var}p^+$. The difference is therefore $\eta^2\cdot\tfrac18-0=\eta^2/8$. For $J'=1-p^2$, $\mathbb EJ'(p^+)=1-(\mathbb Ep^+)^2-\operatorname{Var}p^+$, and the difference is $-\eta^2/8$. $\square$

## Remarks

1. **What is equal and what is not.**
   - In Proposition 4.1 the static evaluation agrees on the whole restricted policy class $\Pi=\lbrace \pi_p\rbrace $ and on the teacher. Learning, however, consumes the teacher's *action frequencies*, which the values do not determine.
   - In Proposition 4.2 the label marginals and the mean gradient agree. The deployment's curvature, however, reads the packet's *second moment*.
   - In both cases the quantity that decides the outcome is the composition of the actual update with the deployment objective. N05 makes this composition the governing object.
2. **Fidelity is not improvement.** The sign of (11) depends on the task: the same variance helps under $p^2$ and hurts under $1-p^2$. Lower gradient variance, higher fidelity and better task performance are distinct properties.
3. **Scope.** Proposition 4.1 concerns the stated aliased class. It does not contradict full Bellman value equivalence over unrestricted policy and value tests. The gap is specific to the learning algorithm: a learner that ignores both datasets and outputs $p=1$ has no gap. That is precisely the point, because sufficiency must be defined relative to the declared learner.

## Dependencies and source

- **Uses:** N01 (the decomposition: these gaps live in what the learner extracts, not in representation).
- **Used by:** N05 (motivates the continuation response as the invariant).
- **Source:** `docs/direction/v015_revision/Environment_Information_Theory_EN.md`, §4, Propositions 5–6, eqs. (10)–(11). Audit: `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md`, A9–A10.
