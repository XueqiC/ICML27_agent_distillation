# N05 · Continuation-response characterization — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N05_continuation_response.md) · [中文](N05_continuation_response_ZH.md)

## Setting and assumptions

**Assumption 5.1 (learning process).**

- The complete learner state $z\in\mathcal Z$ includes the parameters, optimizer moments, buffers, sampler state, the relevant randomness and the interaction history. Full history may be used when no smaller Markov state is available.
- $\mathcal Z$ is finite, or more generally a measurable space, with all functions below bounded and measurable.
- At round $t\in\lbrace 0,\dots,R-1\rbrace $, a feedback law $P_t(\cdot\mid z)$ emits a message $Y_t$. The actual update is $Z_{t+1}=U_t(Z_t,Y_t,\xi_t)$, where $\xi_t$ is the declared update randomness.
- Integrating $Y_t$ and $\xi_t$ gives a Markov kernel $K_t$ on $\mathcal Z$, acting on bounded functions by $(K_tf)(z)=\mathbb E[f(U_t(z,Y,\xi))]$ with $Y\sim P_t(\cdot\mid z)$ and $\xi$ from its declared law.
- $K_t^0$ is the reference protocol and $K_t^S$ an alternative. Both processes start from the same initial law $\nu$.
- Every final student is scored by the same $F_E:\mathcal Z\to[0,1]$, the real task return of the deployed policy $\pi_z$.

**Definitions.**

- Reference continuation: $V_R^0=F_E$ and $V_t^0=K_t^0V_{t+1}^0$ (12).
- Message-level response: $g_{E,f,t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))$ (13).
- Learning value: $L_E(S)=\mathbb E\,F_E(Z_R^S)$, where $Z^S$ is the chain driven by $K^S$ from $Z_0\sim\nu$.
- Occupancy: $d_t^S$ is the law of $Z_t^S$.

**Declared family.** A finite or measurable set $\mathcal I$ of snapshots $(t,z)$, each with its declared reference suffix. The response class is $\mathcal G=\lbrace g_{E,f,t,z}:(t,z)\in\mathcal I\rbrace $.

## Statement

**Theorem 5.1 (exact composition).**
$$L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big].\tag{14}$$
Consequently, if $\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)$ for $d_t^S$-almost every $z$,
$$\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t).\tag{15}$$

**Theorem 5.2 (operational characterization).** A *one-boundary replacement* $I(t,z,P)$ starts at state $z$ at round $t$, draws $Y\sim P$, applies the actual update $U_t(z,Y,\xi_t)$, follows the declared reference suffix to round $R$, and scores $F_E$. Two message laws $P,Q$ yield equal value for every one-boundary replacement in the declared family if and only if
$$\mathbb E_Pg=\mathbb E_Qg\quad\text{for every }g\in\mathcal G.\tag{16}$$

**Theorem 5.3 (stopped composition).** Let $C_t\subseteq\mathcal Z$ be certified sets. Let $\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace $, with $\tau=R$ if there is no such $t$, and let $\rho=\Pr(\tau<R)$. If $\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)$ for all $z\in C_t$, then
$$\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho.\tag{17}$$

**Proposition 5.4 (derived response class).** $\mathcal G$ is generated recursively:

- start from $\mathcal V_R=\lbrace F_E\rbrace $;
- apply the reference kernels, $\mathcal V_t=K_t^0\mathcal V_{t+1}$;
- pull back through the actual update, $g=\mathbb E_\xi\,v\circ U_t(z,\cdot,\xi)$ with $v\in\mathcal V_{t+1}$.

Enlarging the declared task, learner, suffix or snapshot family can only enlarge $\mathcal G$, and therefore can only refine the equivalence (16).

## Proof

**Proof of Theorem 5.1.**

*Endpoints.* Iterating (12) gives $V_0^0=K_0^0K_1^0\cdots K_{R-1}^0F_E$, hence $L_E(0)=\nu V_0^0=\mathbb E\,V_0^0(Z_0^S)$; here the common initial law is used. Also $L_E(S)=\mathbb EF_E(Z_R^S)=\mathbb EV_R^0(Z_R^S)$.

*Telescoping.*
$$L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\Big(\mathbb EV_t^0(Z_t^S)-\mathbb EV_{t+1}^0(Z_{t+1}^S)\Big).$$

*Each term.* By the Markov property of the $S$ chain, $\mathbb E[V_{t+1}^0(Z_{t+1}^S)\mid Z_t^S]=(K_t^SV_{t+1}^0)(Z_t^S)$. By (12), $V_t^0=K_t^0V_{t+1}^0$. Each term therefore equals $\mathbb E[(K_t^0-K_t^S)V_{t+1}^0(Z_t^S)]$, which is (14).

*Bound.* Applying the triangle inequality inside the sum gives (15). $\square$

**Proof of Theorem 5.2.**

*Value of a replacement.* Fix $(t,z)\in\mathcal I$ and a law $P$. After the boundary, the state is $Z_{t+1}=U_t(z,Y,\xi_t)$, and the remaining process follows the reference kernels. By the Markov property and (12), the expected terminal score given $Z_{t+1}$ is $V_{t+1}^0(Z_{t+1})$. Integrating over $\xi_t$ and $Y\sim P$,
$$\mathrm{val}\,I(t,z,P)=\sum_yP(y)\,\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))=\mathbb E_P\,g_{E,f,t,z}.$$
Hence $\mathrm{val}\,I(t,z,P)-\mathrm{val}\,I(t,z,Q)=(P-Q)g_{E,f,t,z}$.

- *Sufficiency.* If (16) holds, every difference vanishes.
- *Necessity.* If $(P-Q)g\ne0$ for some $g=g_{E,f,t,z}\in\mathcal G$, then the replacement $I(t,z,\cdot)$ is itself a member of the family with unequal values.

The same argument applies with state-dependent laws $P_{t,z},Q_{t,z}$. $\square$

**Proof of Theorem 5.3.** $\tau$ is a stopping time: $\lbrace \tau>t\rbrace =\lbrace Z_0\in C_0,\dots,Z_t\in C_t\rbrace $ is $\sigma(Z_0,\dots,Z_t)$-measurable. Decompose
$$L_E(0)-L_E(S)=\mathbb E\big[V_0^0(Z_0)-V_\tau^0(Z_\tau)\big]+\mathbb E\big[V_\tau^0(Z_\tau)-F_E(Z_R)\big].$$

*First term.* Write it as $\mathbb E\sum_{t=0}^{R-1}1\lbrace t<\tau\rbrace \big(V_t^0(Z_t)-V_{t+1}^0(Z_{t+1})\big)$. Since $1\lbrace t<\tau\rbrace $ is $\sigma(Z_{0:t})$-measurable, conditioning on $Z_{0:t}$ gives
$$\mathbb E\big[1\lbrace t<\tau\rbrace (V_t^0(Z_t)-V_{t+1}^0(Z_{t+1}))\big]=\mathbb E\big[1\lbrace t<\tau\rbrace (K_t^0-K_t^S)V_{t+1}^0(Z_t)\big].$$
On $\lbrace t<\tau\rbrace $ we have $Z_t\in C_t$, so this is at most $\mathbb E[1\lbrace t<\tau\rbrace \varepsilon_t(Z_t)]$ in absolute value.

*Second term.* On $\lbrace \tau=R\rbrace $, $V_R^0(Z_R)-F_E(Z_R)=0$. On $\lbrace \tau<R\rbrace $, both $V_\tau^0(Z_\tau)$ and $F_E(Z_R)$ lie in $[0,1]$, so their difference lies in $[-1,1]$. Hence the second term is at most $\Pr(\tau<R)=\rho$ in absolute value.

Adding the two bounds gives (17). $\square$

**Proof of Proposition 5.4.** By (12) and (13), every $g\in\mathcal G$ has the stated form. Conversely, every function of that form is $g_{E,f,t,z}$ for the corresponding snapshot. If $\mathcal I\subseteq\mathcal I'$, then $\mathcal G(\mathcal I)\subseteq\mathcal G(\mathcal I')$. Every law pair that satisfies (16) for $\mathcal G(\mathcal I')$ therefore satisfies it for $\mathcal G(\mathcal I)$. $\square$

## Remarks

1. **The object.** $g$ is the expected *final real-task return* after receiving $y$, performing the *actual* update, and completing the *declared reference learning tail*. It is not a training loss, a teacher confidence or a frozen-policy action value. Every later node (N06–N16) studies the size, approximation and measurement of $\mathcal G$.
2. **Orientation and signs.** The occupancy in (14) is that of the *alternative* process, $d_t^S$. Each drift term can have either sign. Any difference in update rules between the interfaces must be included in the compared kernels.
3. **Pointwise matching is sufficient, not necessary, for one fixed final mean.** Errors at different rounds can cancel. Take two states, $K^0=$ identity, $K^S=$ swap for two rounds, $F=(0,1)$ and initial state $0$. The final gap is $0$ while $\sum_t\mathbb E\lvert\text{drift}\rvert=2$. Theorem 5.2 quantifies over *all* one-boundary replacements, which is why matching every $g$ becomes necessary there.
4. **Exit penalty.** The uncontrolled tail after the first exit contributes at most one bounded-return discrepancy, so the penalty is $\rho$ and not $R\rho$. If $Z_0\notin C_0$, then $\tau=0$ and this event is part of $\rho$. A finite bank of stored checkpoints does not by itself imply that $\rho$ is small for a continuous trajectory; N14 makes this quantitative.
5. **Responsive tails.** The reference suffix must react to the branch's own current student. Coupling random seeds across branches is legitimate. Replaying identical labels or trajectories where the protocol would query a changed student measures a different continuation.

## Dependencies and source

- **Uses:** N04 (why static and mean-update invariants are insufficient).
- **Used by:** N06 (size of $\mathcal G$), N14 (finite audits with transport), N15 (actual-state audits), N16 (coupling responses).
- **Source:** `docs/direction/v015_revision/Environment_Information_Theory_EN.md`, §5, Theorem 7, eqs. (12)–(17), §5.2. Audit: `docs/direction/v014_handoff/Independent_Proof_Audit_EN.md`, A11–A14.
