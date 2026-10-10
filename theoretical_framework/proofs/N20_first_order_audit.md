# N20 · Measuring the first-order statistic at the actual states — Proof

[← Framework map](../README.md) · [Experiments](../experiments/N20_first_order_audit.md) · [中文](N20_first_order_audit_ZH.md)

## Setting and assumptions

Use the setting of node [N18](N18_optimizer_transformed_response.md): complete state $x$, consumed gradient $G_t(x,Y,\xi)$, increment $\Delta_t(x,g)$, mean increment $\mu^P_t(x)$, second increment moment $m^P_{2,t}(x)$, and continuation $V^P_{t+1}$. In this node the norm on $\mathbb R^{p+q}$ is Euclidean, so the space is a real Hilbert space and the dual norm is Euclidean.

**Assumption B (bounded increments).** There is $b<\infty$ with $\lVert\Delta_t(x,G_t(x,Y,\xi))\rVert\le b$ almost surely, for the compared laws at the audited states. Examples:

- clipped SGD on the parameter block: $b=\eta c$;
- sign-type updates on $p$ parameters: $b=\eta\sqrt p$;
- SGD with gradients bounded by $\bar G$: $b=\eta\bar G$.

**Audit samples.** At an audited state $x$, draw $n_P$ independent messages $Y\sim P_t(\cdot\mid x)$ and $n_Q$ independent messages $Y\sim Q_t(\cdot\mid x)$, each with independent update randomness $\xi$. Compute their increments $X=\Delta_t(x,G_t(x,Y,\xi))$. The empirical means are $\bar X_P,\bar X_Q$ and the empirical second moments are $\hat m^P_2,\hat m^Q_2$. Producing them takes $n_P+n_Q$ consumed-gradient evaluations at $x$, and no training tail.

## Statement

**Theorem C5 (dimension-free estimation).** Let $X_1,\dots,X_n$ be IID in a real Hilbert space with $\lVert X\rVert\le b$ almost surely and mean $\mu$. With probability at least $1-\delta$,
$$\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).$$
For two independent samples of sizes $n_P$ and $n_Q$, with probability at least $1-\delta$,
$$\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).$$
Neither bound depends on the dimension.

**Corollary C5′ (first-order audit at an actual state).** Fix an audited state $x$, conditionally on the history that produced it. Suppose Assumptions B and S$_1$ hold there, with constants $\Lambda=\lVert\nabla V^P_{t+1}(x)\rVert$ and $M_2$. With probability at least $1-\delta$ over the audit samples,
$$\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}2\big(\hat m^P_2+\hat m^Q_2+e_2\big),$$
where
$$e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).$$

**Corollary C5″ (run-level audit along actual runs).** Draw $J$ independent runs of protocol $Q$. At every round $t$ of every run, apply C5′ at the visited state with confidence $\delta_{\rm a}/(JR)$, and let $\hat\varepsilon_{j,t}$ be the right-hand side, capped at $1$. Suppose S$_1$ and B hold at every visited state with the stated constants, and that audit samples are drawn without changing the run. Then with probability at least $1-\delta_{\rm a}-\delta_{\rm mc}$,
$$\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.$$

**What is cheap and what is not.** The data-dependent parts are $\bar X_P-\bar X_Q$ and $\hat m_2$. Each needs only consumed-gradient evaluations at the visited state, $n_P+n_Q$ of them per state. The constants $\Lambda$ and $M_2$ are properties of the continuation, not of the messages, and C5 does not estimate them. They must be bounded analytically, or calibrated against a small number of full-tail audits (node [N15](N15_adaptive_audit.md)), with that calibration error added to the bound.

**Theorem C6 (selection correctness by first-order margin).**

- *One boundary.* At a state $x$, take candidate laws $P_j,P_k$ and one continuation $V$ satisfying S$_1$ with constant $M_2$. If
$$\big\lvert\langle\nabla V(x),\mu^{j}-\mu^{k}\rangle\big\rvert>\tfrac{M_2}2\big(m_2^{j}+m_2^{k}\big),$$
then $\operatorname{sign}\big((K^{P_j}-K^{P_k})V(x)\big)=\operatorname{sign}\langle\nabla V(x),\mu^j-\mu^k\rangle$.
- *Run level.* Let $T_{jk}=\sum_t\mathbb E_{d_t^{k}}\langle\nabla V^{j}_{t+1}(X_t),\mu^j_t-\mu^k_t\rangle$ and $B_{jk}=\sum_t\mathbb E_{d_t^{k}}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})$, the first-order term and remainder bound of Theorem C2 with $P_j$ in the role of $P$. If $\lvert T_{jk}\rvert>B_{jk}$, then $\operatorname{sign}\big(L(P_j)-L(P_k)\big)=\operatorname{sign}(T_{jk})$.

**Corollary C6′ (when a cheap score must rank correctly).** Let a score $s(P)$ be used to rank feedback laws at state $x$ for continuation $V$. Suppose:

- (i) $s(P)-s(P_{\rm ref})=\langle\nabla V(x),\mu^P-\mu^{P_{\rm ref}}\rangle$, so the score is the optimizer-transformed mean increment of node [N19](N19_learner_statistics.md) paired with the continuation gradient at the actual state;
- (ii) the margin condition of C6 holds for the pair being compared.

Then the score ranks the pair correctly. A score that drops either requirement has no such guarantee:

- one using the raw mean gradient, or one fitted with coefficients shared across states and tasks, violates (i), because $\nabla V(x)$ changes with the state and the task;
- any score used where the margin fails violates (ii).

## Proof

**Proof of Theorem C5.** Let $f(X_1,\dots,X_n)=\lVert\bar X-\mu\rVert$.

*Mean.* By Jensen's inequality and independence,
$$\mathbb Ef\le(\mathbb Ef^2)^{1/2}=\Big(\frac{\mathbb E\lVert X-\mu\rVert^2}{n}\Big)^{1/2}.$$
In a Hilbert space, $\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2$, so $\mathbb Ef\le b/\sqrt n$. The cross terms $\mathbb E\langle X_i-\mu,X_l-\mu\rangle$, $i\ne l$, vanish by independence, which gives the variance identity used here.

*Deviation.* Replacing $X_i$ by $X_i'$ changes $\bar X$ by $(X_i'-X_i)/n$. By the triangle inequality it changes $f$ by at most $\lVert X_i-X_i'\rVert/n\le2b/n$. McDiarmid's bounded-differences inequality with $c_i=2b/n$ gives
$$\Pr\lbrace f-\mathbb Ef\ge s\rbrace\le\exp\Big(-\frac{2s^2}{\sum_ic_i^2}\Big)=\exp\Big(-\frac{ns^2}{2b^2}\Big).$$
Setting this equal to $\delta$ gives $s=b\sqrt{2\log(1/\delta)/n}$. Hence $f\le b/\sqrt n+b\sqrt{2\log(1/\delta)/n}$ with probability at least $1-\delta$.

*Two samples.* Apply the one-sample bound to each sample with $\delta/2$, use the triangle inequality
$$\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\rVert\le\lVert\bar X_P-\mu^P\rVert+\lVert\bar X_Q-\mu^Q\rVert,$$
and take a union bound. $\square$

**Proof of Corollary C5′.** Allocate $\delta/2$ to the two-sample bound of C5, which gives $e_1$ with $\log(2/(\delta/2))=\log(4/\delta)$. Allocate $\delta/4$ to each second moment. The variables $\lVert X\rVert^2$ lie in $[0,b^2]$, so the one-sided Hoeffding inequality gives $m_2^P\le\hat m_2^P+b^2\sqrt{\log(4/\delta)/(2n_P)}$ with probability at least $1-\delta/4$, and likewise for $Q$. On the intersection of the three events, which has probability at least $1-\delta$:

- $\lVert\mu^P-\mu^Q\rVert\le\lVert\bar X_P-\bar X_Q\rVert+e_1$, by the triangle inequality;
- $m_2^P+m_2^Q\le\hat m_2^P+\hat m_2^Q+e_2$.

Theorem C1 of node N18 gives $\lvert(K^P-K^Q)V(x)\rvert\le\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\rvert+\tfrac{M_2}2(m_2^P+m_2^Q)$. Cauchy–Schwarz gives $\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\rvert\le\Lambda\lVert\mu^P-\mu^Q\rVert$. Substitute the two inequalities. The samples are drawn after the state is fixed, so the argument holds conditionally on the history. $\square$

**Proof of Corollary C5″.** Let $\varepsilon_t(x)=\min\lbrace1,\lvert(K^P_t-K^Q_t)V^P_{t+1}(x)\rvert\rbrace$.

*Audit event.* For each audit $(j,t)$, C5′ holds conditionally on the history with probability at least $1-\delta_{\rm a}/(JR)$. The conditional failure probabilities sum to at most $\delta_{\rm a}$, so with probability at least $1-\delta_{\rm a}$ every audit succeeds. Then $\varepsilon_t(X_{j,t})\le\hat\varepsilon_{j,t}$ for all $j,t$, using also $\varepsilon_t\le1$.

*Telescoping.* The telescoping identity (14) (node [N05](N05_continuation_response.md)), with $P$ as the reference and the triangle inequality inside the expectation, gives $\lvert L(P)-L(Q)\rvert\le\sum_t\mathbb E_{d_t^Q}\varepsilon_t(X_t)$. The cap at $1$ is valid because each term is a difference of two $[0,1]$-valued expectations.

*Monte Carlo.* The variables $S_j=\sum_t\varepsilon_t(X_{j,t})$, $j=1,\dots,J$, are IID in $[0,R]$, with mean $\sum_t\mathbb E_{d_t^Q}\varepsilon_t$. The one-sided Hoeffding inequality gives
$$\sum_t\mathbb E_{d^Q_t}\varepsilon_t\le\frac1J\sum_jS_j+R\sqrt{\log(1/\delta_{\rm mc})/(2J)}$$
with probability at least $1-\delta_{\rm mc}$.

On both events, $\tfrac1J\sum_jS_j\le\tfrac1J\sum_{j,t}\hat\varepsilon_{j,t}$, and a union bound completes the proof. $\square$

**Proof of Theorem C6.**

*One boundary.* By Theorem C1 of node N18 with $k=1$, $(K^{P_j}-K^{P_k})V(x)=s+r$ with $s=\langle\nabla V(x),\mu^j-\mu^k\rangle$ and $\lvert r\rvert\le\tfrac{M_2}2(m_2^j+m_2^k)<\lvert s\rvert$. If $s>0$ then $s+r>s-\lvert s\rvert=0$; if $s<0$ then $s+r<s+\lvert s\rvert=0$. So $s+r$ has the sign of $s$.

*Run level.* Theorem C2 with $P_j$ in the role of $P$ and $P_k$ in the role of $Q$ gives $L(P_j)-L(P_k)=T_{jk}+\mathcal E$ with $\lvert\mathcal E\rvert\le B_{jk}<\lvert T_{jk}\rvert$. The same sign argument applies. $\square$

**Proof of Corollary C6′.** Under (i), $s(P_j)-s(P_k)=\langle\nabla V(x),\mu^j-\mu^k\rangle$, because the reference terms cancel. Under (ii), C6 says this has the sign of $(K^{P_j}-K^{P_k})V(x)$. If (i) fails, the score differs from the first-order term by an amount that C6 does not control. If (ii) fails, C6 does not apply, and the remainder $r$ may reverse the sign. $\square$

## Remarks

1. **Cost.** The Hoeffding allocation of a full-tail audit at one state is $1016$ paired training replicas per candidate at radius $0.1$ (node [N15](N15_adaptive_audit.md)). The first-order audit of C5′ costs $n_P+n_Q$ gradient evaluations per candidate, and its radius does not depend on the parameter count.
   - *Example: clipped SGD.* For $b=\eta c$, the radius $e_1$ is a fixed multiple of $\eta c/\sqrt n$.
   - *Example: sign-type increments.* For sign-type increments on $p$ parameters, $b=\eta\sqrt p$. The first-order differences they produce are of the same scale, so the *relative* accuracy is again governed by $1/\sqrt n$.
2. **What C5′ cannot do alone.** The constants $\Lambda$ and $M_2$ belong to the continuation. C5′ turns the data side of the certificate into a cheap measurement. It does not remove the need for a calibration of the continuation side, which costs a few full-tail audits, not thousands.
3. **Where the margin holds.** By Remark 3 of node [N18](N18_optimizer_transformed_response.md), the remainder is $O(\eta^2)$ per round for SGD, momentum and sign optimizers. For Adam-type optimizers over several rounds it is first order in $\eta$, and it shrinks with warm-up. C6 therefore applies most readily at small step sizes, to non-adaptive learners, or to one-boundary comparisons. In the remaining regime, measured short tails (node [N17](N17_ced_method.md)) must decide.

## Dependencies and source

- **Uses:** [N18](N18_optimizer_transformed_response.md) (Theorems C1, C2), [N19](N19_learner_statistics.md) (which mean increment to estimate), [N15](N15_adaptive_audit.md) (full-tail calibration and cost).
- **Used by:** [N17](N17_ced_method.md).
- **Source:** `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md`, §5 (Theorems C5, C6 and the consequence for cheap proxies). Corollaries C5′ and C5″ make the confidence allocation explicit.
