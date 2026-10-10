# Optimizer-transformed linear response: what a feedback law must carry for a real learner

**Status:** working derivation, 2026-10-10. Proofs are complete for the stated finite-dimensional assumptions. The controlled checks E10–E12 (theory/v015_checks) test them before any of this enters `theoretical_framework/`. Prior work and the novelty boundary are in §7.

This note gives the central result that the framework was missing. The general characterization (Thm. 7 and Thm. 8 of the main theory) says that a feedback law must preserve the expectations of the continuation responses $\mathcal G$. The affine theory computes $\mathcal G$ only for convex-affine learners. This note computes the *first-order part* of $\mathcal G$ for every stateful first-order optimizer used in practice. It then shows that this part is the minimal first-order statistic, catalogs it for SGD, momentum, Adam and AdamW (cold and warm), clipping, sign/Lion optimizers and LoRA, and shows how to estimate and use it at the actual training states.

## 1. Setting

**State.** The complete learner state is $x=(\theta,s)\in\mathcal X\subseteq\mathbb R^{p}\times\mathbb R^{q}$:

- $\theta$ are the trainable parameters, for example LoRA adapters.
- $s$ is the optimizer state, for example $m$ and $v$.
- The step counter is absorbed into the round index $t$.

**Feedback and gradients.** At round $t$, a feedback law $P_t(\cdot\mid x)$ emits a message $Y$. The learner computes a raw consumed gradient $g=G_t(x,Y,\xi)\in\mathbb R^p$. This is after loss normalization and accumulation. Clipping is treated as part of the optimizer below.

**Optimizer.** The optimizer is a measurable increment map
$$x^+=x+\Delta_t(x,g).$$

**Kernel.** The learning kernel of law $P$ is
$$(K_t^Pf)(x)=\mathbb E_{Y\sim P_t(\cdot\mid x),\,\xi}\,f\big(x+\Delta_t(x,G_t(x,Y,\xi))\big).$$
Reference and alternative interfaces share the optimizer and the gradient map. They differ only in the message law. A change of optimizer is a different $\Delta$ and can be compared the same way.

**Increment moments.** For a law $P$ at $(t,x)$, define the message-level mean increment $D_t(x,y)=\mathbb E_\xi\Delta_t(x,G_t(x,y,\xi))$, and
$$\mu_t^P(x)=\mathbb E_{Y\sim P}D_t(x,Y),\qquad m_{k,t}^P(x)=\mathbb E_{Y\sim P,\xi}\lVert\Delta_t(x,G_t(x,Y,\xi))\rVert^k .$$

**Continuation.** $V^P_{t+1}$ is the continuation of protocol $P$ (later updates under $P$, then deployment), with values in $[0,1]$. $L(P)=\mathbb E\,F_E(X_R^P)$.

We fix a norm $\lVert\cdot\rVert$ on $\mathbb R^{p+q}$ with dual norm $\lVert\cdot\rVert_*$.

## 2. The expansion

**Assumption S$_k$ (smooth continuation on the reachable segment).** Let $\mathcal S_{t,x}$ be the convex hull of $x$ and all $x+\Delta_t(x,g)$ with $g$ in the support of the consumed gradients under the compared laws. $V=V_{t+1}$ is $C^{k+1}$ on $\mathcal S_{t,x}$, and $\lvert\nabla^{k+1}V(w)[h,\dots,h]\rvert\le M_{k+1}\lVert h\rVert^{k+1}$ for all $w\in\mathcal S_{t,x}$ and all $h$. The moments $m^P_{k+1},m^Q_{k+1}$ are finite.

**Theorem C1 (optimizer-transformed moment expansion).** Under S$_k$,
$$(K_t^P-K_t^Q)V(x)=\sum_{j=1}^{k}\frac1{j!}\Big\langle\nabla^jV(x),\;\mathbb E_P\Delta^{\otimes j}-\mathbb E_Q\Delta^{\otimes j}\Big\rangle+r_k,\qquad
\lvert r_k\rvert\le\frac{M_{k+1}}{(k+1)!}\big(m_{k+1}^P+m_{k+1}^Q\big).$$
For $k=1$:
$$(K_t^P-K_t^Q)V(x)=\big\langle\nabla V(x),\mu_t^P(x)-\mu_t^Q(x)\big\rangle+r_1,\qquad\lvert r_1\rvert\le\tfrac{M_2}2\big(m_2^P+m_2^Q\big).$$

*Proof.* For each realized increment $\Delta$, the segment $\{x+u\Delta:u\in[0,1]\}$ lies in $\mathcal S_{t,x}$ by convexity. Taylor's theorem with Lagrange remainder gives
$$V(x+\Delta)=V(x)+\sum_{j=1}^k\tfrac1{j!}\nabla^jV(x)[\Delta^{\otimes j}]+R(\Delta),\qquad \lvert R(\Delta)\rvert\le\tfrac{M_{k+1}}{(k+1)!}\lVert\Delta\rVert^{k+1}.$$
Take expectations under the two laws; this is valid because $V$ is bounded and the moments are finite. Subtract. The constant $V(x)$ cancels because both laws are probability measures. Multilinearity moves each expectation inside $\nabla^jV(x)[\cdot]$. Finally $\lvert\mathbb E_PR-\mathbb E_QR\rvert\le\mathbb E_P\lvert R\rvert+\mathbb E_Q\lvert R\rvert$. $\square$

For SGD, where $\Delta=-\eta G$, the case $k=2$ is equation (35) of the main theory. C1 replaces the gradient moments by the moments of the **optimizer's increment**: the order-$j$ information a feedback law must carry is $\mathbb E\Delta^{\otimes j}$.

**Theorem C2 (run-level first-order representation and certificate).** Compare protocols $P$ and $Q$ over $R$ rounds from a common initial law. Assume S$_1$ at every state visited by $Q$, with constants $M_{2,t}(x)$. Then
$$L(P)-L(Q)=\sum_{t=0}^{R-1}\mathbb E_{d_t^Q}\Big\langle\nabla V^P_{t+1}(X_t),\;\mu_t^P(X_t)-\mu_t^Q(X_t)\Big\rangle+\mathcal E,\qquad
\lvert\mathcal E\rvert\le\sum_t\mathbb E_{d_t^Q}\Big[\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big].$$
Consequently, with $\Lambda_t(x)=\lVert\nabla V^P_{t+1}(x)\rVert_*$,
$$\lvert L(P)-L(Q)\rvert\le\min\Big\{1,\ \sum_t\mathbb E_{d_t^Q}\Big[\Lambda_t\,\lVert\mu_t^P-\mu_t^Q\rVert+\tfrac{M_{2,t}}2\big(m^P_{2,t}+m^Q_{2,t}\big)\Big]\Big\}.$$
If S$_1$ holds only on certified sets $C_t$ with first-exit probability $\rho$, the same bounds hold with the sum restricted to $t<\tau$ and $+\rho$ added.

*Proof.* Apply Theorem 7, the telescoping identity (14), with protocol $P$ in the role of the reference and $Q$ as the alternative:
$$L(P)-L(Q)=\sum_t\mathbb E_{d_t^Q}\big[(K_t^P-K_t^Q)V^P_{t+1}(X_t)\big].$$
Apply C1 with $k=1$ at each $X_t$ and sum. Measurability holds because $\mu$, $m_2$ and $\nabla V$ are measurable functions of the state. The certificate follows from $\lvert\langle a,b\rangle\rvert\le\lVert a\rVert_*\lVert b\rVert$. The stopped version is Theorem 5.3 of node N05, applied with $\varepsilon_t(x)$ equal to the per-state bound. $\square$

**Scaling.** Write the parameter block of the increment as $\Delta^\theta=\eta\,\tilde\Delta^\theta$. The optimizer-state block need not scale with $\eta$; Adam's $\Delta m=(1-\beta_1)(G-m)$, for example, is $O(1)$. Use the weighted norm $\lVert(\delta\theta,\delta s)\rVert_\eta=\lVert\delta\theta\rVert+\eta\lVert\delta s\rVert$. If the continuation depends on $s$ only through later parameter increments, each $O(\eta)$, then $\lVert\nabla_sV\rVert=O(\eta)$ and $M_2$ is $O(1)$ in this norm on a region of bounded radius. Under these scalings $m_2=O(\eta^2)$. A first-order mismatch $\mu^P-\mu^Q\ne0$ in the weighted dual pairing then produces a gap of order $\eta$, while a first-order match leaves $O(\eta^2)$ per round. The same split governs every optimizer below. The constants are properties of the continuation, not of the data.

## 3. Necessity and minimality of the first-order statistic

Fix a boundary $(t,x)$ and a declared family $\mathcal V$ of continuations, each $C^2$ on $\mathcal S_{t,x}$ with Hessian bound $M_2$. The family covers the deployment tasks and suffixes of interest. Let $U=\operatorname{span}\{\nabla V(x):V\in\mathcal V\}$. The **first-order response class** is
$$\mathcal G^{(1)}_{t,x}=\big\{\,y\mapsto\langle\nabla V(x),D_t(x,y)\rangle:\ V\in\mathcal V\,\big\}.$$

**Theorem C3 (first-order sufficiency, necessity and minimality).** Let $\Pi_U$ be the orthogonal projection onto $U$; Euclidean pairing is used here.

1. *Sufficiency.* If $\Pi_U(\mu^P-\mu^Q)=0$, then $\lvert(K^P-K^Q)V(x)\rvert\le\tfrac{M_2}2(m_2^P+m_2^Q)$ for every $V\in\mathcal V$.
2. *Necessity.* If $\Pi_U(\mu^P-\mu^Q)\ne0$, some $V\in\mathcal V$ has $\langle\nabla V(x),\mu^P-\mu^Q\rangle\ne0$, and
$$\lvert(K^P-K^Q)V(x)\rvert\ge\lvert\langle\nabla V(x),\mu^P-\mu^Q\rangle\rvert-\tfrac{M_2}2(m_2^P+m_2^Q).$$
   Under the scaling of §2 the first term is $\Theta(\eta)$ and the second is $O(\eta^2)$, so the mismatch is detected for all sufficiently small $\eta$.
3. *Minimality.* On a finite message alphabet of size $m$, the minimal number of linear expectation statistics that determine every first-order term $\langle\nabla V(x),\mu^P-\mu^Q\rangle$, $V\in\mathcal V$, for all law pairs is
$$d_0^{(1)}=\dim\operatorname{span}\big(\{1\}\cup\mathcal G^{(1)}_{t,x}\big)-1\le\min\{\dim U,\ m-1\}.$$

*Proof.* (1) follows from C1, since $\langle\nabla V(x),\mu^P-\mu^Q\rangle=\langle\nabla V(x),\Pi_U(\mu^P-\mu^Q)\rangle=0$ for $\nabla V(x)\in U$.

(2) Suppose $\langle\nabla V(x),w\rangle=0$ for every $V\in\mathcal V$, where $w=\mu^P-\mu^Q$. Then $w\perp U$, so $\Pi_Uw=0$, a contradiction. The inequality is C1 together with the reverse triangle inequality.

(3) The first-order term equals $(P-Q)g$ for $g=\langle\nabla V(x),D(x,\cdot)\rangle\in\mathcal G^{(1)}$. Theorem 8(a), node N06, applied to the class $\mathcal G^{(1)}$, gives the exact count. Each $g\in\mathcal G^{(1)}$ is a linear functional of the vector function $D(x,\cdot)$ along a direction in $U$. Hence $\operatorname{span}\mathcal G^{(1)}\subseteq\{y\mapsto\langle u,D(x,y)\rangle:u\in U\}$, whose dimension modulo constants is at most $\dim U$. It is also at most $m-1$. $\square$

**Reading.** To first order, the statistic that a feedback law must preserve for a given learner and task family is the **$U$-projection of the mean optimizer increment**. It is not the mean gradient, the loss or the teacher's accuracy. Which features of the gradient law enter it is decided entirely by the optimizer map $\Delta$. Section 4 computes it.

## 4. What each optimizer makes the feedback carry

All statements below concern $D_t(x,y)$. We write $G$ for the consumed gradient of message $y$ and $\Gamma$ for a full-weight gradient. Message-independent terms (weight decay, momentum decay, bias-correction constants) cancel in $\mu^P-\mu^Q$ and are omitted.

**C4(a) SGD.** $\Delta^\theta=-\eta G$, so $\mu^P-\mu^Q=-\eta(\mathbb E_PG-\mathbb E_QG)$. First-order statistic: the mean consumed gradient. Second order (C1 with $k=2$): $\mathbb E\,GG^\top$. This recovers equation (35).

**C4(b) Heavy-ball and Nesterov momentum.** With $m^+=\beta m+G$ and $\theta^+=\theta-\eta m^+$, we have $\Delta^\theta=-\eta(\beta m+G)$ and $\Delta^m=(\beta-1)m+G$. With Nesterov, $\Delta^\theta=-\eta(G+\beta m^+)$. Both are affine in $G$. First-order statistic: $\mathbb E\,G$. It enters through $\langle-\eta\nabla_\theta V+\nabla_mV,\ \mathbb E_PG-\mathbb E_QG\rangle$, so momentum changes the *weight* of the mean gradient but not the statistic.

**C4(c) Adam and AdamW at a warm state** ($v_i>0$ for all $i$; notation of the warm-Adam note).

- The increments are $\Delta^m=(1-\beta_1)(G-m)$, $\Delta^v=(1-\beta_2)(G\odot G-v)$ and $\Delta^\theta=-\eta\lambda\theta-\eta\,n(G)/(d(G)+\epsilon)$.
- By Lemma W0, $\Delta^\theta=\text{const}-\eta c_1PG+e(G)$, with $c_1=(1-\beta_1)/(1-\beta_1^t)$, $P=\operatorname{diag}(1/(a+\epsilon))$ and $\lvert e_i\rvert\le\eta u_i\min\{\kappa_i/2,1\}$. Therefore
$$\mu^P-\mu^Q=\Big(-\eta c_1P\,\delta_1+(\mathbb E_Pe-\mathbb E_Qe),\ \ (1-\beta_1)\delta_1,\ \ (1-\beta_2)\delta_2\Big),\qquad \delta_1=\mathbb E_PG-\mathbb E_QG,\quad \delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}.$$
- First-order statistic: the pair $(\mathbb E\,G,\ \mathbb E\,G^{\odot2})$, up to a warmth error $\eta\,\mathbb E[u\min\{\kappa/2,1\}]$. The mean gradient enters through $\theta$ (preconditioned by $P$) and through $m$. The *coordinatewise second moment* enters at first order through $v$, weighted by $(1-\beta_2)\nabla_vV$.
- This is sharper than Proposition W1: the $L_v\kappa$ residual of W1 is replaced by an explicit linear term in $\mathbb E\,G^{\odot2}$. Equal-mean laws with unequal gradient second moments are therefore *not* first-order equivalent for warm Adam unless $\nabla_vV\perp\delta_2$.

**C4(d) Adam at a cold state** ($m=v=0$, first step).

- The increments are $\Delta^\theta=-\eta\lambda\theta-\eta h_\epsilon(G)$, $\Delta^m=(1-\beta_1)G$ and $\Delta^v=(1-\beta_2)G^{\odot2}$. First-order statistic: $(\mathbb E\,h_\epsilon(G),\ \mathbb E\,G,\ \mathbb E\,G^{\odot2})$.
- The parameter block sees only the sign law as $\epsilon\to0$ (Prop. A2). The moment blocks retain the mean and the second moment (Prop. A4). The $\theta$-weight $\eta\nabla_\theta V$ multiplies an $O(1)$ sign difference. For equal-mean laws with different sign laws this is a first-order gap, which is what C23 measures.
- **Prediction.** For R=1 the C23 Adam gap should match $\eta\langle\nabla_\theta V,\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle$ to $O(\eta^2)$.

**C4(e) Gradient clipping** (global norm $c$, before any base optimizer). Let $\operatorname{clip}_c(G)=G\min\{1,c/\lVert G\rVert\}$.

- With clipped SGD, $\Delta^\theta=-\eta\operatorname{clip}_c(G)$. First-order statistic: the **clipped mean** $\mathbb E\operatorname{clip}_c(G)$. It depends on the joint law of direction and norm.
- *Witness.* Take $c=1$. Let $P$ put $G=4e_1$ with probability $\tfrac14$ and $G=-\tfrac43e_1$ with probability $\tfrac34$, and let $Q=\delta_0$. Then $\mathbb E_PG=\mathbb E_QG=0$, but $\mathbb E_P\operatorname{clip}_1G=-\tfrac12e_1$. The first-order gap is $\tfrac\eta2\lvert\partial_{\theta_1}V\rvert$.
- *Always-active clipping.* If $\lVert G\rVert>c$ almost surely under both laws, then $\mathbb E\operatorname{clip}_cG=c\,\mathbb E[G/\lVert G\rVert]$. The statistic becomes the **mean unit direction**: gradient magnitudes are first-order irrelevant.
- With a clipped base optimizer, replace $G$ by $\operatorname{clip}_c(G)$ in C4(a)–(d) and (f). Our Seed-SFT recipe clips at global norm 1; the main-table K20 run recorded a pre-clip norm of 5.28 at its final update. Clipping is therefore active, and every statistic above is taken of the clipped gradient.

**C4(f) Sign optimizers with momentum** (Lion form: $u=\operatorname{sign}(\beta_1m+(1-\beta_1)G)$, $\theta^+=\theta-\eta u$, $m^+=\beta_2m+(1-\beta_2)G$).

- $\Delta^\theta_i=-\eta\operatorname{sign}(\beta_1m_i+(1-\beta_1)G_i)$, so $\mathbb E\,\Delta^\theta_i=-\eta\operatorname{sign}(m_i)\big(1-2\pi_i\big)$ for $m_i\ne0$. Here $\pi_i=\Pr\{\operatorname{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\operatorname{sign}(m_i)\}$ is the **flip probability**, counting a zero as half a flip.
- The parameter block's first-order statistic is the flip-probability vector. If momentum dominates, i.e. $(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert$ almost surely under both laws, then $\pi_i=0$ and coordinate $i$'s parameter increment is message-independent. The feedback is first-order irrelevant there except through $\Delta^m=(1-\beta_2)(G-m)$, which carries $\mathbb E\,G$ into later steps.

**C4(g) LoRA.** For an adapted matrix $W=W_0+\alpha BA$ ($A\in\mathbb R^{r\times k}$, $B\in\mathbb R^{d\times r}$), the adapter gradients are $G_A=\alpha B^\top\Gamma$ and $G_B=\alpha\Gamma A^\top$, where $\Gamma=\nabla_W\ell$.

- At the standard initialization $B=0$, $G_A\equiv0$ for every message. Any optimizer that leaves a zero-gradient coordinate fixed from a cold state (SGD, momentum from $m=0$, Adam with $h_\epsilon(0)=0$) therefore does not move $A$ in step 1, under every law.
- The $B$-block statistic is $\mathbb E[\Gamma]A_0^\top$ for SGD and $\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)$ for cold Adam.
- So in the first step the feedback can matter only through the rank-$r$ random sketch $\Gamma A_0^\top$ of its full-weight gradient. Two laws whose mean full-weight gradients differ only by $\Delta\Gamma$ with $\Delta\Gamma A_0^\top=0$ are first-order equivalent at step 1 under SGD. Under cold Adam the same holds for laws whose sketched gradients $\Gamma A_0^\top$ have the same sign law.
- In later steps $B\ne0$ and $G_A$ carries $B^\top\Gamma$, so the sketch dimension grows with training.

*Proof of C4.* Each case is a direct computation of $D_t(x,y)$ from the stated update, followed by subtraction of the two expectations. Message-independent terms cancel. (c) uses Lemma W0 for the parameter block. (e)'s witness: $\operatorname{clip}_1(4e_1)=e_1$ and $\operatorname{clip}_1(-\tfrac43e_1)=-e_1$, giving $\tfrac14-\tfrac34=-\tfrac12$. (f): for $m_i\ne0$ the sign equals $\operatorname{sign}(m_i)$ except on the flip event. (g) uses the product rule for $W=W_0+\alpha BA$ and $B_0=0$. $\square$

## 5. Using the statistic at the actual states

**Theorem C5 (dimension-free estimation).** Let $X_1,\dots,X_n$ be IID in a real Hilbert space with $\lVert X\rVert\le b$ almost surely and mean $\mu$. With probability at least $1-\delta$,
$$\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).$$
For two independent samples of sizes $n_P,n_Q$, with probability at least $1-\delta$,
$$\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).$$
The bound does not depend on the dimension. It applies directly to the parameter-increment block, with $b=\eta\,c$ for clipped SGD and $b=\eta\sqrt p$ for sign-type increments.

*Proof.* Let $f=\lVert\bar X-\mu\rVert$. By Jensen and independence, $\mathbb Ef\le(\mathbb E f^2)^{1/2}=(\mathbb E\lVert X-\mu\rVert^2/n)^{1/2}\le b/\sqrt n$, since $\mathbb E\lVert X-\mu\rVert^2=\mathbb E\lVert X\rVert^2-\lVert\mu\rVert^2\le b^2$. Replacing one sample changes $f$ by at most $\lVert X_i-X_i'\rVert/n\le2b/n$. McDiarmid's inequality gives $\Pr\{f-\mathbb Ef\ge s\}\le\exp(-ns^2/(2b^2))$, and $s=b\sqrt{2\log(1/\delta)/n}$ proves the first claim. For the second, apply it to each sample with $\delta/2$, use the triangle inequality and take a union bound. $\square$

Combined with C2, the first-order part of the certificate needs, at each audited actual state, $n$ consumed-gradient evaluations per law. It needs no training tail. The constants $\Lambda_t$ and $M_{2,t}$ are properties of the continuation. They must be bounded separately, or calibrated against a few full-tail audits (node N15); C5 does not estimate them.

**Theorem C6 (selection correctness by first-order margin).** At a state $x$, take candidate laws $P_j$ and $P_k$ and one continuation $V$ satisfying S$_1$. If
$$\big\lvert\langle\nabla V(x),\mu^{j}-\mu^{k}\rangle\big\rvert>\tfrac{M_2}2\big(m_2^{j}+m_2^{k}\big),$$
then $\operatorname{sign}\big((K^{P_j}-K^{P_k})V(x)\big)=\operatorname{sign}\langle\nabla V(x),\mu^j-\mu^k\rangle$. The first-order score then ranks the two candidates correctly. Over $R$ rounds, the same holds for $L(P_j)-L(P_k)$ when the summed first-order term of C2, with $P_j$ in the role of $P$, exceeds the summed remainder bound.

*Proof.* By C1, $(K^{P_j}-K^{P_k})V(x)=s+r$ with $s$ the first-order term and $\lvert r\rvert\le\tfrac{M_2}2(m_2^j+m_2^k)<\lvert s\rvert$. So $s+r$ has the sign of $s$. The run-level statement is C2 applied in the same way. $\square$

**Consequence for cheap proxies.** A static statistic predicts the response sign only if it is (i) the optimizer-transformed mean of §4, (ii) paired with the continuation gradient at the actual state, and (iii) used where the margin dominates the remainder. The E6 proxies were raw or update features fitted by one ridge model across states and tasks. They violate (ii), because $\nabla V$ changes with the state and the task. Where the remainder dominates they also violate (iii). C6 predicts that the oracle first-order score is never wrong where the margin dominates. Check E12 tests this on the frozen E6 panels.

## 6. Higher order: the optimizer-transformed moment hierarchy

C1 with general $k$ gives the analogue of Theorems 11 and 17 for any optimizer. If two laws agree on $\mathbb E\Delta^{\otimes j}$ for $j\le k$ at the visited states, the per-round discrepancy is at most $\tfrac{M_{k+1}}{(k+1)!}(m_{k+1}^P+m_{k+1}^Q)$, which is $O(\eta^{k+1})$ under the scaling of §2.

- For SGD these are the gradient moments, which is the classical case.
- For cold Adam they are the moments of $h_\epsilon(G)$ in the parameter block. For $\epsilon\to0$ these are joint sign moments; the joint sign law is what A2 requires for the immediate output.
- Under always-active clipping, the parameter block involves only moments of the direction $G/\lVert G\rVert$.

The information order that the feedback must carry is thus a property of the **optimizer-transformed increment law**, not of the raw data.

## 7. Prior work and novelty boundary

- *First-order training influence.* TracIn (Pruthi et al., NeurIPS 2020) sums $\eta\langle\nabla\ell_{\rm test},\nabla\ell_{\rm train}\rangle$ along an SGD trajectory. This is C2's first-order term for SGD with the test loss in place of $V$. Hara et al. (NeurIPS 2019) give SGD-influence by unrolling. SOURCE (Bae et al., 2024) approximates unrolled differentiation, including Adam.
- *Optimizer-aware data selection.* LESS (Xia et al., ICML 2024) selects data by Adam-preconditioned gradient features computed at checkpoints after a LoRA warmup. This is the parameter block of C4(c) at a warm state.
- *Sign and Adam.* Kingma and Ba; Balles and Hennig (2018); Bernstein et al. (signSGD, 2018); Chen et al. (Lion, 2023).

**What is claimed as new, to be checked against the full texts:**

1. The *minimal* first-order sufficient statistic of a feedback law for an arbitrary stateful optimizer is the projection of the complete-state increment mean, including the optimizer-state blocks (C3), and it is necessary as well as sufficient.
2. The catalog consequences of C4: the second moment enters at first order for warm Adam through $v$; clipping turns the statistic into the clipped mean, or the unit direction when clipping is always active; momentum-dominated sign optimizers are first-order blind in the parameter block; LoRA at initialization sees only a rank-$r$ sketch.
3. The margin criterion C6, which states when a cheap statistic must rank correctly.
4. The connection to the framework's sufficiency theory (Thm. 8), its audit theory (Thms. 19 and 20) and the few-shot cold window (N12).

None of the mathematics is deep in isolation. It is Taylor's theorem, telescoping, LP duality and McDiarmid's inequality. The claim to be earned is that this object predicts real distillation outcomes.
