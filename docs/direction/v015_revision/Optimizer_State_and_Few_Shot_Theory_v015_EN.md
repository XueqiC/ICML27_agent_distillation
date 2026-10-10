# Environment Information Theory: Coverage, Optimizer State, and Few-Shot Acquisition

**Version:** 0.15 research addendum, 10 October 2026 UTC.  
**Companion:** `Environment_Information_Theory_EN.md`, corrected version 0.14.1.  
**Evidence status:** Mathematical derivations and finite regression checks; the user's external v0.14 validation and controlled-neural C23 results are reported evidence. Their source code and raw results were not available for an independent rerun here. This document does not report Gemma, LoRA, or external-benchmark training. Independent review means a separate model-agent review, not formal verification or external peer review.

## 1. What the validation changes

The report supports the finite core under its assumptions and exposes a genuine omitted hypothesis in Theorem 19 (39). Its example invalidates the original statement, even though the intended local calculation is correct. The previous proof audit also missed that omission. The corrected main document makes state-to-cell transport explicit; the archived v0.14 package remains unchanged.

The Adam results concern a second issue: transferring an affine-SGD specialization to a different learning operator. They do not contradict the general response characterization. A measured gap against an affine-SGD comparator does not, by itself, distinguish optimizer nonlinearity, retained optimizer state, nonlinear prediction/loss, and changing neural features. These mechanisms require separate controls.

The present extension keeps one governing object. For task environment \(E\), optimizer/recipe \(o\), complete learner state \(z\), feedback \(y\), and declared reference suffix,
\[
\psi_{E,o,t,z}(y)
=\mathbb E_\xi V^{0,o}_{t+1}(U^o_t(z,y,\xi)).
\tag{A1}
\]
The environment must supply distinctions that the permitted acquisition process can discover and that this response family can use. Demonstrations determine what remains uncertain; the update and suffix determine which uncertainty affects learning; the audit determines which claims about those effects are supported at visited states.

These are three different requirements. Demonstration coverage is not learner-state audit coverage. Matching a parameter delta is not matching a complete next state. A neural predictor of a response is not automatically a uniform certificate for that response.

## 2. The corrected finite-audit bridge

### Proposition A1: transport is the missing hypothesis

Use the frozen audit and its simultaneous event from Theorem 19. Write
\(\bar D_j=\widehat D_j+e_0+e_S\).
At each certified complete state \(z\) at time \(t\), specify an audited cell \(x(t,z)\), a common message coordinate system, and its fixed features \(\phi_x\). Let
\[
\left\|(P^0_{t,z}-P^j_{t,z})\phi_x
       -(P^0_x-P^j_x)\phi_x\right\|_\infty
\le a_{t,j}(z).
\tag{A2}
\]
The laws on the left are the actual conditional laws at \(z\); the laws on the right are those sampled in the audit. Suppose the actual response (A1) has a uniform decomposition
\[
\psi_{E,o,t,z}=c_{t,z}+b_{t,z}^{\mathsf T}\phi_x+r_{t,z},
\quad \|b_{t,z}\|_1\le B_t,
\quad \|r_{t,z}\|_\infty\le\tau_t.
\tag{A3}
\]
For fresh training with the selected frozen provider, let \(\sigma\) be first exit from the certified set, taking \(\infty\) if there is none. Suppose \(\Pr(\sigma<R)\le\rho\), conditional on the frozen audit result. On its audit event,
\[
|L_E(0)-L_E(j)|
\le \sum_{t=0}^{R-1}\mathbb E_j\!\left[
1\{t<\sigma\}
\{B_t(\bar D_j+a_{t,j}(Z_t))+2\tau_t\}\right]+\rho.
\tag{A4}
\]
All needed coverage, approximation, and coefficient bounds must hold for every provider selectable on that event. Independently estimated bounds require their own simultaneous confidence allocation. The bound may be capped at one.

**Proof.** Constants integrate to zero under a probability-law difference. Equation (A2), the audit event, and Hölder's inequality bound the feature contribution by \(B_t(\bar D_j+a_{t,j}(z))\). Two expectations of a residual bounded by \(\tau_t\) contribute at most \(2\tau_t\). To justify the stopped form, follow provider \(j\) until its first uncertified state and use the reference recipe thereafter. The full and switched processes can be coupled identically until that exit, so their expected bounded terminal utilities differ by at most \(\rho\). The telescoping identity applied to the switched process contains only pre-exit discrepancies; its pre-exit occupancy agrees with provider \(j\). Integrating the local bounds gives (A4). \(\square\)

Exact conditional-law matching implies \(a=0\), but is stronger than necessary. Matching the discrepancy vector in (A2) suffices. If no bound exists, the state is uncertified and its first visit contributes to exit probability. Mapping to the nearest weight checkpoint does not prove (A2).

If practical features vary by state, and a common coordinate comparison gives
\(\sup_y\|\phi_z(y)-\phi_x(y)\|_\infty\le\alpha_{zx}\), then
\[
\|(P^0_{t,z}-P^j_{t,z})\phi_z\|_\infty
\le D_j+a_{t,j}(z)+2\alpha_{zx}.
\tag{A5}
\]
This follows by adding and subtracting the fixed-cell features; each probability expectation of their difference is bounded by \(\alpha_{zx}\). Apply (A3) with \(\phi_z\) when using this version. Empirical feature proximity on sampled messages is not the required supremum bound.

**Counterexample and regression.** Audit state 0 where both sources emit 0; train at state 1 where reference emits 0 and candidate emits 1; set \(\psi(y)=y\). The old bound with \(B=1,\tau=\rho=0\), \(L=M=d=1\), \(n_0=n_S=10{,}000\), and \(\delta=0.05\) is about 0.02960 although the gap is one. The new bound either includes \(a=1\) or declares immediate exit with \(\rho=1\). This is an expected-failure regression for the old statement, not a failure to hide or relax.

Equation (40) remains valid under Theorem 14's state-independent IID affine block-source assumptions and required moment features. In that setting the source law and its moment discrepancy apply at every state. It cannot be applied to arbitrary adaptive teacher-on-student-prefix providers merely because some checkpoints were audited.

## 3. What Adam requires: three distinct levels of sufficiency

We use standard bias-corrected Adam, with epsilon outside the square root, no AMSGrad, and gradients defined **after** any declared accumulation, rescaling, or clipping. The optimizer consumes \(G_t\):
\[
m_t=\beta_1m_{t-1}+(1-\beta_1)G_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)G_t^{\odot2},
\]
\[
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\tag{A6}
\]
Division and square root are coordinatewise; \(0\le\beta_1,\beta_2<1\), \(\eta_t>0\); \(\lambda=0\) gives Adam, while the displayed fixed decoupled decay gives this AdamW convention. Other implementations must be derived from their executed update. The formula and its first-step consequence are established optimizer facts [O1–O4].

### Proposition A2: the cold parameter-output quotient

Fix \(\theta_0\), \(m_0=v_0=0\), and step counter zero. Put \(c=(1-\eta\lambda)\theta_0\). Then
\[
\theta_1=c-\eta h_\epsilon(G),\qquad
h_\epsilon(g)_i=\frac{g_i}{|g_i|+\epsilon}.
\tag{A7}
\]
Define the **ideal sign update** at \(\epsilon=0\) using \(h_0(0)=0\). This convention is necessary at zero coordinates; a literal floating-point Adam implementation with epsilon zero may produce \(0/0\).

For this ideal update, two gradient laws give identical expected values for **all bounded measurable utilities of the immediate parameter output** if and only if their joint sign-pattern laws agree. On a finite gradient alphabet with \(s\) reachable sign patterns, the minimum number of unrestricted exact linear expectation statistics sufficient for this entire utility family is \(s-1\). Here \(s\le3^p\) for \(p\) coordinates, or \(s\le2^p\) if zeros are excluded. A restricted task family may need fewer statistics.

In real arithmetic, for every \(\epsilon>0\), the map in (A7) is coordinatewise injective with
\[
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-|u_i|},\qquad |u_i|<1.
\tag{A8}
\]
Thus equality for all bounded measurable immediate-parameter utilities is equivalent to equality of the **full consumed-gradient law**. On an unrestricted finite gradient alphabet of size \(m\), the exact linear-statistic dimension is \(m-1\).

**Proof.** First-step bias corrections give \(\widehat m_1=G\) and \(\widehat v_1=G^{\odot2}\), proving (A7). At zero epsilon the map from a sign pattern to \(c-\eta s\) is injective. Indicator utilities of its reachable output points identify every pattern probability, proving necessity; equality of the pushforward law proves sufficiency. On a finite quotient of size \(s\), its function space modulo constants has dimension \(s-1\). If fewer features were used, their span together with constants would miss a quotient indicator; a zero-mass signed perturbation annihilating the features separates its expectation, and can be added with small positive and negative coefficients to an interior probability law. This contradicts sufficiency. The converse uses \(s-1\) pattern indicators. Solving (A7) for \(g\) yields (A8), and the same argument on all distinct gradients completes the positive-epsilon case. \(\square\)

This is an exact-statistics statement, not a recommendation to estimate a complete high-dimensional gradient distribution. Finite-precision rounding or an additional projection can merge outputs and changes this exact quotient. Inverse conditioning near \(|u|=1\) is poor; bounded regular task responses can have much cheaper approximate descriptions.

**Joint versus marginal signs.** In two dimensions, let one law put equal mass on \((+,+),(-,-)\) and another on \((+,-),(-,+)\). Every coordinate has the same sign marginal and mean update. The bounded output test that asks whether the two parameter displacements have the same sign has expectations one and zero. Coordinatewise sign frequencies therefore do not characterize a general nonlinear task response.

### Proposition A3: a margin-sensitive approximate quotient

Let \(F\in[0,1]\) obey the coordinatewise Lipschitz condition
\[
|F(u)-F(v)|\le\sum_i L_i|u_i-v_i|
\]
on both sets of parameter outputs and the needed connecting region, with \(L_i\ge0\). For two consumed-gradient laws \(P,Q\), define
\[
\beta_{P,i}(a)=P(0<|G_i|<a),\quad a>0,
\]
and define \(\Delta_{\rm sign}(F)\) as the absolute expected utility difference under the ideal sign outputs. Then
\[
\begin{split}
|\mathbb E_PF(c-\eta h_\epsilon(G))-
  \mathbb E_QF(c-\eta h_\epsilon(G))|
\le {}&\Delta_{\rm sign}(F)\\
&+\eta\sum_iL_i\left[
\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}
\right].
\end{split}
\tag{A9}
\]
The bound can be capped at one. Moreover, \(\Delta_{\rm sign}(F)\le\operatorname{TV}(P_{\rm sign},Q_{\rm sign})\).

**Proof.** At \(g_i=0\) the two maps agree. For \(g_i\ne0\),
\(|h_\epsilon(g_i)-\operatorname{sign}(g_i)|=\epsilon/(|g_i|+\epsilon)\).
Split its expectation into \(0<|g_i|<a\), where it is at most one, and \(|g_i|\ge a\), where it is at most \(\epsilon/(a+\epsilon)\). Apply the Lipschitz bound separately under \(P\) and \(Q\), then a triangle inequality through the two ideal-sign expected utilities. The TV statement is the defining bound for a test in \([0,1]\). \(\square\)

Equations (A8)–(A9) separate exact sufficiency from approximate sufficiency. For any positive epsilon, arbitrary utilities can distinguish gradient magnitudes. For regular utilities and little mass near zero, a sign-based approximation can still be accurate. This is precisely why the original framework needs approximation error and coefficient stability, not only an exact information order.

### Proposition A4: retained optimizer state defeats general sign-only sufficiency

For \(\beta_1<1\), the first complete optimizer state contains
\(m_1=(1-\beta_1)G\), so it retains the entire consumed gradient even when the parameter step is ideal sign descent. Equality of first-step signs alone does not preserve a nonempty learning continuation.

A concrete counterexample uses one coordinate, \(\beta_1=\beta_2=1/2\), \(\lambda=0\), \(\epsilon=0\), \(\eta>0\), and cold state. Two first gradients are \(G_1=1\) and \(G_1'=2\). Both produce \(\theta_1=\theta_0-\eta\). Apply the same second gradient \(G_2=G_2'=-3/4\). The two raw first moments at time two are \(-1/8\) and \(+1/8\), the bias-corrected first moments are \(-1/6\) and \(+1/6\), and the bias-corrected second moments are \(17/24\) and \(41/24\). Therefore
\[
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad
\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}.
\tag{A10}
\]
Their second steps have opposite directions. The bounded test \(F(\theta)=1\{\theta>\theta_1\}\) separates them by one. Smooth sigmoid tests also separate them, though with a smaller gap.

**Proof and realizability.** Substitute the four gradients into (A6), noting \(1-(1/2)^2=3/4\). At \(\theta_0=0\), squared losses \(\frac12(\theta-a)^2\) with first targets \(a=-1,-2\) realize the first gradients. Since both first parameters coincide, a common second target \(a=\theta_1+3/4\) realizes the common second gradient. Thus the example does not require inconsistent branch-dependent future gradients. \(\square\)

Both branches even have the same gradient-sign sequence \((+,-)\). The issue is hidden magnitude-bearing state, not only missing a second-step sign statistic. With positive epsilon the first parameter outputs differ slightly already; the clean equality in (A10) concerns the ideal zero-epsilon update.

This establishes a counterexample, not a claim that every magnitude difference affects every task. For arbitrary complete-state utilities, injectivity of \(m_1\) forces full-gradient equality; for actual deployment tasks, the appropriate smaller object remains (A1).

At a warm state the numerator changes sign at
\(G_i=-\beta_1m_i/(1-\beta_1)\), while the denominator depends on \(v_i\) and \(G_i^2\). Consequently even the relevant threshold depends on the state. For a scalar cross-entropy gradient \(G=p-Z\), cold sign probabilities concern the masses below, at, and above the current \(p\); allowing all thresholds identifies a CDF. No single fixed threshold or marginal sign feature is generally sufficient across changing states.

## 4. Make the number of demonstrations explicit

Condition throughout on a declared common pretrained information state \(I_{\rm pre}\). Extra teacher feedback must not be counted as part of the original \(K\) demonstrations. For a nested experiment that observes \(K\) IID trajectories and is allowed to ignore observations, the information-limited Bayes risk is nonincreasing in \(K\): any rule available at \(K\) remains available at \(K+1\). This does not imply that a fixed neural training recipe improves monotonically, nor that there is a universal decay rate.

### Proposition A5: an exact few-shot coverage model

Within one task family, let \(\Theta_1,\ldots,\Theta_d\) be independent fair bits conditional on \(I_{\rm pre}\). Deployment context \(j\) has probability \(w_j\ge0\), \(\sum_jw_j=1\), and the correct decision is \(\Theta_j\). The student class contains all bit tables, so representation error is zero.

Each trusted teacher trajectory reveals a subset \(S\subseteq\{1,\ldots,d\}\) and exactly the labels \(\Theta_S\). The subsets are IID across trajectories and independent of \(\Theta\). Any additional transcript content has a conditional law depending only on the revealed transcript and independent randomness, so it leaks no unrevealed labels. Dependencies among reveal events within a trajectory are permitted. Let \(q_j=\Pr(j\in S)\).

After \(K\) complete trajectories, the minimum achievable expected regret is
\[
\boxed{R_{\rm info}(K)=\frac12\sum_{j=1}^d w_j(1-q_j)^K.}
\tag{A11}
\]
At \(K=0\), interpret \((1-q)^0=1\), including \(q=1\).

**Proof.** Given the full transcript, a revealed bit is known. Every unrevealed bit is still fair and independent of the transcript: the reveal mechanism is independent of all labels, and the additional content cannot leak it. Its conditional Bayes error is therefore \(1/2\). Bit \(j\) is unrevealed with probability \((1-q_j)^K\), by independence across trajectories. Weighted expectation gives the lower bound. Memorizing revealed bits and guessing unrevealed bits attains it because the student class contains all tables. \(\square\)

The one-demonstration marginal decrease is
\[
R_{\rm info}(K)-R_{\rm info}(K+1)
=\frac12\sum_j w_jq_j(1-q_j)^K.
\tag{A12}
\]
Coordinates with \(q_j=0\) cause an irreducible passive error; small positive \(q_j\) can make convergence arbitrarily slow. Thus no rate depending only on \(K\) follows without structural coverage assumptions. Correlated labels, informative non-revelation, non-IID demonstrations, or useful pretrained knowledge change (A11). It is a Bayes information floor, not the achieved neural student's risk.

### Proposition A6: extra queries repair missing information in this model

After observing the demonstrations, permit at most \(n\) additional noiseless, unit-cost queries, each freely revealing one chosen bit. This is a **hard conditional budget**, not merely an expected budget. Let \(U_K\) be the set of unseen bits, and list its weights in decreasing order as \(w_{(1)},\ldots,w_{(|U_K|)}\). Then
\[
R_{\rm info}(K,n)
=\frac12\mathbb E\left[
\sum_{r=n+1}^{|U_K|} w_{(r)}\right],
\tag{A13}
\]
where an empty sum is zero. Querying the \(n\) largest-weight unseen bits is optimal. Labels do not alter the ranking because hidden bits are independent and all unseen conditional errors are \(1/2\).

**Proof.** Conditional on the transcript, a query to unseen bit \(j\) reduces expected regret by exactly \(w_j/2\). A query to an already known bit has zero benefit. With independent bits no query reduces uncertainty about another. Any decision rule using at most \(n\) queries can therefore remove at most the sum of the \(n\) largest unseen weights, regardless of label-dependent adaptation; querying those bits attains the bound. Average over the initial transcript. \(\square\)

For equal weights and a demonstration revealing one uniform context, let \(M_K=|U_K|\). Then
\[
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},
\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K.
\tag{A14}
\]
One must not replace \(\mathbb E[(M_K-n)_+]\) by \((\mathbb E M_K-n)_+\). The latter is generally strictly smaller. Direct queries may be impossible in sequential environments: access and reachability costs then belong in the acquisition model and (A13) no longer applies.

### Proposition A7: demonstration information credits an adaptive query budget

Consider two task worlds \(v\in\{0,1\}\) requiring different correct final decisions. Condition on the fixed common \(I_{\rm pre}\) as above; all demonstration/query KL bounds and decision error bounds in this proposition are conditional on that state. In direction \(v\) versus \(1-v\), each IID initial demonstration has KL divergence at most \(\kappa_D^{(v)}\). Following \(K\) demonstrations, an adaptive learner selects queries by a common history-dependent rule and stops at \(N\le N_{\max}<\infty\). For every allowable query and preceding history on the relevant support, assume its conditional observation law has directional KL at most \(\kappa_X^{(v)}\). The learner's private randomization is world-independent. Suppose its final decision has error at most \(\delta\) under each world, with \(0<\delta<1/2\).

Take finite nonnegative upper bounds \(\kappa_D^{(v)},\kappa_X^{(v)}\). Then
\[
K\kappa_D^{(v)}+
\kappa_X^{(v)}\mathbb E_vN
\ge \operatorname{kl}(1-\delta,\delta).
\tag{A15}
\]
For \(0<\kappa_X^{(v)}<\infty\),
\[
\mathbb E_vN\ge
\frac{[\operatorname{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}
{\kappa_X^{(v)}}.
\tag{A16}
\]
If \(\kappa_X^{(v)}=0\) and the demonstration term is too small, no allowed query budget can achieve the stated errors. A zero right-hand side in (A16) does not establish achievability.

**Proof.** Pad the transcript with a fixed null symbol after stopping. Conditional query-selection, randomization, and stopping kernels coincide across worlds at a common history, so they contribute no additional KL. The finite-horizon KL chain rule gives transcript divergence at most \(K\kappa_D^{(v)}+\mathbb E_v\sum_{t=1}^N\kappa_X^{(v)}\). Apply data processing to the event that the output equals the correct decision for world \(v\). Its probabilities are at least \(1-\delta\) under \(v\) and at most \(\delta\) under the other world. Binary KL is minimized over that rectangle at \((1-\delta,\delta)\), giving (A15), and rearrangement gives (A16). \(\square\)

These results close the missing explicit \(K\) bookkeeping in transparent finite models. They use classical missing-mass and change-of-measure reasoning [I1–I3]; they are supporting specializations, not standalone novelty claims.

## 5. How the new pieces compose

The original Bayes decomposition separates representation error, residual information error, and learner error. Proposition A5 or A7 characterizes the second term for a declared acquisition problem. Acquiring sufficient information alone does not determine the third term. The update-relative response (A1) is the bridge: it defines what a compiler must preserve for a particular learner to use that information.

For a reference recipe with measured or independently bounded benefit \(\gamma_K=L_E(0;K)-J_E(\pi_{\rm init})\), a candidate with certificate \(\varepsilon_K\) from (A4) obeys
\[
L_E(j;K)-J_E(\pi_{\rm init})\ge\gamma_K-\varepsilon_K.
\tag{A17}
\]
The proof is subtraction. Its content depends on both premises: an effective reference and a valid response/transport certificate. The information floor in (A11) cannot substitute for an achieved \(\gamma_K\). Likewise, a proxy correlation cannot substitute for the uniform terms in (A4).

This gives a coherent interpretation of failure:

| Failure | Mathematical symptom | Appropriate intervention |
|---|---|---|
| Required environmental facts remain unknown | Positive acquisition lower bound or unseen weighted mass | Obtain new allowed evidence; replay alone is insufficient |
| Available feedback is miscompiled for the actual learner | Mismatch against (A1), e.g. Adam state effects missed by affine moments | Change feedback selection/representation using the executed update and suffix |
| Frozen audit is extrapolated to new states | Large or unsupported transport term or exit probability | Audit actual states or prove transport/stability on the visited region |
| Reference training itself is ineffective | Small or negative \(\gamma_K\) | Improve the reference recipe before treating fidelity as success |
| A proxy misses task consequences | Poor held-out contrast prediction or wrong selection signs | Revise/reject the proxy; retain a narrower empirical claim |

There is no theorem here that uniformly compresses arbitrary AdamW training on arbitrary tasks into a few cheap statistics. Propositions A2 and A4 show why such a theorem needs restrictions on the task responses, state region, or approximation tolerance.

## 6. Method consequence: an optimizer-aware instance of CED

Keep Consequence-Calibrated Experience Distillation (CED) as the general method family. Its next implementation should have the following concrete loop, conditioned on the \(K\) initial demonstrations and separately accounted acquisition budget.

1. **Expose uncertainty.** Use the student and allowed task interface to identify decisions whose required environmental evidence is missing. The teacher may label student-visited prefixes if the task permits it. Record genuinely new evidence separately from replay, paraphrase, and synthetic descendants.
2. **Generate executable feedback candidates.** Include the raw teacher reference, ordinary sequence distillation, state-targeted feedback, and any proposed compressed/selected provider. Coupling or regrouping is an optional controlled candidate, with its single-step additive null retained.
3. **Measure the actual learner response.** At restorable pilot states, apply the actual optimizer update. Candidate features may include task probes, \(\Delta\theta\), \(\Delta m\), \(\Delta v\), and their joint projections. A full next-state law is sufficient but usually intractable; compressed features must earn their relevance against declared task continuations.
4. **Fit and calibrate a bounded response model.** Learn provider contrasts against independent full-tail deployments on a limited audit set. Optimize a surrogate of the coefficient-stability/approximation tradeoff in Theorem 18. Without a justified uniform residual and state transport, label it a calibrated predictor rather than a certificate.
5. **Select under measured cost and test transfer.** Compare the selected provider with matched-budget reference and baseline arms on held-out complete training runs, tasks, seeds, and query budgets. Freeze selection before final evaluation. The final student deploys without teacher help.

A provisional selection score may combine predicted task benefit, an out-of-sample uncertainty penalty, and measured acquisition/training cost. Its weights, fit, and selection rule must be fixed on development data. This is a practical hypothesis; the present mathematics does not prove that the learned score is optimal or that it improves Gemma.

The novel contribution must ultimately be a substantive, falsifiable result about this acquisition–update–task relationship. Adding Adam features to a data selector is insufficient by itself. The stronger target is a response family with a useful approximation guarantee, an accessible way to obtain its required environmental information, and a method that preserves task value across held-out conditions at a demonstrated cost advantage.

## 7. Experiments that discriminate the theory

The companion coding-agent document gives runnable obligations and output schemas. Its order is intentional.

| Stage | Question | Essential controls | Evidence required |
|---|---|---|---|
| Finite correction | Does the repaired certificate expose missing coverage? | Original counterexample; exact transport; nonzero transport; exit repair | Exact regression retains violation of old statement and verifies corrected terms |
| Optimizer mechanism | Is C23 caused by normalization, memory, or feature change? | SGD; explicit sign update; exact finite-epsilon cold Adam; retained-state Adam; optimizer-matched fixed-Jacobian model | Same source laws, raw gradients, parameter/moment states, exact support expectations where possible |
| Few-shot acquisition | Does \(K\) reduce the right uncertainty? | Independent demonstrations; replay; synthetic descendants; separately budgeted direct queries | Exact (A11)–(A14) first, then real-task curves with access assumptions declared |
| Neural proxy validity | Do inexpensive features predict full-tail task contrasts? | Task-loss/gradient baseline; parameter-only; parameter-plus-moments; task-response features | Held-out sign errors, calibration, tail-length sensitivity, task/recipe/state transfer |
| Actual student pilot | Does the method survive the intended model and optimizer? | Matched evidence, teacher calls, tokens, updates, and evaluations | Actual checkpoint, actual LoRA/AdamW execution, task outcomes, resource receipts |
| Cross-task result | Is the learned relation broader than one trained selector? | Leave-task-family-out and leave-recipe/state-region-out evaluation | No retraining on test consequences; uncertainty and negative-transfer reporting |

The initialization-tangent affine-SGD baseline must not be called an Adam predictor. A fixed-Jacobian model using nonlinear cross-entropy and Adam is generally not affine in source statistics either. Preserve the same loss, optimizer state recurrence, clipping, and sampling in the optimizer-matched comparator; its remaining mismatch with the neural run measures more than one effect unless other nonlinear components are also controlled.

For smooth local approximations, a one-step \(O(\eta^2)\) remainder does not alone imply a uniform \(O(R\eta^2)\) training error. If an error recursion is \(e_{t+1}\le(1+\eta L)e_t+C\eta^2\), then
\(e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s\) for \(e_0=0\). Linear-in-\(R\) simplification requires an appropriate stability or short-horizon condition. A neural response theorem must also control state/feature drift and the relevant continuation regularity.

### Certificate cost is a design question, not a proved impossibility

For the current fixed-sample Hoeffding radius in Theorem 20, \(M=4,\delta=0.05,e=0.1\) requires
\[
n\ge\left\lceil\frac{2\log(2M/\delta)}{e^2}\right\rceil=1016
\]
independent paired replicas per audited candidate/state. This is a sufficient distribution-free allocation, not a lower bound on every valid certificate and not a measured 12B runtime.

Useful alternatives are paired common-randomness experiments with correct marginals, variance-sensitive valid bounds, independently justified transport, a small number of full-tail anchor audits, and randomized complete-training comparisons for a narrower population-average method claim. Optional stopping or adaptive replication needs a valid sequential confidence method. Reference reuse across candidates can reduce work but does not create independence across reused contrasts. Shorter tails and learned critics change the estimand unless a separate tail/critic error bound is established.

Whole-run randomized comparisons can validate the mean outcome of a frozen method on a declared task/seed population without certifying every adaptive local decision. They cannot be relabeled as local Theorem 20 guarantees. This distinction allows an honest affordable empirical paper while the stronger certificate problem remains open.

## 8. What remains open, and what is now established

The coverage erratum is a resolved mathematical defect under the new explicit assumptions. Propositions A2–A4 give a rigorous boundary between immediate-parameter, complete-state, and task-specific sufficiency for Adam. Propositions A5–A7 make \(K\) and additional queries explicit in finite acquisition models. Their proofs are included, their finite witnesses are executable, and their novelty boundaries are recorded in the separate audit.

The work still lacks a nonvacuous neural response approximation/transport theorem on a useful training region; a validated inexpensive proxy for full-tail task effects; and real few-shot agent gains that transfer across tasks at matched evidence and compute. These are separate obligations, not a percentage-complete estimate. Nothing in this addendum establishes an ICML best-paper-level contribution.

An appropriate next central theorem would identify a nontrivial optimizer/task/state family for which (i) attainable acquisition controls a specific response uncertainty, (ii) a compact measurable representation approximates that response with stable coefficients, and (iii) the resulting task-value guarantee remains useful along adaptive training. The candidate theorem must improve on the existing general duality and coverage tools rather than simply restate them. Whether such a theorem holds is a research question; do not treat this agenda as a proved result.

## References and attribution

- **[O1]** Kingma and Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015. Algorithm 1. https://arxiv.org/abs/1412.6980
- **[O2]** Loshchilov and Hutter. *Decoupled Weight Decay Regularization*. ICLR 2019. https://arxiv.org/abs/1711.05101
- **[O3]** Balles and Hennig. *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients*. ICML 2018. https://proceedings.mlr.press/v80/balles18a.html
- **[O4]** Abuduweili and Liu. *Revisiting the Initial Steps in Adaptive Gradient Descent Optimization*. CPAL 2025. Section 2.1. https://arxiv.org/abs/2412.02153
- **[O5]** Bernstein et al. *signSGD: Compressed Optimisation for Non-Convex Problems*. ICML 2018. https://proceedings.mlr.press/v80/bernstein18a.html
- **[I1]** Rajaraman et al. *Toward the Fundamental Limits of Imitation Learning*. NeurIPS 2020. Missing-mass analysis and lower-bound construction. https://arxiv.org/abs/2009.05990
- **[I2]** Judah et al. *Active Imitation Learning: Formal and Practical Reductions to I.I.D. Learning*. JMLR 2014. https://www.jmlr.org/papers/v15/judah14a.html
- **[I3]** Zhang et al. *When Does Online Imitation Learning Help in LLM Post-Training? The Role of (Non-)Realizability Beyond Horizon*. 2026 preprint. https://arxiv.org/abs/2606.30445

The supporting optimizer identities, probability-law characterization, missing-mass formulas, and KL chain-rule arguments are not presented as discoveries of these underlying mechanisms. The v0.14 reference list and the targeted v0.15 novelty audit provide the broader context.
