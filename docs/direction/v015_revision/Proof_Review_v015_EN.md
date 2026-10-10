# Independent Proof Review: v0.15 Corrections and Extensions

**Date:** 2026-10-10.  
**Review type:** independent model-agent mathematical review, not external peer review or formal verification.  
**Scope:** the four corrective/additional derivations supplied for v0.15. The frozen v0.14 audit is not edited.

## Assessment

The proposed transport correction, cold-Adam characterization, coverage model, and combined demonstration/query lower bound are valid under the conditions recorded below.

Two substantive findings require explicit action:

1. The reported state-to-audited-cell gap in v0.14 Theorem 19 is genuine. Membership in a “certified” state set did not itself connect a state's conditional feature discrepancy to the finite audited cell discrepancies. My earlier integration review did not identify that missing premise. The corrected theorem must add the transport bound below or restrict its certified set to states with exact audited laws.
2. The proposed two-step Adam counterexample used the wrong second gradient for its claimed opposite signs. With first gradients 1 and 2, \(\beta_1=1/2\), and second gradient \(-3/2\), both second first moments are negative. Replacing the second gradient by \(-3/4\) gives the intended counterexample.

The derivations remain specializations of the general environment–learner response framework. They do not establish novel decision theory, a new Adam identity, or a general neural performance certificate.

## V15.1. Transport from an actual learner state to an audited cell

### The missing premise is necessary

In v0.14, the finite audit controlled feature discrepancies at cells \(x\). The learning-value bound then used those discrepancies at actual states \(z\). A response expansion in the audited feature coordinates is insufficient without a relation between the conditional laws at \(z\) and the laws audited at \(x\).

A one-round finite witness makes the problem explicit. Let the message be \(Y\in\{0,1\}\), the feature be \(\phi(Y)=Y\), and the actual update set the final policy parameter to \(p=Y\). Let deployment utility be \(F(p)=p\), so the response is exactly \(g(Y)=Y\), with coefficient bound \(B=1\) and residual zero.

At the audited cell \(x_a\), both providers output 0. Their true audited discrepancy is zero. At the actual initial state \(z_b\), the reference outputs 1 and the candidate outputs 0. If \(z_b\) is declared “certified” merely because its response has the same feature expansion, the exit probability is zero while the actual learning-value difference is 1. Even an exact audit at \(x_a\) cannot certify zero error at \(z_b\).

This is an assumption gap in the stated application of the audit, not a failure of the telescoping identity.

### Corrected local statement

For each relevant time/state and candidate \(j\), specify:

- an audited cell \(x=x_t(z)\);
- common feature coordinates, including any explicit transport needed when message spaces or state-dependent feature definitions differ;
- the actual conditional feature-discrepancy vector
  \[
  \Delta_{t,j}(z)
  =
  \mathbb E_{0,t,z}\phi_{t,z}(Y)
  -
  \mathbb E_{j,t,z}\phi_{t,z}(Y);
  \]
- the corresponding audited-cell discrepancy vector
  \[
  \Delta^{\rm audit}_j(x)
  =
  \mu_{0,x}-\mu_{j,x};
  \]
- a valid bound
  \[
  \|\Delta_{t,j}(z)-\Delta^{\rm audit}_j(x_t(z))\|_\infty
  \le a_{t,j}(z).
  \tag{V15.1}
  \]

The feature vectors in this display must have the same declared meaning and dimension. Simply assigning a nearby checkpoint's name to \(z\) does not establish this premise.

Let
\[
D_j=\max_x\|\Delta^{\rm audit}_j(x)\|_\infty.
\]
If the actual response admits
\[
g_{t,z}=c_{t,z}+b_{t,z}^{\top}\phi_{t,z}+r_{t,z},
\qquad
\|b_{t,z}\|_1\le B_t(z),\quad
\|r_{t,z}\|_\infty\le\tau_t(z),
\]
then
\[
|(\mathbb E_{0,t,z}-\mathbb E_{j,t,z})g_{t,z}|
\le
B_t(z)\{D_j+a_{t,j}(z)\}+2\tau_t(z).
\tag{V15.2}
\]

**Proof.** The transport assumption and triangle inequality bound the actual discrepancy norm by \(D_j+a_{t,j}(z)\). Constants cancel; the linear term is bounded by the coefficient \(\ell_1\) norm times that discrepancy; the two residual expectations contribute at most \(2\tau_t(z)\). No additional approximation argument is hidden.

The bound may be truncated at 1 for responses in \([0,1]\).

### Corrected propagation and statistical audit

On the fixed-provider audit event from v0.14,
\[
D_j\le\widehat D_j+e_0+e_S
\]
simultaneously over the frozen candidates. Let \(\widehat j\) be the selected candidate. Stop at the first state where the cell mapping, transport bound, response approximation, or other certification conditions cease to hold; cap the stopping time at the training horizon. If the exit probability conditional on the frozen audit result is at most \(\rho_{\widehat j}\) on the stated certification event, then
\[
\begin{aligned}
|L_E(0)-L_E(S_{\widehat j})|
\le{}&
\mathbb E_{S_{\widehat j}}
\sum_{t<\tau_{\rm exit}}
\Big[
B_t(Z_t)
\{\widehat D_{\widehat j}+e_0+e_S+
a_{t,\widehat j}(Z_t)\}
+2\tau_t(Z_t)
\Big]
+\rho_{\widehat j}.
\end{aligned}
\tag{V15.3}
\]

This is the original stopped telescoping bound with the corrected local discrepancy. The expectation is over the selected fresh training process conditional on the frozen audit result, including the exit history. An exit probability averaged over different audit outcomes is not enough for a bound asserted on each certified audit outcome. Shared initialization and the same terminal real-environment utility remain required.

Transport, residual, and exit-probability bounds must hold for the selected provider. If they are estimated statistically, their confidence failures must be included in the total probability accounting; candidate selection cannot evade that requirement.

**Exact cases and limits.**

- Exact conditional feature-law matching between \(z\) and its audited cell for both providers gives \(a_{t,j}(z)=0\). In fact equality of their discrepancy vectors is enough; full-law equality is stronger than needed.
- A state with no justified transport belongs outside the certified set. It cannot be assigned zero transport error by convention.
- Small geometric distance between neural checkpoints is not itself a transport bound.
- In the counterexample above, the correct transport error is 1, recovering the exact value gap; alternatively the actual state is uncertified and the exit penalty is 1.
- The IID affine certificate, v0.14 equation (40), remains valid under its explicitly state-independent block laws and fixed update conditions. It does not require interpolation from a bank of conditional neural states. The moment law actually used in training must still be the audited fixed law.

**Verdict:** the repair is valid and necessary. The correction supersedes the earlier unqualified integration acceptance of the state-to-cell step.

## V15.2. Cold bias-corrected Adam/AdamW: exact and approximate information

### Algorithm and arithmetic assumptions

Use ideal real-arithmetic Adam with denominator \(\sqrt{\widehat v}+\epsilon\), or AdamW with decoupled deterministic weight decay. The initial moments and step counter are cold:
\[
m_0=v_0=0,\qquad t=0,
\]
with \(\beta_1,\beta_2\in[0,1)\), fixed step size \(\eta>0\), and fixed initial parameter \(\theta\). Let \(G\) be the gradient actually supplied to the moment update, including any declared preprocessing such as clipping. There is no additional projection or parameter quantization in the mathematical update.

At the first step,
\[
m_1=(1-\beta_1)G,\quad
v_1=(1-\beta_2)G^{\odot2},\quad
\widehat m_1=G,\quad
\widehat v_1=G^{\odot2}.
\]
Thus
\[
\theta^+=(1-\eta\lambda)\theta-\eta h_\epsilon(G),
\qquad
h_\epsilon(g)=\frac{g}{|g|+\epsilon}
\tag{V15.4}
\]
coordinatewise. For ordinary Adam without decoupled decay, take \(\lambda=0\); any coupled regularization is part of \(G\).

This formula does not automatically apply to warm moments, another epsilon convention, a changed step counter, or a projected/quantized optimizer.

### Exact zero-epsilon parameter-only result

For \(\epsilon=0\), either assume no zero gradient coordinates or explicitly define \(h_0(0)=0\), so \(h_0=\operatorname{sign}\). Literal division by zero in a standard implementation is not silently repaired by the mathematical theorem.

Fix the initial parameter and all deterministic update settings. For two gradient laws \(P,Q\), equality of
\[
\mathbb E_P F(\theta^+)
=
\mathbb E_Q F(\theta^+)
\]
for every bounded measurable final-parameter utility \(F\) is equivalent to equality of the **joint** sign-vector laws.

**Proof.** The parameter update is an injective affine map of the sign vector because \(\eta>0\). Matching joint sign laws therefore matches final parameter laws. Conversely, all bounded indicator utilities identify the final parameter distribution, and the inverse affine map recovers the sign-vector law.

If the admissible gradient domain has \(s\) attainable joint sign patterns and arbitrary probability laws over these patterns are permitted, the minimum number of independent linear expectation statistics sufficient for all such utilities is \(s-1\). Indicators of all but one pattern suffice, and the finite-alphabet annihilator argument proves necessity.

This is a parameter-only one-step result. Matching coordinatewise sign frequencies alone need not match joint signs. If some coordinates have zero step size, only their actively updated counterparts can be identified in this way.

### Positive epsilon and arbitrary utilities

For \(\epsilon>0\), each coordinate map \(h_\epsilon:\mathbb R\to(-1,1)\) is a continuous bijection, with
\[
h_\epsilon^{-1}(u)=\frac{\epsilon u}{1-|u|}.
\]

Consequently, matching expected utility for every bounded measurable final-parameter utility is equivalent to matching the entire law of \(G\), on its attainable domain. This follows from the same pushforward/inverse-map argument.

For unrestricted continuous gradient laws, no fixed finite collection of linear expectation statistics can determine all those laws. On a finite attainable gradient alphabet of size \(s\), the corresponding unrestricted minimum is \(s-1\).

The apparent discontinuity at \(\epsilon=0\) is genuine for arbitrary measurable utilities: such a utility can distinguish arbitrarily close parameter values. It is not a claim that tiny epsilon causes an operationally large difference for smooth task utilities.

### Lipschitz approximation by signs

Suppose
\[
|F(u)-F(v)|\le\sum_iL_i|u_i-v_i|
\]
on the relevant parameter domain. Define, for \(a>0\),
\[
\beta_{P,i}(a)=P(0<|G_i|\le a),
\]
and similarly for \(Q\). The looser definition \(P(|G_i|\le a)\) also gives a valid bound.

For any \(g\ne0\),
\[
|h_\epsilon(g)-\operatorname{sign}(g)|
=\frac{\epsilon}{|g|+\epsilon},
\]
while the difference is zero at \(g=0\) under the stated sign convention. Hence
\[
\mathbb E_P|h_\epsilon(G_i)-\operatorname{sign}(G_i)|
\le \beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}.
\]

If \(P,Q\) have identical joint sign laws, their zero-epsilon utility expectations agree, and triangle inequality gives
\[
\boxed{
\left|
\mathbb E_PF(\theta_\epsilon^+)
-
\mathbb E_QF(\theta_\epsilon^+)
\right|
\le
\eta\sum_iL_i
\left[
\beta_{P,i}(a)+\beta_{Q,i}(a)
+\frac{2\epsilon}{a+\epsilon}
\right].
}
\tag{V15.5}
\]

Without joint sign matching, add the actual zero-epsilon utility discrepancy. For \(F\in[0,1]\), that extra discrepancy is at most the total variation distance between joint sign laws. The Lipschitz constants and the margin probabilities are assumptions or quantities requiring valid estimation, not automatically known neural metrics.

### Why sign information is insufficient for a suffix

The next **full optimizer state** includes
\[
m_1=(1-\beta_1)G,\qquad
v_1=(1-\beta_2)G^{\odot2}.
\]
It retains magnitude information even when the first parameter update uses only signs. In particular, \(m_1\) is injective in \(G\) for \(\beta_1<1\).

The initially proposed opposite-sign example with second gradient \(-3/2\) is incorrect. The corrected example is:

- scalar initial parameter, \(\lambda=0,\eta>0,\epsilon=0\);
- \(\beta_1=\beta_2=1/2\);
- first gradient \(G_1=1\) versus \(G_1=2\);
- second gradient \(G_2=-3/4\) in both processes.

Both first parameter updates equal \(\theta_1=\theta_0-\eta\), but at step 2:

| First gradient | \(m_1\) | \(m_2\) | \(\widehat m_2\) | \(\widehat v_2\) |
|---|---:|---:|---:|---:|
| \(1\) | \(1/2\) | \(-1/8\) | \(-1/6\) | \(17/24\) |
| \(2\) | \(1\) | \(1/8\) | \(1/6\) | \(41/24\) |

The denominators are positive, so the second updates move in opposite directions from their common \(\theta_1\). A bounded final-parameter utility can distinguish them. The same second gradient is feasible, for example by using a linear-loss second example.

With the original \(G_2=-3/2\), the two raw \(m_2\) values would instead be \(-1/2\) and \(-1/4\); both are negative.

**Verdict:** the cold one-step characterization and Lipschitz bound are valid with the qualifications above. The corrected example proves that preserving first-step parameter information is weaker than preserving the information required for future optimization. This is directly consistent with the framework's complete-state requirement.

## V15.3. Few-shot coverage and optimal noiseless follow-up queries

### Precisely stated model

Let \(\Theta_1,\ldots,\Theta_d\) be independent fair bits. A deployment context \(j\) has known nonnegative weight \(w_j\), with \(\sum_jw_j=1\). The student can execute a full context-to-action table. Its utility is weighted correctness:
\[
J_\Theta(\pi)=\sum_jw_j\,\Pr_\pi(A_j=\Theta_j\mid j),
\]
so an informed optimal student has utility 1.

Each demonstration reveals a random subset \(S\) and the noiseless labels \(\Theta_S\). The subset law is independent of all bits, and the \(K\) subsets are IID. The observed transcript contains no additional information about the unobserved bits, except independent noise. Write \(q_j=\Pr(j\in S)\).

Let \(U_K\) be the coordinates not revealed by any of the \(K\) demonstrations. Conditional on the transcript, every bit in \(U_K\) remains independent and fair. The Bayes-optimal expected regret is therefore
\[
\frac12\sum_{j\in U_K}w_j.
\]
Since \(\Pr(j\in U_K)=(1-q_j)^K\),
\[
\boxed{
\mathcal R_{\rm info}(K)
=
\frac12\sum_{j=1}^dw_j(1-q_j)^K.
}
\tag{V15.6}
\]

No independence among coordinates within a single subset is needed. Independence across demonstrations and from the hidden bits is needed.

This is an information lower bound for the specified prior/model, achieved by using known labels and arbitrary guesses on unseen bits. It does not establish that a neural optimizer learns the observed table or that actual teacher-trajectory coverage is independent of task difficulty.

### Optimal additional queries

Conditional on the observed transcript, allow at most \(n\) noiseless unit-cost queries, each revealing any chosen unobserved bit. Here \(n\) is a hard per-transcript integer budget, not merely an expected budget.

Revealing unseen coordinate \(j\) reduces posterior Bayes regret by exactly \(w_j/2\), independently of its revealed value. Therefore an optimal query policy reveals the \(\min(n,|U_K|)\) largest unseen weights. Value-adaptive choices cannot improve this result because all remaining unseen bits remain independent and fair.

If \(T_n(U_K)\) denotes those selected coordinates, the optimal final risk is
\[
\boxed{
\mathcal R_{\rm info}(K,n)
=
\frac12\mathbb E
\sum_{j\in U_K\setminus T_n(U_K)}w_j.
}
\tag{V15.7}
\]

If every demonstration exposes one uniformly drawn context, and \(w_j=1/d\), let \(M_K=|U_K|\). Then
\[
\boxed{
\mathcal R_{\rm info}(K,n)
=\frac{1}{2d}\mathbb E[(M_K-n)_+].
}
\tag{V15.8}
\]

The positive part must remain **inside** the expectation. For \(d=3,K=3,n=1\), exact enumeration gives
\[
\mathbb E M_K=8/9,\qquad
\mathcal R_{\rm info}(K,0)=4/27,\qquad
\mathcal R_{\rm info}(K,1)=1/54.
\]
Incorrectly substituting \((\mathbb E M_K-1)_+\) would give zero.

**Limits.**

- Correlated or nonfair bits change the posterior uncertainty and the value of queries.
- Selection of \(S\) based on hidden bits can itself reveal information.
- Noisy or unequal-cost queries require a different optimization.
- A constraint only on expected query count permits allocation across transcripts and is not the hard-budget problem above.
- The model assumes any unobserved coordinate can actually be queried.
- \(K\) counts complete demonstrations under the declared subset-exposure process. It does not count task families or optimizer epochs.

**Verdict:** the coverage and query formulas are exact in this finite model.

## V15.4. Demonstration information plus adaptive query information

### Correct directional statement

Let the unknown world be \(v\in\{+,-\}\). For each direction \(v\), suppose
\[
\mathrm{KL}(P_v^{\rm demo}\|P_{-v}^{\rm demo})
\le\kappa_{D,v},
\]
where the \(K\) demonstrations are conditionally IID given the world. Their initial transcript divergence is at most \(K\kappa_{D,v}\).

After those demonstrations, a common algorithm chooses queries adaptively. For every permitted query and relevant history, assume
\[
\mathrm{KL}\!\left(
P_v(\cdot\mid{\rm history},x)
\|P_{-v}(\cdot\mid{\rm history},x)
\right)
\le\kappa_{X,v}.
\]
The algorithm's private randomization has no additional world information. All informative observations are included in these transcripts. Suppose its stopping query count \(N\) has finite expectation in both worlds, and its final binary decision has error at most \(\delta\) in each world, with \(0<\delta<1/2\).

Then
\[
\boxed{
K\kappa_{D,v}
+\kappa_{X,v}\mathbb E_vN
\ge
\operatorname{kl}(1-\delta,\delta)
}
\tag{V15.9}
\]
for each direction. Common upper bounds \(\kappa_D,\kappa_X\) can replace the directional bounds.

**Proof.** The transcript KL chain rule adds the initial demonstration divergence and the expected sum of conditional query divergences. The common adaptive query/stop rules add no separate divergence. Thus the full transcript divergence is at most the left side. Data processing to the final binary decision gives at least \(\operatorname{kl}(1-\delta,\delta)\). Combining the bounds proves the result. Truncation and the finite-expectation assumptions justify the stopped-transcript argument.

For \(0<\kappa_{X,v}<\infty\),
\[
\boxed{
\mathbb E_vN
\ge
\frac{
[\operatorname{kl}(1-\delta,\delta)-K\kappa_{D,v}]_+
}{\kappa_{X,v}}.
}
\tag{V15.10}
\]

The “demo KL” used here must be its actual directional value or an **upper** bound. A lower bound on demonstration KL cannot be substituted into this inequality.

### Edge cases and interpretation

- If \(\kappa_{X,v}=0\) and \(K\kappa_{D,v}<\operatorname{kl}(1-\delta,\delta)\), the declared observations cannot support such a uniformly correct algorithm, regardless of the finite number of these uninformative queries. Do not divide by zero.
- If \(K\kappa_{D,v}\) exceeds the testing threshold, the query lower bound becomes zero. This does not prove that demonstrations alone are sufficient or provide an attaining algorithm.
- Infinite KL upper bounds make the associated inequality potentially vacuous.
- Uniformly correct binary identification is the target here. A policy-learning version requires disjoint near-optimal output sets or an explicit reduction from successful policies to the binary decision.
- Adaptive or dependent demonstrations require their own conditional KL sum; the IID factor \(K\kappa_D\) cannot be retained automatically.
- Each counted query must be the declared full information-bearing experiment. A multi-step branch cannot be split or merged without changing the relevant divergence accounting.
- In the noiseless bit-coverage model, distinct worlds may have infinite demonstration KL whenever a differing bit has positive probability of being exposed. The present finite-KL bound may then be vacuous; this does not contradict the exact Bayes coverage formula. They analyze different aspects under different assumptions.

**Verdict:** the lower bound and positive-part form are valid with the stated orientation, access, stopping, and error conditions. They are standard change-of-measure consequences.

## Independent numerical checks in this review

Separate exact-arithmetic calculations, independent of the proposed extension's implementation, checked:

| Check | Result |
|---|---|
| Corrected two-step Adam moments | \(\widehat m_2=-1/6,+1/6\); \(\widehat v_2=17/24,41/24\) |
| Original proposed Adam sign claim | Rejected: \(G_2=-3/2\) gives two negative \(m_2\) values |
| Uniform coverage, \(d=3,K=3\) | Passive risk \(4/27\) |
| Same model with one optimal query | Risk \(1/54\), not zero |
| Weighted subset example, weights \(1/2,1/3,1/6\), subsets \(\{1\}\) or \(\{2,3\}\) each with probability \(1/2\), \(K=2\) | Passive risk \(1/8\); one-query risk \(1/48\) |

These finite checks support arithmetic and implementation consistency. The proofs and conditions above, not numerical examples alone, justify the general statements.

## Relationship to the unified framework

The correction and extensions occupy distinct positions in the same logical chain:

- The coverage model determines what the initial \(K\) demonstrations reveal and what additional queries can resolve.
- The cold-Adam analysis computes a response quotient for one explicitly specified update and shows how utility regularity changes approximate information sufficiency.
- The two-step counterexample demonstrates why continuation responses must depend on the complete optimizer state.
- The transport correction connects an audited response coordinate system to actual visited states, which is required before a finite-feature audit can propagate into a learning-value statement.
- The two-world inequality quantifies a necessary tradeoff between initial demonstration information and additional permitted experiments.

None of these steps removes the need to verify real neural response metrics, access assumptions, or cross-task performance. In particular, the state-to-cell error must be bounded, included in an exit event, or avoided by direct actual-state auditing. It cannot be renamed away.

## Final exact-text integration review and signoff

**Date:** 2026-10-10. The corrected main document and the complete written v0.15 addendum were checked against the derivations above. Both substantive findings listed at the beginning of this review have been incorporated: the main theorem now states the transport premise, and the addendum uses the corrected second gradient \(-3/4\). No further specific mathematical correction is outstanding in the reviewed additions.

| Written statements | Integration verdict and necessary scope |
|---|---|
| Main Theorem 19, (39), (39a), (40); addendum (A1)–(A5) | Valid with the explicit transport premise, common feature coordinates, simultaneous candidate coverage, fresh training law, and exit probability conditional on the frozen audit result. The stopped proof switches permanently to the reference after first exit and incurs one exit penalty. Equation (40) remains restricted to the declared state-independent IID affine model. |
| Addendum Proposition A2, (A6)–(A8) | Valid for the executed cold bias-corrected update in real arithmetic. The zero-epsilon result uses the declared zero-coordinate convention and immediate parameter utilities. Positive epsilon gives the stated injective map; rounding or projection can change the quotient. The finite dimensions concern unrestricted probability laws and exact linear expectation statistics. |
| Addendum Proposition A3, (A9) | Valid for the stated Lipschitz utilities with nonnegative \(L_i\). The strict near-zero event \(0<|G_i|<a\) used in the final text is correct; zero atoms contribute no sign-approximation error. The nonmatching-sign term and total variation qualification are retained. |
| Addendum Proposition A4, (A10) | The corrected moment values and opposite second-step directions are correct, and the common second squared-loss target realizes the asserted gradient. This is an existence counterexample to sign-only continuation sufficiency, not a claim about every task. |
| Addendum Propositions A5–A6, (A11)–(A14) | Valid conditional on the fixed common pretrained information state, independent fair hidden bits, the stated nonleaking reveal mechanism, an executable full student table, and the hard conditional query budget. The positive part remains inside the expectation. |
| Addendum Proposition A7, (A15)–(A16) | Valid with the written finite nonnegative directional KL upper bounds, world-independent algorithm, conditional error guarantees, and bounded stopping \(N\le N_{\max}<\infty\). The final document explicitly conditions all premises on the same fixed \(I_{\rm pre}\). Bounded stopping is a sufficient stronger condition than the finite-expectation extension discussed earlier in this audit. Finite bounds avoid undefined zero-times-infinity boundary expressions. |
| Addendum (A17) and method implications | The subtraction identity is valid. A measured reference benefit and a valid response/transport certificate remain separate premises. No neural performance or best-paper-level novelty claim follows from this identity. |

The implementation review examined the posterior/query dynamic program, adaptive transcript-KL enumeration, and stopped-occupancy calculation in the primary-authored finite verifier. The dynamic program correctly operates on unnormalized posterior masses and permits at most the query budget; it does not merely restate the top-weight formula. The stopped calculation removes exited mass once and weights local errors by surviving candidate occupancy. The \(K=0\) empty-product calculation uses an exact Fraction start. No relaxed tolerance was needed for the exact posterior equalities.

I separately executed those three existing verifier groups; all passed. This is a targeted re-execution and review of primary-authored code, distinct from the separately derived exact-arithmetic examples recorded earlier. The primary result artifact reports 13 passing finite groups. Neither activity reproduces the user's external C23 suite or establishes real neural training performance. The recorded result artifact was not overwritten by this review.

### Exact reviewed versions

SHA-256 hashes identify the source text and finite artifacts covered by this integration record:

| File | SHA-256 |
|---|---|
| v014/Environment_Information_Theory_EN.md (version 0.14.1) | 7fd67fa9b05ce716a5ab8767ec652b76470d6ec9fa75797881993b4313ef13a3 |
| v015/Optimizer_State_and_Few_Shot_Theory_v015_EN.md | c65ecc610ab028946f8ed05e1c0c8e0f36a623df755be403a92790499d7621f3 |
| v015/verify_v015_extensions.py | 695370fdf48576f447e76b9a32e57aacd0fdb2e5050a4e38113dcd8037f51dc6 |
| v015/results_v015.json | a3ee3d80d7fdd7cbea84b6e8e6a3952f0bb1172e0e6f869604737e5fa09afe9e |

This signoff is an independent model-agent review of the listed mathematics and integration, not formal verification, external peer review, or a certification of originality. The earlier v0.14 audit remains frozen; its acceptance of the missing state-to-cell step is superseded by this explicit correction. The present audit is now frozen.

