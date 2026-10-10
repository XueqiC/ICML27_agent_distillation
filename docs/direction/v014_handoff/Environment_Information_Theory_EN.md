# What Must an Environment Communicate to a Learning Agent?

## A theory of decision-relevant acquisition and learner-relative information sufficiency

**Version:** 0.14, 10 October 2026 UTC.  
**Status:** English research framework with explicit proofs, independently reviewed finite claims, and an executable empirical research specification. No real neural-agent training result is asserted. Independent reviews are model-agent reviews, not external peer review or formal verification.  
**Central object:** an environment-to-learner information interface. A world model, replay buffer, teacher, tool, human, simulator, or structured feedback channel is a possible implementation.

### Abstract

What information must an environment transmit for a particular learner to acquire a desired capability? Answering this requires separating what the student can execute, what the available experiments can reveal, and what the update procedure can exploit. We formulate these distinctions and derive an operational characterization of information sufficiency: a feedback interface is interchangeable with a reference interface for every declared one-boundary learning intervention precisely when it preserves the expectations of the corresponding continuation-response functions. Their span determines exact linear-statistic sufficiency; their uniform approximation width determines approximate sufficiency; their composition through updates determines which feedback interactions matter. A finite-horizon, affine-learning specialization gives necessary and sufficient joint-moment conditions, sharp hierarchy witnesses, a training-round-independent discrepancy bound, and a first-unmatched-cumulant decay law. Polynomial updates explain how learning can amplify information order. Scalar approximation and multivariate parity constructions show why exact degree alone does not measure practical information complexity. Finite-sample certificates and a cross-family transfer criterion connect the theory to measurement. We derive Consequence-Calibrated Experience Distillation as a family of acquisition, feedback-compilation, and training-interface procedures. Its neural implementation is a research hypothesis requiring actual updates, reference continuations, held-out calibration, and cross-task evaluation.

This is one deductive chain, not a claim that each mathematical component is original. Classical decision theory, duality, polynomial algebra, weak approximation, and concentration are credited as tools. The proposed integrated contribution and its stronger specializations remain subject to the accompanying novelty audit.

## 1. The question, the logical chain, and the scope of completeness

A deployed agent acts on the information it receives. A learning agent also changes through a particular update procedure. Two interfaces may describe the world equally well for a fixed policy yet train different students. Conversely, preserving an entire world distribution may be unnecessary for a restricted learning objective.

The governing question is:

> Relative to specified deployment tasks, permitted experiments, and learning procedures, which distinctions between feedback laws can change the eventual real task performance?

The following chain answers that question at progressively more constructive levels.

| Link | Question | Mathematical object or result | Why the next link is necessary |
|---|---|---|---|
| Feasibility | What can the student execute? | Deployment policy class and representation gap | No training signal can remove a prohibited execution capability |
| Acquisition | What facts must be learned? | Decision ambiguity under allowed experiments | Reformatting uninformative observations cannot resolve conflicting decisions |
| Learning consequence | How does feedback affect capability? | Feedback → actual update → remaining learning → deployment | Static policy values and teacher quality do not determine training outcomes |
| Sufficiency | Which distinctions may be discarded? | Continuation-response equivalence | A usable interface should preserve relevant effects, not every world detail |
| Complexity | How many statistics and what order are needed? | Exact span, approximate width, update composition | “Sufficient information” must become an analyzable requirement |
| Verification | Can sufficiency be checked with finite data? | Explicit feature, sampling, coverage, and transfer terms | An unobserved response function is not already an empirical metric |
| Method | How should feedback be acquired and compiled? | Consequence-calibrated interface selection | Fidelity must connect to useful learning and held-out task performance |

Completeness here means that assumptions, implications, counterexamples, and empirical obligations are explicit for the declared finite models. It does not mean a finished theory of arbitrary foundation-model learning, a universal efficient algorithm, or an award-level novelty determination.

### 1.1 Distinct quantities

- \(K\): initial complete trusted teacher demonstrations **per task family**.
- \(n_{\rm query}\): additional permitted environment experiments.
- \(n\): independent experience blocks.
- \(R\): learning updates or declared macro-updates.
- \(N\): deployment decisions.
- \(d\): number of independent linear expectation statistics.
- \(\varepsilon\): tolerated task-performance discrepancy.

These are not interchangeable notions of sample complexity. Replaying one episode \(n\) times does not produce \(n\) independent environmental observations. A low-dimensional real-valued expectation interface is not a bit-complexity result.

### 1.2 Conventions

All returns are normalized to \([0,1]\). Finite spaces are used for the core decision, duality, and statistical statements. Probability-table examples use compact cubes with continuous objectives. Every maximum below is assumed attained; these conditions ensure it in the stated settings. Extensions to general measurable spaces need their own measurability, integrability, compactness, or measurable-selection conditions.

An environment includes task distribution, state transitions, feedback semantics, scoring, and a fixed execution harness. A harness cannot secretly read student parameters except through declared actions. The final student is evaluated alone in the real/reference environment. A reference learning protocol is not automatically optimal merely because its teacher is strong.

## 2. First principles: feasibility, identification, and use

Let \(E\) be an environment, \(\Pi\subseteq\Pi_{\rm ref}\) the student's deployment class, and
\[
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad
J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi).
\]
The restriction \(\Pi\) includes observation rights, memory, representation, and execution constraints. Its gap cannot generally be separated into architecture and observation effects without an additional intervention.

Condition on common initial information \(I_0\), including the pretrained model, known rules, and the \(K\) demonstrations. An adaptive experiment \(x\) returns a message from \(P_E(\cdot\mid x)\). All additional accessible information is included in the transcript \(D\). Given \(D,I_0\), a learner's remaining private randomness has no additional access to the unknown \(E\).

### Proposition 1: an exact decomposition

For a stated prior on a finite environment family, define
\[
B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D].
\]
Then
\[
\begin{aligned}
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]
={}&
\underbrace{\mathbb E[J_E^{\rm ref}-J_E^\Pi]}_{\mathcal R_{\rm rep}}
+\underbrace{\mathbb E J_E^\Pi-\mathbb E_D B(D)}_{\mathcal R_{\rm info}}\\
&+\underbrace{\mathbb E_D\!\left[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\right]}_{\mathcal R_{\rm alg}} .
\end{aligned}
\tag{1}
\]
All three terms are nonnegative.

**Proof.** Add and subtract the two intermediate expectations. Class inclusion proves the first inequality. Conditional maximization cannot exceed the conditional expectation of the environmentwise maximum, proving the second. A randomized learner's conditional value is a mixture of policy values and cannot exceed their maximum, proving the third. \(\square\)

This is a decision-theoretic decomposition, not a new statistical principle. It prevents two invalid inferences: one optimizer's failure is not an information-theoretic lower bound; additional information need not improve a fixed optimizer. Separately maximizing the three terms does not preserve the displayed equality.

### Proposition 2: decision ambiguity, rather than full environment recovery

In the finite case, \(\mathcal R_{\rm info}=0\) if and only if, for almost every \(D\),
\[
\bigcap_{E:\Pr(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\ne\varnothing.
\tag{2}
\]

**Proof.** Choose a posterior-optimal policy. Its conditional gap is a weighted sum of nonnegative environmentwise gaps. The sum is zero exactly when this policy is optimal in every environment of positive posterior mass. \(\square\)

For a known finite ambiguity class \(C\), the corresponding robust quantity is
\[
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}
\left[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\right].
\tag{3}
\]
Finite minimax duality gives
\[
r(C)=\max_{\mu\in\Delta(C)}
\left[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\right].
\tag{4}
\]
To prove (4), apply the finite zero-sum minimax theorem to the matrix
\(M_{E,\pi}=J_E^\Pi-J_E(\pi)\). This is a classical minimax application.

If all allowed experiments have the same law in every \(E\in C\), every learner induces a common output distribution over \(\Pi\). Its worst-case regret is therefore at least \(r(C)\). If the class is known, choosing its minimax mixture attains this value. In particular, \(r(C)=0\) exactly when a common optimal policy exists. This last statement follows because an average of nonnegative gaps can vanish only if every policy in the mixture's support is optimal in every environment.

A confidence set \(C_D\) containing the true environment with probability at least \(1-\delta\), together with a policy mixture whose worst-case gap on \(C_D\) is at most \(\varepsilon\), gives a \(1-\delta\) data-event guarantee for the mixture's **expected** regret over its policy randomization. It does not automatically guarantee that one sampled policy has regret at most \(\varepsilon\). Constructing such a confidence set and optimizing this criterion are separate, potentially difficult problems. The result does not assert that a finite noisy transcript identifies its observational-equivalence class.

### Proposition 3: generation cannot create conditionally absent facts

If \(D'\) is randomized postprocessing of \(D\), then
\[
\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D']
\le \mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D].
\tag{5}
\]
This follows from the tower property and convexity of the maximum. In particular, adding \(Z\perp E\mid(D,I_0)\) cannot reduce \(\mathcal R_{\rm info}\). It may reduce \(\mathcal R_{\rm alg}\) by making existing information easier to learn. A pretrained teacher's additional knowledge belongs in \(I_0\); new external queries enlarge \(D\).

## 3. A complete acquisition example: perfect demonstrations can be insufficient

Let \(v\in\{-1,+1\}\) be unknown and \(0<\Delta\le1/8\) known. A hidden state \(s\in\{0,1\}\) is uniform. The teacher sees \(s\); the student has one aliased context and chooses \(A\) with probability \(p\).

| State | Success probability of \(A\) | Success probability of \(B\) |
|---|---:|---:|
| \(s=0\) | \(1\) | \(1/2-2v\Delta\) |
| \(s=1\) | \(1/2+2v\Delta\) | \(1\) |

The teacher always chooses \(A\) at state 0 and \(B\) at state 1. It is strictly optimal and always succeeds in both worlds. Even demonstrations revealing \(s\) have identical distributions across the two worlds. Yet
\[
J_v(p)=3/4+v\Delta(2p-1),\qquad J_v^\Pi=3/4+\Delta.
\tag{6}
\]

### Theorem 4: passive impossibility and active attainability in the same family

Any learner using only passive teacher demonstrations and their randomized postprocessing satisfies
\[
\max_v\mathbb E[J_v^\Pi-J_v(\widehat p)]\ge\Delta.
\tag{7}
\]
If the permitted interface additionally allows resetting to \(s=1\), executing \(A\), and receiving its Bernoulli outcome, then, for \(0<\delta<1/2\), identifying a student with regret at most \(\Delta/2\) with probability at least \(1-\delta\) **in each of the two worlds** requires
\[
\mathbb E_v n_{\rm query}\ge
\frac{\operatorname{kl}(1-\delta,\delta)}
{\operatorname{kl}(1/2+2\Delta,1/2-2\Delta)}
\ge \frac{\operatorname{kl}(1-\delta,\delta)}{64\Delta^2}.
\tag{8}
\]
Conversely,
\[
n_{\rm query}=
\left\lceil\frac{\log(1/\delta)}{8\Delta^2}\right\rceil
\tag{9}
\]
independent such queries suffice to identify the optimal endpoint \(p=1\) or \(0\) with probability at least \(1-\delta\).

**Proof.** Passive data induce a common output mean \(m=\mathbb E\widehat p\); the two regrets are \(2\Delta(1-m)\) and \(2\Delta m\). Their maximum is at least \(\Delta\), attained by \(p=1/2\).

For (8), successful outputs satisfy \(\widehat p\ge3/4\) in world \(+1\) and \(\widehat p\le1/4\) in world \(-1\). The output therefore gives a binary test. The adaptive transcript KL equals the expected sum of queried conditional KL divergences; passive observations contribute zero. Data processing to the output test gives the first bound. For \(|u|\le1/4\),
\(\operatorname{kl}(1/2+u,1/2-u)\le16u^2\); insert \(u=2\Delta\). Finite expected stopping time and the stated query interface are required. For (9), threshold the sample mean at \(1/2\). One-sided Hoeffding bounds its error by \(\exp(-8n_{\rm query}\Delta^2)\). \(\square\)

This is a standard change-of-measure argument instantiated in a student-information mismatch. It does not claim a new testing inequality or unknown-gap optimal algorithm.

The acquired fact can be compiled into the actual frequency learner: choose target probability \(q=7/8\) or \(1/8\) according to the estimated sign, generate \(m\) Bernoulli target labels, and let the student be their frequency. If
\[
m\ge\left\lceil32\log(2/\delta_L)\right\rceil,
\]
then \(|\widehat p-q|\le1/8\) with probability at least \(1-\delta_L\); the resulting regret is at most \(\Delta/2\) with probability at least \(1-\delta-\delta_L\). These targets are compiled labels, not new authentic teacher trajectories. Acquisition complexity and compilation complexity are distinct.

## 4. Why static evaluation is not learning sufficiency

### Proposition 5: identical policy values, different distilled students

Consider four states with common reward rules and an optimal state-observing teacher.

| State | Reference frequency | Alternative frequency | \(r(A)\) | \(r(B)\) |
|---|---:|---:|---:|---:|
| \(u\) | \(3/4\) | \(0\) | \(19/30\) | \(1/2\) |
| \(v\) | \(1/4\) | \(0\) | \(3/10\) | \(1/2\) |
| \(w\) | \(0\) | \(1/4\) | \(9/10\) | \(1/2\) |
| \(x\) | \(0\) | \(3/4\) | \(13/30\) | \(1/2\) |

For every aliased student \(\pi_p\), both sources give
\[
J(\pi_p)=1/2+p/20,
\qquad J(T)=3/5.
\]
But their teacher-action frequencies are \(3/4\) and \(1/4\). Frequency/empirical cross-entropy distillation on any \(m\ge1\) independent demonstrations, evaluated in the same reference environment, yields expected returns
\[
43/80\quad\text{and}\quad41/80,
\tag{10}
\]
a constant gap \(1/40\).

**Proof.** Sum the statewise rewards and teacher choices. The learned frequency is unbiased and the reference return is affine in it. \(\square\)

This separates static return equivalence on the stated restricted policy class from learning equivalence. It is not a counterexample to full Bellman value equivalence over unrestricted policy/value tests.

### Proposition 6: identical mean updates, different learning values

Compare two binary-label packets: two independent Bernoulli\((1/2)\) labels versus a uniformly ordered pair \((0,1)\). Each label has the same marginal in both sources. Let
\[
p^+=(1-\eta)/2+\eta(Y_1+Y_2)/2,\qquad0<\eta\le1.
\]
Both sources have the same mean gradient and mean \(p^+\), but for a two-decision deployment with \(J(p)=p^2\),
\[
\mathbb E_{\rm independent}J(p^+)
-\mathbb E_{\rm paired}J(p^+)=\eta^2/8.
\tag{11}
\]
The proof is \(\mathbb E p^2=(\mathbb Ep)^2+\operatorname{Var}p\), with packet-average variances \(1/8\) and \(0\). Replacing \(J\) by \(1-p^2\) reverses the preference. Lower gradient variance, greater fidelity, and better task performance are not synonymous.

## 5. The central operational characterization

### 5.1 Complete learning state and continuation response

Let \(z\) include parameters, optimizer moments, buffers, sampler state, relevant randomness, and interaction history. Full history can be used when a smaller Markov state is unavailable. The recipe \(f\) specifies initialization, loss, update rule, learning horizon \(R\), and deployment objective.

At round \(t\), a feedback law \(P_t(\cdot\mid z)\) supplies \(Y_t\), and the actual update is \(U_t(z,Y_t,\xi_t)\). Integrate any declared update randomness \(\xi_t\) into the learning kernel \(K_t\). Let \(K_t^0\) denote the reference protocol and \(K_t^S\) an alternative. Both training processes start from the same initial complete-state law; otherwise an initial-value difference must be added to the comparison. All final students are evaluated with the same
\[
F_E(z)=J_E(\pi_z)\in[0,1].
\]
The reference continuation is
\[
V_R^0=F_E,\qquad V_t^0=K_t^0V_{t+1}^0.
\tag{12}
\]
Its message-level pullback is
\[
g_{E,f,t,z}(y)=
\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t)).
\tag{13}
\]
This is the expected final real task return after receiving \(y\), performing the actual update, and completing the declared reference learning tail. It is not a training loss, teacher confidence, or frozen-policy action value.

### Theorem 7: exact composition and operational necessity

Let \(L_E(S)=\mathbb E F_E(Z_R^S)\), and let \(d_t^S\) be the actual alternative learner-state occupancy. Then
\[
L_E(0)-L_E(S)=
\sum_{t=0}^{R-1}\mathbb E_{d_t^S}
[(K_t^0-K_t^S)V_{t+1}^0(Z_t)].
\tag{14}
\]
Thus if the absolute conditional response discrepancy is bounded by
\(\varepsilon_t(z)\) on visited states,
\[
|L_E(0)-L_E(S)|
\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t).
\tag{15}
\]

For a declared family of snapshots, tasks, recipes, and suffixes, two message laws are interchangeable for **every one-boundary replacement followed by a declared suffix** if and only if
\[
\mathbb E_P g=\mathbb E_Q g
\quad\text{for every corresponding response }g.
\tag{16}
\]

**Proof.** Along the \(S\) process,
\[
\mathbb E[V_t^0(Z_t)-V_{t+1}^0(Z_{t+1})]
=\mathbb E[(K_t^0-K_t^S)V_{t+1}^0(Z_t)].
\]
Summing telescopes to (14); the triangle inequality gives (15). The value difference of a one-boundary intervention is exactly \((P-Q)g\). Equality for every test proves sufficiency; a nonzero test difference supplies its own witnessing intervention, proving necessity. \(\square\)

This is a simulation/telescoping argument on the learning process. Its mathematical machinery is classical. Its role is to fix the correct object and quantifiers for all subsequent results. Pointwise matching is not necessary for one fixed final mean, since errors at different rounds can cancel.

If guarantees hold only inside a certified state set and \(\rho\) bounds the probability of first exit, stopping the telescoping argument at that exit gives
\[
|L_E(0)-L_E(S)|
\le\mathbb E_S\sum_{t<\tau_{\rm exit}}\varepsilon_t(Z_t)+\rho.
\tag{17}
\]
The uncontrolled tail contributes at most one bounded-return discrepancy; the penalty is not \(R\rho\). A finite neural checkpoint bank does not by itself prove small \(\rho\).

### 5.2 The response class is derived, not chosen for convenience

Let \(\mathcal G\) collect (13) over the declared family. It is obtained by pulling terminal task functions backward through actual updates and reference learning kernels. Enlarging the task, learner, suffix, or snapshot family can only enlarge \(\mathcal G\), refine equivalence, and make uniform sufficiency harder.

One can recursively construct the class from terminal functions \(\mathcal V_R=\{F_E\}\), continuation operators \(K_t^0\), and pullbacks \(v\circ U_t(z,\cdot)\). The recursion is the connection between the operational characterization and the explicit algebra in Section 7. Defining an arbitrary feature bank first and calling it “capability information” would reverse this logic.

The reference tail must be responsive to its branch's current student. Coupling random seeds is legitimate; replaying identical labels or trajectories when the protocol would query a changed student generally changes the continuation being measured.

## 6. Exact and approximate information complexity

Fix a finite message alphabet \(\mathcal Y\), \(|\mathcal Y|=m\), and a response class \(\mathcal G\subseteq[0,1]^{\mathcal Y}\). Let an interface preserve means of fixed features \(\phi_1,\ldots,\phi_d\), and write
\[
H_\phi=\operatorname{span}\{1,\phi_1,\ldots,\phi_d\}.
\]

### Theorem 8: exact sufficiency, sharp approximation duality, and transfer

**(a) Exact sufficiency.** Equality of feature means determines all response means for all distributions if and only if
\[
\operatorname{span}(\mathcal G)\subseteq H_\phi.
\tag{18}
\]
The minimum number of independent expectation statistics is
\[
d_0(\mathcal G)=\dim\operatorname{span}(\{1\}\cup\mathcal G)-1.
\tag{19}
\]

**(b) Approximate sufficiency.** Define the worst-pair unresolved response discrepancy
\[
\mathfrak D_\phi(\mathcal G)=
\sup_{\substack{P,Q\in\Delta(\mathcal Y)\\
\mathbb E_P\phi=\mathbb E_Q\phi}}
\sup_{g\in\mathcal G}|(P-Q)g|.
\]
Then
\[
\boxed{\mathfrak D_\phi(\mathcal G)=
2\sup_{g\in\mathcal G}\inf_{h\in H_\phi}\|g-h\|_\infty.}
\tag{20}
\]
Consequently the optimal \(d\)-feature unresolved discrepancy is twice the smallest uniform approximation radius of \(\mathcal G\) by a subspace containing constants with dimension at most \(d+1\).

**(c) Cross-family transfer.** Suppose every target-family response satisfies
\[
g=c+\sum_{\ell=1}^d b_\ell\phi_\ell+r,\qquad
\|b\|_1\le B,\quad\|r\|_\infty\le\tau.
\]
Then
\[
\sup_{g\in\mathcal G_{\rm target}}|(P-Q)g|
\le B\|\mathbb E_P\phi-\mathbb E_Q\phi\|_\infty+2\tau.
\tag{21}
\]
If a target response is outside \(H_\phi\), some matching-feature \(P,Q\) separate that response. Therefore matching a training-family test bank alone cannot guarantee an arbitrary unseen task or learner.

**Proof.** For (a), linearity proves sufficiency. Otherwise there is a nonzero vector \(v\) orthogonal to \(H_\phi\) but not to some response \(g\). Perturb a strictly positive distribution by \(\pm av\) with sufficiently small \(a\); these distributions agree on features but disagree on \(g\). Counting the dimension modulo constants proves (19).

For (b), each admissible pair annihilates \(H_\phi\), hence
\(|(P-Q)g|\le2\inf_h\|g-h\|_\infty\). Conversely, finite-dimensional norm duality gives
\[
\inf_{h\in H_\phi}\|g-h\|_\infty
=\max_{\substack{v\perp H_\phi\\\|v\|_1\le1}}v^\top g.
\]
Since constants are in \(H_\phi\), \(v\) has zero sum. Its positive and negative masses are equal and at most \(1/2\). The vectors \(2v_+\) and \(2v_-\), completed with the same nonnegative leftover probability mass, form admissible \(P,Q\) with difference \(2v\). This attains the reverse bound. Taking suprema over \(g\) proves (20). Part (c) follows by annihilating the constant, bounding the linear term, and bounding two residual expectations by \(2\tau\); the converse is (a). \(\square\)

Equation (20) uses **arbitrary pairs sharing each other's feature means**. It is not an equality for an externally fixed reference \(P\) or one prescribed moment fiber. Restricting feasible distributions changes the dual problem. The theorem is an application of classical linear algebra and approximation duality, not a claim to invent either.

Define \(d_\varepsilon(\mathcal G)\) as the smallest number of fixed linear expectation features with \(\mathfrak D_\phi(\mathcal G)\le\varepsilon\). This is an operational approximate dimension, not a universal information rate. Finite-sample usefulness also requires coefficient stability in (21); small approximation residual alone is insufficient.

### Corollary 9: unrestricted learning can require the full distribution

Suppose a task has two policies with values \(a\) and \(a+\gamma\), and the learner family contains, for every event \(A\subseteq\mathcal Y\), the rule “choose the better policy exactly when \(Y\in A\).” Its response class contains \(a+\gamma1_A\), so
\[
\sup_A|(P-Q)(a+\gamma1_A)|=\gamma\,{\rm TV}(P,Q),
\qquad d_0=m-1.
\tag{22}
\]
This follows from the event definition of total variation and the spanning property of singleton indicators. It rules out a uniformly small reusable information interface for arbitrary future tasks and arbitrary learners. It does not prove that actual neural SGD attains this worst case.

## 7. An explicit closed theory: deployment dependence composed with learning

The previous characterization is general but may require unknown response functions. This section derives them for a nonempty, realizable learner/task class.

### 7.1 Model

The student has \(k\) binary-action contexts and probabilities \(p\in[0,1]^k\). At a decision whose context is \(j\), it chooses action 1 with conditional probability \(p_j\). Contexts may depend on previous actions and feedback, and the same parameter may be reused. A finite environment/harness makes at most \(N\) decisions and returns a bounded terminal reward.

An independent complete experience block gives \(Z\in[0,1]^k\). Within-block coordinates may be arbitrarily dependent. Every coordinate used must actually be observed or otherwise have a justified construction; missing labels are not automatically zeros.

The update is a predetermined convex affine average,
\[
p_R=w_0p_0+\sum_{i=1}^R w_iZ_i,\qquad
w_i\ge0,\quad\sum_{i=0}^Rw_i=1.
\tag{23}
\]
Blocks are IID from \(P\) or \(Q\), weights are fixed independently of the data, and \(p_0\) is common and fixed. This includes frequency learning and fixed-step convex averaging. It excludes general neural SGD, data-dependent stopping, and adaptive queries.

### Lemma 10: deployment produces a polynomial response

The value has a unique polynomial representation
\[
J_E(p)=\sum_{|\alpha|\le N}c_\alpha p^\alpha,\qquad
\sum_\alpha|c_\alpha|\le3^N.
\tag{24}
\]

**Proof.** Each path contains at most \(N\) factors \(p_j\) or \(1-p_j\); all non-policy factors are independent of \(p\). Summing finite paths gives the degree bound. For the coefficient bound, use backward induction on histories. If both action-branch continuation polynomials have coefficient \(\ell_1\) norm at most \(B\), their mixture \(p_jV_1+(1-p_j)V_0\) has norm at most \(3B\). Environment expectation does not increase this bound. Terminal rewards have norm at most 1. Polynomial uniqueness holds on a cube with nonempty interior. \(\square\)

The actual degree can be much smaller than \(N\); the coefficient bound is deliberately conservative.

### Theorem 11: necessary and sufficient joint information order

The following are equivalent:

1. \(\mathbb E_PZ^\alpha=\mathbb E_QZ^\alpha\) for every \(|\alpha|\le N\).
2. For every deployment task in Section 7.1 and every affine learner (23),
\[
\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R).
\tag{25}
\]

**Proof.** Substitute (23) into (24). The resulting polynomial in the independent blocks has total degree at most \(N\); every monomial expectation factors into within-block moments of degree at most \(N\). This proves sufficiency. For necessity, choose a task that visits context \(j\) exactly \(\alpha_j\) times and succeeds only if all actions are 1. It has value \(p^\alpha\). Choose one update \(p=Z\); its two learning values are precisely the two moments. \(\square\)

The theorem quantifies over the entire stated family. A fixed task can require fewer moment combinations. It preserves expected final task return, not the whole learned-policy distribution or a failure-probability tail absent from the test class.

For example, equal coordinate marginals do not preserve \(\mathbb E Z_1Z_2\): equal mixtures of \((0,0),(1,1)\) and of \((0,1),(1,0)\) have identical marginals and cross moments \(1/2\) and \(0\). The corresponding two-context task distinguishes their trained students.

### Theorem 12: sharp exact hierarchy and a constant-gap joint witness

For \(N\ge1\), put \(z_i=i/N\). Give even \(i\) mass \(\binom Ni/2^{N-1}\) under \(P_N\), and odd \(i\) that mass under \(Q_N\). Then
\[
\mathbb E_{P_N}Z^\ell=\mathbb E_{Q_N}Z^\ell\quad(0\le\ell<N),
\qquad
\Delta_N:=\mathbb E_{P_N}Z^N-\mathbb E_{Q_N}Z^N
=\frac{(-1)^N N!}{2^{N-1}N^N}.
\tag{26}
\]
Thus every exact order is necessary for some scalar task.

**Proof.** Even and odd binomial sums are \(2^{N-1}\). The \(N\)-th finite difference annihilates polynomials of degree below \(N\), and
\(\sum_i(-1)^i\binom Ni i^N=(-1)^NN!\). Scaling gives (26). The task \(J(p)=p^N\), with one update \(p=Z\), realizes its difference. \(\square\)

This scalar discrepancy becomes small as \(N\) grows. It alone does not establish fixed-accuracy hardness. For that distinction, set \(k=N\), let \(P\) and \(Q\) be uniform on even- and odd-parity binary vectors, and use
\[
J_{\rm even}(p)=\frac{1+\prod_{j=1}^N(1-2p_j)}2.
\tag{27}
\]
This is the probability that the \(N\) deployed actions have even parity. All joint moments of total degree below \(N\) match, because every proper coordinate subset is uniform under both laws. One update \(p=Z\) gives expected returns 1 and 0. This is a constant-gap witness at every order.

The constructions use standard finite differences and parity independence. Their role is to establish the precise operational hierarchy and to distinguish exact from approximate complexity.

### Theorem 13: sharp linear-statistic counts and a broad-task barrier

On the grid \(\mathcal Y=\{0,1/N,\ldots,1\}^k\), the smallest number of independent expectation statistics sufficient for all tasks and learners in Theorem 11 is
\[
\binom{k+N}{N}-1.
\tag{28}
\]
On \(\mathcal Y=\{0,1\}^k\), with \(N\ge k\) and arbitrary terminal rewards on the \(k\)-action vector, every feature interface with fewer than \(2^k-1\) independent statistics has worst-pair discrepancy 1. Thus tolerating any discrepancy strictly below 1 does not reduce this worst-case dimension.

**Proof.** For (28), monomials of total degree at most \(N\) are linearly independent on the grid: a polynomial vanishing on the full grid vanishes identically by successive univariate interpolation, since each coordinate degree is at most \(N\). Their number is \(\binom{k+N}{N}\). Theorem 11 provides an upper bound, while the one-update monomial tasks and Theorem 8 give the lower bound.

For the binary claim, one update \(p=Z\) produces deterministic actions, so any function \(f:\mathcal Y\to[0,1]\) is realized by setting the task's terminal reward to \(f\) of those actions. Too few features leave a nonzero signed vector annihilating constants and all features. Normalize its positive and negative parts separately to distributions \(P,Q\). They have disjoint supports and identical feature means. Choosing \(f\) to indicate the positive support gives value gap 1. \(\square\)

The second result is deliberately over a very broad task family. It explains why task-family structure is necessary for efficient transfer; it does not predict that ordinary agent benchmarks require exponentially many measurements.

### Theorem 14: a discrepancy bound independent of training rounds

Let
\[
\delta_N(P,Q)=\max_{1\le|\alpha|\le N}
|\mathbb E_PZ^\alpha-\mathbb E_QZ^\alpha|,
\qquad C_J=\sum_\alpha|c_\alpha||\alpha|.
\]
For every affine learner (23),
\[
|\mathbb E_{P^R}J_E(p_R)-\mathbb E_{Q^R}J_E(p_R)|
\le C_J\delta_N(P,Q)\le N3^N\delta_N(P,Q).
\tag{29}
\]
The bound may also be truncated at 1.

**Proof.** Expand a monomial \(p_R^\alpha\). Convex nonnegative weights give nonnegative expansion coefficients whose sum is at most 1, as seen by setting all block coordinates to 1. Each term's expectation is a product of at most \(|\alpha|\) within-block moments in \([0,1]\). Product telescoping bounds its change by \(|\alpha|\delta_N\). Sum the nonnegative expansion coefficients and then the absolute coefficients of \(J\). \(\square\)

The independence of \(R\) is a structural consequence of convex affine averaging, not a general neural-training guarantee. Large degree and coefficient constants may make the generic bound uninformative.

### Theorem 15: the first unmatched interaction controls averaging decay

Let \(\bar p_n=n^{-1}\sum_{i=1}^nZ_i\), let \(P,Q\) share mean \(\mu\), and suppose their joint cumulants agree below order \(r\), where \(2\le r\le N\). Then
\[
\mathbb E_{P^n}J(\bar p_n)-\mathbb E_{Q^n}J(\bar p_n)
=\frac{\langle D^rJ(\mu),\,\kappa_r(P)-\kappa_r(Q)\rangle}
{r!\,n^{r-1}}+O(n^{-r}).
\tag{30}
\]
If the displayed contraction vanishes, the rate may be faster. A nonzero coefficient is required for a \(\Theta(n^{1-r})\) statement.

**Proof.** The Taylor expansion of the finite-degree polynomial \(J\) at \(\mu\) terminates exactly. Centered average cumulants of order \(s\ge2\) equal \(\kappa_s/n^{s-1}\); the first cumulant is zero. Expand each centered moment through set partitions. Any term that differs between the two laws must contain a block of size at least \(r\). The only term of order \(n^{-(r-1)}\) is the single \(r\)-block in the \(r\)-th moment. A larger block or any additional nonempty block contributes at most order \(n^{-r}\). There are finitely many terms and all moments are bounded. \(\square\)

For the scalar laws (26) and \(J(p)=p^N\), the equality is exact:
\[
\mathbb E_{P_N^n}\bar p_n^N-\mathbb E_{Q_N^n}\bar p_n^N
=\Delta_N/n^{N-1}.
\tag{31}
\]
If means differ, the limit is \(J(\mu_P)-J(\mu_Q)\), with an \(O(n^{-1})\) correction; this limit can still vanish for a particular task. The independent block count \(n\), not epochs, controls these laws.

## 8. Exact order is not approximate information complexity

Consider \(J(p)=p^N\) on \([0,1]\). Its Chebyshev expansion is
\[
p^N=4^{-N}\binom{2N}{N}
+2\,4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1).
\tag{32}
\]
This follows by setting \(2p-1=\cos\theta\), writing
\(p^N=\cos^{2N}(\theta/2)\), and grouping conjugate terms of the binomial Fourier expansion.

### Proposition 16: two degree-\(N\) tasks can require very different accuracy orders

Truncating (32) after order \(q<N\) gives a polynomial \(h_q\) with
\[
\|p^N-h_q\|_\infty
=\Pr\{|B-N|>q\}
\le2e^{-q^2/N},\qquad B\sim{\rm Binomial}(2N,1/2).
\tag{33}
\]
Therefore matching moments through degree \(q\) controls the scalar one-update value gap by \(4e^{-q^2/N}\). It gives the same bound after any fixed affine average (23), because the final scalar distributions still match moments through degree \(q\).

**Proof.** All Chebyshev coefficients are nonnegative, \(|T_j|\le1\), and at \(p=1\) all \(T_j\) equal 1; hence the truncation error equals the coefficient tail. The binomial identity gives the stated tail, and Hoeffding supplies the inequality. Moment equality cancels \(h_q\), leaving two residual expectations. \(\square\)

Thus a sufficient order for this scalar task is on the scale
\(\sqrt{N\log(1/\varepsilon)}\), capped at \(N\), although exact equality requires order \(N\). This is an upper bound, not a claim of a matching approximate-order lower bound.

In contrast, the best degree-\(q<N\) polynomial approximation to the parity value (27) on \([0,1]^N\) has uniform error exactly \(1/2\). The constant \(1/2\) attains it. The even/odd laws agree on every lower-degree polynomial but their target values differ by 1, forcing at least \(1/2\) error by the triangle inequality.

This distinction is a direct consequence of the operational width in Theorem 8. Equal deployment horizon and equal exact degree need not imply equal compressibility. Neither scalar compressibility nor multivariate hardness can be transferred to an arbitrary task class without proof.

**Stability limitation.** Exact lower-order moment matching and a small polynomial approximation residual do not by themselves yield a useful finite-sample raw-moment certificate: approximation coefficients can amplify estimation error. Bounded Chebyshev features avoid a power-basis amplification for a single scalar message; after several affine updates, their pullback to block statistics needs its own coefficient bound. In general use the pair \((B,\tau)\) in (21), not approximation degree alone.

## 9. Learning dynamics can amplify the information requirement

Suppose independent blocks \(Z_i\) are processed by polynomial updates
\(p_i=U_i(p_{i-1},Z_i)\) that keep \(p_i\in[0,1]^k\). Each coordinate of \(U_i\) has degree at most \(a_i\) in the old parameter vector, holding \(Z_i\) fixed, and degree at most \(b_i\) in the new block, holding the old parameters fixed. Use \(a_i,b_i\ge1\) as upper bounds.

### Theorem 17: information order is a composition property

For deployment degree at most \(N\), the final response has degree in block \(i\) at most
\[
D_i=N b_i\prod_{t=i+1}^R a_t.
\tag{34}
\]
Matching the joint moments of block \(i\) through \(D_i\) for each \(i\) therefore preserves expected final value. If all blocks share one source law, matching through \(\max_iD_i\) suffices.

**Proof.** Pull the terminal polynomial backward through the updates. Substituting \(U_i\) into a polynomial of degree \(d\) produces old-parameter degree at most \(a_i d\) and new-block degree at most \(b_i d\). Induction gives (34). Independence factors the final blockwise polynomial expectations. \(\square\)

This upper bound can be attained within the permitted class. Set \(p_1=Z_1\), then \(p_i=p_{i-1}^2\), and deploy once with \(J(p)=p\). The final response is \(Z_1^{2^{R-1}}\). The laws (26), with \(N\) replaced by \(2^{R-1}\), match all lower moments and still disagree on final value. Later blocks are simply unused.

Even a one-step deployment does not impose a first-moment requirement for arbitrary learners. For \(p=1\{Z>1/2\}\), compare
\(P(Z=0)=1/3,\ P(Z=3/4)=2/3\) with \(Q(Z=1/2)=1\). Both means are \(1/2\), but \(J(p)=p\) has learned values \(2/3\) and \(0\).

Polynomial degree is coordinate-dependent. An invertible reparameterization can change the representation of an update. The intrinsic operational object is its response class and approximation width; degree is a tractable certificate in the explicitly chosen probability coordinates.

For smooth SGD, a useful local specialization is
\[
(K^P-K^Q)V(\theta)
=-\eta\nabla V^\top(\mu_P-\mu_Q)
+\frac{\eta^2}{2}\langle\nabla^2V,M_P-M_Q\rangle+\mathcal E,
\tag{35}
\]
where \(M=\mathbb E GG^\top\), and
\[
|\mathcal E|\le\frac{M_3\eta^3}{6}
\left(\mathbb E_P\|G\|^3+\mathbb E_Q\|G\|^3\right)
\]
if the third derivative norm of the true continuation \(V\) is at most \(M_3\) on relevant segments. Taylor's theorem proves the formula. Conditional update-moment methods and weak approximation have substantial prior work [R2]. Replacing \(V\) by training loss or ignoring AdamW state requires separate justification.

## 10. From approximate information to measurable error

Approximation width alone does not specify which statistics can be estimated accurately. The following noisy version closes that logical link on a finite alphabet.

### Theorem 18: an exact stability–approximation tradeoff

For fixed features and nonnegative tolerances \(\epsilon_1,\ldots,\epsilon_d\), define
\[
W_{\boldsymbol\epsilon}(g)=
\sup_{\substack{P,Q\in\Delta(\mathcal Y)\\
|(P-Q)\phi_j|\le\epsilon_j\ \forall j}}|(P-Q)g|.
\]
Then
\[
\boxed{
W_{\boldsymbol\epsilon}(g)
=\inf_{b,c}\left[
2\left\|g-c-\sum_jb_j\phi_j\right\|_\infty
+\sum_j\epsilon_j|b_j|
\right]
=\inf_b\left[
\operatorname{osc}\!\left(g-\sum_jb_j\phi_j\right)
+\sum_j\epsilon_j|b_j|
\right].
}
\tag{36}
\]
Here \(\operatorname{osc}(h)=\max h-\min h\). In particular \(W_{\boldsymbol\epsilon}(g)\le\operatorname{osc}(g)\le1\).

**Proof.** The set of probability differences is exactly
\(\{v:1^\top v=0,\|v\|_1\le2\}\): the converse is obtained by completing its positive and negative masses with a common probability component. The constrained set is symmetric, so the absolute value can be removed when maximizing. Dualize the constraints \(|\phi_j^\top v|\le\epsilon_j\) in this finite feasible bounded linear program. The signed dual multipliers \(b_j\) contribute \(\sum_j\epsilon_j|b_j|\); maximizing the remaining zero-mass \(\ell_1\)-bounded vector against \(g-\sum b_j\phi_j\) gives its oscillation. Finite LP strong duality applies. Finally \(\operatorname{osc}(h)=2\inf_c\|h-c\|_\infty\). \(\square\)

This is classical LP/norm duality applied to the same operational discrepancy as Theorem 8. It shows precisely why low approximation error with unstable coefficients is insufficient. It also provides a constructive finite optimization objective when response values are known or independently bounded. Equation (36), like (20), is sharp over arbitrary pairs of message laws. For restricted feasible interfaces it remains an upper bound unless their constraints are included in the optimization.

### Theorem 19: frozen finite-interface selection

Using only independent pilot data, freeze \(M\) executable providers, \(L\) query cells, and \(d\) features \(\phi_{x,\ell}\in[0,1]\). A cell specifies its full learner state, task, query, update, and reference suffix. Collect \(n_0\) independent reference messages per cell and \(n_S\) independent candidate messages per candidate/cell. Define
\[
D_j=\max_{x,\ell}|\mu_{0,x,\ell}-\mu_{j,x,\ell}|,
\qquad
\widehat D_j=\max_{x,\ell}|\widehat\mu_{0,x,\ell}-\widehat\mu_{j,x,\ell}|.
\]
Let
\[
e_0=\sqrt{\frac{\log(4Ld/\delta)}{2n_0}},\qquad
e_S=\sqrt{\frac{\log(4MLd/\delta)}{2n_S}}.
\]
With probability at least \(1-\delta\), simultaneously for all candidates,
\[
|D_j-\widehat D_j|\le e_0+e_S.
\tag{37}
\]
If \(\widehat j\) minimizes \(\widehat D_j\) up to known error \(\eta_{\rm opt}\), then
\[
D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
\tag{38}
\]

**Proof.** Apply two-sided Hoeffding to each feature mean, and union bound the \(Ld\) reference means and \(MLd\) candidate means with failure budgets \(\delta/2\) each. Triangle inequalities prove (37); apply it to the selected and best candidates for (38). Features evaluated on a common message may be correlated. Samples within each mean must satisfy the declared independent-replica assumption. \(\square\)

To connect these observed feature errors to actual learning, suppose every relevant response at a certified state admits
\[
g_{f,t,z}=c+b^\top\phi_x+r,\qquad
\|b\|_1\le B_t,\quad\|r\|_\infty\le\tau_t.
\]
If subsequent training with the selected frozen provider uses a **fresh** declared training stream and exits the certified set with probability at most \(\rho\), then on the audit event
\[
|L_E(0)-L_E(S_{\widehat j})|
\le\sum_t B_t(\widehat D_{\widehat j}+e_0+e_S)
+2\sum_t\tau_t+\rho.
\tag{39}
\]
Use (36) for sharper coordinatewise tolerances when available. In the affine class, replace the sum by the sharper
\[
|L_E(0)-L_E(S_{\widehat j})|
\le C_J(\widehat D_{\widehat j}+e_0+e_S).
\tag{40}
\]

These are high-probability statements about the population expected value of a training procedure selected by an independent audit. They are not high-probability statements about one training seed's score. If a provider is an empirical block bank, freeze that bank before an independent audit, then use fresh independent draws from the audited bank law and account for its discrepancy from the reference. Creating a new bank from the audit observations changes the provider: it requires a new independent audit or a valid uniform correction for that data-dependent construction. Merely freezing and resampling such a bank does not inherit the original fixed-provider confidence event. Do not silently apply an IID future-stream theorem to a selected fixed training batch.

Additional confidence failures for coverage, coefficients, or residuals must be added to the total failure probability. More candidate-generated data can reduce \(e_S\); it does not reduce uncertainty \(e_0\) about the reference.

### 10.1 Estimating the response without an oracle

At a restorable complete state, draw a candidate message, apply the actual update, run the full specified reference learning suffix, and deploy the resulting student in the real task. Repeated independent replicas estimate the provider-integrated response \(\mathbb E_{Y\sim P}g(Y)\), or a paired provider contrast. To estimate the pointwise response \(g(y)\) in (13), instead fix a realizable message \(y\) and reproduce its valid update/branch before repeating update randomness, the suffix, and deployment. This may be expensive; every query, teacher call, gradient step, restore, and deployment is part of the method's cost.

For finite fully accessible message/state/task families, uniformly estimate all needed response values. Fit a bounded-coefficient approximation on that entire finite set and add the uniform response-estimation radius to its residual. That gives a finite, implementable certificate. It assumes the relevant branches or messages can actually be produced/restored. Algebraic access to a state is not permission or physical capability to reset to it.

For continuous neural states, a predictor's validation residual on sampled checkpoints is not a uniform \(\tau_t\). A short learning tail estimates a different response. Teacher loss, gradient features, and learned critics are candidate proxies until independently validated against actual full-tail consequences.

## 11. Fidelity, improvement, and adaptive verification

If reference learning improves the initial student by \(\gamma\), and the interface's one-sided discrepancy is at most \(\varepsilon\), then
\[
L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon.
\tag{41}
\]
This is obtained by subtraction. Its substantive premise is that the reference recipe actually works. Fidelity alone can reproduce an ineffective recipe.

A separate method can optimize signed local advantage relative to the reference:
\[
A_t(z,j)=(K_t^j-K_t^0)V_{t+1}^0(z).
\tag{42}
\]
The signed version of (14) gives
\[
L_E(S)-L_E(0)=\sum_t\mathbb E_{d_t^S}A_t(Z_t,J_t).
\tag{43}
\]
This motivates conservative provider selection, while preserving the distinction between approximate replication and deliberate improvement.

### Theorem 20: actual-state adaptive auditing without a checkpoint-coverage assumption

Consider a finite training horizon and a fixed reference recipe. At each audited round, conditional on all pre-audit history, freeze the actual complete state \(Z_t\), a finite set of \(M_t\) nonreference candidate kernels, a replica count \(n_t\), and a confidence allocation \(\delta_t\). Assume \(\sum_t\delta_t\le\delta\) almost surely.

For each candidate, independently repeat paired experiments from that state: candidate versus reference update, each followed by the **full** reference-learning suffix and real deployment. The two paired marginals must be correct, and the contrast lies in \([-1,1]\). Define
\[
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},
\qquad
\ell_t(j)=\widehat A_t(j)-e_t,\quad \ell_t(0)=0.
\]
Select a maximizer and let \(B_t=\max_j\ell_t(j)\in[0,1]\). Apply a **fresh** selected update, discarding audit-trained replicas. Unaudited rounds use the reference and have \(B_t=0\).

If audits leave the actual learner/environment state unchanged and the reference ignores auxiliary audit records, then:

1. With probability at least \(1-\delta\), the sum of selected true conditional drifts is at least \(\sum_tB_t\).
2. Unconditionally over both audits and training,
\[
L_E({\rm adaptive})-L_E(0)\ge-\delta,
\tag{44}
\]
and
\[
L_E({\rm adaptive})-L_E(0)
\ge\mathbb E\sum_tB_t-2\delta.
\tag{45}
\]

**Proof.** Conditional Hoeffding and a finite union bound give a current audit event \(G_t\) with conditional failure probability at most \(\delta_t\), on which the selected advantage \(A_t\ge B_t\ge0\). The conditional union bound gives simultaneous success probability at least \(1-\delta\).

Let \(\mathcal F_t\) contain the current audit and choice but precede the fresh real update. Then
\[
A_t=\mathbb E[V_{t+1}^0(Z_{t+1})\mid\mathcal F_t]-V_t^0(Z_t).
\]
Writing the update deviation from this conditional expectation as a martingale difference \(D_{t+1}\), telescoping gives
\[
F_E(Z_R)-V_0^0(Z_0)=\sum_tA_t+\sum_tD_{t+1}.
\]
The second sum has unconditional mean zero. Always \(A_t\in[-1,1]\) and \(B_t\in[0,1]\), so
\(A_t\ge-1_{G_t^c}\) and \(A_t\ge B_t-2\,1_{G_t^c}\).
Taking unconditional expectations proves (44)–(45). \(\square\)

The high-probability statement concerns the predictable drift, not the realized terminal return. Conditioning terminal expectations on “all audits succeeded” is invalid in general because future audit events depend on reached states. A generic additional martingale tail bound is possible, but its \(\sqrt R\) scale can be vacuous.

This theorem removes a specific neural-state coverage extrapolation by measuring at the actual state. It does not make full-tail audits cheap, nor establish that any candidate has a positive lower bound. For \(M=4,\delta=0.05\), achieving the distribution-free radius \(e\le0.1\) takes at least \(\lceil2\log(160)/0.01\rceil=1016\) independent paired training replicas **per candidate** at one state: 4064 pairs, or 8128 branch executions without reference reuse. Many deployment episodes of one trained replica do not substitute for these independent update/tail draws. The small pilot in the execution plan measures feasibility and prediction quality; it is not sized to deliver this worst-case certificate.

A proxy can replace the gold contrast only with an independently justified uniform bias allowance, subtracted from its lower bound. Without that allowance the adaptive neural gate is empirical. Within-audit adaptive stopping requires a valid sequential confidence method; the fixed-\(n_t\) radius cannot be reused after arbitrary stopping.

## 12. One derived mechanism: coupling valid experience units

The theory permits multiple interventions: acquiring new facts, changing feedback content, preserving dependencies, choosing curricula, and adapting an interface across learner/task families. Coupling is one especially controlled test of the distinction between marginal information and learning-relevant joint information.

Fix finite valid unit sets \(\mathcal Y_1,\mathcal Y_2\) and strictly positive slot marginals \(\mu,\nu\). Assume every Cartesian-product pair is a valid packet, and let \(\mathcal C(\mu,\nu)\) be the full coupling polytope with those marginals. Let
\[
H_z(i,j)=\mathbb E[\text{real terminal return after the actual joint-packet update and reference suffix}].
\]

### Proposition 21: exactly when dependence can matter

The coupling width
\[
\max_{\Gamma\in\mathcal C}\mathbb E_\Gamma H_z
-\min_{\Gamma\in\mathcal C}\mathbb E_\Gamma H_z
\]
is zero if and only if \(H_z(i,j)=a_i+b_j\) on the positive-support Cartesian product. Moreover,
\[
|(\Gamma-\Gamma')H_z|
\le2\inf_{a,b}\|H_z-a-b\|_\infty.
\tag{46}
\]

**Proof.** Additive functions have expectations fixed by the marginals. Conversely, the product coupling \(\mu\nu^\top\) is strictly positive. Small signed perturbations along any \(2\times2\) rectangle preserve marginals. Zero width therefore forces
\(H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\) for every rectangle. Fixing a row and column yields the additive representation. Coupling differences annihilate additive functions and have \(\ell_1\) norm at most 2, proving (46). \(\square\)

The inequality is generally strict; a fixed-marginal coupling set is not the unrestricted pair-law set of Theorem 8. If some pairings are forbidden, zero width only means that the response annihilates the actually feasible coupling differences; the additive “if and only if” need not hold. The result is standard coupling linear algebra. It supplies an experimentally meaningful question: is there a nonadditive **learning-consequence** component beyond samplewise quality, gradient variance, or action-level cost?

A strict null is essential. If exactly the same examples are differentiated at one common parameter state, linearly accumulated with identical weights, and followed by one deterministic additive SGD step, regrouping cannot change the update. Nonzero effects in that null reveal implementation or randomness differences. Multiple sequential updates and AdamW can yield real interactions; they do not isolate covariance alone.

## 13. The derived few-shot method: Consequence-Calibrated Experience Distillation

CED is a proposed family of procedures, not a validated new algorithm. Its central design is to choose an environment-information interface relative to the actual learner's continuation responses. It does not require a world model.

### 13.1 Acquisition and compilation are separate operations

Start from \(K\) complete trusted demonstrations per task family. Maintain a separate ledger of additional teacher outputs, environment branches, observed failures, feedback, and synthetic transformations.

When the initial transcript leaves decision-relevant ambiguity, use permitted training-environment experiments to resolve it. In a tractable model, (3) gives an explicit stopping target; in a neural agent, disagreement, failure localization, or uncertainty scores are acquisition heuristics to be evaluated, not substitutes for an identification theorem.

Compile acquired evidence into complete valid feedback units usable by the actual student objective. Candidate units can include teacher-labeled student prefixes, authentic recovery branches, success/failure contrasts, or complete episodes. A synthetic rewrite carries provenance; it is never reported as a newly observed environmental fact.

### 13.2 Select interfaces using the response they induce

At reproducible training states, compare a finite provider menu. Depending on the experiment, candidates may change:

- which authentic evidence is supplied;
- how that evidence is expressed or supervised;
- which contexts and recoveries are covered;
- how intact units are grouped or ordered;
- how the interface adapts to a specified learner recipe or task panel.

Measure actual update-plus-tail consequences on training/validation tasks. Use independent data for feature discovery, candidate selection/calibration, and final reporting. Where affordable, use direct response tests; elsewhere learn a small predictor and compare it against gradient and rollout baselines on held-out real updates.

Use discrepancy minimization when replicating a useful reference, and calibrated signed advantage when deliberately improving it. These are different objectives and must be logged separately. A finite robust task/recipe panel may use the minimum calibrated gain across cells; this protects only that panel, not arbitrary unseen tasks.

The optional coupling branch preserves slot marginals, unit validity, token weights, total examples, and optimizer steps while changing their joint organization. Its failure rejects that branch, not the unified theory or other CED branches.

### 13.3 Deploy and test the actual distilled student

The final student receives no teacher, selector, reference-tail evaluator, or extra privileged state at deployment. Measure success and failure modes on unseen instances, held-out task families, and changed learning recipes. Separate:

1. **Method transfer:** freeze the selector/protocol, then distill a new student on a new family with its allowed \(K\) demonstrations.
2. **Policy transfer:** freeze the distilled student itself and deploy on a new family without new adaptation.

Neither result implies the other. The transfer condition (21) explains what a theoretical guarantee would require; absent a verified residual bound, cross-task results are empirical.

The detailed implementation, baseline access parity, model identifiers, state restoration, metric validation, and resource gates are in \`CED_Method_EN.md\` and \`Coding_Agent_Execution_Plan_EN.md\`.

## 14. What would make this one substantive theory rather than a stack?

The contribution cannot be established by counting named theorems. The same object must do work at every link:

1. The acquisition problem identifies decision conflicts that available observations cannot resolve.
2. Once facts are available, the response class identifies exactly how they can affect the declared learning procedure.
3. Its quotient and noisy approximation width characterize discarded distinctions.
4. Pulling that class through deployment and update operators produces explicit necessity witnesses, complexity bounds, and rate predictions.
5. The same responses determine the certificate and the method's validation target.

The most concrete result package currently consists of Theorems 7–20 together with realizable finite witnesses. The broad characterization uses established tools. Its research value must rest on the operational target, nontrivial specializations, and predictions that change what is measured or trained. A paper that only renames the telescoping identity and approximation duality would be inadequate.

### 14.1 Realizability and access restrictions

The unrestricted-law dualities quantify over all distributions on a specified finite feedback alphabet. Their lower-bound witnesses need not be attainable by one particular real service or dataset. For a restricted interface family, replace the unrestricted feasible set by that family's actual difference set.

The moment/parity examples are realizable finite feedback protocols: an allowed reset experiment draws a finite latent index and emits its associated complete block; a grid coordinate \(i/N\) can be represented by the frequency of \(i\) ones in \(N\) actually emitted labels. The student uses the specified frequency/affine update, and the deployment process performs the declared action sequence and terminal scoring. This supplies valid constructed environments for necessity claims. It does not establish that an arbitrary benchmark permits these reset experiments or generates arbitrary target block laws.

The scalar Chebyshev result is a tractable approximation example. The unrestricted binary-task lower bound is a deliberately broad impossibility example. Neither alone demonstrates efficient general neural information compilation.

### 14.2 Prior art and contribution boundaries

| Part of this framework | Existing foundation | Candidate contribution to investigate |
|---|---|---|
| Student feasibility and decision ambiguity | Statistical decision theory; asymmetric imitation | A precise separation between acquisition and learnable compilation in the declared agent setting |
| Response equivalence and composition | Value equivalence, machine teaching, weak approximation [R1–R2, R9] | A reusable environment-interface theory relative to complete learning procedures |
| Width and feature certificates | Approximation/LP duality; finite-loss dataset-distillation guarantees [R4] | Operational necessity, stable measurement, and transfer through the same response class |
| Moment hierarchy and update composition | Polynomial algebra and cumulants | Explicit learner/task-dependent information requirements, tight witnesses, and falsifiable rates as one characterization |
| Accuracy-sensitive complexity | Classical polynomial approximation and parity independence | Contrasting equal-horizon/equal-degree tasks under the operational information criterion |
| Post-update utility optimization | Behaviour Distillation, meta-reweighting and MetaOPD [R3, R5] | A concrete useful information-compilation procedure with independently validated responses |
| Gradient or packet predictors | GRACE, LESS, antithetic sampling and ordering [R6 and audit] | Incremental measurement and selection value on actual agents, not merely a new name |
| Useful information selection at framework level | Rate-distortion learning and value-equivalent sampling [R7–R8] | The exact declared-learner continuation target and its acquisition/verification bridge, if demonstrably distinct |

An integrated theory can be original even when its ingredients are classical. Conversely, putting correct existing results in one document does not establish a major contribution. The independent novelty audit examines both levels and leaves specific fulltext comparisons open.

## 15. Experiments that can falsify the chain

| Theory claim | Controlled finite experiment | Required neural-agent counterpart | Failure interpretation |
|---|---|---|---|
| Identification differs from imitation quality | Two perfect-teacher worlds in Section 3 | Equal-\(K\) demonstrations; extra authentic feedback versus replay-only control | Query heuristic may fail even when the finite lower bound is correct |
| Static quality does not determine learning quality | Proposition 5 | Match source/teacher quality as closely as feasible, compare real post-update gains | No neural separation weakens practical relevance, not the algebraic witness |
| Mean updates can omit relevant dependence | Proposition 6 and parity | Same-feedback controlled batching branch; strict additive null | Null violation invalidates implementation; no effect rejects the branch |
| Accuracy complexity depends on task structure | Scalar powers versus parity, same exact degree | Structured compositional tasks plus unstructured controls | Generic horizon claims are rejected if structure is ignored |
| Updates alter information requirements | Affine versus polynomial/threshold learners | Changed optimizer, step size, trainable layers, tail length | A metric tied to one recipe must not be advertised as universal |
| More independent data differs from replay | Exact cumulant slopes | Independent environment blocks versus repeated same-block exposure | Replay improvements are optimization effects, not the stated \(n\)-law |
| A metric measures the right consequence | Exact response enumeration | Held-out full-update/tail gain prediction, sign error and selection regret | High correlation without useful selection is insufficient |
| Fidelity differs from improvement | Reverse \(J\) in Proposition 6 | Check reference gain and signed candidate gain separately | Low discrepancy to a weak reference is not method success |
| Transfer requires response coverage | Omitted-test witnesses in Theorem 8 | Freeze selector before task-family and recipe transfer | Negative transfer is evidence about missing response coverage |

The full plan requires real parameter updates and final student-only deployment in external agent environments. Finite arithmetic checks validate examples and code consistency; they are neither neural experiments nor proofs of empirical efficacy.

## 16. Claim ledger and remaining obligations

| Claim | Current status |
|---|---|
| Finite mathematical statements through the operational, moment, approximation, coupling, and adaptive-audit results | Explicit proofs; independent model-agent review of the substantive statements and repairs |
| General finite-law minimax/duality machinery is new | Not claimed |
| Entire integrated theory is novel relative to all prior work | Not established; bounded independent literature audit completed |
| A cheap response statistic uniformly certifies continuous neural training | Not proved |
| Neural CED improves few-shot agent distillation | Proposed and testable; no training result yet |
| Frozen selector transfers across tasks and learning recipes | Requires the specified experiments; no result yet |
| ICML best-paper-level contribution | An ambition, not an assessed outcome |

The important open research obligation is a useful bridge between the exact characterization and realistically measurable neural responses. It could be fulfilled by a new structured response approximation theorem, a tight acquisition/verification result for an important adaptive learner class, or a method whose distinctive theoretical predictions survive controlled real-agent tests. These are alternative routes within the same theory, not a requirement to force the project into one coupling mechanism.

The proof audit has corrected four especially consequential issues: fresh training after frozen selection; exact degree versus approximate complexity; fidelity versus improvement; and the probability statement for adaptive actual-state auditing. Neural interpolation, response-proxy bias, resource cost, and out-of-family generalization remain explicit assumptions or empirical questions.

## 17. Selected primary references

This register supports the concise comparisons above. The independent novelty audit contains the broader 31-entry register, source-level review depth, and unresolved fulltexts.

- **[R1]** Grimm et al. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS 2020. https://arxiv.org/abs/2011.03506
- **[R2]** Li, Tai, E. *Stochastic Modified Equations and Dynamics of Stochastic Gradient Algorithms I: Mathematical Foundations.* JMLR 2019. https://jmlr.org/papers/v20/17-526.html
- **[R3]** Lupu et al. *Behaviour Distillation.* ICLR 2024. https://arxiv.org/abs/2406.15042
- **[R4]** Gupta, Saket, Raghuveer. *Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets.* https://arxiv.org/abs/2512.00536
- **[R5]** Wang et al. *MetaOPD: Meta-Learned Token Weighting for On-Policy Distillation.* Preprint, 8 October 2026. https://arxiv.org/abs/2610.11989
- **[R6]** Panigrahi et al. *In Good GRACEs: Principled Teacher Selection for Knowledge Distillation.* https://arxiv.org/abs/2511.02833
- **[R7]** Arumugam, Van Roy. *Deciding What to Learn: A Rate-Distortion Approach.* ICML 2021. https://proceedings.mlr.press/v139/arumugam21a.html
- **[R8]** Arumugam, Van Roy. *Deciding What to Model: Value-Equivalent Sampling for Reinforcement Learning.* NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b18d368150474ac6fc9bb665d3eb3da-Abstract-Conference.html
- **[R9]** Lessard, Zhang, Zhu. *An Optimal Control Approach to Sequential Machine Teaching.* AISTATS 2019. https://proceedings.mlr.press/v89/lessard19a.html
- **[R10]** Banerjee, Montúfar. *The Variational Deficiency Bottleneck.* https://arxiv.org/abs/1810.11677
- **[R11]** Voelcker et al. *Calibrated Value-Aware Model Learning with Probabilistic Environment Models.* https://arxiv.org/abs/2505.22772

## Appendix A. Reading the package

Read this framework first, then \`Independent_Proof_Audit_EN.md\` and \`Independent_Novelty_Audit_EN.md\`. The method specification turns the theory into candidate training interventions. The coding-agent plan and configuration specify implementation and evaluation obligations. \`START_HERE_Coding_Agent_EN.md\` is the direct execution handoff.

The accompanying verification scripts and JSON outputs report finite checks only. They must remain separate from future neural-run manifests and results. The runtime audit records why no actual agent training was performed in this workspace.

