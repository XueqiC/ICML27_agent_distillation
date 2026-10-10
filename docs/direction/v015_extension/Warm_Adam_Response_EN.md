# Optimizer warmth and the information a feedback message must carry

**Status:** working derivation, 2026-10-10. Extends the v0.15 addendum (`docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md`, Props. A2–A4) and the controlled findings in `docs/direction/2026-10-10_v015_verification_ZH.md` (E1: warm moments remove most of the C23 Adam gap; E2: the gap survives a frozen Jacobian).
Proofs are complete for the stated finite-dimensional setting; the neural/LoRA relevance is an empirical question tested separately (A-pilot). Precedents are listed in §6.

## 1. Setting

The setting is bias-corrected Adam with decoupled weight decay \(\lambda\) and \(\epsilon\) outside the square root, at round \(t\ge1\), from a complete learner state \(z=(\theta,m,v,t-1)\). A feedback message \(y\) produces a **consumed gradient** \(G(y)\in\mathbb R^p\), taken after accumulation, loss normalization and clipping. The coordinatewise update is
\[
\theta^+_i(y)=(1-\eta\lambda)\theta_i-\eta\,\frac{n_i(y)}{d_i(y)+\epsilon},\qquad
n_i(y)=\frac{\beta_1m_i+(1-\beta_1)G_i(y)}{1-\beta_1^t},\qquad
d_i(y)=\sqrt{\frac{\beta_2v_i+(1-\beta_2)G_i(y)^2}{1-\beta_2^t}}.
\]
The other components of the next complete state are \(m^+=\beta_1m+(1-\beta_1)G(y)\) and \(v^+=\beta_2v+(1-\beta_2)G(y)^{\odot2}\).

**Warmth.** For \(v_i>0\), define
\[
\kappa_i(y)=\frac{(1-\beta_2)\,G_i(y)^2}{\beta_2\,v_i},\qquad a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^t}},
\]
and set \(\kappa_i=\infty\) when \(v_i=0\). Then \(d_i(y)=a_i\sqrt{1+\kappa_i(y)}\) and \(v^+_i=\beta_2v_i(1+\kappa_i(y))\). In words, \(\kappa_i\) is the relative perturbation that the new gradient makes to the existing second moment.

**Frozen-denominator surrogate.** Define
\[
\tilde\theta^+_i(y)=(1-\eta\lambda)\theta_i-\eta\,\frac{n_i(y)}{a_i+\epsilon},\qquad
\tilde z^+(y)=(\tilde\theta^+(y),\,m^+(y),\,\beta_2v,\,t).
\]
\(\tilde z^+\) is **affine in \(G(y)\)**: it is one step of momentum SGD with the fixed diagonal preconditioner \(P=\operatorname{diag}(1/(a+\epsilon))\).

The one-boundary response (A1) is \(\psi(y)=\mathbb E_\xi V(z^+(y))\). Here \(V\) is the declared reference learning suffix followed by deployment, with values in \([0,1]\).

## 2. Warm one-step approximation

**Lemma W0 (coordinatewise).** Let \(u_i(y)=|n_i(y)|/(a_i+\epsilon)\). Then
\[
|\theta^+_i(y)-\tilde\theta^+_i(y)|\le\eta\,u_i(y)\,\min\{\kappa_i(y)/2,\;1\}.
\]

*Proof.* The difference equals \(\eta|n_i|\cdot|1/(a_i\sqrt{1+\kappa_i}+\epsilon)-1/(a_i+\epsilon)|\).

- The first form of the bound: this quantity is \(\eta|n_i|\,a_i(\sqrt{1+\kappa_i}-1)/[(a_i+\epsilon)(a_i\sqrt{1+\kappa_i}+\epsilon)]\). Use \(\sqrt{1+\kappa}-1\le\kappa/2\) and \(a_i/(a_i\sqrt{1+\kappa_i}+\epsilon)\le1\).
- The second form: both reciprocals lie in \([0,1/(a_i+\epsilon)]\), so their difference is at most \(1/(a_i+\epsilon)\). \(\square\)

**Proposition W1 (warm Adam is first-order affine in the consumed gradient).** Fix the state \(z\) and a message set \(Y\). Write \(\Delta^\theta(y)=\tilde\theta^+(y)-(1-\eta\lambda)\theta+\eta\beta_1 m/((1-\beta_1^t)(a+\epsilon))\). This is the gradient-dependent part, and it is linear: \(\Delta^\theta(y)=-\eta\,c_1P\,G(y)\) with \(c_1=(1-\beta_1)/(1-\beta_1^t)\). Assume, on a region containing all the states involved:

- (L) \(|V(\theta,m,v)-V(\theta',m,v)|\le\sum_iL_i|\theta_i-\theta'_i|\);
- (S) \(|V(\theta,m,v\odot(1+k))-V(\theta,m,v)|\le L_v\max_i\min(k_i,\bar k)\) for nonnegative \(k\);
- (H) the map \(F(g)=V(\tilde z^+(g))\), where \(\tilde z^+(g)\) is the frozen-denominator next state for consumed gradient \(g\), is twice differentiable on the convex hull of \(\{0\}\cup\{G(y):y\in Y\}\) with \(|s^\top\nabla^2F(g)\,s|\le H_G\|s\|_\infty^2\). Because \(F\) depends on \(g\) only through \(\tilde\theta^+\) (scaled by \(\eta\)) and \(m^+\) (which enters later updates scaled by \(\eta\)), \(H_G\) is typically \(O(\eta^2)\) for a smooth suffix. That is an expectation about the suffix, not part of the assumption.

Then there are constants \(c\) and \(b\), depending on \(z\) but not on \(y\), such that for every \(y\in Y\)
\[
\Bigl|\psi(y)-c-b^\top G(y)\Bigr|
\le \underbrace{\eta\sum_iL_iu_i(y)\min\{\tfrac{\kappa_i(y)}2,1\}}_{\text{normalization}}
+\underbrace{L_v\max_i\min\{\kappa_i(y),\bar k\}}_{\text{second moment}}
+\underbrace{\tfrac12H_G\,\|G(y)\|_\infty^2}_{\text{curvature}}
=: r(y).
\]
Hence, for any two message laws \(P_0\) and \(P_1\) on \(Y\),
\[
\bigl|\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\bigr|\le |b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)|+\mathbb E_{P_0}r+\mathbb E_{P_1}r.
\]
In particular, **equal mean consumed gradients imply an expected-response gap of at most \(\mathbb E_{P_0}r+\mathbb E_{P_1}r\).**

*Proof.* Pass from \(z^+(y)\) to \(\tilde z^+(y)\) in two moves.

1. Replace \(\theta^+\) by \(\tilde\theta^+\). By (L) and Lemma W0, this costs at most the normalization term.
2. Replace \(v^+=\beta_2v\odot(1+\kappa)\) by \(\beta_2v\). By (S), this costs at most the second-moment term.

The remaining function is \(F(G(y))\). Taylor-expand \(F\) around \(g=0\) along the segment to \(G(y)\), with \(c=F(0)\) and \(b=\nabla F(0)\). The integral remainder is at most \(\tfrac12H_G\|G(y)\|_\infty^2\) by (H). The two-law statement follows by integrating and applying the triangle inequality. \(\square\)

The bound can be capped at one. When \(\kappa\to0\) and \(H_G=O(\eta^2)\), the residual is \(O(\eta\kappa+L_v\kappa+\eta^2)\). This is the same first-order information requirement as preconditioned SGD (eq. (35) of the main theory, with \(P\) inserted): **the mean consumed gradient**.

## 3. Cold start separates equal-mean feedback at first order

**Proposition W2 (cold first-order separation).** Take a cold state (\(m=v=0\), \(t=1\)), \(\epsilon\to0\) (the ideal sign map of A2), and the linear utility \(V(\theta)=\tfrac12+w^\top\theta\). For every \(\eta>0\) there are two laws of a scalar consumed gradient with **equal means** (indeed all moments up to any fixed order \(N\) equal, by the construction of `law_scalar`) whose expected responses differ by \(\Theta(\eta)\).

Concrete example: \(P_0\) puts \(G=1\) with probability \(1/4\) and \(G=-1/3\) with probability \(3/4\), so its mean is \(0\); \(P_1\) puts \(G=0\) with probability one. Then \(\mathbb E_{P_0}\Delta\theta=-\eta(\tfrac14-\tfrac34)=\eta/2\) and \(\mathbb E_{P_1}\Delta\theta=0\), so the response gap is \(|w|\eta/2\). With a warm state (\(\kappa\ll1\)), W1 gives a gap of \(O(\eta\kappa+\eta^2)\).

*Proof.* Proposition A2 gives \(\theta_1=\theta_0-\eta\operatorname{sign}(G)\). Substitute. \(\square\)

**Reading.** The first-order information that a feedback law must match **changes with optimizer warmth**:

| Optimizer state | What must match | Example |
|---|---|---|
| Cold Adam | Joint sign law of \(G\) (Prop. A2), magnitudes retained in \(m\) (Prop. A4) | first update of every fresh fine-tuning run |
| Warm Adam, \(\kappa\ll1\) | Mean preconditioned consumed gradient \(P\,\mathbb E G\), to \(O(\eta\kappa+\eta^2)\) | after enough steps or an lr=0 moment warm-up |
| SGD | Mean consumed gradient, to \(O(\eta^2)\) | classical affine/Taylor regime (eq. (35)) |

## 4. How warmth evolves, and a free way to warm the state

If successive consumed gradients have a stationary coordinatewise second moment \(\sigma_i^2\), then \(v_i\approx(1-\beta_2^{\,s})\sigma_i^2\) after \(s\) gradient evaluations. For a fresh gradient \(G\) at evaluation \(s+1\),
\[
\kappa_i\approx\frac{1-\beta_2}{\beta_2(1-\beta_2^{\,s})}\frac{G_i^2}{\sigma_i^2}\;\approx\;\frac{G_i^2/\sigma_i^2}{s}\qquad(s\ll(1-\beta_2)^{-1}=1000).
\]

**Corollary W3 (cold window).** Apply W1 at every boundary of the stopped telescoping identity (Theorem 7, with the transport of (39a) for audited claims). The first-order non-affine contribution then accumulates like \(\sum_{s=1}^{R}\min(1,c/s)\approx c\log R\) for a fresh run. With \(W\) prior gradient evaluations it accumulates like \(\sum_s c/(W+s)\approx c\log(1+R/W)\). The few-shot regime (a few hundred updates of a fresh AdamW state; K=20 Seed-SFT runs 233 updates) is exactly where the cold window is a non-negligible fraction of training.

**Moment warm-up (method hypothesis).** Before the first parameter update, run \(W\) AdamW steps with learning rate exactly 0 on the same \(K\) demonstrations.

- Parameters are unchanged and no new information is used. The cost is \(W\) extra forward/backward passes, counted as compute, not data.
- \(m\), \(v\) and the step counter become warm, so the first real updates are already in the W1 regime.

Predictions that can falsify this:

| Prediction | Test |
|---|---|
| P1 | Update non-additivity \(\nu=\|\Delta\theta_{A\cup B}-(w_A\Delta\theta_A+w_B\Delta\theta_B)\|/\|\Delta\theta_{A\cup B}\|\) on real LoRA scales roughly like \(\min(1,c/(W+s))\), and is near 0 for SGD without clipping |
| P2 | Feedback sources with matched mean preconditioned gradients become interchangeable after warm-up, and cheap first-order proxies (mean gradient, \(\Delta\theta\)) become better predictors of the response (controlled: rerun E6 with warm-up) |
| P3 | Final task performance changes are an empirical matter. Warm-up removes the sign-regime steps, whose effect may be helpful, harmful or neutral; standard LR warmup is the practical control |

## 5. What this is not

- W1 is a local, one-boundary statement. Its constants \(L\), \(L_v\) and \(H\) are properties of the continuation value. They are not estimated by a single training run, and a certificate would still need the transport and coverage terms of (39a).
- The affine-in-\(G\) structure does not make the response affine in the **message** when \(G(y)\) is itself nonlinear in \(y\). Examples are sequence-level teacher forcing, where the prefix changes the Jacobian, and active clipping. The information statistic is the mean **consumed** gradient, which a cheap proxy can estimate only to the extent that gradients are measurable at the visited state.
- Clipping (global norm 1.0 in our recipe) makes the consumed gradient piecewise nonlinear in the raw gradient. The probe reports clip-active fractions, and the theory is stated in consumed gradients.

## 6. Precedents and novelty boundary

- Kingma & Ba (2015): bias correction and the cold-start sign behavior of the first step.
- Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond* (RAdam, ICLR 2020): the adaptive learning rate has large variance at early steps, which motivates warmup or rectification. Balles & Hennig (2018): Adam as sign descent with variance adaptation. Practical warmup of learning rates is standard.
- The lr=0 moment warm-up itself is a simple, plausibly previously used trick. It is not claimed as new.

**Claimed contribution (to be earned):** the information-theoretic statement that the *first-order feedback statistic a distillation method must control* switches from the joint sign law (cold) to the mean preconditioned consumed gradient (warm), with an explicit warmth-controlled residual. A second claim is that this switch changes which cheap proxies can select feedback. Both need the controlled checks and the real-model probe before any paper claim.
