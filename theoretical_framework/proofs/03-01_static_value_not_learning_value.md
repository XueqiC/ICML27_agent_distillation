# Proposition 3.1 · Static value is not learning value

*Proof*

[← Framework](../README.md) · [Statement](../statements/03-01_static_value_not_learning_value.md) · [Proof](03-01_static_value_not_learning_value.md) · [Experiments](../experiments/03-01_static_value_not_learning_value.md) · [中文](03-01_static_value_not_learning_value_ZH.md)

**What is proved here.** Both parts are explicit computations. In (a), the two state laws give action $`A`$ the same mean reward but the teacher different action frequencies; in (b), the two packets have equal means but different variances, which a curved deployment value reads.

**Setting used on this page.**

- *Aliased student* $`\pi_p`$: one context, plays $`A`$ with probability $`p`$. *Frequency distillation:* $`\widehat p`$ = fraction of $`A`$ among $`m`$ teacher demonstrations. *One SGD step* of size $`\eta`$ on the squared loss.

**Statement ([Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1)).** *Construction (a): four-state sources.*

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

**Proof.**

**(a)**

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

## Related work

- Part (b) uses only the first two moments of the packet. First-order influence estimates ([Koh & Liang, 2017](#ref-koh2017understanding); [Pruthi et al., 2020](#ref-pruthi2020estimating)) are linear in the training gradient, so the equal first moments shown above make them equal for the two sources, while the learning values differ by $`\eta^2/8`$.
