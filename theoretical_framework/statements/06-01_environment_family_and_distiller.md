# Definition 6.1 · Environment family, data experiment, distiller, audited states

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-01_environment_family_and_distiller.md) · [中文](06-01_environment_family_and_distiller_ZH.md)

**In one paragraph.** This definition fixes the objects of Theorem A: a family of environments indexed by an unknown parameter, the data a distiller can obtain (demonstrations and queries), what counts as a distiller, and the learner states at which the effect of feedback is audited. Every result of §6 is stated for these objects.

**Setting used on this page.**

- **Complete learner state** $`z=(\theta,o)`$: parameters and optimizer state ([Definition 1.3](../README.md#s1-3)); a **feedback law** emits the messages the learner consumes, and the actual update maps the state and a message to the next state.
- **Information and distillers** ([Definition 1.2](../README.md#s1-2)): the common initial information (the pretrained student, known rules, the $`K`$ trusted demonstrations) and the transcript of permitted experiments; a distiller maps them, with private randomness, to a feedback protocol.

<a id="s6-1"></a>

**Definition 6.1 (Environment family, data experiment, distiller, audited states).**

- **Environment family.** $`\lbrace E_\theta:\theta\in\Theta\rbrace`$ with $`\Theta\subseteq\mathbb R^p`$ open; the true $`\theta`$ is unknown.
- **Oracle feedback.** $`P^*_\theta=(P^*_{\theta,t})_t`$ is the feedback law one would use if $`\theta`$ were known, for example the teacher's feedback at the states the student visits.
- **Data experiment.**
  - $`K`$ demonstrations $`O_1,\dots,O_K`$ are i.i.d. with law $`P^D_\theta`$.
  - $`n`$ queries may be adaptive: the query point $`X_i`$ in the query space $`\mathcal X_Q`$ is a function of the history and of private randomness, and the outcome is $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$.
  - The law of the whole transcript $`D`$ is $`\mathsf P^{K,n}_\theta`$.
- **Distiller.** Any measurable map from $`(D,U)`$ to a feedback law $`Q=(Q_t)_t`$, where the private randomness $`U`$ has a law that does not depend on $`\theta`$.
  - Replay, rewriting and synthesis are distillers, as long as they acquire no further $`\theta`$-dependent information.
  - The pretrained knowledge of a teacher model is common initial information, the same for every $`\theta`$. A new teacher call that observes a new state is a query.
- **Audited states.** Fixed complete learner states $`z_1,\dots,z_T`$ ([Definition 1.3](../README.md#s1-3)), for example the states the reference training reaches at checkpoints. They do not depend on $`\theta`$.

**What it is.**

- $`\theta`$ indexes everything that differs between the candidate environments; the teacher model's pretrained knowledge is the same for every $`\theta`$.
- A distiller may do anything with the transcript (replay it, rewrite it, synthesize from it) but cannot learn about $`\theta`$ except through the transcript.
- A new teacher call that observes a new state is a query, so it is part of the transcript.

**Role in the framework.**

- It turns "replay, rewriting and synthesis cannot create information" into a statement about all measurable maps of the transcript, which [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3) then quantifies.

**Uses.**

- [Definition 1.2](../README.md#s1-2): information and distillers.
- [Definition 1.3](../README.md#s1-3): complete learner state and mean increment.

**Used by.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm.md#s6-2): builds the functional on these objects.
- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound.md#s6-3): lower bound for every distiller.
- [Theorem 6.5](../statements/06-05_achievability.md#s6-5): plug-in distiller that attains it.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): aliasing and replay.
- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): state-coverage specialization.

**Status.** definition.
