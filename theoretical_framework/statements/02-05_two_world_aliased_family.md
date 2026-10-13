# Definition 2.5 · Two-world aliased family

*Statement*

[← Framework](../README.md) · [Statement](02-05_two_world_aliased_family.md) · [中文](02-05_two_world_aliased_family_ZH.md)

**In one paragraph.** This definition builds the smallest family in which a perfect teacher's demonstrations are not enough. The teacher is optimal in both of two worlds and behaves identically in them, but the aliased student's best action differs between the worlds, so the student can only find out by acting itself.

**Setting used on this page.**

- [Definition 1.1](../README.md#s1-1): values $`J(\pi)\in[0,1]`$ of deployed policies; [Definition 1.2](../README.md#s1-2): demonstrations and further experiments.
- *Aliased student:* the student has a single context, so it plays the same action distribution whatever the hidden state is.

<a id="s2-5"></a>

**Definition 2.5 (Two-world aliased family).**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects the world. A gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform.
- The student has one aliased context, so it cannot see $`s`$, and makes one decision: action $`A`$ with probability $`p`$, otherwise $`B`$.
- The teacher observes $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$.
- Success probabilities:

| State | Success of $`A`$ | Success of $`B`$ |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

Two interfaces are considered.

- **Passive interface.** The learner sees only teacher demonstrations, possibly with $`s`$ revealed, and randomized post-processing of them.
- **Query interface.** In addition, the learner may reset to $`s=1`$, execute $`A`$ and observe the Bernoulli outcome. Queries may be chosen adaptively. The number of queries $`N`$ is a stopping time with $`\mathbb E_vN<\infty`$. The learner's algorithm and private randomness are the same in both worlds.

Write $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ for the binary KL divergence.

*Values in this family.* Averaging over $`s`$,

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big]=\tfrac34+v\Delta(2p-1).
```

- $`J_v`$ is linear in $`p`$, so $`J_v^\Pi=\tfrac34+\Delta`$, attained at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- The teacher always plays an action with success probability 1. Every other state–action success probability equals $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$ because $`2\Delta\le\tfrac14`$. The teacher is therefore strictly optimal in both worlds.

**What it says.**

- Two worlds differ only in how well each action works in the state where the teacher does not use it.
- The teacher never reveals that difference, because it always takes the action that succeeds for sure.

**Intuition.**

- The teacher's perfect play hides exactly the fact the student needs: which of the two imperfect actions is better for a student that cannot see the state.

**Role in the framework.**

- It is the setting of [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) (passive impossibility, query complexity) and [Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7) (compilation cost).

**Uses.**

- none

**Used by.**

- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): passive impossibility and query complexity.
- [Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7): the cost of compiling the acquired fact.

## Related work

- The family is a two-point construction in the sense of Le Cam's method: two worlds that the passive data cannot separate but whose optimal decisions differ ([Le Cam, 1973](#ref-lecam1973convergence)).
- A teacher that acts on information the student cannot observe, so that imitation marginalizes that information, produces the *imitation gap* of [Weihs et al., 2021](#ref-weihs2021bridging). The family is a minimal instance in which the missing fact is a single sign.
