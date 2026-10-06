# Mechanism analysis plan: why distilled students fail (pre-registered 2026-10-04, before any distilled-student result)

User request (2026-10-04 16:52): separate the reasons a distilled student fails into at least two testable mechanisms:

- **M1. The knowledge is in the experience, but the student did not absorb it or does not invoke it.**
- **M2. The student needs more explicit decision information**: better teaching, or learning to legally re-observe / query.

This file fixes the populations, measurements, interventions and readout before the first Seed-SFT or baseline
number exists, so the analysis cannot be fitted to the results. Changes after results are appended as dated
amendments, never edited in place.

## 1. Populations (dev = valid_seen 140; valid_unseen stays sealed)

For each analysed student S (Seed-SFT at each K, then each baseline at its dev-selected checkpoint):

- **Gap set G(S)**: dev games where the teacher run succeeds and S fails (greedy). Primary population.
- **Regression set R(S)**: games where the base student succeeds and S fails. Reported separately (a distillation
  side effect, not a missing capability).
- Games where the teacher fails are excluded from the mechanism tests (teacher error is a confound, Section 5).

Greedy decoding is deterministic up to serving noise. Every branch replay first re-runs the unmodified prefix and
checks that S reproduces its own logged action at the branch point (replay check, Section 3, arm I0).

## 2. Failure taxonomy (automatic, auditable)

Computed from the event log; audit fields (PDDL facts, won) are used for labelling only, never in any prompt.
Primary label = first matching rule in this order; all matching rules are also kept.

| Label | Rule |
|---|---|
| F-format | the episode ends with ≥3 consecutive parser errors, or ≥25% of its steps are parser errors or inadmissible |
| F-goal | S places or acts on an object or receptacle whose type does not match the goal (misread goal) |
| F-search | a goal object is never visible in any observation of the episode |
| F-precondition | the goal object was held, but a required transformation or order is wrong (e.g. heat/cool/clean missing, attempted at the wrong receptacle, put before transform) |
| F-loop | the same (action, observation) pair occurs ≥3 times, or a cycle of length ≤4 repeats ≥3 times |
| F-cap | none of the above, 40-step limit reached |

## 3. First critical error and branch interventions

**First critical error (FCE).** For an episode in G(S), T(t) = 1 if the teacher, taking over after S's first t
steps (same real environment, reset + action-prefix replay with consistency checks), succeeds. The FCE is the
first t with T(t) = 1 and T(t+1) = 0, located by a scan at t = 0, 4, 8, … and then refined step by step inside the
first interval where T drops (2 teacher samples per probe point; T = 1 if either succeeds). If T(0) = 0 the game
is moved to "teacher-unsolvable from the start" and excluded.

**Interventions at the FCE state** (before S's erroneous action at step t+1). Each arm changes only the input of that
one step; afterwards S continues on its own under the unchanged harness. Outcome = success of the continuation
(greedy; plus 4 samples at temperature 0.7 for a rescue rate with a confidence interval).

| Arm | Change at the FCE step | What a rescue means |
|---|---|---|
| I0 | none (replay check) | must reproduce the failure; otherwise the episode is flagged as non-deterministic |
| I1 restate | append a deterministic summary of facts already in the visible history (receptacles visited, objects seen where, what is held) | information was present in context but not used |
| I2 re-observe | execute one legal information action in the real environment (`look`, `inventory`, or `examine` of the current receptacle, chosen by a fixed rule) and show its real result; costs one step | the student lacks the habit of legally re-observing |
| I3 teacher advice | the teacher sees the same visible history and writes a short plan/thought without an action; inserted as "Advice:"; S chooses the action | more explicit decision information (better teaching) suffices |
| I4 teacher action | the teacher's action replaces S's action at this step; S continues | upper bound: a correct decision at this step suffices |
| I5 experience in context | insert the most similar seed trajectory (same task type, from the K seeds S was trained on) as a worked example | the knowledge exists in the experience and is usable when in context |

The same arms are run for the base student on the same states where meaningful (I5 on base = pure in-context use of
the seeds; it separates "not absorbed by training" from "not usable at all").

## 4. Knowledge probes and coverage (M1 evidence outside the episode)

- **Coverage.** For each FCE, mark whether some seed trajectory of S's training set demonstrates the needed step
  pattern (e.g. "heat X with microwave 1" after taking X; finding X in a drawer or cabinet). Rule-based, from the
  seed event logs.
- **Probes.** Templated questions built only from the seeds (procedure: "To heat a mug, what must you do after
  taking it?"; location priors: "Where did you find a <obj> in the examples?"), scored against the seed text.
  Run on base and S without environment context. Probe accuracy splits M1 into **M1a not absorbed** (probe fails)
  and **M1b absorbed but not invoked** (probe passes, behaviour fails).
- **Training-set fit.** Teacher-action agreement of S on its own training states (teacher forcing) and on fresh
  rollouts of its training tasks; low fit points to dose/optimisation rather than to either mechanism.

## 5. Confounds checked before any mechanism claim

1. Format: F-format episodes are reported but analysed separately; the parser-error rate is reported per student.
2. Dose: the student is its dev-selected checkpoint; the same analysis is repeated on one earlier and one later
   checkpoint of the same run.
3. Coverage: task types absent from the seeds are reported separately (no M1 claim without coverage).
4. Teacher error: games the teacher fails are excluded; FCE requires T(t) = 1 before the error.
5. Noise: rescues are counted only if greedy and the sampled rate (≥3/4) agree; paired McNemar for student-level
   comparisons; dev140 resolves only about ±8 pp per arm, so arm-level differences under ~10 pp are not claimed.

## 6. Readout (fixed in advance)

Per student: |G|, taxonomy shares, FCE position, and the rescue rate of each arm with a 95% Wilson interval.
Decision table per FCE episode:

| | I5 or I1 rescues | neither rescues |
|---|---|---|
| **I3 or I2 rescues** | both mechanisms contribute | **M2** (needs explicit decision information) |
| **neither rescues** | **M1** (knowledge in experience; M1a/M1b by probe) | only I4 rescues → execution/capacity; none → unresolved |

An M1 label also requires coverage (Section 4). The headline statement is the share of gap episodes in each cell,
per K, with intervals, and how the shares move as K grows.

## 7. Budget

Teacher calls: about 6 probe points × 2 samples × ~10 steps per gap episode for the FCE, plus one call per I3/I4
arm. With ~40–60 gap episodes per student this is a few thousand calls per student, cached, with the usage ledger.
Student branch replays run on our own GPUs.

## Amendment 1 (committed 2026-10-04 22:26 CDT) — taxonomy definitions, before any FCE or intervention result

Made after reviewing the Stage 4a classifier's labelled samples on the real base / K=20 s0 3-pass / teacher dev140 runs
(reports/stage4a_validation.md). Only the descriptive taxonomy (Section 2) changes; populations, FCE, interventions,
confounds and the Section 6 readout are unchanged.

1. **F-format = parser errors only**: ≥3 consecutive parser errors at the end, or ≥25% of steps are parser errors.
   Well-formed but inadmissible commands are not a format failure (reviewed example: "go to drawer 5" repeated while
   already at drawer 5, 80% of steps).
2. **New F-inadmissible**, placed after F-format: ≥25% of steps are inadmissible commands.
3. **F-goal counts executed (legal) actions only**, and excludes the task type's own transformation appliance:
   a legal move of a goal-type object into a receptacle whose type is neither the target receptacle nor, for heat /
   cool / clean tasks, the microwave / fridge / sinkbasin; or a legal heat / cool / clean / use applied to an object of
   the wrong type. Reviewed examples that the old rule mislabelled: an inadmissible "move plate 1 to stoveburner 1"
   (never executed), and "move tomato 3 to microwave 1" on a heat task (a procedure error → F-precondition).

New order: F-format, F-inadmissible, F-goal, F-search, F-precondition, F-loop, F-cap.

## Amendment 2 (committed 2026-10-04 22:35 CDT) — serving noise; I0 becomes the control arm (before any FCE or intervention result)

Measured: the same base model on dev140 under two vLLM server instances (rai GPU1, greedy, 16–32 concurrent requests)
gave 59/140 both times, but only 8/140 trajectories were identical (median first divergence at step 2) and 12 outcomes
flipped (6 each way). Greedy decoding under batched serving is therefore not reproducible per episode.

1. **I0 is a control arm, not only a replay check.** It uses the same continuation protocol as I1–I5 (greedy + 4 samples
   at T = 0.7). An arm's rescue rate is reported next to I0's on the same FCE states, and an arm counts as rescuing only
   in excess of I0 (paired per state: arm rescues and I0 does not). The decision table (Section 6) uses these paired
   excess rescues.
2. Student-level comparisons report the cross-server base-vs-base discordance (12/140 here) as the serving-noise
   reference beside every McNemar test.
3. If a batch-invariant serving mode (vLLM `VLLM_BATCH_INVARIANT=1`) is verified to give identical trajectories across
   instances at acceptable cost, all paper evaluations and branch replays switch to it, all affected numbers are
   re-run under it, and this amendment records the switch.

### Amendment 2 implementation update (2026-10-05): batch-invariant protocol switch

The rai base dev40 pilot with `VLLM_BATCH_INVARIANT=1` reported 40/40 identical trajectories across two server
instances, at approximately 2.8× serving time (read-only artifacts: `reports/runs/eval/bi-pilot-dev40-{1,2}`).
Following the explicit Stage 4b instruction, evaluation and mechanism launchers now default to
`BATCH_INVARIANT=1`. Server and suite provenance record the setting. Paper evaluations and branch replays use
this mode; affected historical numbers require new runs. This engineering change does not re-run experiments,
modify old artifacts, or establish reproducibility for all adapters/hardware.

I0 retains action-divergence metadata but no longer excludes states or suppresses intervention arms. Raw arm
and I0 rescues each require greedy success and at least 3/4 sampled successes. Paired excess rescues require
both complete outcomes on the same state, with arm rescue true and I0 rescue false. Unknown I0 outcomes remain
unknown. Wilson intervals use the paired binary counts, and Section 6 uses those paired excess rescues.
Historical non-batch-invariant base-vs-base discordance (12/140) remains labelled beside McNemar comparisons.

## Addendum A (committed 2026-10-04 23:45 CDT) — gap stratification by environment traits (user request, before any stratified result)

User, 2026-10-04: the purpose is to see honestly what current distillation baselines achieve, what our student can be trained
to, and which gaps specific environment traits amplify or shrink. Every gap (teacher − base, each method − base, teacher −
each method) is therefore also reported per trait bucket. Traits are computed once per dev game from the game files and the
initial audit state (never from any policy's outcome), frozen to a file before the first stratified readout:

| Trait | Definition | Buckets |
|---|---|---|
| task type | ALFWorld template | 6 types; also simple (pick) / transform (heat, cool, clean) / multi-object (two) / look |
| hidden target | every goal-type object starts inside a receptacle that must be opened | yes / no |
| room size | number of receptacles in the initial observation | tertiles over dev140 |
| distractors | goal-type objects in the room beyond the number required | 0 / 1 / ≥2 |
| minimum length | steps of the teacher's shortest successful dev trajectory for that game (unknown if the teacher failed) | tertiles; unknown separate |
| state change | the goal requires heat / cool / clean | yes / no |
| seed coverage | number of the method's K seeds of the same task type | 0 / 1 / ≥2 (per K and support seed) |

Readout per trait bucket: success of each row with Wilson 95% intervals and the paired difference against base and teacher
with exact McNemar p-values. dev140 buckets are small (often 15–50 games), so only differences whose interval excludes zero
AND whose direction repeats across support seeds are stated as findings; everything else is a hypothesis for the sealed test.

**Amendment 2, point 3 executed (committed 2026-10-04 23:52 CDT):** VLLM_BATCH_INVARIANT=1 verified on rai GPU1 — base on dev40: 40/40 identical
trajectories across two server instances (18/40 both); LoRA k20s0-p20 on dev40: 40/40 identical (27/40 both), equivalence gate
passes (error 0.040 vs noise 0.051). All paper evaluations and all branch replays use batch-invariant serving from now on; every
number reported earlier without it (rai ladder, first dev140 rows) is preliminary and is re-run under it.

## Amendment 3 (committed 2026-10-05 00:50 CDT) — intervention point = closest approach, not the budget drop (before any intervention result)

Observed on the first live FCE task (k20s0-p20, look_at_obj_in_light-Bowl-None-DeskLamp-301): teacher takeover succeeds from every
prefix t ≤ 39 and fails only at t = 40, where no turns remain, so Section 3's FCE = 39 marks where the 40-turn budget ran out, not an
error. ALFWorld states are almost always recoverable given turns; students fail by wasting turns. In this episode the teacher needs
~20 more turns from t = 4…20 (the student searches drawers for the lamp), then exactly **1** turn from t = 24 onward (lamp on, bowl
on the dresser: "take bowl 1 from dresser 1" wins), while the student types "look" sixteen times. The informative state is t = 24.

1. **Teacher steps-to-go** D(t) = the fewest turns a successful teacher takeover from prefix t used (over the samples at that probe;
   undefined if none succeeded). Already recorded by every FCE probe.
2. **Intervention state t\*** = the earliest probed prefix attaining the minimum D over the episode (the student's closest approach
   to success), after a dense probe of every prefix in [t_c − 3, t_c] around the earliest coarse minimum t_c (2 samples each).
   Ties → earliest. If no probe succeeded, the episode is "teacher-unsolvable from the student's states" and excluded.
3. Interventions I0–I5 (Section 3 table, amendment 2 pairing) are applied at t\* instead of the budget FCE. The budget FCE, D(t) curves,
   and the student's turns spent after t\* without progress are reported per episode (descriptive).
4. Readout additions: D(0) (teacher distance from the start), D(t\*) (how close the student got), and turns remaining at t\*.

## Amendment 4 (committed 2026-10-05 10:51 CDT) — Section 4 operationalisation (before any coverage or probe result)

Decided by me (not by the implementation). Rationale for replacing Section 4's templated questions with action agreement on the exact
seed states: the question format is not the student's input distribution (a free-text answer can be right while the policy never acts
on it, and scoring free text needs a judge), whereas the seed states where the teacher executed the needed pattern test the trained
policy in its own evaluation prompt. Coverage is decided on any task type because the question is whether the knowledge exists anywhere in
the K seeds; same-type evidence is still reported. The implementation ran a CPU sanity pass that assigned coverage statuses to the 24 run-2 gap
episodes; I had not inspected those statuses when writing this amendment. Implementation details as follows.

Section 4 now uses the shortest successful executed teacher takeover at amendment-3 t* to define needed actions
(ties: lowest sample index). Patterns retain the verb, object type and receptacle/appliance type; instance numbers
are removed using the goals.py type normalization. Coverage requires every non-navigation pattern in the student's
verified training seeds. Same-task-type evidence is preferred for display; both same-type and any-type coverage
are reported, with any-type coverage deciding the gate. Go-to/look/examine/inventory patterns are reported but
excluded from the gate. Unknown action patterns and missing execution evidence remain unknown; navigation-only
takeovers provide no substantive knowledge evidence for M1.

The requested absorption probe replaces the templated questions with greedy student action agreement on exact
logged, teacher-forced seed prefixes, using the shared evaluation prompt. A pattern passes at >= 1/2 agreement
over all matching seed states (same-type results also reported). An episode passes when every non-navigation
needed pattern passes. Incomplete calls remain unknown. The same calls measure training-set fit on all seed
states per task type; this stage performs no fresh rollouts or environment transitions.

M1 attribution requires coverage: M1a = covered and not absorbed; M1b = covered and absorbed. An uncovered M1
rescue cell is reported as knowledge not in experience. In the both-rescue cell, uncovered episodes retain the
M2 signal with the missing-experience qualifier. Rescue cells, I0 pairing, cohort exclusions and denominators
stay unchanged; the readout also reports the coverage-qualified decision table and attribution counts. These
are descriptive operational tests, not proof of general knowledge, matched preconditions, or a scientific finding.

## Amendment 5 (committed 2026-10-05 11:14 CDT) — two coverage levels; the M1 gate uses procedural coverage

Made after seeing the first coverage output (run 2: 2/24 covered, 21 not covered, 1 navigation-only), before seeing any probe or M1/M2
attribution result. Inspection showed that amendment 4's patterns kept the object type for every verb, so e.g. "clean {obj:dishsponge}
with {sinkbasin}" counted as uncovered although the seeds clean other objects in the sinkbasin. Section 4's own examples abstract the
object for procedures ("heat X with microwave 1 after taking X") and keep it only for location knowledge ("finding X in a drawer or
cabinet"). Therefore:

1. **Procedural coverage (decides the M1 gate):** every non-navigation needed step, with the object abstracted to its role (goal object /
   other), i.e. verb + receptacle/appliance type (e.g. "clean {goal} with {sinkbasin}", "take {goal} from {countertop}", "move {goal} to
   {cart}", "use {desklamp}"), is demonstrated in at least one of the student's seeds (any task type).
2. **Instance coverage (reported, stricter):** the amendment-4 patterns with the object type kept.
3. Absorption probes run on the seed states matched at the level that decides coverage (procedural); instance-level agreement is reported
   alongside.

---
Note (2026-10-05 12:05 CDT): the time stamps in the amendment headers were first written from a misread clock (the machine runs on
EDT) and were up to ~1 h off; they are now the git commit times in CDT, which are the authoritative pre-registration evidence
(`git log -- docs/MECHANISM_PLAN.md`). No amendment text changed.

## Amendment 6 (committed 2026-10-06, time = this commit, CDT) — FCE episodes that exhaust the teacher cap (before any intervention result on k5s0-t400k)

The k5s0-t400k run (G = 43) finished FCE for 41 episodes; 2 episodes (tasks 7b4784eef3de0929, fc2e9c60d66f3535) used their full
operational cap of 700 teacher HTTP attempts and stayed `incomplete`, so the launcher (correctly) refused to start interventions.
The caps were operational settings of the launch script, never part of Sections 1–7. Rule, fixed now and applied to every student:
1. An FCE episode that exhausts the per-episode cap is retried once with twice the cap (1,400), in a fresh workspace whose teacher
   cache is seeded read-only from the first run (identical prompts and sampling → identical responses; only the extra calls are new).
   The run cap is raised to 40,000 to make room; total cost is reported.
2. If it is still incomplete it is reported as **FCE-incomplete (cap)**, excluded from the intervention arms, kept in every
   denominator as unresolved, and listed by task type.
3. No other FCE result may change; the readout states which episodes needed the larger cap.
